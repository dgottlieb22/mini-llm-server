"""Prefix caching on the paged KV cache.

A KV block's contents are a pure function of the tokens in it and every
token before it. So a full block of prompt tokens can be named by a hash of
(hash of the previous block, the block's tokens), and two prompts that share
a prefix share those blocks: the second request points its block table at
the first request's physical blocks and prefills only what comes after.

Why this needs the paged design: the shared blocks are plain physical
blocks with a reference count, and a sequence's block table can mix shared
blocks with its own. The slot core could not express that, since a slot is
one contiguous region owned by one sequence.

Rules:
- Only full blocks are cached (a partial block's contents depend on tokens
  the next request may not share). Only prompt blocks are cached; blocks a
  sequence fills while decoding are its own (they could be cached too, as
  vLLM does, but no workload here would hit them).
- A cached block has a refcount: the sequences whose tables hold it plus
  one if it is in the cache index. Freeing a sequence decrements its blocks;
  a block at refcount zero goes back to the allocator. A block that is
  indexed but no longer used by any sequence stays resident (refcount 1) as
  a hit for later requests, until the pool runs low: then the eviction
  policy drops the least recently used cached block(s) to make room.
- Admission reserves prompt + output as before, minus the blocks a cache
  hit would provide. The hit is computed at can_admit() time and locked
  (refcounted) at admit() time, so a block cannot be evicted between the
  decision and the prefill.

Like vLLM's "automatic prefix caching", this is transparent to the client:
nothing in the request says which prefix is shared. Unlike vLLM, the hash
is Python's over tuples, in a dict, which is fine at this scale.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .paged import BlockTable, PagedGPT2Core


class PrefixCachedGPT2Core(PagedGPT2Core):
    def __init__(self, hf_model, budget_tokens: int, block_size: int = 16, max_seq_len: int = 1024) -> None:
        super().__init__(hf_model, budget_tokens, block_size, max_seq_len)
        self.refcount: Dict[int, int] = {}                 # physical block -> refs
        self.block_hash: Dict[int, int] = {}               # physical block -> its content hash
        self.index: "OrderedDict[int, int]" = OrderedDict()  # content hash -> physical block, LRU order
        self.cache_hits_blocks = 0
        self.cache_miss_blocks = 0
        self.evictions = 0

    # ---- hashing ---------------------------------------------------------

    def _prefix_hashes(self, token_ids: Sequence[int]) -> List[int]:
        """Content hash of each FULL block of the prompt, chained."""
        out: List[int] = []
        h = 0
        for start in range(0, len(token_ids) - len(token_ids) % self.block_size, self.block_size):
            h = hash((h, tuple(token_ids[start:start + self.block_size])))
            out.append(h)
        return out

    def _lookup(self, token_ids: Sequence[int]) -> Tuple[List[int], int]:
        """(physical blocks for the longest cached prefix, number of cached tokens)."""
        hit: List[int] = []
        for h in self._prefix_hashes(token_ids):
            b = self.index.get(h)
            if b is None:
                break
            hit.append(b)
        # Never let the whole prompt hit: the last position's logits are what
        # prefill returns, so at least one token must be computed.
        if hit and len(hit) * self.block_size >= len(token_ids):
            hit.pop()
        return hit, len(hit) * self.block_size

    # ---- allocation with sharing -----------------------------------------

    def _free_after_eviction(self, need: int) -> bool:
        """Evict unreferenced cached blocks (LRU) until `need` blocks are free."""
        while self.allocator.num_free < need:
            victim = None
            for h, b in self.index.items():  # oldest first
                if self.refcount.get(b, 0) == 1:  # held only by the index
                    victim = (h, b)
                    break
            if victim is None:
                return False
            h, b = victim
            del self.index[h]
            del self.refcount[b]
            del self.block_hash[b]
            self.allocator.free([b])
            self.evictions += 1
        return True

    def _evictable(self) -> int:
        return sum(1 for b in self.index.values() if self.refcount.get(b, 0) == 1)

    def blocks_needed_for(self, token_ids: Sequence[int], max_new_tokens: int) -> int:
        _, cached = self._lookup(token_ids)
        return self.allocator.blocks_needed(len(token_ids) + max_new_tokens) - cached // self.block_size

    def can_admit_prompt(self, token_ids: Sequence[int], max_new_tokens: int) -> bool:
        if not self.fits(len(token_ids), max_new_tokens):
            return False
        need = self.blocks_needed_for(token_ids, max_new_tokens)
        return need <= self.allocator.num_free + self._evictable()

    def admit_prompt(self, token_ids: Sequence[int], max_new_tokens: int) -> BlockTable:
        hit, cached = self._lookup(token_ids)
        need = self.allocator.blocks_needed(len(token_ids) + max_new_tokens) - len(hit)
        if not self._free_after_eviction(need):
            raise RuntimeError("admit_prompt() called without can_admit_prompt()")
        for b in hit:
            self.refcount[b] += 1
            self.index.move_to_end(self.block_hash[b])  # recently used
        fresh = self.allocator.allocate(need)
        table = BlockTable(hit + fresh, self.block_size)
        table.cached_tokens = cached  # type: ignore[attr-defined]
        self.cache_hits_blocks += len(hit)
        self.cache_miss_blocks += len(self._prefix_hashes(token_ids)) - len(hit)
        return table

    def free(self, handle: BlockTable) -> None:
        to_allocator = []
        for b in handle.blocks:
            if b in self.refcount:
                self.refcount[b] -= 1
                if self.refcount[b] == 0:  # not indexed and no sequence: gone
                    del self.refcount[b]
                    to_allocator.append(b)
            else:
                to_allocator.append(b)
        self.allocator.free(to_allocator)
        handle.blocks = []

    def allocated_tokens(self, handles: Sequence[BlockTable]) -> int:
        """Physical blocks held by these sequences, shared blocks counted once."""
        blocks = set()
        for h in handles:
            blocks.update(h.blocks)
        return len(blocks) * self.block_size

    def reset_stats(self) -> None:
        super().reset_stats()
        self.cache_hits_blocks = self.cache_miss_blocks = self.evictions = 0

    def clear_cache(self) -> None:
        """Drop every indexed block no sequence holds (used after warm-up)."""
        for h, b in list(self.index.items()):
            if self.refcount.get(b, 0) == 1:
                del self.index[h]
                del self.refcount[b]
                del self.block_hash[b]
                self.allocator.free([b])

    def _register(self, handle: BlockTable, token_ids: Sequence[int]) -> None:
        """After a prefill, index this prompt's newly computed full blocks."""
        hashes = self._prefix_hashes(token_ids)
        first_new = getattr(handle, "cached_tokens", 0) // self.block_size
        for i in range(first_new, len(hashes)):
            b = handle.blocks[i]
            if hashes[i] in self.index:
                continue  # a concurrent prefill registered the same content first
            self.index[hashes[i]] = b
            self.block_hash[b] = hashes[i]
            self.refcount[b] = 2  # the index and this sequence
        # Blocks that were hits already carry their refcount from admit.

    # ---- the engine still calls the generic names ------------------------
    # can_admit/admit take lengths; with caching, admission depends on the
    # tokens. The engine detects `can_admit_prompt` and uses it.

    @property
    def cached_blocks(self) -> int:
        return len(self.index)

    # ---- prefill only the uncached tail -----------------------------------

    @torch.inference_mode()
    def prefill(self, handle: BlockTable, token_ids: Sequence[int]) -> torch.Tensor:
        cached = getattr(handle, "cached_tokens", 0)
        if cached == 0:
            out = super().prefill(handle, token_ids)
            self._register(handle, token_ids)
            return out
        # The shared blocks hold K/V for positions [0, cached). Compute the
        # tail [cached, T) with attention over the shared prefix + itself.
        T = len(token_ids)
        tail = torch.as_tensor(token_ids[cached:], dtype=torch.long, device=self.device)
        n_tail = T - cached
        x = (self.wte[tail] + self.wpe[cached:T]).unsqueeze(0)  # [1, n_tail, d]
        nb_total = self.allocator.blocks_needed(T)
        nb_cached = cached // self.block_size
        new_blocks = torch.as_tensor(handle.blocks[nb_cached:nb_total], dtype=torch.long, device=self.device)
        prefix_blocks = torch.as_tensor(handle.blocks[:nb_cached], dtype=torch.long, device=self.device)
        pad = nb_total * self.block_size - T
        # Keys visible to tail position i (absolute cached+i): all of [0, cached+i].
        total_cols = nb_total * self.block_size
        cols = torch.arange(total_cols, device=self.device)
        rows = torch.arange(cached, T, device=self.device)
        mask = (cols[None, :] <= rows[:, None]).view(1, 1, n_tail, total_cols)
        for layer, L in enumerate(self.layers):
            q, k, v = self._qkv(self._ln(x, L["ln1_w"], L["ln1_b"]), L)  # [1, H, n_tail, hd]
            for cache, t in ((self.k_cache, k), (self.v_cache, v)):
                tt = t[0].transpose(0, 1)  # [n_tail, H, hd]
                if pad:
                    tt = torch.nn.functional.pad(tt, (0, 0, 0, 0, 0, pad))
                cache[layer][new_blocks] = tt.reshape(nb_total - nb_cached, self.block_size, self.n_head, self.head_dim)
            # Attention over prefix blocks + new blocks, in block order = position order.
            all_blocks = torch.cat([prefix_blocks, new_blocks])
            K = self.k_cache[layer][all_blocks].view(1, total_cols, self.n_head, self.head_dim).transpose(1, 2)
            V = self.v_cache[layer][all_blocks].view(1, total_cols, self.n_head, self.head_dim).transpose(1, 2)
            att = torch.nn.functional.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            x = self._finish_block(x, att, L)
        self._register(handle, token_ids)
        return self._logits(x[:, -1])

    def prefill_batch(self, handles: Sequence[BlockTable], prompts: Sequence[Sequence[int]]) -> torch.Tensor:
        """Batched prefill is only used for groups with no cache hits; a hit prefills alone."""
        if any(getattr(h, "cached_tokens", 0) for h in handles):
            return torch.cat([self.prefill(h, p) for h, p in zip(handles, prompts)])
        out = super().prefill_batch(handles, prompts)
        for h, p in zip(handles, prompts):
            self._register(h, p)
        return out

    def stats(self) -> Dict[str, object]:
        s = super().stats()
        s.update(
            prefix_cache_blocks=len(self.index),
            prefix_hit_blocks=self.cache_hits_blocks,
            prefix_miss_blocks=self.cache_miss_blocks,
            prefix_evictions=self.evictions,
        )
        return s
