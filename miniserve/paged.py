"""Paged KV cache: block-granular allocation behind the same core interface.

The stage 3 core reserves max_seq_len positions per sequence; on the
workloads measured there, 70-93% of the reserved cache was empty at the
busiest moment, and the reservation was what capped concurrency. Here the
cache is a pool of fixed-size blocks (block_size tokens each). A sequence
holds a block table (its logical block i lives in physical block
table[i]) and only ever holds enough blocks for the tokens it has stored,
rounded up to a block. A finished sequence's blocks go back to the pool for
anyone.

The analogy to virtual memory holds for: the block table is a page table;
the sequence's token index is a virtual address; a physical block is a
frame; the only waste is inside a sequence's last block (internal
fragmentation, at most block_size - 1 tokens per sequence). It breaks in
two places: there is no address translation hardware, so every attention
step has to gather a sequence's blocks itself (below), and nothing is ever
paged out to a slower tier unless a scheduler adds preemption.

Admission policy here is conservative: a request is admitted only if the
pool has blocks for its whole prompt plus its whole requested output, so a
running sequence can never run out of blocks. That relies on knowing the
output length, which this benchmark fixes and a real server only has as an
upper bound (max_tokens). The optimistic policy, admit on the prompt alone
and grow block by block, preempting when the pool is empty, is a later
stage; with it, block-granular allocation also stops needing the output
length up front.

Attention over blocks. A sequence's blocks are not contiguous, and torch's
attention wants [B, heads, T, head_dim] per batch. So each decode step
gathers every active sequence's blocks into a padded contiguous batch
(index_select on the block dimension), then runs the same masked attention
as the slot core. That copy is the live KV of the batch, every step. On a
GPU it is cheap next to the matmuls and vLLM avoids even that with a kernel
that reads the block table directly; on this CPU it is a cost the stage 4
notes measure. The engine computes only active sequences here (there is no
slot array to fill), so the step cost scales with how many are running.

Block 0 is reserved as a zero block that pads short block tables; masked
positions never read it, but the gather needs a valid index.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .gpt2 import GPT2Core


class BlockAllocator:
    """A free list of physical block ids. Block 0 is never handed out."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 2 or block_size < 1:
            raise ValueError("need num_blocks >= 2 (block 0 is reserved) and block_size >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: List[int] = list(range(num_blocks - 1, 0, -1))  # 1..num_blocks-1
        self.allocations = 0
        self.frees = 0

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_usable(self) -> int:
        return self.num_blocks - 1

    def blocks_needed(self, tokens: int) -> int:
        return -(-tokens // self.block_size)  # ceil

    def allocate(self, n: int) -> List[int]:
        if n > len(self._free):
            raise RuntimeError(f"out of blocks: need {n}, have {len(self._free)}")
        out = [self._free.pop() for _ in range(n)]
        self.allocations += n
        return out

    def free(self, blocks: Sequence[int]) -> None:
        for b in blocks:
            if b == 0:
                raise ValueError("block 0 is reserved")
            self._free.append(b)
        self.frees += len(blocks)


class BlockTable:
    """One sequence's logical-to-physical block mapping."""

    def __init__(self, blocks: List[int], block_size: int) -> None:
        self.blocks = blocks
        self.block_size = block_size

    @property
    def capacity(self) -> int:
        return len(self.blocks) * self.block_size

    def physical(self, position: int):
        """(block id, offset) of a token position."""
        return self.blocks[position // self.block_size], position % self.block_size


class PagedGPT2Core(GPT2Core):
    """GPT2Core with block-granular storage. Handles are BlockTables."""

    def __init__(self, hf_model, budget_tokens: int, block_size: int = 16, max_seq_len: int = 1024) -> None:
        # Let the parent build weights and sizes with a 1-slot cache, then replace the cache.
        super().__init__(hf_model, max_slots=1, max_seq_len=max_seq_len)
        if budget_tokens < block_size:
            raise ValueError("budget smaller than one block")
        self.block_size = block_size
        self.num_blocks = budget_tokens // block_size + 1  # +1 for the reserved zero block
        self.allocator = BlockAllocator(self.num_blocks, block_size)
        # Token-major inside a block: [block, token, head, head_dim]. Gathering
        # a block table then gives [B, nb, block, head, hd], which VIEWS as
        # [B, nb*block, head, hd] and transposes (a stride change, no copy)
        # into attention's [B, head, T, hd]. The first version stored
        # [block, head, token, hd] and needed a second copy to reorder;
        # scripts/probe_paged_step.py has both numbers.
        shape = (self.n_layer, self.num_blocks, block_size, self.n_head, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        del self._free_slots, self._all_slots
        # Reusable destination for the per-step gather. A fresh tensor per
        # call is page-faulted by the allocator once it passes about 32 MB
        # (glibc's mmap threshold; jemalloc's cliff is lower), which turned a
        # 90 us gather into 5 ms: scripts/probe_gather_kernels.py.
        self._gather_rows = 0
        self._gk = self._gv = None

    def _gather_buffers(self, rows: int):
        if rows > self._gather_rows:
            shape = (rows, self.block_size, self.n_head, self.head_dim)
            self._gk = torch.empty(shape, dtype=self.dtype, device=self.device)
            self._gv = torch.empty(shape, dtype=self.dtype, device=self.device)
            self._gather_rows = rows
        return self._gk[:rows], self._gv[:rows]

    # ---- allocation ------------------------------------------------------

    @property
    def budget_tokens(self) -> int:
        return self.allocator.num_usable * self.block_size

    def fits(self, prompt_len: int, max_new_tokens: int) -> bool:
        total = prompt_len + max_new_tokens
        return total <= self.max_seq_len and self.allocator.blocks_needed(total) <= self.allocator.num_usable

    def can_admit(self, prompt_len: int, max_new_tokens: int) -> bool:
        return self.fits(prompt_len, max_new_tokens) and \
            self.allocator.blocks_needed(prompt_len + max_new_tokens) <= self.allocator.num_free

    def admit(self, prompt_len: int, max_new_tokens: int) -> BlockTable:
        if not self.can_admit(prompt_len, max_new_tokens):
            raise RuntimeError("admit() called without can_admit()")
        n = self.allocator.blocks_needed(prompt_len + max_new_tokens)
        return BlockTable(self.allocator.allocate(n), self.block_size)

    def free(self, handle: BlockTable) -> None:
        self.allocator.free(handle.blocks)
        handle.blocks = []

    def allocated_tokens(self, handles: Sequence[BlockTable]) -> int:
        return sum(h.capacity for h in handles)

    # ---- optimistic (grow-as-you-go) allocation, stage 7 ------------------

    def admit_prompt_only(self, prompt_len: int) -> Optional[BlockTable]:
        """Reserve blocks for the prompt plus one decode position, or None if the pool cannot."""
        n = self.allocator.blocks_needed(prompt_len + 1)
        if n > self.allocator.num_free:
            return None
        return BlockTable(self.allocator.allocate(n), self.block_size)

    def needs_block(self, handle: BlockTable, next_position: int) -> bool:
        """Will writing a token at next_position overflow the table?"""
        return next_position >= handle.capacity

    def grow(self, handle: BlockTable) -> bool:
        """Append one block; False if the pool is empty."""
        if self.allocator.num_free < 1:
            return False
        handle.blocks.extend(self.allocator.allocate(1))
        return True

    @property
    def free_blocks(self) -> int:
        return self.allocator.num_free

    @property
    def cache_bytes(self) -> int:
        return 2 * self.k_cache.numel() * self.k_cache.element_size()

    # ---- model calls -----------------------------------------------------

    @torch.inference_mode()
    def prefill(self, handle: BlockTable, token_ids: Sequence[int]) -> torch.Tensor:
        T = len(token_ids)
        if T > handle.capacity:
            raise ValueError(f"prompt of {T} tokens exceeds the block table's capacity {handle.capacity}")
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        x = (self.wte[ids] + self.wpe[:T]).unsqueeze(0)  # [1, T, d]
        nb = self.allocator.blocks_needed(T)
        blocks = torch.as_tensor(handle.blocks[:nb], dtype=torch.long, device=self.device)
        pad = nb * self.block_size - T
        for layer, L in enumerate(self.layers):
            q, k, v = self._qkv(self._ln(x, L["ln1_w"], L["ln1_b"]), L)  # [1, H, T, hd]
            # Scatter K/V into this sequence's blocks: [H, T, hd] -> [nb, block, H, hd].
            for cache, t in ((self.k_cache, k), (self.v_cache, v)):
                tt = t[0].transpose(0, 1)  # [T, H, hd]
                if pad:
                    tt = F.pad(tt, (0, 0, 0, 0, 0, pad))
                cache[layer][blocks] = tt.reshape(nb, self.block_size, self.n_head, self.head_dim)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = self._finish_block(x, att, L)
        return self._logits(x[:, -1])

    @torch.inference_mode()
    def prefill_batch(self, handles: Sequence[BlockTable], prompts: Sequence[Sequence[int]]) -> torch.Tensor:
        """Several sequences in one right-padded forward (see GPT2Core.prefill_batch)."""
        B = len(handles)
        lens = [len(p) for p in prompts]
        P = max(lens)
        for h, n in zip(handles, lens):
            if n > h.capacity:
                raise ValueError(f"prompt of {n} tokens exceeds the block table's capacity {h.capacity}")
        ids = torch.zeros((B, P), dtype=torch.long, device=self.device)
        for i, p in enumerate(prompts):
            ids[i, :len(p)] = torch.as_tensor(p, dtype=torch.long, device=self.device)
        x = self.wte[ids] + self.wpe[:P].unsqueeze(0)
        plans = []  # per row: (block ids tensor, number of blocks, pad)
        for h, n in zip(handles, lens):
            nb = self.allocator.blocks_needed(n)
            plans.append((torch.as_tensor(h.blocks[:nb], dtype=torch.long, device=self.device), nb,
                          nb * self.block_size - n))
        for layer, L in enumerate(self.layers):
            q, k, v = self._qkv(self._ln(x, L["ln1_w"], L["ln1_b"]), L)  # [B, H, P, hd]
            for i, (blocks, nb, pad) in enumerate(plans):
                n = lens[i]
                for cache, t in ((self.k_cache, k), (self.v_cache, v)):
                    tt = t[i, :, :n].transpose(0, 1)  # [n, H, hd]
                    if pad:
                        tt = F.pad(tt, (0, 0, 0, 0, 0, pad))
                    cache[layer][blocks] = tt.reshape(nb, self.block_size, self.n_head, self.head_dim)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = self._finish_block(x, att, L)
        last = torch.as_tensor([n - 1 for n in lens], device=self.device)
        return self._logits(x[torch.arange(B, device=self.device), last])

    @torch.inference_mode()
    def decode(self, handles: Sequence[BlockTable], tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """One step for the active sequences only. Returns logits [len(handles), vocab]."""
        B = len(handles)
        pos = torch.as_tensor(list(positions), dtype=torch.long, device=self.device)
        tok = torch.as_tensor(list(tokens), dtype=torch.long, device=self.device)
        if int(pos.max()) >= self.max_seq_len:
            raise ValueError("a sequence has outgrown max_seq_len")
        # Block tables padded to the longest, with the zero block as filler.
        nb_each = [self.allocator.blocks_needed(int(p) + 1) for p in positions]
        max_nb = max(nb_each)
        table = torch.zeros((B, max_nb), dtype=torch.long, device=self.device)
        for i, (h, nb) in enumerate(zip(handles, nb_each)):
            if nb > len(h.blocks):
                raise ValueError("sequence outgrew its block table")
            table[i, :nb] = torch.as_tensor(h.blocks[:nb], dtype=torch.long, device=self.device)
        flat_table = table.reshape(-1)
        write_block = torch.as_tensor([h.blocks[int(p) // self.block_size] for h, p in zip(handles, positions)],
                                      dtype=torch.long, device=self.device)
        write_off = pos % self.block_size
        T = max_nb * self.block_size
        self.attn_positions_computed += B * T
        self.attn_positions_live += sum(int(p) + 1 for p in positions)
        mask = (torch.arange(T, device=self.device)[None, :] <= pos[:, None]).view(B, 1, 1, T)
        x = (self.wte[tok] + self.wpe[pos]).unsqueeze(1)  # [B, 1, d]
        gk, gv = self._gather_buffers(B * max_nb)
        for layer, L in enumerate(self.layers):
            q, k, v = self._qkv(self._ln(x, L["ln1_w"], L["ln1_b"]), L)  # [B, H, 1, hd]
            self.k_cache[layer][write_block, write_off] = k[:, :, 0]
            self.v_cache[layer][write_block, write_off] = v[:, :, 0]
            # The gather: one copy of the batch's live KV per layer into a reused buffer, then free views.
            torch.index_select(self.k_cache[layer], 0, flat_table, out=gk)
            torch.index_select(self.v_cache[layer], 0, flat_table, out=gv)
            K = gk.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            V = gv.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            x = self._finish_block(x, att, L)
        return self._logits(x[:, 0])

    def reset_stats(self) -> None:
        self.attn_positions_computed = self.attn_positions_live = 0
        self.allocator.allocations = self.allocator.frees = 0

    def stats(self) -> Dict[str, object]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.allocator.num_usable,
            "free_blocks": self.allocator.num_free,
            "block_allocations": self.allocator.allocations,
            "block_frees": self.allocator.frees,
            "attn_positions_computed": self.attn_positions_computed,
            "attn_positions_live": self.attn_positions_live,
        }
