"""GPT-2 forward pass with an explicit, slot-indexed KV cache.

From stage 3 on, the engine owns the generation loop, so it needs a model
that can (a) prefill one new sequence into a cache slot and (b) run one
decode step for every slot at once, with sequences of different lengths
and no padding. Hugging Face's forward assumes a padded batch whose cache
grows by concatenation; it cannot drop one sequence from the middle of a
running batch or admit another without rebuilding the whole cache. So the
forward pass lives here: ~120 lines, weights loaded from the Hugging Face
checkpoint, numerically the same computation (checked in tests).

Cache layout: k_cache[layer] is [max_slots, n_head, max_seq_len, head_dim]
(v_cache the same). Slot s owns the contiguous region [s, :, 0:max_seq_len].
Every sequence uses positions 0..L-1 of its own region, so position
embeddings and the attention mask never see padding. This is the naive
allocator: one maximum-length reservation per sequence. Stage 4 replaces it
with block-granular allocation.

Decode computes all slots every step, idle ones included (fed a dummy token
at position 0 and masked to a single key). The alternative, gathering only
the active slots' KV into a contiguous batch, copies the live cache every
step (113 MB per step at 8 x 192 tokens for GPT-2 small in fp32), which on a
CPU costs about as much as the step itself and would confound the
scheduling measurement. Computing idle slots is cheap on this host because
the step cost is nearly flat in batch size (notes/stage0.md); the waste is
counted in the engine's slot_utilization stat and revisited at stage 4,
where paging makes a gather unavoidable.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


class GPT2Core:
    def __init__(self, hf_model, max_slots: int, max_seq_len: int) -> None:
        cfg = hf_model.config
        if cfg.model_type != "gpt2":
            raise ValueError(f"GPT2Core needs a gpt2 checkpoint, got {cfg.model_type}")
        if max_seq_len > cfg.n_positions:
            raise ValueError(f"max_seq_len {max_seq_len} exceeds the model's {cfg.n_positions} positions")
        self.n_layer, self.n_head, self.d_model = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.head_dim = self.d_model // self.n_head
        self.eps = cfg.layer_norm_epsilon
        self.eos_token_id = cfg.eos_token_id
        self.max_slots, self.max_seq_len = max_slots, max_seq_len

        t = hf_model.transformer
        self.dtype = t.wte.weight.dtype
        self.device = t.wte.weight.device
        self.wte = t.wte.weight.detach()          # [vocab, d]; also the LM head (tied)
        self.wpe = t.wpe.weight.detach()          # [n_positions, d]
        self.lnf_w, self.lnf_b = t.ln_f.weight.detach(), t.ln_f.bias.detach()
        self.layers: List[Dict[str, torch.Tensor]] = []
        for blk in t.h:
            # Conv1D weights are [in, out]: y = x @ W + b.
            self.layers.append(dict(
                ln1_w=blk.ln_1.weight.detach(), ln1_b=blk.ln_1.bias.detach(),
                attn_w=blk.attn.c_attn.weight.detach(), attn_b=blk.attn.c_attn.bias.detach(),
                proj_w=blk.attn.c_proj.weight.detach(), proj_b=blk.attn.c_proj.bias.detach(),
                ln2_w=blk.ln_2.weight.detach(), ln2_b=blk.ln_2.bias.detach(),
                fc_w=blk.mlp.c_fc.weight.detach(), fc_b=blk.mlp.c_fc.bias.detach(),
                mlp_w=blk.mlp.c_proj.weight.detach(), mlp_b=blk.mlp.c_proj.bias.detach(),
            ))
        shape = (self.n_layer, max_slots, self.n_head, max_seq_len, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self._all_slots = torch.arange(max_slots, device=self.device)

    # ---- sizes -----------------------------------------------------------

    @property
    def kv_bytes_per_token(self) -> int:
        return self.n_layer * 2 * self.n_head * self.head_dim * self.k_cache.element_size()

    @property
    def cache_bytes(self) -> int:
        return 2 * self.k_cache.numel() * self.k_cache.element_size()

    # ---- building blocks -------------------------------------------------

    def _ln(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.d_model,), w, b, self.eps)

    def _qkv(self, a: torch.Tensor, L: Dict[str, torch.Tensor]):
        """a: [B, T, d] -> q, k, v each [B, n_head, T, head_dim]."""
        B, T, _ = a.shape
        qkv = torch.addmm(L["attn_b"], a.reshape(-1, self.d_model), L["attn_w"]).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _finish_block(self, x: torch.Tensor, att: torch.Tensor, L: Dict[str, torch.Tensor]) -> torch.Tensor:
        """att: [B, n_head, T, head_dim] -> residual add, MLP, residual add."""
        B, _, T, _ = att.shape
        att = att.transpose(1, 2).reshape(B * T, self.d_model)
        x = x + torch.addmm(L["proj_b"], att, L["proj_w"]).view(B, T, self.d_model)
        a = self._ln(x, L["ln2_w"], L["ln2_b"]).reshape(B * T, self.d_model)
        m = F.gelu(torch.addmm(L["fc_b"], a, L["fc_w"]), approximate="tanh")  # GPT-2's gelu_new
        return x + torch.addmm(L["mlp_b"], m, L["mlp_w"]).view(B, T, self.d_model)

    def _logits(self, x_last: torch.Tensor) -> torch.Tensor:
        """x_last: [B, d] -> [B, vocab]."""
        return self._ln(x_last, self.lnf_w, self.lnf_b) @ self.wte.t()

    # ---- the two operations the engine calls -----------------------------

    @torch.inference_mode()
    def prefill(self, slot: int, token_ids: Sequence[int]) -> torch.Tensor:
        """Run the prompt of one sequence into `slot`; return logits at its last position."""
        T = len(token_ids)
        if T > self.max_seq_len:
            raise ValueError(f"prompt of {T} tokens exceeds max_seq_len {self.max_seq_len}")
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        x = (self.wte[ids] + self.wpe[:T]).unsqueeze(0)  # [1, T, d]
        for layer, L in enumerate(self.layers):
            q, k, v = self._qkv(self._ln(x, L["ln1_w"], L["ln1_b"]), L)
            self.k_cache[layer, slot, :, :T] = k[0]
            self.v_cache[layer, slot, :, :T] = v[0]
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = self._finish_block(x, att, L)
        return self._logits(x[:, -1])  # [1, vocab]

    @torch.inference_mode()
    def decode(self, slots: Sequence[int], tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """One step for the active `slots`: token i goes in at positions[i] of slot i.

        Returns logits [len(slots), vocab]. All max_slots slots are computed;
        idle slots get a dummy token at position 0 and attend only to key 0.
        """
        S = self.max_slots
        tok = torch.zeros(S, dtype=torch.long, device=self.device)
        pos = torch.zeros(S, dtype=torch.long, device=self.device)
        slot_idx = torch.as_tensor(list(slots), dtype=torch.long, device=self.device)
        tok[slot_idx] = torch.as_tensor(list(tokens), dtype=torch.long, device=self.device)
        pos[slot_idx] = torch.as_tensor(list(positions), dtype=torch.long, device=self.device)
        if int(pos.max()) >= self.max_seq_len:
            raise ValueError("a sequence has outgrown max_seq_len")
        T = int(pos.max()) + 1
        # key j is visible to slot s iff j <= pos[s]; idle slots see only key 0.
        mask = (torch.arange(T, device=self.device)[None, :] <= pos[:, None]).view(S, 1, 1, T)
        x = (self.wte[tok] + self.wpe[pos]).unsqueeze(1)  # [S, 1, d]
        for layer, L in enumerate(self.layers):
            q, k, v = self._qkv(self._ln(x, L["ln1_w"], L["ln1_b"]), L)  # [S, H, 1, hd]
            self.k_cache[layer][self._all_slots, :, pos] = k[:, :, 0]
            self.v_cache[layer][self._all_slots, :, pos] = v[:, :, 0]
            att = F.scaled_dot_product_attention(
                q, self.k_cache[layer, :, :, :T], self.v_cache[layer, :, :, :T], attn_mask=mask
            )
            x = self._finish_block(x, att, L)
        return self._logits(x[:, 0])[slot_idx]


def argmax_tokens(logits: torch.Tensor, suppress: Sequence[int] = ()) -> List[int]:
    """Greedy pick per row, optionally never choosing `suppress` (used to force length)."""
    if suppress:
        logits = logits.clone()
        logits[:, list(suppress)] = -math.inf
    return logits.argmax(dim=-1).tolist()
