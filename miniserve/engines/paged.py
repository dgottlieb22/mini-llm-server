"""Stage 4: continuous batching over a paged KV cache.

Same loop as stage 3 (miniserve/engines/continuous.py); the only change is
the storage behind can_admit()/admit()/free(): fixed-size blocks from a pool
sized by kv_budget_tokens, instead of one max_seq_len slot per sequence.
Concurrency is no longer a configured number of slots; it is however many
sequences the pool can hold at their actual lengths.
"""
from __future__ import annotations

from typing import Dict, Optional

from ..model import ModelRunner
from .continuous import ContinuousBatchEngine


class PagedEngine(ContinuousBatchEngine):
    name = "paged"
    stage = 4

    def __init__(self, runner: ModelRunner, kv_budget_tokens: int = 8192, block_size: int = 16,
                 max_seq_len: int = 1024, max_live_tokens: Optional[int] = None) -> None:
        if kv_budget_tokens < block_size or block_size < 1:
            raise ValueError("need kv_budget_tokens >= block_size >= 1")
        self.kv_budget_tokens, self.block_size = kv_budget_tokens, block_size
        # max_slots is only used by the parent for its own stats; the budget is what binds here.
        super().__init__(runner, max_slots=kv_budget_tokens // block_size, max_seq_len=max_seq_len,
                         max_live_tokens=max_live_tokens)
        self._kv_peak_fragmentation = 0

    def _make_core(self):
        return self.runner.make_paged_core(self.kv_budget_tokens, self.block_size, self.max_seq_len)

    def _warmup_count(self, prompt_len: int, gen_len: int) -> int:
        return max(1, self.kv_budget_tokens // (prompt_len + gen_len))

    def _note_kv_usage(self) -> None:
        super()._note_kv_usage()
        resident = sum(s.cache_len + 1 for s in self._running.values())
        allocated = self.core.allocated_tokens([s.handle for s in self._running.values()])
        # Reserved for running sequences but not holding a token: the output
        # reservation plus the unused tail of each last block.
        self._kv_peak_fragmentation = max(self._kv_peak_fragmentation, allocated - resident)

    def _reset_stats(self) -> None:
        super()._reset_stats()
        self._kv_peak_fragmentation = 0

    def _extra_stats(self) -> Dict[str, object]:
        s = dict(self.core.stats())
        s["kv_peak_reserved_but_empty_tokens"] = self._kv_peak_fragmentation
        return s
