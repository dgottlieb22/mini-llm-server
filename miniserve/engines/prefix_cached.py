"""Stage 6: paged engine with prefix caching.

The loop and the allocator are stage 4/5's; the core (miniserve/prefix_cache.py)
indexes full prompt blocks by content and lets a new request reuse a resident
prefix instead of recomputing it.
"""
from __future__ import annotations

from .paged import PagedEngine


class PrefixCachedEngine(PagedEngine):
    name = "prefix_cached"
    stage = 6

    def _make_core(self):
        return self.runner.make_prefix_cached_core(self.kv_budget_tokens, self.block_size, self.max_seq_len)
