import asyncio

import pytest
import torch

from miniserve.bench.harness import run_workload
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.prefix_cached import PrefixCachedEngine
from miniserve.model import HFModelRunner
from miniserve.prefix_cache import PrefixCachedGPT2Core
from miniserve.request import GenerationRequest


@pytest.fixture(scope="module")
def tiny():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=97, n_positions=64, n_embd=32, n_layer=2, n_head=4,
                     bos_token_id=96, eos_token_id=96)
    return GPT2LMHeadModel(cfg).eval()


def hf_logits(model, ids):
    with torch.inference_mode():
        return model(torch.tensor([ids])).logits[0]


PREFIX = list(range(1, 13))          # 12 tokens = 3 full blocks of 4
A = PREFIX + [50, 51, 52]            # 15 tokens
B = PREFIX + [60, 61, 62, 63, 64]    # 17 tokens, shares 3 blocks with A


def test_second_prompt_hits_the_prefix_and_matches_hf(tiny):
    core = PrefixCachedGPT2Core(tiny, budget_tokens=64, block_size=4, max_seq_len=64)
    ha = core.admit_prompt(A, 4)
    assert ha.cached_tokens == 0
    out_a = core.prefill(ha, A)
    assert torch.allclose(out_a[0], hf_logits(tiny, A)[-1], atol=1e-4)
    assert core.cached_blocks == 3                 # A's three full blocks indexed
    hb = core.admit_prompt(B, 4)
    assert hb.cached_tokens == 12 and hb.blocks[:3] == ha.blocks[:3]
    out_b = core.prefill(hb, B)                    # computes only tokens 12..16
    assert torch.allclose(out_b[0], hf_logits(tiny, B)[-1], atol=1e-4)
    assert core.cached_blocks == 4                 # B's 4th full block added
    # Decoding both afterwards is still exact.
    logits = core.decode([ha, hb], [70, 71], [len(A), len(B)])
    assert torch.allclose(logits[0], hf_logits(tiny, A + [70])[-1], atol=1e-4)
    assert torch.allclose(logits[1], hf_logits(tiny, B + [71])[-1], atol=1e-4)
    assert core.cache_hits_blocks == 3 and core.cache_miss_blocks == 3 + 1


def test_a_full_hit_still_computes_the_last_block(tiny):
    core = PrefixCachedGPT2Core(tiny, budget_tokens=64, block_size=4, max_seq_len=64)
    h1 = core.admit_prompt(PREFIX, 2)
    core.prefill(h1, PREFIX)
    h2 = core.admit_prompt(PREFIX, 2)              # identical prompt
    assert h2.cached_tokens == 8                   # 2 of 3 blocks reused; last one recomputed for its logits
    out = core.prefill(h2, PREFIX)
    assert torch.allclose(out[0], hf_logits(tiny, PREFIX)[-1], atol=1e-4)


def test_refcounts_blocks_survive_their_first_owner_and_get_evicted_under_pressure(tiny):
    core = PrefixCachedGPT2Core(tiny, budget_tokens=32, block_size=4, max_seq_len=64)  # 8 blocks
    ha = core.admit_prompt(A, 1)                   # 4 blocks
    core.prefill(ha, A)
    core.free(ha)
    assert core.allocator.num_free == 5            # 3 indexed blocks stay resident, A's partial one returned
    assert core.cached_blocks == 3
    hb = core.admit_prompt(B, 1)                   # hit on 3, needs ceil(18/4)=5 -> 2 fresh
    assert hb.cached_tokens == 12
    core.prefill(hb, B)
    assert core.allocator.num_free == 3
    # Now a 24-token unrelated prompt needs 6 blocks: only 3 free, 1 evictable (B holds the other 3 + its own).
    big = list(range(70, 94))
    assert not core.can_admit_prompt(big, 1)       # 3 free + 1 evictable < 7
    core.free(hb)
    assert core.cached_blocks == 4 and core.allocator.num_free == 4  # B's partial block returned, 4 indexed
    assert core.can_admit_prompt(big, 1)           # 4 free + 4 evictable >= 7
    hc = core.admit_prompt(big, 1)
    assert core.evictions >= 3
    assert core.allocator.num_free + len(hc.blocks) + core.cached_blocks == core.allocator.num_usable
    core.prefill(hc, big)
    assert torch.allclose(core.decode([hc], [5], [24])[0], hf_logits(tiny, big + [5])[-1], atol=1e-4)


def test_free_never_returns_a_block_someone_else_holds(tiny):
    core = PrefixCachedGPT2Core(tiny, budget_tokens=64, block_size=4, max_seq_len=64)
    ha = core.admit_prompt(A, 1)
    core.prefill(ha, A)
    hb = core.admit_prompt(B, 1)
    core.prefill(hb, B)
    shared = set(ha.blocks[:3])
    core.free(ha)
    assert all(core.refcount[b] == 2 for b in shared)   # index + B
    core.free(hb)
    assert all(core.refcount[b] == 1 for b in shared)   # index only
    assert all(b not in core.allocator._free for b in shared)


def test_shared_blocks_are_counted_once_and_warmup_leaves_no_trace(tiny):
    core = PrefixCachedGPT2Core(tiny, budget_tokens=64, block_size=4, max_seq_len=64)
    ha = core.admit_prompt(A, 1)
    core.prefill(ha, A)
    hb = core.admit_prompt(B, 1)
    core.prefill(hb, B)
    # A: 4 blocks, B: 5 blocks, 3 shared -> 6 physical blocks = 24 tokens
    assert core.allocated_tokens([ha, hb]) == 24
    core.free(ha)
    core.free(hb)
    assert core.cache_hits_blocks == 3
    core.clear_cache()
    core.reset_stats()
    assert core.cached_blocks == 0 and core.cache_hits_blocks == 0
    assert core.allocator.num_free == core.allocator.num_usable


def _spec(prefix, **over):
    base = dict(
        name="t", num_requests=8, prompt_len=LengthSpec("uniform", low=20, high=28),
        gen_len=LengthSpec("fixed", value=3), arrival="instant", seed=5, shared_prefix_len=prefix, vocab_size=90,
    )
    base.update(over)
    return WorkloadSpec(**base)


@pytest.mark.model
def test_engine_outputs_match_without_caching_on_a_shared_prefix_workload(tiny):
    from miniserve.engines.paged import PagedEngine

    runner = HFModelRunner.__new__(HFModelRunner)
    # Hand-build a runner around the tiny model so the real cores run in the engine.
    runner.model, runner.config, runner.name, runner.device = tiny, tiny.config, "tiny", "cpu"
    runner.eos_token_id, runner.vocab_size, runner.dtype = tiny.config.eos_token_id, 97, torch.float32
    spec = _spec(prefix=16)
    _, plain = asyncio.run(run_workload(PagedEngine(runner, kv_budget_tokens=256, block_size=4, max_seq_len=64),
                                        spec, warmup=False))
    cached_engine = PrefixCachedEngine(runner, kv_budget_tokens=256, block_size=4, max_seq_len=64)
    _, cached = asyncio.run(run_workload(cached_engine, spec, warmup=False))
    for a, b in zip(plain, cached):
        assert a.result.output_ids == b.result.output_ids
    s = cached_engine.stats()
    assert s["cached_prompt_tokens"] > 0 and s["prefix_hit_blocks"] > 0
    assert cached_engine.core.allocator.num_free + cached_engine.core.cached_blocks == cached_engine.core.allocator.num_usable
