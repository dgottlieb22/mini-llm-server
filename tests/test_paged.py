import asyncio

import pytest
import torch

from conftest import hf_logits
from miniserve.bench.harness import run_workload
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.continuous import ContinuousBatchEngine
from miniserve.engines.paged import PagedEngine
from miniserve.model import FakeModelRunner
from miniserve.paged import BlockAllocator, BlockTable, PagedGPT2Core
from miniserve.request import GenerationRequest


# ---- allocator -------------------------------------------------------------

def test_allocator_hands_out_every_block_but_zero_and_takes_them_back():
    a = BlockAllocator(num_blocks=5, block_size=4)
    assert a.num_usable == 4 and a.num_free == 4
    got = a.allocate(4)
    assert sorted(got) == [1, 2, 3, 4] and a.num_free == 0
    with pytest.raises(RuntimeError):
        a.allocate(1)
    a.free(got[:2])
    assert a.num_free == 2
    with pytest.raises(ValueError):
        a.free([0])
    assert a.blocks_needed(1) == 1 and a.blocks_needed(4) == 1 and a.blocks_needed(5) == 2


def test_block_table_addressing():
    t = BlockTable([7, 3, 9], block_size=4)
    assert t.capacity == 12
    assert t.physical(0) == (7, 0) and t.physical(5) == (3, 1) and t.physical(11) == (9, 3)


# ---- paged core against HF ---------------------------------------------------


def test_paged_core_matches_hf_across_block_boundaries(tiny):
    core = PagedGPT2Core(tiny, budget_tokens=64, block_size=4, max_seq_len=64)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]   # 9 tokens: 3 blocks, last one partial
    b = [40, 41, 42]                  # 3 tokens: 1 block
    ha = core.admit(len(a), 6)
    hb = core.admit(len(b), 6)
    assert len(ha.blocks) == 4 and len(hb.blocks) == 3  # ceil(15/4), ceil(9/4)
    out_a = core.prefill(ha, a)
    out_b = core.prefill(hb, b)
    assert torch.allclose(out_a[0], hf_logits(tiny, a)[-1], atol=1e-4)
    assert torch.allclose(out_b[0], hf_logits(tiny, b)[-1], atol=1e-4)
    seqs = {id(ha): list(a), id(hb): list(b)}
    nxt = {id(ha): 11, id(hb): 12}
    for step in range(5):  # crosses block boundaries for both sequences
        handles = [ha, hb]
        tokens = [nxt[id(ha)], nxt[id(hb)]]
        positions = [len(seqs[id(ha)]), len(seqs[id(hb)])]
        logits = core.decode(handles, tokens, positions)
        for i, h in enumerate(handles):
            seqs[id(h)].append(tokens[i])
            assert torch.allclose(logits[i], hf_logits(tiny, seqs[id(h)])[-1], atol=1e-4), f"step {step}"
        nxt = {id(ha): 20 + step, id(hb): 50 + step}
    core.free(ha)
    core.free(hb)
    assert core.allocator.num_free == core.allocator.num_usable


def test_paged_core_reuses_freed_blocks_cleanly(tiny):
    core = PagedGPT2Core(tiny, budget_tokens=16, block_size=4, max_seq_len=64)
    h1 = core.admit(8, 8)            # takes all 4 blocks
    assert not core.can_admit(1, 1)
    core.prefill(h1, list(range(1, 9)))
    core.free(h1)
    h2 = core.admit(3, 1)
    out = core.prefill(h2, [5, 6, 7])
    assert torch.allclose(out[0], hf_logits(tiny, [5, 6, 7])[-1], atol=1e-4)
    logits = core.decode([h2], [8], [3])
    assert torch.allclose(logits[0], hf_logits(tiny, [5, 6, 7, 8])[-1], atol=1e-4)


def test_paged_core_fits_and_can_admit(tiny):
    core = PagedGPT2Core(tiny, budget_tokens=32, block_size=8, max_seq_len=64)
    assert core.budget_tokens == 32
    assert core.fits(16, 16) and not core.fits(17, 16)      # 33 tokens needs 5 blocks of 4
    assert not core.fits(40, 30)                             # beyond max_seq_len too
    h = core.admit(8, 8)
    assert core.can_admit(8, 8) and not core.can_admit(9, 8)  # 2 blocks left
    core.free(h)


# ---- engine ------------------------------------------------------------------

def _spec(**over):
    base = dict(
        name="t", num_requests=12, prompt_len=LengthSpec("uniform", low=3, high=9),
        gen_len=LengthSpec("uniform", low=2, high=12), arrival="instant", seed=7,
    )
    base.update(over)
    return WorkloadSpec(**base)


def test_outputs_match_the_slot_engine():
    spec = _spec()
    _, slots = asyncio.run(run_workload(ContinuousBatchEngine(FakeModelRunner(), max_slots=4), spec, warmup=False))
    _, paged = asyncio.run(run_workload(PagedEngine(FakeModelRunner(), kv_budget_tokens=64, block_size=4),
                                        spec, warmup=False))
    for a, b in zip(slots, paged):
        assert a.request.request_id == b.request.request_id
        assert a.result.output_ids == b.result.output_ids


def test_more_concurrency_than_slots_at_equal_budget():
    # Budget 128 tokens, max_seq_len 32: the slot engine fits 4 sequences; requests of
    # 4 prompt + 4 output tokens fit 16 at a time in blocks of 4.
    spec = _spec(num_requests=16, prompt_len=LengthSpec("fixed", value=4), gen_len=LengthSpec("fixed", value=4))
    slot_engine = ContinuousBatchEngine(FakeModelRunner(decode_cost_per_token=0.001), max_slots=4, max_seq_len=32)
    paged_engine = PagedEngine(FakeModelRunner(decode_cost_per_token=0.001), kv_budget_tokens=128, block_size=4,
                               max_seq_len=32)
    asyncio.run(run_workload(slot_engine, spec, warmup=False))
    asyncio.run(run_workload(paged_engine, spec, warmup=False))
    assert slot_engine.stats()["kv_budget_tokens"] == paged_engine.stats()["kv_budget_tokens"] == 128
    assert max(len(c) for c in slot_engine.core.decode_calls) == 4
    assert max(paged_engine.core.decode_calls) == 16
    assert paged_engine.stats()["decode_steps"] < slot_engine.stats()["decode_steps"]


def test_blocks_come_back_on_finish_and_on_cancel():
    runner = FakeModelRunner(decode_cost_per_token=0.002)
    engine = PagedEngine(runner, kv_budget_tokens=32, block_size=4, max_seq_len=32)

    async def go():
        await engine.start()
        a = asyncio.ensure_future(engine.generate(GenerationRequest("a", [1, 2], 14)))  # 16 tokens: 4 blocks, ~28 ms
        b = asyncio.ensure_future(engine.generate(GenerationRequest("b", [3, 4], 30)))  # 32 tokens: all 8, waits
        await asyncio.sleep(0.01)
        assert engine.core.allocator.num_free == 4  # only 'a' admitted, b needs all 8
        await a
        await asyncio.sleep(0.01)
        assert engine.core.allocator.num_free == 0  # b took everything once a finished
        b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await b
        c = await engine.generate(GenerationRequest("c", [5], 2))
        await engine.stop()
        return c

    c = asyncio.run(go())
    assert c.num_generated == 2
    assert engine.core.allocator.num_free == engine.core.allocator.num_usable  # no leak
    s = engine.stats()
    assert s["block_allocations"] == s["block_frees"]
    assert s["cancelled"] == 1 and s["admission_waits"] >= 1


def test_oversize_request_is_rejected_not_queued_forever():
    engine = PagedEngine(FakeModelRunner(), kv_budget_tokens=16, block_size=4, max_seq_len=64)

    async def go():
        await engine.start()
        with pytest.raises(ValueError, match="cannot fit"):
            await engine.generate(GenerationRequest("big", [1] * 10, 7))  # 17 tokens > 16
        ok = await engine.generate(GenerationRequest("ok", [1] * 10, 6))
        await engine.stop()
        return ok

    assert asyncio.run(go()).num_generated == 6


def test_fragmentation_stat_counts_reserved_but_empty():
    engine = PagedEngine(FakeModelRunner(), kv_budget_tokens=64, block_size=4, max_seq_len=64)
    spec = _spec(num_requests=1, prompt_len=LengthSpec("fixed", value=5), gen_len=LengthSpec("fixed", value=6))
    asyncio.run(run_workload(engine, spec, warmup=False))
    s = engine.stats()
    # 11 tokens -> 3 blocks = 12 reserved; after prefill 6 resident (5 prompt + first token) -> 6 empty.
    assert s["kv_peak_allocated_tokens"] == 12
    assert s["kv_peak_reserved_but_empty_tokens"] == 6


def test_live_token_bound_limits_concurrency_even_with_free_blocks():
    # 1024 tokens of blocks (plenty), but at most 40 live tokens: requests of
    # 10 prompt + 10 output fit 3-4 at a time by tokens, 51 by blocks.
    spec = _spec(num_requests=12, prompt_len=LengthSpec("fixed", value=10), gen_len=LengthSpec("fixed", value=10))
    unbounded = PagedEngine(FakeModelRunner(decode_cost_per_token=0.001), kv_budget_tokens=1024, block_size=4,
                            max_seq_len=64)
    bounded = PagedEngine(FakeModelRunner(decode_cost_per_token=0.001), kv_budget_tokens=1024, block_size=4,
                          max_seq_len=64, max_live_tokens=64)
    asyncio.run(run_workload(unbounded, spec, warmup=False))
    asyncio.run(run_workload(bounded, spec, warmup=False))
    assert max(unbounded.core.decode_calls) == 12
    assert max(bounded.core.decode_calls) <= 6
    assert bounded.stats()["admission_waits"] >= 1
    with pytest.raises(ValueError):
        PagedEngine(FakeModelRunner(), kv_budget_tokens=1024, block_size=4, max_seq_len=64, max_live_tokens=32)


def test_attention_accounting_on_the_real_core(tiny):
    core = PagedGPT2Core(tiny, budget_tokens=64, block_size=4, max_seq_len=64)
    ha, hb = core.admit(8, 2), core.admit(3, 2)
    core.prefill(ha, list(range(1, 9)))
    core.prefill(hb, [40, 41, 42])
    core.decode([ha, hb], [11, 12], [8, 3])
    # padded to 3 blocks (12 positions) for both: 24 computed; live 9 + 4 = 13
    assert core.attn_positions_computed == 24 and core.attn_positions_live == 13


def test_invalid_config():
    with pytest.raises(ValueError):
        PagedEngine(FakeModelRunner(), kv_budget_tokens=2, block_size=4)
