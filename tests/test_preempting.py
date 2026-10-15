import asyncio

import pytest

from miniserve.bench.harness import run_workload
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.paged import PagedEngine
from miniserve.engines.preempting import PreemptingEngine
from miniserve.request import GenerationRequest


def _spec(**over):
    base = dict(
        name="t", num_requests=12, prompt_len=LengthSpec("uniform", low=6, high=14),
        gen_len=LengthSpec("uniform", low=4, high=20), arrival="instant", seed=11, vocab_size=90,
    )
    base.update(over)
    return WorkloadSpec(**base)


def test_admits_on_prompt_alone_and_grows_block_by_block(tiny_runner):
    # 64-token pool, block 4. Conservative admission of (8 prompt + 40 output) needs 12 blocks;
    # optimistic needs 3 and then grows.
    engine = PreemptingEngine(tiny_runner, kv_budget_tokens=64, block_size=4, max_seq_len=64)

    async def go():
        await engine.start()
        r = await engine.generate(GenerationRequest("a", list(range(1, 9)), 40))
        await engine.stop()
        return r

    r = asyncio.run(go())
    assert r.num_generated == 40 and r.finish_reason == "length"
    s = engine.stats()
    assert s["preemptions"] == 0
    assert s["block_allocations"] == 12  # 3 at admission + 9 grown, one per block boundary
    assert engine.core.allocator.num_free + engine.core.cached_blocks == engine.core.allocator.num_usable


def test_preemption_preserves_output_and_fifo(tiny_runner):
    # Pool of 64 tokens. Two requests whose combined growth exceeds it: the
    # younger is preempted at least once and resumed; both outputs must equal
    # what the conservative engine produces one at a time.
    spec = _spec(num_requests=2, prompt_len=LengthSpec("fixed", value=8), gen_len=LengthSpec("fixed", value=30))
    ref_engine = PagedEngine(tiny_runner, kv_budget_tokens=256, block_size=4, max_seq_len=64)
    _, ref = asyncio.run(run_workload(ref_engine, spec, warmup=False))
    eng = PreemptingEngine(tiny_runner, kv_budget_tokens=64, block_size=4, max_seq_len=64, growth_reserve=0)
    _, out = asyncio.run(run_workload(eng, spec, warmup=False))
    assert [r.result.output_ids for r in ref] == [r.result.output_ids for r in out]
    s = eng.stats()
    assert s["preemptions"] >= 1 and s["recomputed_tokens"] > 0
    first, second = out
    # The older request (admitted first) was never preempted: its token times have no gap
    # longer than the younger one's resume prefill, and it finished first.
    assert first.result.finish_time <= second.result.finish_time
    assert first.result.num_generated == second.result.num_generated == 30
    # Timestamps stay monotonic across the stitch.
    tt = second.result.token_times
    assert all(b >= a for a, b in zip(tt, tt[1:]))
    assert eng.core.allocator.num_free + eng.core.cached_blocks == eng.core.allocator.num_usable


def test_many_requests_small_pool_all_complete_correctly(tiny_runner):
    spec = _spec(num_requests=12)
    ref_engine = PagedEngine(tiny_runner, kv_budget_tokens=1024, block_size=4, max_seq_len=64)
    _, ref = asyncio.run(run_workload(ref_engine, spec, warmup=False))
    eng = PreemptingEngine(tiny_runner, kv_budget_tokens=96, block_size=4, max_seq_len=64)
    _, out = asyncio.run(run_workload(eng, spec, warmup=False))
    assert [r.result.output_ids for r in ref] == [r.result.output_ids for r in out]
    assert all(r.result.finish_reason == "length" for r in out)
    assert eng.stats()["served"] == 12
    assert eng.core.allocator.num_free + eng.core.cached_blocks == eng.core.allocator.num_usable


def test_growth_reserve_reduces_preemptions(tiny_runner):
    spec = _spec(num_requests=12)
    counts = {}
    for reserve in (0, 1, 2):
        eng = PreemptingEngine(tiny_runner, kv_budget_tokens=96, block_size=4, max_seq_len=64, growth_reserve=reserve)
        asyncio.run(run_workload(eng, spec, warmup=False))
        counts[reserve] = eng.stats()["preemptions"]
    assert counts[0] >= counts[1] >= counts[2]
    with pytest.raises(ValueError):
        PreemptingEngine(tiny_runner, kv_budget_tokens=96, block_size=4, growth_reserve=-1)


def test_cancelling_a_preempted_request_frees_everything(tiny_runner):
    eng = PreemptingEngine(tiny_runner, kv_budget_tokens=32, block_size=4, max_seq_len=64, growth_reserve=0)

    async def go():
        await eng.start()
        a = asyncio.ensure_future(eng.generate(GenerationRequest("a", [1, 2, 3, 4], 28)))
        b = asyncio.ensure_future(eng.generate(GenerationRequest("b", [5, 6, 7, 8], 28)))
        # Wait until b (the younger) has been preempted at least once, then cancel it
        # while its history sits in _resumed and its resumed request is queued or running.
        for _ in range(500):
            await asyncio.sleep(0.002)
            if eng.stats()["preemptions"] >= 1 and not b.done():
                break
        assert eng.stats()["preemptions"] >= 1 and not b.done()
        b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await b
        ra = await a
        await eng.stop()
        return ra

    ra = asyncio.run(go())
    assert ra.num_generated == 28
    assert eng._resumed == {}
    assert eng.core.allocator.num_free + eng.core.cached_blocks == eng.core.allocator.num_usable
