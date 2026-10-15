import asyncio

import pytest

from miniserve.bench.harness import run_workload
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.async_serial import AsyncSerialEngine
from miniserve.engines.continuous import ContinuousBatchEngine
from miniserve.model import FakeModelRunner
from miniserve.request import GenerationRequest


def _spec(**over):
    base = dict(
        name="t", num_requests=10, prompt_len=LengthSpec("uniform", low=3, high=9),
        gen_len=LengthSpec("uniform", low=2, high=12), arrival="instant", seed=7,
    )
    base.update(over)
    return WorkloadSpec(**base)


def test_outputs_match_the_serial_engine():
    spec = _spec(num_requests=12)
    _, serial = asyncio.run(run_workload(AsyncSerialEngine(FakeModelRunner()), spec, warmup=False))
    _, cont = asyncio.run(run_workload(ContinuousBatchEngine(FakeModelRunner(), max_slots=4), spec, warmup=False))
    for a, b in zip(serial, cont):
        assert a.request.request_id == b.request.request_id
        assert a.result.output_ids == b.result.output_ids
        assert b.result.finish_reason == "length"


async def test_a_finished_sequence_frees_its_slot_for_the_next_request_mid_batch():
    runner = FakeModelRunner(decode_cost_per_token=0.002)
    engine = ContinuousBatchEngine(runner, max_slots=2)
    await engine.start()
    short = GenerationRequest("short", [1, 2], 2)
    long = GenerationRequest("long", [3, 4], 20)
    third = GenerationRequest("third", [5, 6], 2)
    rs, rl, rt = await asyncio.gather(engine.generate(short), engine.generate(long), engine.generate(third))
    await engine.stop()
    # 'short' returned as soon as its own 2 tokens were done, well before 'long'.
    assert rs.finish_time < rl.token_times[5]
    # 'third' was admitted into short's slot while 'long' was still running: it
    # started before long finished and reused slot 0 (short's slot).
    assert rt.service_start < rl.finish_time
    assert engine.core.prefills == [0, 1, 0]
    # and the engine never ran more than 2 slots at once
    assert max(len(c) for c in engine.core.decode_calls) <= 2
    assert engine.stats()["served"] == 3


async def test_results_return_immediately_not_with_the_batch():
    engine = ContinuousBatchEngine(FakeModelRunner(decode_cost_per_token=0.003), max_slots=4)
    await engine.start()
    reqs = [GenerationRequest(f"r{n}", [n], n) for n in (1, 5, 10)]
    results = await asyncio.gather(*(engine.generate(r) for r in reqs))
    await engine.stop()
    finishes = [r.finish_time for r in results]
    assert finishes == sorted(finishes) and finishes[0] < finishes[1] < finishes[2]
    for r in results:
        assert r.finish_time == r.token_times[-1]  # no holding after the last token


async def test_cancellation_frees_the_slot_at_a_step_boundary():
    runner = FakeModelRunner(decode_cost_per_token=0.003)
    engine = ContinuousBatchEngine(runner, max_slots=1)
    await engine.start()
    victim = asyncio.ensure_future(engine.generate(GenerationRequest("victim", [1], 100)))
    await asyncio.sleep(0.03)  # let it get admitted and run a few steps
    victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await victim
    # The single slot must come back: this request can only run if it did.
    r = await asyncio.wait_for(engine.generate(GenerationRequest("after", [2], 3)), timeout=2.0)
    await engine.stop()
    assert r.num_generated == 3
    assert engine.stats()["cancelled"] == 1
    assert engine.core.prefills == [0, 0]


async def test_cancelled_while_queued_is_skipped():
    engine = ContinuousBatchEngine(FakeModelRunner(decode_cost_per_token=0.003), max_slots=1)
    await engine.start()
    blocker = asyncio.ensure_future(engine.generate(GenerationRequest("blocker", [1], 10)))
    await asyncio.sleep(0.005)
    queued = asyncio.ensure_future(engine.generate(GenerationRequest("queued", [2], 10)))
    await asyncio.sleep(0.005)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await blocker
    await engine.stop()
    assert engine.core.prefills == [0]  # the cancelled request never touched the model
    assert engine.stats()["cancelled"] == 1


async def test_admission_control_rejects_what_cannot_fit():
    engine = ContinuousBatchEngine(FakeModelRunner(), max_slots=2, max_seq_len=16)
    await engine.start()
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        await engine.generate(GenerationRequest("big", [1] * 10, 7))
    ok = await engine.generate(GenerationRequest("fits", [1] * 10, 6))
    await engine.stop()
    assert ok.num_generated == 6
    assert engine.stats()["rejected"] == 1 and engine.stats()["served"] == 1


def test_fifo_admission_no_starvation():
    # 10 requests, 3 slots: admission order must be arrival order even though
    # short requests free slots early.
    runner = FakeModelRunner(decode_cost_per_token=0.001)
    spec = _spec(num_requests=10)
    _, records = asyncio.run(run_workload(ContinuousBatchEngine(runner, max_slots=3), spec, warmup=False))
    starts = [r.result.service_start for r in records]  # records are in arrival order
    assert starts == sorted(starts)


def test_stats_and_kv_accounting():
    runner = FakeModelRunner(kv_bytes_per_token=100)
    engine = ContinuousBatchEngine(runner, max_slots=2, max_seq_len=32)
    spec = _spec(num_requests=4, prompt_len=LengthSpec("fixed", value=5), gen_len=LengthSpec("fixed", value=4))
    asyncio.run(run_workload(engine, spec, warmup=False))
    s = engine.stats()
    assert s["served"] == 4 and s["prefill_count"] == 4 and s["prefill_tokens"] == 20
    assert s["kv_budget_tokens"] == 64 and s["kv_cache_bytes"] == 100 * 64
    # two sequences of 5 prompt + up to 3 cached output tokens each
    assert 10 <= s["kv_peak_used_tokens"] <= 16
    assert 0 < s["slot_utilization"] <= 1.0
    assert s["decode_steps"] >= 3


def test_warmup_fills_every_slot_and_resets_stats():
    engine = ContinuousBatchEngine(FakeModelRunner(), max_slots=3, max_seq_len=32)
    spec = _spec(num_requests=2, prompt_len=LengthSpec("fixed", value=4), gen_len=LengthSpec("fixed", value=3))
    asyncio.run(run_workload(engine, spec, warmup=True))
    assert sorted(engine.core.prefills[:3]) == [0, 1, 2]
    assert engine.stats()["served"] == 2


def test_invalid_config():
    with pytest.raises(ValueError):
        ContinuousBatchEngine(FakeModelRunner(), max_slots=0)
