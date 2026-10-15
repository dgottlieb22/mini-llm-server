import asyncio

import pytest

from miniserve.bench.harness import run_workload
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.async_serial import AsyncSerialEngine
from miniserve.engines.naive import NaiveEngine
from miniserve.model import FakeModelRunner
from miniserve.request import GenerationRequest


async def _ticks_during(engine, request, tick=0.005):
    """How many times a side coroutine gets to run while one request is served."""
    await engine.start()
    count = 0
    done = asyncio.Event()

    async def ticker():
        nonlocal count
        while not done.is_set():
            count += 1
            await asyncio.sleep(tick)

    t = asyncio.ensure_future(ticker())
    await engine.generate(request)
    done.set()
    await t
    await engine.stop()
    return count


async def test_event_loop_stays_responsive_while_model_runs():
    req = GenerationRequest("r", [1, 2, 3], max_new_tokens=20)
    # 20 tokens x 5 ms = 100 ms of model time per request.
    blocked = await _ticks_during(NaiveEngine(FakeModelRunner(decode_cost_per_token=0.005)), req)
    free = await _ticks_during(AsyncSerialEngine(FakeModelRunner(decode_cost_per_token=0.005)), req)
    assert blocked <= 2          # the naive engine starves the loop
    assert free >= 10            # stage 1 keeps accepting work during the 100 ms


async def test_fifo_and_serial():
    runner = FakeModelRunner(decode_cost_per_token=0.002)
    engine = AsyncSerialEngine(runner)
    await engine.start()
    reqs = [GenerationRequest(f"r{i}", [i + 1], 3) for i in range(5)]
    results = await asyncio.gather(*(engine.generate(r) for r in reqs))
    await engine.stop()
    assert runner.calls == [r.request_id for r in reqs]
    for a, b in zip(results, results[1:]):
        assert b.service_start >= a.finish_time
    assert engine.stats()["served"] == 5
    assert engine.stats()["max_queue_depth"] >= 4


async def test_model_exception_propagates_to_caller_and_engine_keeps_going():
    class Boom(FakeModelRunner):
        def generate_one(self, request):
            if request.request_id == "bad":
                raise RuntimeError("bad luck")
            return super().generate_one(request)

    engine = AsyncSerialEngine(Boom())
    await engine.start()
    with pytest.raises(RuntimeError, match="bad luck"):
        await engine.generate(GenerationRequest("bad", [1], 1))
    ok = await engine.generate(GenerationRequest("good", [1], 2))
    await engine.stop()
    assert ok.num_generated == 2


def test_harness_results_match_naive_on_the_same_workload():
    spec = WorkloadSpec(
        name="t", num_requests=6, prompt_len=LengthSpec("fixed", value=8),
        gen_len=LengthSpec("fixed", value=4), arrival="instant", seed=1,
    )
    _, naive = asyncio.run(run_workload(NaiveEngine(FakeModelRunner()), spec))
    _, async_ = asyncio.run(run_workload(AsyncSerialEngine(FakeModelRunner()), spec))
    assert [r.result.output_ids for r in naive] == [r.result.output_ids for r in async_]
    assert [r.request.request_id for r in naive] == [r.request.request_id for r in async_]
