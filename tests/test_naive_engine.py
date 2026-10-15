import asyncio

import pytest

from miniserve.bench.harness import run_workload
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.naive import NaiveEngine
from miniserve.model import FakeModelRunner
from miniserve.request import GenerationRequest


def _spec(**over):
    base = dict(
        name="t", num_requests=6, prompt_len=LengthSpec("fixed", value=8),
        gen_len=LengthSpec("fixed", value=5), arrival="instant", seed=1,
    )
    base.update(over)
    return WorkloadSpec(**base)


async def test_naive_engine_is_fifo_and_serial():
    runner = FakeModelRunner(decode_cost_per_token=0.002)
    engine = NaiveEngine(runner)
    reqs = [GenerationRequest(f"r{i}", [i + 1], 3) for i in range(5)]
    results = await asyncio.gather(*(engine.generate(r) for r in reqs))
    assert runner.calls == [r.request_id for r in reqs]
    # Strictly serial: each request starts after the previous one finished.
    for a, b in zip(results, results[1:]):
        assert b.service_start >= a.finish_time
    assert all(r.num_generated == 3 and r.finish_reason == "length" for r in results)
    assert engine.stats() == {"served": 5}


def test_instant_arrivals_queue_behind_each_other():
    runner = FakeModelRunner(decode_cost_per_token=0.002)
    spec = _spec(arrival="instant")
    t0, records = asyncio.run(run_workload(NaiveEngine(runner), spec))
    assert len(records) == 6 and all(r.result and not r.result.error for r in records)
    assert all(r.arrival_time == t0 for r in records)
    waits = [r.result.service_start - r.arrival_time for r in records]
    assert waits == sorted(waits)  # later in line, longer wait
    assert waits[0] < 0.05 and waits[-1] > 5 * 0.002 * 4  # the last one waited for five others


def test_scheduled_arrivals_are_recorded_even_when_the_loop_is_blocked():
    # Each request costs ~20 ms; arrivals come every 1 ms, so the event loop is
    # blocked when most of them are due. Their arrival must still be the
    # scheduled time, not when the coroutine happened to run.
    runner = FakeModelRunner(decode_cost_per_token=0.005)
    spec = _spec(arrival="burst", burst_size=1, burst_interval=0.001)
    t0, records = asyncio.run(run_workload(NaiveEngine(runner), spec))
    offsets = [r.arrival_time - t0 for r in records]
    assert offsets == pytest.approx([0.0, 0.001, 0.002, 0.003, 0.004, 0.005], abs=1e-9)
    assert records[-1].result.service_start - records[-1].arrival_time > 0.05


def test_closed_loop_arrival_is_previous_finish_and_service_is_fifo():
    runner = FakeModelRunner(decode_cost_per_token=0.001)
    spec = _spec(arrival="closed", concurrency=2, num_requests=6)
    t0, records = asyncio.run(run_workload(NaiveEngine(runner), spec))
    assert len(records) == 6
    firsts = [r for r in records if r.arrival_time == t0]
    assert len(firsts) == 2  # both clients start at t0
    finishes = {r.result.finish_time for r in records}
    for r in records:
        assert r.arrival_time == t0 or r.arrival_time in finishes
    # records are sorted by arrival; the engine must have served them in that order
    assert runner.calls == [r.request.request_id for r in records]


def test_engine_exception_is_recorded_not_raised():
    class Boom(FakeModelRunner):
        def generate_one(self, request):
            if request.request_id.endswith("2"):
                raise RuntimeError("bad luck")
            return super().generate_one(request)

    spec = _spec(num_requests=4)
    _, records = asyncio.run(run_workload(NaiveEngine(Boom()), spec))
    errors = [r for r in records if r.result.error]
    assert len(errors) == 1 and "bad luck" in errors[0].result.error
    assert sum(1 for r in records if not r.result.error) == 3
