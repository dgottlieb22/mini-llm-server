import asyncio

import pytest

from miniserve.bench.harness import run_workload
from miniserve.bench.run import parse_engine_kwargs
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.async_serial import AsyncSerialEngine
from miniserve.engines.static_batch import StaticBatchEngine
from miniserve.model import FakeModelRunner
from miniserve.request import GenerationRequest


def _spec(**over):
    base = dict(
        name="t", num_requests=10, prompt_len=LengthSpec("uniform", low=3, high=9),
        gen_len=LengthSpec("uniform", low=2, high=6), arrival="instant", seed=7,
    )
    base.update(over)
    return WorkloadSpec(**base)


def test_batches_form_from_the_queue_in_fifo_order_up_to_the_cap():
    runner = FakeModelRunner(decode_cost_per_token=0.001)
    spec = _spec(num_requests=10)
    _, records = asyncio.run(run_workload(StaticBatchEngine(runner, max_batch_size=4), spec))
    assert [len(b) for b in runner.batches] == [4, 4, 2]
    flat = [rid for b in runner.batches for rid in b]
    assert flat == [r.request.request_id for r in records]  # FIFO across and within batches
    stats = StaticBatchEngine(runner, max_batch_size=4).stats()
    assert stats["max_batch_size"] == 4


def test_batch_cap_of_one_degenerates_to_stage_1():
    spec = _spec(num_requests=5)
    _, serial = asyncio.run(run_workload(AsyncSerialEngine(FakeModelRunner()), spec))
    runner = FakeModelRunner()
    _, batched = asyncio.run(run_workload(StaticBatchEngine(runner, max_batch_size=1), spec))
    assert all(len(b) == 1 for b in runner.batches)
    assert [r.result.output_ids for r in serial] == [r.result.output_ids for r in batched]


def test_outputs_match_unbatched_generation_per_request():
    spec = _spec(num_requests=8)
    _, serial = asyncio.run(run_workload(AsyncSerialEngine(FakeModelRunner()), spec))
    _, batched = asyncio.run(run_workload(StaticBatchEngine(FakeModelRunner(), max_batch_size=8), spec))
    for a, b in zip(serial, batched):
        assert a.request.request_id == b.request.request_id
        assert a.result.output_ids == b.result.output_ids
        assert b.result.num_generated == b.request.max_new_tokens


def test_padding_and_slot_accounting():
    runner = FakeModelRunner()
    engine = StaticBatchEngine(runner, max_batch_size=8)
    reqs = [GenerationRequest("a", [1] * 3, 2), GenerationRequest("b", [1] * 7, 5)]
    engine._account(reqs)
    s = engine.stats()
    assert s["prompt_tokens_real"] == 10 and s["prompt_tokens_padded"] == 14
    assert s["decode_slots_real"] == 7 and s["decode_slots_padded"] == 10
    assert s["prompt_pad_fraction"] == pytest.approx(4 / 14)
    assert s["decode_slot_waste_fraction"] == pytest.approx(3 / 10)
    assert s["batch_sizes"] == {"2": 1}


async def test_short_member_waits_for_the_longest_member():
    runner = FakeModelRunner(decode_cost_per_token=0.005)
    engine = StaticBatchEngine(runner, max_batch_size=4)
    await engine.start()
    short = GenerationRequest("short", [1, 2], 2)
    long = GenerationRequest("long", [3, 4], 20)
    rs, rl = await asyncio.gather(engine.generate(short), engine.generate(long))
    await engine.stop()
    assert runner.batches == [["short", "long"]]
    # 'short' produced its last token early but its result returned with the batch.
    assert rs.token_times[-1] < rl.token_times[-1]
    assert rs.finish_time == rl.finish_time
    assert rs.finish_time - rs.token_times[-1] > 10 * 0.005


async def test_batch_failure_reaches_every_member_and_engine_continues():
    class Boom(FakeModelRunner):
        def generate_batch(self, requests):
            if any(r.request_id == "bad" for r in requests):
                raise RuntimeError("bad batch")
            return super().generate_batch(requests)

    engine = StaticBatchEngine(Boom(), max_batch_size=4)
    await engine.start()
    results = await asyncio.gather(
        engine.generate(GenerationRequest("bad", [1], 1)),
        engine.generate(GenerationRequest("victim", [2], 1)),
        return_exceptions=True,
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    ok = await engine.generate(GenerationRequest("later", [3], 2))
    await engine.stop()
    assert ok.num_generated == 2


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError):
        StaticBatchEngine(FakeModelRunner(), max_batch_size=0)


def test_engine_kwargs_parser():
    assert parse_engine_kwargs(["max_batch_size=8", "x=0.5", "name=foo"]) == {
        "max_batch_size": 8, "x": 0.5, "name": "foo"}
    assert parse_engine_kwargs([]) == {}
    with pytest.raises(SystemExit):
        parse_engine_kwargs(["novalue"])
