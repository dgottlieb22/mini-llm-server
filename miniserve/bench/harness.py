"""Replay a WorkloadSpec against an Engine and collect per-request records.

Open-loop modes pre-compute arrival offsets; each request gets a coroutine
that sleeps until its arrival and then calls engine.generate(). The recorded
arrival is the scheduled time, not the moment the coroutine happened to run,
so an engine that blocks the event loop (stage 0) is charged for the wait it
causes.

Closed-loop mode runs `concurrency` workers that each keep one request in
flight; a request's arrival is the previous request's finish time, which is
exactly when a closed-loop client would send it.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import List, Optional, Sequence, Tuple

from ..engines.base import Engine
from ..request import GenerationRequest, GenerationResult, RequestRecord
from .workload import WorkloadSpec, arrival_offsets, generate_requests


def _error_result(request_id: str, started: float, exc: BaseException) -> GenerationResult:
    return GenerationResult(
        request_id=request_id,
        output_ids=[],
        service_start=started,
        token_times=[],
        finish_time=time.perf_counter(),
        finish_reason="error",
        error=f"{type(exc).__name__}: {exc}",
    )


async def _serve(engine: Engine, rec: RequestRecord) -> None:
    started = time.perf_counter()
    try:
        rec.result = await engine.generate(rec.request)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 (we record every failure)
        rec.result = _error_result(rec.request.request_id, started, exc)


async def run_workload(
    engine: Engine,
    spec: WorkloadSpec,
    requests: Optional[Sequence[GenerationRequest]] = None,
    warmup: bool = True,
) -> Tuple[float, List[RequestRecord]]:
    """Returns (t0, records). t0 is the perf_counter() run start.

    With warmup=True the engine first serves a dummy workload at the largest
    prompt and output length in the spec, before t0. A fresh process pays
    page-fault and allocator costs on its first call that a running server
    never sees again (see results/stage2/probes.md), so the measurement
    starts from a warm engine.
    """
    reqs = list(requests) if requests is not None else generate_requests(spec)
    records: List[RequestRecord] = []
    await engine.start()
    try:
        if warmup:
            await engine.warmup(max(r.prompt_len for r in reqs), max(r.max_new_tokens for r in reqs))
        t0 = time.perf_counter()
        if spec.open_loop:
            offsets = arrival_offsets(spec)
            assert offsets is not None and len(offsets) == len(reqs)

            async def client(req: GenerationRequest, offset: float) -> None:
                arrival = t0 + offset
                delay = arrival - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                rec = RequestRecord(request=req, arrival_time=arrival)
                records.append(rec)
                await _serve(engine, rec)

            await asyncio.gather(*(client(r, o) for r, o in zip(reqs, offsets)))
        else:
            pending = deque(reqs)

            async def worker() -> None:
                prev_finish = t0
                while pending:
                    req = pending.popleft()
                    rec = RequestRecord(request=req, arrival_time=prev_finish)
                    records.append(rec)
                    # Yield once so every client that has "sent" by now gets
                    # to submit before this one re-enters the engine. Without
                    # this, a blocking engine lets one worker drain the queue
                    # while the other clients' first requests never start.
                    await asyncio.sleep(0)
                    await _serve(engine, rec)
                    assert rec.result is not None
                    prev_finish = rec.result.finish_time

            await asyncio.gather(*(worker() for _ in range(spec.concurrency)))
    finally:
        await engine.stop()
    records.sort(key=lambda r: (r.arrival_time, r.request.request_id))
    return t0, records


def peak_live_tokens(records: Sequence[RequestRecord]) -> int:
    """Largest number of tokens resident at once, from the timestamps.

    A request holds prompt_len tokens from service_start, one more per
    generated token, and releases all of them at finish. This is the KV
    footprint the engine needed, independent of how it allocates.
    """
    events = []  # (time, order, delta); releases sort before acquires at a tie
    for rec in records:
        r = rec.result
        if r is None or r.error:
            continue
        events.append((r.service_start, 1, rec.request.prompt_len))
        for t in r.token_times:
            events.append((t, 1, 1))
        events.append((r.finish_time, 0, -(rec.request.prompt_len + r.num_generated)))
    events.sort()
    live = peak = 0
    for _, _, d in events:
        live += d
        peak = max(peak, live)
    if live != 0 or math.isnan(peak):
        raise AssertionError(f"token accounting did not return to zero: {live}")
    return peak
