"""Metric definitions. Every stage is scored by this file and nothing else.

Per request (all in seconds, all relative to the request's own arrival):
  queue_wait       service_start - arrival
  ttft             first token time - arrival. This is what a user sees; it
                   includes queueing.
  prefill_latency  first token time - service_start. The model's own cost of
                   producing the first token, queueing excluded.
  tpot             (last token time - first token time) / (n_generated - 1);
                   undefined for single-token outputs.
  e2e_latency      finish - arrival

Per run:
  makespan         last finish - first arrival
  requests_per_s   completed requests / makespan
  output_tok_per_s generated tokens / makespan
  total_tok_per_s  (prompt + generated tokens) / makespan
  busy_fraction    sum of per-request service time / makespan. For a serial
                   engine this is server utilization; for a batching engine
                   members overlap and it exceeds 1 when the engine is full.
  occupancy        fraction of the makespan with at least one request in
                   service. This is server utilization for any engine.
  mean_in_service  busy_fraction / occupancy: average number of requests in
                   service while the engine was busy (1 for a serial engine).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..request import RequestRecord


@dataclass
class RequestMetrics:
    request_id: str
    prompt_len: int
    gen_len: int
    arrival: float
    service_start: float
    first_token: float
    finish: float
    queue_wait: float
    ttft: float
    prefill_latency: float
    tpot: Optional[float]
    e2e_latency: float
    finish_reason: str


def per_request(records: Sequence[RequestRecord], t0: float) -> List[RequestMetrics]:
    out = []
    for rec in records:
        r = rec.result
        if r is None or r.error:
            continue
        n = r.num_generated
        tpot = (r.token_times[-1] - r.token_times[0]) / (n - 1) if n > 1 else None
        out.append(
            RequestMetrics(
                request_id=rec.request.request_id,
                prompt_len=rec.request.prompt_len,
                gen_len=n,
                arrival=rec.arrival_time - t0,
                service_start=r.service_start - t0,
                first_token=r.first_token_time - t0,
                finish=r.finish_time - t0,
                queue_wait=r.service_start - rec.arrival_time,
                ttft=r.first_token_time - rec.arrival_time,
                prefill_latency=r.first_token_time - r.service_start,
                tpot=tpot,
                e2e_latency=r.finish_time - rec.arrival_time,
                finish_reason=r.finish_reason,
            )
        )
    return out


def percentiles(values: Sequence[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {}
    a = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def occupancy(intervals: Sequence[Tuple[float, float]], makespan: float) -> float:
    """Fraction of the makespan during which at least one request was in service."""
    if not intervals or makespan <= 0:
        return 0.0
    covered = 0.0
    cur_start, cur_end = None, None
    for s, e in sorted(intervals):
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                covered += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    covered += cur_end - cur_start
    return covered / makespan


def summarize(metrics: Sequence[RequestMetrics], num_submitted: int) -> Dict[str, object]:
    if not metrics:
        return {"completed": 0, "submitted": num_submitted}
    first_arrival = min(m.arrival for m in metrics)
    last_finish = max(m.finish for m in metrics)
    makespan = max(last_finish - first_arrival, 1e-9)
    gen_tokens = sum(m.gen_len for m in metrics)
    prompt_tokens = sum(m.prompt_len for m in metrics)
    service_time = sum(m.finish - m.service_start for m in metrics)
    occ = occupancy([(m.service_start, m.finish) for m in metrics], makespan)
    return {
        "completed": len(metrics),
        "submitted": num_submitted,
        "makespan_s": makespan,
        "requests_per_s": len(metrics) / makespan,
        "output_tok_per_s": gen_tokens / makespan,
        "total_tok_per_s": (gen_tokens + prompt_tokens) / makespan,
        "busy_fraction": service_time / makespan,
        "occupancy": occ,
        "mean_in_service": (service_time / makespan) / occ if occ > 0 else 0.0,
        "queue_wait_s": percentiles([m.queue_wait for m in metrics]),
        "ttft_s": percentiles([m.ttft for m in metrics]),
        "prefill_latency_s": percentiles([m.prefill_latency for m in metrics]),
        "tpot_s": percentiles([m.tpot for m in metrics if m.tpot is not None]),
        "e2e_latency_s": percentiles([m.e2e_latency for m in metrics]),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
    }


def rows(metrics: Sequence[RequestMetrics]) -> List[Dict[str, object]]:
    return [asdict(m) for m in metrics]
