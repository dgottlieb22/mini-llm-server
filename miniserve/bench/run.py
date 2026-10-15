"""Benchmark entry point.

    python -m miniserve.bench.run --engine naive --workload uniform --out results/raw

Writes one JSON record per run (spec, environment, per-request rows, summary)
and prints the summary. The record is the unit every later comparison reads.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional

from ..engines import ENGINES
from ..model import FakeModelRunner, HFModelRunner
from . import metrics
from .envinfo import environment
from .harness import peak_live_tokens, run_workload
from .workload import WORKLOADS, WorkloadSpec


def build_runner(args: argparse.Namespace):
    if args.model == "fake":
        return FakeModelRunner(
            prefill_cost_per_token=args.fake_prefill_cost,
            decode_cost_per_token=args.fake_decode_cost,
        )
    return HFModelRunner(
        model_name=args.model, device=args.device, dtype=args.dtype, torch_threads=args.threads
    )


def parse_engine_kwargs(items: Optional[List[str]]) -> Dict[str, object]:
    """key=value pairs; values parsed as int, then float, then left as str."""
    out: Dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--engine-kw expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def run_once(
    engine_name: str,
    spec: WorkloadSpec,
    runner,
    tag: Optional[str] = None,
    engine_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    engine_cls = ENGINES[engine_name]
    engine = engine_cls(runner, **(engine_kwargs or {}))
    t0, records = asyncio.run(run_workload(engine, spec))
    per_req = metrics.per_request(records, t0)
    summary = metrics.summarize(per_req, num_submitted=len(records))
    errors = [r.result.error for r in records if r.result is not None and r.result.error]
    peak_tokens = peak_live_tokens(records)
    summary["peak_live_tokens"] = peak_tokens
    summary["peak_kv_bytes"] = peak_tokens * runner.kv_bytes_per_token
    summary["errors"] = errors
    return {
        "schema": 1,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tag": tag,
        "engine": {
            "name": engine.name,
            "stage": engine.stage,
            "kwargs": engine_kwargs or {},
            "stats": engine.stats(),
        },
        "model": {
            "name": runner.name,
            "kv_bytes_per_token": runner.kv_bytes_per_token,
            "dtype": getattr(runner, "dtype", None) and str(runner.dtype),
            "device": getattr(runner, "device", "cpu"),
            "torch_threads": getattr(runner, "torch_threads", None),
        },
        "workload": spec.to_dict(),
        "env": environment(),
        "summary": summary,
        "requests": metrics.rows(per_req),
    }


def _fmt_pct(d: Dict[str, float], scale: float = 1000.0, unit: str = "ms") -> str:
    if not d:
        return "n/a"
    return f"p50 {d['p50']*scale:8.1f}  p95 {d['p95']*scale:8.1f}  p99 {d['p99']*scale:8.1f} {unit}"


def summary_text(record: Dict[str, object]) -> str:
    s = record["summary"]
    w = record["workload"]
    lines = [
        f"engine={record['engine']['name']} (stage {record['engine']['stage']})  model={record['model']['name']}  "
        f"workload={w['name']}  completed={s['completed']}/{s['submitted']}",
        f"  makespan {s['makespan_s']:.1f} s   {s['requests_per_s']:.3f} req/s   "
        f"{s['output_tok_per_s']:.1f} out tok/s   {s['total_tok_per_s']:.1f} total tok/s   "
        f"busy {s['busy_fraction']:.2f}",
        f"  queue wait  {_fmt_pct(s['queue_wait_s'])}",
        f"  TTFT        {_fmt_pct(s['ttft_s'])}",
        f"  prefill     {_fmt_pct(s['prefill_latency_s'])}",
        f"  TPOT        {_fmt_pct(s['tpot_s'])}",
        f"  e2e latency {_fmt_pct(s['e2e_latency_s'], 1.0, 's ')}",
        f"  peak live tokens {s['peak_live_tokens']}  ({s['peak_kv_bytes']/2**20:.1f} MiB KV)",
    ]
    if s["errors"]:
        lines.append(f"  errors: {len(s['errors'])} (first: {s['errors'][0]})")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=sorted(ENGINES), default="naive")
    p.add_argument("--workload", choices=sorted(WORKLOADS), default="smoke")
    p.add_argument("--model", default="gpt2", help="HF model id, or 'fake' for the test runner")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    p.add_argument("--fake-prefill-cost", type=float, default=0.0, help="s/token for --model fake")
    p.add_argument("--fake-decode-cost", type=float, default=0.0, help="s/token for --model fake")
    p.add_argument("--out", default="results/raw", help="directory for the JSON record")
    p.add_argument("--tag", default=None, help="free-form label stored in the record")
    p.add_argument("--engine-kw", action="append", default=[], metavar="KEY=VALUE",
                   help="engine constructor argument, e.g. max_batch_size=8 (repeatable)")
    args = p.parse_args(argv)

    spec = WORKLOADS[args.workload]
    runner = build_runner(args)
    engine_kwargs = parse_engine_kwargs(args.engine_kw)
    record = run_once(args.engine, spec, runner, tag=args.tag, engine_kwargs=engine_kwargs)
    os.makedirs(args.out, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    kw = "".join(f"_{k}{v}" for k, v in sorted(engine_kwargs.items()))
    path = os.path.join(args.out, f"{args.engine}{kw}_{args.workload}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    print(summary_text(record))
    print(f"  wrote {path}")
    return 0 if not record["summary"]["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
