"""Cross-stage summary tables for the README, from the committed result records.

    .venv/bin/python scripts/summary_table.py

One row per stage configuration, one table per workload. The configuration
each row represents is fixed here (engine name, kwargs, tag prefix) so a
row always means the same thing; the newest matching record wins when a
workload was re-run.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, engine name, required kwargs, tag prefix)
ROWS = [
    ("0 naive", "naive", {}, "stage0"),
    ("1 async intake", "async_serial", {}, "stage1"),
    ("2 static batching, B=8", "static_batch", {"max_batch_size": 8}, "stage2"),
    ("3 continuous, 8 slots", "continuous", {"max_slots": 8}, "stage3"),
    ("4 paged, 8192 tokens", "paged", {"kv_budget_tokens": 8192, "block_size": 16}, "stage4"),
    ("5 + batched prefill", "paged", {"kv_budget_tokens": 8192, "prefill_batch_tokens": 4096}, "stage5"),
    ("6 + prefix cache", "prefix_cached", {"kv_budget_tokens": 8192, "prefill_batch_tokens": 4096}, "stage6"),
    ("7 + preemption", "preempting", {"kv_budget_tokens": 8192, "prefill_batch_tokens": 4096}, "stage7"),
]
WORKLOADS = ["offline", "closed8", "burst", "uniform", "hetero", "uniform_hi", "hetero_hi",
             "uniform_8x", "hetero_4x", "prefix_2x", "prefix_4x"]


def load_all():
    recs = []
    for path in glob.glob(os.path.join(ROOT, "results", "stage*", "*.json")):
        with open(path) as f:
            r = json.load(f)
        r["_path"] = path
        recs.append(r)
    return recs


def pick(recs, engine, kwargs, tag_prefix, workload):
    cands = []
    for r in recs:
        e = r["engine"]
        if e["name"] != engine or r["workload"]["name"] != workload:
            continue
        if not (r.get("tag") or "").startswith(tag_prefix):
            continue
        kw = e.get("kwargs", {})
        if any(kw.get(k) != v for k, v in kwargs.items()):
            continue
        if "max_live_tokens" in kw or kw.get("growth_reserve") == 0:
            continue  # side experiments, not the stage's configuration
        cands.append(r)
    return max(cands, key=lambda r: r["timestamp"]) if cands else None


def main() -> int:
    recs = load_all()
    for w in WORKLOADS:
        rows = [(label, pick(recs, eng, kw, tag, w)) for label, eng, kw, tag in ROWS]
        rows = [(l, r) for l, r in rows if r is not None]
        if not rows:
            continue
        print(f"\n**{w}**: {rows[0][1]['workload'].get('description', '')}\n")
        print("| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for label, r in rows:
            s = r["summary"]
            print(f"| {label} | {s['requests_per_s']:.2f} | {s['output_tok_per_s']:.0f} | "
                  f"{1000*s['ttft_s']['p50']:.0f} ms | {1000*s['ttft_s']['p99']:.0f} ms | "
                  f"{1000*s['tpot_s']['p50']:.0f} ms | {s['e2e_latency_s']['p99']:.1f} s |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
