"""Compare benchmark records.

    python -m miniserve.bench.compare results/stage0/*.json results/stage1/*.json

Prints one markdown table per workload with a row per engine, plus a
head-of-line (HOL) blocking summary per record: how long requests waited in
the queue relative to their own service time. A request that waits longer
than its own service time is paying for other people's work.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

COLUMNS = [
    ("req/s", lambda s: f"{s['requests_per_s']:.3f}"),
    ("out tok/s", lambda s: f"{s['output_tok_per_s']:.1f}"),
    ("TTFT p50", lambda s: _ms(s["ttft_s"], "p50")),
    ("TTFT p99", lambda s: _ms(s["ttft_s"], "p99")),
    ("TPOT p50", lambda s: _ms(s["tpot_s"], "p50")),
    ("queue p50", lambda s: _ms(s["queue_wait_s"], "p50")),
    ("queue p99", lambda s: _ms(s["queue_wait_s"], "p99")),
    ("e2e p50", lambda s: _sec(s["e2e_latency_s"], "p50")),
    ("e2e p99", lambda s: _sec(s["e2e_latency_s"], "p99")),
    ("busy", lambda s: f"{s['busy_fraction']:.2f}"),
    ("peak tok", lambda s: str(s["peak_live_tokens"])),
]


def _ms(d: Dict[str, float], key: str) -> str:
    return f"{d[key]*1000:.0f} ms" if d else "n/a"


def _sec(d: Dict[str, float], key: str) -> str:
    return f"{d[key]:.2f} s" if d else "n/a"


def load(paths: Iterable[str]) -> List[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        rec["_path"] = p
        out.append(rec)
    return out


def label(rec: dict) -> str:
    e = rec["engine"]
    kw = "".join(f" {k}={v}" for k, v in sorted(e.get("kwargs", {}).items()))
    tag = f" [{rec['tag']}]" if rec.get("tag") else ""
    return f"stage {e['stage']} {e['name']}{kw}{tag}"


def table(records: List[dict]) -> str:
    by_workload: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_workload[r["workload"]["name"]].append(r)
    lines = []
    for wname, recs in by_workload.items():
        recs.sort(key=lambda r: (r["engine"]["stage"], r["timestamp"]))
        lines.append(f"\n### {wname}: {recs[0]['workload'].get('description', '')}\n")
        lines.append("| engine | " + " | ".join(c for c, _ in COLUMNS) + " |")
        lines.append("|---|" + "|".join("---:" for _ in COLUMNS) + "|")
        for r in recs:
            s = r["summary"]
            lines.append(f"| {label(r)} | " + " | ".join(fn(s) for _, fn in COLUMNS) + " |")
    return "\n".join(lines)


def hol_summary(rec: dict) -> Dict[str, object]:
    """Queue wait relative to each request's own service time."""
    rows = rec["requests"]
    if not rows:
        return {}
    wait = np.array([r["queue_wait"] for r in rows])
    service = np.array([r["finish"] - r["service_start"] for r in rows])
    gen = np.array([r["gen_len"] for r in rows])
    multiple = wait / np.maximum(service, 1e-9)
    short = gen <= np.percentile(gen, 25)
    return {
        "requests": len(rows),
        "waited_longer_than_own_service": int((multiple > 1).sum()),
        "wait_over_service_p50": float(np.median(multiple)),
        "wait_over_service_max": float(multiple.max()),
        "shortest_quartile_gen_len_max": int(gen[short].max()),
        "shortest_quartile_queue_wait_p50_s": float(np.median(wait[short])),
        "shortest_quartile_service_p50_s": float(np.median(service[short])),
        "queue_share_of_e2e_p50": float(np.median(wait / np.array([r["e2e_latency"] for r in rows]))),
    }


def hol_text(records: List[dict]) -> str:
    lines = ["\n### Head-of-line blocking\n",
             "| engine | workload | waited > own service | wait/service p50 | wait/service max | "
             "shortest-quartile wait p50 | its service p50 | queue share of e2e p50 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(records, key=lambda r: (r["workload"]["name"], r["engine"]["stage"])):
        h = hol_summary(r)
        if not h:
            continue
        lines.append(
            f"| {label(r)} | {r['workload']['name']} | {h['waited_longer_than_own_service']}/{h['requests']} | "
            f"{h['wait_over_service_p50']:.1f}x | {h['wait_over_service_max']:.1f}x | "
            f"{h['shortest_quartile_queue_wait_p50_s']:.2f} s | {h['shortest_quartile_service_p50_s']:.2f} s | "
            f"{h['queue_share_of_e2e_p50']*100:.0f}% |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--no-hol", action="store_true")
    args = p.parse_args(argv)
    records = load(args.paths)
    print(table(records))
    if not args.no_hol:
        print(hol_text(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
