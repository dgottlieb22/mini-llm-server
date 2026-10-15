"""Recompute the summary block of existing result records from their per-request rows.

    python -m miniserve.bench.resummarize results/stage*/*.json

Used when a summary metric is added or redefined. The per-request rows are
the raw measurement and are never rewritten; only `summary` is replaced,
with peak_live_tokens, peak_kv_bytes and errors carried over because they
are not derivable from the rows alone.
"""
from __future__ import annotations

import json
import sys

from .metrics import RequestMetrics, summarize


def resummarize(path: str) -> None:
    with open(path) as f:
        rec = json.load(f)
    rows = [RequestMetrics(**r) for r in rec["requests"]]
    old = rec["summary"]
    new = summarize(rows, num_submitted=old["submitted"])
    for key in ("peak_live_tokens", "peak_kv_bytes", "errors"):
        new[key] = old[key]
    rec["summary"] = new
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)


def main(argv=None) -> int:
    for p in (argv or sys.argv[1:]):
        resummarize(p)
        print(f"resummarized {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
