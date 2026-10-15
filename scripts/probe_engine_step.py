"""Where does an engine step's wall time go? Run the real stage 3 engine on
the offline workload and split each step into: time inside core.decode on
the worker thread, the executor round trip around it, and everything else
the loop does between steps.

    .venv/bin/python scripts/probe_engine_step.py --threads 32 --slots 8
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniserve.bench.harness import run_workload  # noqa: E402
from miniserve.bench.workload import WORKLOADS  # noqa: E402
from miniserve.engines.continuous import ContinuousBatchEngine  # noqa: E402
from miniserve.model import HFModelRunner  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--workload", default="offline")
    args = p.parse_args()
    runner = HFModelRunner(args.model, torch_threads=args.threads)
    engine = ContinuousBatchEngine(runner, max_slots=args.slots)
    core = engine.core

    inner, outer_end, outer_start = [], [], []
    real_decode = core.decode

    def timed_decode(slots, tokens, positions):
        t = time.perf_counter()
        out = real_decode(slots, tokens, positions)
        inner.append((time.perf_counter() - t) * 1000)
        return out

    core.decode = timed_decode  # type: ignore[assignment]

    real_step = engine._step

    async def timed_step():
        t = time.perf_counter()
        outer_start.append(t)
        await real_step()
        outer_end.append(time.perf_counter())

    engine._step = timed_step  # type: ignore[assignment]

    asyncio.run(run_workload(engine, WORKLOADS[args.workload]))
    s = engine.stats()
    step_wall = [(e - st) * 1000 for st, e in zip(outer_start, outer_end)]
    between = [(outer_start[i + 1] - outer_end[i]) * 1000 for i in range(len(outer_end) - 1)]
    n = s["decode_steps"]
    print(f"workload={args.workload} slots={args.slots} steps={n}")
    print(f"core.decode inside worker: median {statistics.median(inner[-n:]):.2f} ms  "
          f"(min {min(inner[-n:]):.2f}, max {max(inner[-n:]):.2f})")
    print(f"engine decode_time (executor round trip): mean {s['decode_step_mean_ms']:.2f} ms")
    print(f"_step wall (incl. pick + bookkeeping): median {statistics.median(step_wall):.2f} ms")
    print(f"gap between steps (admit/prefill/loop): median {statistics.median(between):.2f} ms, "
          f"max {max(between):.1f} ms, total {sum(between)/1000:.2f} s")
    print(f"prefill: {s['prefill_count']} x {1000*s['prefill_time_s']/s['prefill_count']:.1f} ms")


if __name__ == "__main__":
    main()
