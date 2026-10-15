"""Decode step cost in a tight loop versus with gaps between steps.

    .venv/bin/python scripts/probe_step_gaps.py --threads 32 --slots 8

The engine's recorded decode step (17-19 ms) is slower than the same
GPT2Core.decode call in a tight loop (14.8 ms, probe_core_steps.py). Between
engine steps the worker thread is idle for a few milliseconds while the
asyncio loop does bookkeeping. This times the step with controlled idle gaps
and from a worker thread, to see whether the gap itself costs time (OpenMP
worker threads going to sleep and being woken, caches cooling).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from miniserve.gpt2 import GPT2Core  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--context", type=int, default=160)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--gaps-ms", type=float, nargs="+", default=[0, 0.5, 1, 2, 5, 20])
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    S = args.slots
    core = GPT2Core(hf, max_slots=S, max_seq_len=1024)
    ids = list(range(100, 100 + args.context))
    for s in range(S):
        core.prefill(s, ids)
    slots = list(range(S))

    def timed_steps(gap_s: float):
        xs = []
        for _ in range(args.repeats):
            if gap_s:
                time.sleep(gap_s)
            t = time.perf_counter()
            core.decode(slots, [7] * S, [args.context] * S)
            xs.append((time.perf_counter() - t) * 1000)
        return statistics.median(xs[3:])

    print(f"model={args.model} threads={torch.get_num_threads()} slots={S} context={args.context}")
    print("main thread, idle gap before each step:")
    for g in args.gaps_ms:
        print(f"  gap {g:5.1f} ms: step {timed_steps(g / 1000):6.2f} ms")
    print("worker thread (ThreadPoolExecutor), main thread idle:")
    with cf.ThreadPoolExecutor(1) as ex:
        for g in args.gaps_ms:
            print(f"  gap {g:5.1f} ms: step {ex.submit(timed_steps, g / 1000).result():6.2f} ms")

    # The engine's actual pattern: the worker runs decode, then the MAIN thread
    # runs a small torch op on the result (clone + argmax over [S, vocab]),
    # then the worker runs the next step. Two OpenMP teams in one process.
    print("worker thread steps, main thread runs clone+argmax on the logits between steps:")
    with cf.ThreadPoolExecutor(1) as ex:
        xs = []
        for i in range(args.repeats):
            t = time.perf_counter()
            logits = ex.submit(core.decode, slots, [7] * S, [args.context] * S).result()
            xs.append((time.perf_counter() - t) * 1000)
            picks = logits.clone().argmax(dim=-1).tolist()  # main thread
        print(f"  step {statistics.median(xs[3:]):6.2f} ms (measured around the worker call only)")
        xs = []
        for i in range(args.repeats):
            t = time.perf_counter()
            picks = ex.submit(lambda: core.decode(slots, [7] * S, [args.context] * S).argmax(dim=-1).tolist()).result()
            xs.append((time.perf_counter() - t) * 1000)
        print(f"  step {statistics.median(xs[3:]):6.2f} ms when argmax runs on the worker thread too")
    for var in ("OMP_NUM_THREADS", "KMP_BLOCKTIME", "GOMP_SPINCOUNT", "OMP_WAIT_POLICY"):
        print(f"{var}={os.environ.get(var)}")
    print(torch.__config__.parallel_info())


if __name__ == "__main__":
    main()
