"""Step costs of our own forward (GPT2Core) versus Hugging Face.

    .venv/bin/python scripts/probe_core_steps.py --threads 32 --slots 8

Decode: one step with all slots active at context ~160, and with only one
slot active (the engine computes every slot regardless). Prefill: one
sequence of 16..512 tokens. Compare against results/stage0/probes.md.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from miniserve.gpt2 import GPT2Core  # noqa: E402


def med(fn, repeats):
    fn()
    xs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs), min(xs), max(xs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--context", type=int, default=160)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    print(f"model={args.model} threads={torch.get_num_threads()} context={args.context}")
    print("decode step, all slots active (and with one slot active, same max_slots):")
    for S in args.slots:
        core = GPT2Core(hf, max_slots=S, max_seq_len=1024)
        ids = list(range(100, 100 + args.context))
        for s in range(S):
            core.prefill(s, ids)
        slots = list(range(S))
        all_active = lambda: core.decode(slots, [7] * S, [args.context] * S)  # noqa: E731
        one_active = lambda: core.decode([0], [7], [args.context])  # noqa: E731
        m, lo, hi = med(all_active, args.repeats)
        m1, _, _ = med(one_active, args.repeats)
        print(f"  max_slots {S:>2}: all active {m:6.2f} ms (min {lo:.2f} max {hi:.2f})  "
              f"= {m/S:5.2f} ms per sequence-token;  one active {m1:6.2f} ms")
    print("prefill, one sequence:")
    core = GPT2Core(hf, max_slots=1, max_seq_len=1024)
    for n in (16, 64, 128, 256, 512):
        ids = list(range(100, 100 + n))
        m, lo, hi = med(lambda: core.prefill(0, ids), 7)
        print(f"  {n:>4} tokens: {m:7.1f} ms (min {lo:.1f} max {hi:.1f})")


if __name__ == "__main__":
    main()
