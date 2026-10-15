"""Decode step cost of the paged core (gather every step) versus the slot core.

    .venv/bin/python scripts/probe_paged_step.py --threads 32

For each batch size, both cores hold B sequences at the same context; one
step is timed. The paged core is also timed at several block sizes, since
block size sets the gather granularity (and the internal fragmentation the
notes discuss). Prefill is timed too: the paged core scatters K/V into
blocks after computing them.
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
from miniserve.paged import PagedGPT2Core  # noqa: E402


def med(fn, repeats):
    fn()
    xs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--block-sizes", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--context", type=int, default=160)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = list(range(100, 100 + args.context))
    print(f"model={args.model} threads={torch.get_num_threads()} context={args.context}")
    header = f"{'B':>3} {'slots ms':>9} " + " ".join(f"paged bs={bs:<3} ms" for bs in args.block_sizes)
    print("decode step:")
    print(header)
    for B in args.batches:
        slot_core = GPT2Core(hf, max_slots=B, max_seq_len=1024)
        for s in range(B):
            slot_core.prefill(s, ids)
        slot_ms = med(lambda: slot_core.decode(list(range(B)), [7] * B, [args.context] * B), args.repeats)
        row = [f"{B:>3} {slot_ms:>9.2f} "]
        for bs in args.block_sizes:
            core = PagedGPT2Core(hf, budget_tokens=B * 1024, block_size=bs, max_seq_len=1024)
            handles = [core.admit(args.context, 64) for _ in range(B)]
            for h in handles:
                core.prefill(h, ids)
            ms = med(lambda: core.decode(handles, [7] * B, [args.context] * B), args.repeats)
            row.append(f"{ms:>16.2f}")
        print("".join(row))
    print("prefill of one sequence (slots vs paged bs=16):")
    slot_core = GPT2Core(hf, max_slots=1, max_seq_len=1024)
    core = PagedGPT2Core(hf, budget_tokens=2048, block_size=16, max_seq_len=1024)
    for n in (16, 128, 512):
        pids = list(range(100, 100 + n))
        h = core.admit(n, 1)
        a = med(lambda: slot_core.prefill(0, pids), 7)
        b = med(lambda: core.prefill(h, pids), 7)
        print(f"  {n:>4} tokens: slots {a:6.1f} ms   paged {b:6.1f} ms")
        core.free(h)


if __name__ == "__main__":
    main()
