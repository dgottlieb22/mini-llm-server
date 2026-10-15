"""Per-operator profile of one decode step: paged core vs slot core.

    .venv/bin/python scripts/profile_paged_step.py --threads 32 --batch 2 --context 160

At low concurrency the paged step measured 7-8 ms slower than the slot step
in the stage 4 runs although the gathered volume is tiny; this attributes
the difference to operators.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from miniserve.gpt2 import GPT2Core  # noqa: E402
from miniserve.paged import PagedGPT2Core  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--context", type=int, default=160)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = list(range(100, 100 + args.context))
    B = args.batch

    slot = GPT2Core(hf, max_slots=B, max_seq_len=1024)
    for s in range(B):
        slot.prefill(s, ids)
    paged = PagedGPT2Core(hf, budget_tokens=B * 1024, block_size=args.block_size, max_seq_len=1024)
    handles = [paged.admit(args.context, 64) for _ in range(B)]
    for h in handles:
        paged.prefill(h, ids)

    for name, fn in (("slots", lambda: slot.decode(list(range(B)), [7] * B, [args.context] * B)),
                     ("paged", lambda: paged.decode(handles, [7] * B, [args.context] * B))):
        for _ in range(3):
            fn()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(args.steps):
                fn()
        total = sum(e.self_cpu_time_total for e in prof.key_averages()) / args.steps / 1000
        print(f"\n=== {name}: batch {B}, context {args.context}: {total:.1f} ms self-time per step ===")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top))


if __name__ == "__main__":
    main()
