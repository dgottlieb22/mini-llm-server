"""Attribute one decode step's CPU time to operators, for batch sizes 1 and 2.

    .venv/bin/python scripts/profile_decode_step.py --threads 32

Uses torch.profiler over 10 steps of a populated-cache forward and prints the
operators with the most self CPU time. Answers: is the batch-1 step bound by
the matmuls (weight bandwidth) or by everything around them?
"""
from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, DynamicCache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--context", type=int, default=160)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    for batch in args.batches:
        ids = torch.randint(0, 50000, (batch, args.context))
        with torch.inference_mode():
            cache = DynamicCache()
            out = model(ids, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(3):  # warm-up
                model(nxt, past_key_values=cache, use_cache=True)
                cache.crop(args.context)
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                for _ in range(args.steps):
                    model(nxt, past_key_values=cache, use_cache=True)
                    cache.crop(args.context)
        total_us = sum(e.self_cpu_time_total for e in prof.key_averages())
        print(f"\n=== batch {batch}: {total_us/args.steps/1000:.1f} ms per step (profiled, self time sum) ===")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top))


if __name__ == "__main__":
    main()
