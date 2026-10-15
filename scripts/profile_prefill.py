"""Profile one prefill forward at several batch sizes (fixed 128-token prompts).

    .venv/bin/python scripts/profile_prefill.py --batches 8 32

Prefill cost per prompt token halves from B=8 to B=32 in the stage 2 runs
(9.2 to 4.1 tokens/ms). This attributes the forward to operators at each
shape so the notes can say what grew, instead of guessing.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[8, 32])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--profile", action="store_true", help="also print the per-operator profile")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    for batch in args.batches:
        ids = torch.randint(0, 50000, (batch, args.prompt_len))
        mask = torch.ones_like(ids)

        def fwd():
            with torch.inference_mode():
                model(ids, attention_mask=mask, use_cache=True, logits_to_keep=1)

        fwd()
        xs = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fwd()
            xs.append((time.perf_counter() - t0) * 1000)
        ms = statistics.median(xs)
        tokens = batch * args.prompt_len
        print(f"\n=== batch {batch} x {args.prompt_len} = {tokens} tokens: {ms:.0f} ms median, "
              f"min {min(xs):.0f}, max {max(xs):.0f} over {args.repeats} "
              f"({tokens/ms:.1f} tok/ms, {2*124e6*tokens/ms/1e9:.2f} TFLOP/s) ===")
        if not args.profile:
            continue
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            fwd()
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top))


if __name__ == "__main__":
    main()
