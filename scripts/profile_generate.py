"""Profile model.generate() at a given batch size, prefill and decode separately.

    .venv/bin/python scripts/profile_generate.py --batch 32 --prompt-len 128 --gen-len 9

Times the prefill step (the first streamer put after the prompt) and the
decode steps from the streamer timestamps, then re-runs under torch.profiler
and prints the heaviest operators. Compares against a bare forward at the
same batch and context so the generate() overhead is visible.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, DynamicCache


class S:
    def __init__(self):
        self.times = []

    def put(self, v):
        self.times.append(time.perf_counter())

    def end(self):
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=9)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    eos = model.config.eos_token_id
    ids = torch.randint(0, 50000, (args.batch, args.prompt_len))
    mask = torch.ones_like(ids)

    def gen(streamer):
        with torch.inference_mode():
            model.generate(ids, attention_mask=mask, max_new_tokens=args.gen_len, min_new_tokens=args.gen_len,
                           do_sample=False, pad_token_id=eos, streamer=streamer)

    gen(S())  # warm-up
    s = S()
    t0 = time.perf_counter()
    gen(s)
    total = time.perf_counter() - t0
    prefill = s.times[1] - t0
    steps = [(b - a) * 1000 for a, b in zip(s.times[1:], s.times[2:])]
    print(f"batch={args.batch} prompt={args.prompt_len} gen={args.gen_len} threads={torch.get_num_threads()}")
    print(f"generate(): total {total*1000:.0f} ms, prefill {prefill*1000:.0f} ms, "
          f"decode step median {statistics.median(steps):.1f} ms (min {min(steps):.1f}, max {max(steps):.1f})")

    # Bare forward at the same shapes.
    with torch.inference_mode():
        cache = DynamicCache()
        t0 = time.perf_counter()
        out = model(ids, attention_mask=mask, past_key_values=cache, use_cache=True)
        bare_prefill = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        out = model(ids, attention_mask=mask, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
        bare_prefill_1logit = (time.perf_counter() - t0) * 1000
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        cache = out.past_key_values
        xs = []
        for _ in range(args.gen_len):
            t0 = time.perf_counter()
            out = model(nxt, past_key_values=cache, use_cache=True)
            xs.append((time.perf_counter() - t0) * 1000)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    print(f"bare forward: prefill {bare_prefill:.0f} ms (all logits), {bare_prefill_1logit:.0f} ms (logits_to_keep=1); "
          f"decode step median {statistics.median(xs[1:]):.1f} ms")

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        gen(S())
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top))


if __name__ == "__main__":
    main()
