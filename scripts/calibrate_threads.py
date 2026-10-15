"""Measure prefill and decode cost of one request versus torch thread count.

    .venv/bin/python scripts/calibrate_threads.py --model gpt2 --threads 4 8 16 32 64

On a many-core CPU, torch defaults to one intra-op thread per core, which is
usually wrong for the small matmuls of a small model. This picks the thread
count every later benchmark uses, so the choice is recorded, not guessed.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM


def time_one(model, prompt_len: int, gen_len: int, repeats: int):
    ids = torch.randint(0, 50000, (1, prompt_len))
    mask = torch.ones_like(ids)
    prefill, decode = [], []
    for _ in range(repeats):
        times = []

        class S:
            def put(self, v):
                times.append(time.perf_counter())

            def end(self):
                pass

        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(
                ids, attention_mask=mask, max_new_tokens=gen_len, min_new_tokens=gen_len,
                do_sample=False, pad_token_id=model.config.eos_token_id, streamer=S(),
            )
        prefill.append(times[1] - t0)  # times[0] is the prompt put()
        decode.append((times[-1] - times[1]) / (gen_len - 1))
    return statistics.median(prefill), statistics.median(decode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    print(f"model={args.model} prompt={args.prompt_len} gen={args.gen_len} repeats={args.repeats}")
    print(f"{'threads':>7} {'prefill_ms':>11} {'decode_ms_per_tok':>18}")
    for n in args.threads:
        torch.set_num_threads(n)
        time_one(model, args.prompt_len, 4, 1)  # warm-up
        pf, dc = time_one(model, args.prompt_len, args.gen_len, args.repeats)
        print(f"{n:>7} {pf*1000:>11.1f} {dc*1000:>18.2f}")


if __name__ == "__main__":
    main()
