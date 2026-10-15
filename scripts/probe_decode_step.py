"""Where does a decode step's time go: the model forward, or the loop around it?

    .venv/bin/python scripts/probe_decode_step.py --model gpt2 --threads 32

Measures (a) one bare forward pass of a single token with a populated KV
cache, (b) the per-token cost inside model.generate(), and (c) the forward
pass with batch size 1..32 so we can see whether a step's cost grows with
batch size (the question static batching hinges on).
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache


def median_ms(fn, repeats):
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
    p.add_argument("--context", type=int, default=160, help="tokens already in the KV cache")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--thread-sweep", type=int, nargs="*", default=[])
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    eos = model.config.eos_token_id

    def prefill(batch: int):
        ids = torch.randint(0, 50000, (batch, args.context))
        with torch.inference_mode():
            cache = DynamicCache()
            out = model(ids, past_key_values=cache, use_cache=True)
        return out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)

    def time_step(batch: int) -> float:
        cache, nxt = prefill(batch)
        xs = []
        for _ in range(args.repeats + 1):
            t = time.perf_counter()
            with torch.inference_mode():
                model(nxt, past_key_values=cache, use_cache=True)
            xs.append((time.perf_counter() - t) * 1000)
            cache.crop(args.context)  # untimed: keep the cache length fixed between repeats
        return statistics.median(xs[1:])  # drop the warm-up step

    print(f"model={args.model} threads={torch.get_num_threads()} context={args.context}")
    print("(a) bare forward, one new token, KV cache populated:")
    for batch in args.batches:
        ms = time_step(batch)
        print(f"    batch {batch:>2}: {ms:7.2f} ms/step  = {ms/batch:6.2f} ms per sequence-token")
    if args.thread_sweep:
        print("(a') batch-1 and batch-2 step vs thread count:")
        for n in args.thread_sweep:
            torch.set_num_threads(n)
            print(f"    threads {n:>2}: batch 1 {time_step(1):7.2f} ms   batch 2 {time_step(2):7.2f} ms")
        torch.set_num_threads(args.threads)

    print("(b) model.generate(), batch 1, per generated token:")
    ids = torch.randint(0, 50000, (1, args.context))
    times = []

    class S:
        def put(self, v):
            times.append(time.perf_counter())

        def end(self):
            pass

    with torch.inference_mode():
        model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.repeats + 1,
                       min_new_tokens=args.repeats + 1, do_sample=False, pad_token_id=eos, streamer=S())
    steps = [(b - a) * 1000 for a, b in zip(times[1:], times[2:])]
    print(f"    {statistics.median(steps):7.2f} ms per token (median of {len(steps)} steps)")

    print("(c) prefill cost vs prompt length, batch 1:")
    for n in (16, 64, 128, 256, 512):
        ids = torch.randint(0, 50000, (1, n))

        def pf():
            with torch.inference_mode():
                model(ids, use_cache=True)

        pf()
        print(f"    {n:>4} tokens: {median_ms(pf, 5):7.1f} ms")


if __name__ == "__main__":
    main()
