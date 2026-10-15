"""Prefill cost with and without a cached prefix, same prompt.

    .venv/bin/python scripts/probe_prefix_prefill.py --threads 32

For prompt lengths 288..512 with a 256-token shared prefix: time the full
prefill (cold cache) and the tail-only prefill (prefix resident), and the
plain paged core's prefill of the same prompt for reference.
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

from miniserve.paged import PagedGPT2Core  # noqa: E402
from miniserve.prefix_cache import PrefixCachedGPT2Core  # noqa: E402


def med(fn, repeats):
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
    p.add_argument("--prefix", type=int, default=256)
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    prefix = [(i * 7919 + 17) % 50000 for i in range(args.prefix)]
    plain = PagedGPT2Core(hf, budget_tokens=4096, block_size=16, max_seq_len=1024)
    cached = PrefixCachedGPT2Core(hf, budget_tokens=4096, block_size=16, max_seq_len=1024)
    print(f"model={args.model} threads={torch.get_num_threads()} shared prefix={args.prefix}")
    print(f"{'prompt':>6} {'tail':>5} {'paged full ms':>14} {'cached, cold ms':>16} {'cached, hit ms':>15}")
    for total in (288, 320, 384, 448, 512):
        prompt = prefix + [(i * 104729 + 3) % 50000 for i in range(total - args.prefix)]

        def plain_full():
            h = plain.admit(total, 1)
            plain.prefill(h, prompt)
            plain.free(h)

        def cached_cold():
            # empty the index so this is a miss
            for h in list(cached.index.keys()):
                b = cached.index.pop(h)
                cached.refcount.pop(b, None)
                cached.block_hash.pop(b, None)
                cached.allocator.free([b])
            h = cached.admit_prompt(prompt, 1)
            cached.prefill(h, prompt)
            cached.free(h)

        def cached_hit():
            h = cached.admit_prompt(prompt, 1)
            assert h.cached_tokens >= args.prefix, h.cached_tokens
            cached.prefill(h, prompt)
            cached.free(h)

        a = med(plain_full, args.repeats)
        b = med(cached_cold, args.repeats)
        cached_cold()  # leave the prefix resident
        c = med(cached_hit, args.repeats)
        print(f"{total:>6} {total-args.prefix:>5} {a:>14.1f} {b:>16.1f} {c:>15.1f}")


if __name__ == "__main__":
    main()
