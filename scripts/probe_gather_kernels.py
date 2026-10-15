"""Which torch op should gather KV blocks on this CPU?

    .venv/bin/python scripts/probe_gather_kernels.py --threads 32

Gathers B*nb blocks of [16, 12, 64] fp32 (49 KB each, GPT-2 small, block
size 16) out of a 512-block cache for small and large batches, with the
three obvious formulations. The paged decode path does this 24 times per
step, so fixed per-call cost matters at low concurrency and throughput at
high concurrency.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


def med(fn, repeats=50):
    fn()
    xs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1e6)
    return statistics.median(xs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--block-size", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    bs, H, hd, nblocks = args.block_size, 12, 64, 513
    cache = torch.randn(nblocks, bs, H, hd)
    cache2d = cache.view(nblocks, -1)
    print(f"threads={torch.get_num_threads()} block={bs} tokens x {H} heads x {hd} = {bs*H*hd*4/1024:.0f} KB per block")
    print(f"{'B x nb':>8} {'rows':>5} {'MB':>6} {'index us':>9} {'index_select us':>16} {'2D index_select us':>19} {'embedding us':>13}")
    for B, nb in ((1, 11), (2, 11), (8, 11), (16, 11), (32, 11), (8, 48), (16, 48)):
        idx = torch.randint(1, nblocks, (B * nb,))
        mb = B * nb * bs * H * hd * 4 / 1e6
        a = med(lambda: cache[idx])
        b = med(lambda: torch.index_select(cache, 0, idx))
        c = med(lambda: torch.index_select(cache2d, 0, idx))
        d = med(lambda: F.embedding(idx, cache2d))
        print(f"{B:>3}x{nb:<4} {B*nb:>5} {mb:>6.1f} {a:>9.0f} {b:>16.0f} {c:>19.0f} {d:>13.0f}")


if __name__ == "__main__":
    main()
