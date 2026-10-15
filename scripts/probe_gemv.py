"""Is a batch-1 linear layer threaded on this torch build?

    .venv/bin/python scripts/probe_gemv.py

Times x @ W for GPT-2-sized weights with M = 1, 2, 4 rows under several thread
counts, and sums the bytes of weights GPT-2 small reads per decode step. If
M=1 does not speed up with threads while M=2 does, batch-1 decode is a
single-threaded, memory-bound GEMV and batching is "free" up to the point
where the threaded GEMM catches up.
"""
from __future__ import annotations

import statistics
import time

import torch

print(torch.__config__.parallel_info().split("\n")[0])
print("BLAS:", "MKL" if torch.backends.mkl.is_available() else "other", "| mkldnn:", torch.backends.mkldnn.is_available())

K, N = 768, 3072  # GPT-2 small MLP up-projection
W = torch.randn(K, N)
params_per_step = sum(p.numel() for p in torch.nn.Linear(K, N).parameters())
print(f"W is {W.numel()*4/2**20:.1f} MiB; GPT-2 small reads ~{124e6*4/2**20:.0f} MiB of weights per decode step")


def t(fn, repeats=50):
    fn()
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(xs)


print(f"{'threads':>7} {'M=1 us':>9} {'M=2 us':>9} {'M=4 us':>9}  M=1 GB/s")
for n in (1, 4, 16, 32):
    torch.set_num_threads(n)
    row = []
    for m in (1, 2, 4):
        x = torch.randn(m, K)
        row.append(t(lambda: x @ W))
    gbs = W.numel() * 4 / (row[0] * 1e-6) / 1e9
    print(f"{n:>7} {row[0]:>9.0f} {row[1]:>9.0f} {row[2]:>9.0f}  {gbs:6.1f}")

# Does any M=1 formulation get threaded? LM-head sized (768 x 50257, 154 MiB:
# bigger than L3, so this one is DRAM-bound like the real step).
K2, N2 = 768, 50257
W2 = torch.randn(N2, K2)  # nn.Linear layout: [out, in]
x1 = torch.randn(1, K2)
v1 = torch.randn(K2)
print(f"\nLM-head sized W [{N2}x{K2}] = {W2.numel()*4/2**20:.0f} MiB")
print(f"{'threads':>7} {'linear(M=1) us':>15} {'mv us':>9} {'linear(M=2) us':>15}")
for n in (1, 32):
    torch.set_num_threads(n)
    a = t(lambda: torch.nn.functional.linear(x1, W2), repeats=10)
    b = t(lambda: torch.mv(W2, v1), repeats=10)
    c = t(lambda: torch.nn.functional.linear(torch.randn(2, K2), W2), repeats=10)
    print(f"{n:>7} {a:>15.0f} {b:>9.0f} {c:>15.0f}   (M=1 linear: {W2.numel()*4/(a*1e-6)/1e9:.1f} GB/s)")
