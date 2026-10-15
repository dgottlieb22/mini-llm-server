"""Check that batched generation produces the same tokens as one-at-a-time.

    .venv/bin/python scripts/check_batched_equivalence.py --threads 32

Builds a heterogeneous batch (prompt lengths 16-200, output lengths 8-40),
runs it through HFModelRunner.generate_batch and each request through
generate_one, and compares token ids. Greedy decoding on fp32 can still
differ between batched and single kernels at a near-tie in the logits, so
the report counts exact matches and the first differing position rather
than asserting bitwise equality.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniserve.model import HFModelRunner  # noqa: E402
from miniserve.request import GenerationRequest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    runner = HFModelRunner(args.model, torch_threads=args.threads)
    reqs = [
        GenerationRequest(f"r{i}", [rng.randrange(50000) for _ in range(rng.randint(16, 200))], rng.randint(8, 40))
        for i in range(args.batch)
    ]
    t0 = time.perf_counter()
    batched = runner.generate_batch(reqs)
    t_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    singles = [runner.generate_one(r) for r in reqs]
    t_single = time.perf_counter() - t0
    exact = 0
    for r, b, s in zip(reqs, batched, singles):
        same = b.output_ids == s.output_ids
        exact += same
        first_diff = next((k for k, (x, y) in enumerate(zip(b.output_ids, s.output_ids)) if x != y), None)
        print(f"{r.request_id}: prompt {r.prompt_len:>3} gen {r.max_new_tokens:>2}  "
              f"{'same' if same else f'differs from token {first_diff}'}  "
              f"(batched {len(b.output_ids)} tokens, single {len(s.output_ids)})")
    print(f"{exact}/{len(reqs)} identical. batch {t_batch:.2f} s vs singles {t_single:.2f} s "
          f"({t_single/t_batch:.1f}x), {max(r.max_new_tokens for r in reqs)} lockstep steps")


if __name__ == "__main__":
    main()
