"""Cold versus warm: time HFModelRunner.generate_batch repeatedly in one process.

    .venv/bin/python scripts/probe_cold_warm.py --batch 32 --repeats 4

The offline workload at max_batch_size=32 is a single generate() call in a
fresh process, so if the first call in a process is slower than later ones,
that benchmark measures a cold engine. This times the exact benchmark call
path (generate_batch with its stopping criteria and logits processor).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniserve.model import HFModelRunner  # noqa: E402
from miniserve.request import GenerationRequest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    runner = HFModelRunner(args.model, torch_threads=args.threads)
    reqs = [GenerationRequest(f"r{i}", [(i * 7919 + j) % 50000 for j in range(args.prompt_len)], args.gen_len)
            for i in range(args.batch)]
    for k in range(args.repeats):
        results = runner.generate_batch(reqs)
        r = results[0]
        prefill = (r.token_times[0] - r.service_start) * 1000
        steps = [(b - a) * 1000 for a, b in zip(r.token_times, r.token_times[1:])]
        total = (r.finish_time - r.service_start)
        print(f"call {k+1}: prefill {prefill:6.0f} ms   step median {statistics.median(steps):5.1f} ms  "
              f"(first 5 steps: {' '.join(f'{s:.0f}' for s in steps[:5])} ms)   total {total:.2f} s")


if __name__ == "__main__":
    main()
