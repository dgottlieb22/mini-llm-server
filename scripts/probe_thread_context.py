"""Is a batched generate() slower on a worker thread than on the main thread?

    .venv/bin/python scripts/probe_thread_context.py --batch 32

The engine runs the model on a ThreadPoolExecutor thread while the asyncio
loop idles on the main thread. This times the same generate() call from the
main thread, from a worker thread, and from a worker thread while the main
thread runs an asyncio loop, to see whether the calling context matters.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import statistics
import time

import torch
from transformers import AutoModelForCausalLM


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
    p.add_argument("--gen-len", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    eos = model.config.eos_token_id
    ids = torch.randint(0, 50000, (args.batch, args.prompt_len))
    mask = torch.ones_like(ids)

    def gen():
        s = S()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(ids, attention_mask=mask, max_new_tokens=args.gen_len, min_new_tokens=args.gen_len,
                           do_sample=False, pad_token_id=eos, streamer=s)
        steps = [(b - a) * 1000 for a, b in zip(s.times[1:], s.times[2:])]
        return (s.times[1] - t0) * 1000, statistics.median(steps), torch.get_num_threads()

    def report(label, r):
        print(f"{label:<40} prefill {r[0]:7.0f} ms   decode step median {r[1]:6.1f} ms   torch threads {r[2]}")

    gen()
    report("main thread", gen())
    report("main thread (again)", gen())
    with cf.ThreadPoolExecutor(1) as ex:
        report("worker thread", ex.submit(gen).result())
        report("worker thread (again)", ex.submit(gen).result())

    async def via_loop():
        loop = asyncio.get_running_loop()
        with cf.ThreadPoolExecutor(1) as ex:
            r1 = await loop.run_in_executor(ex, gen)
            r2 = await loop.run_in_executor(ex, gen)
        return r1, r2

    r1, r2 = asyncio.run(via_loop())
    report("worker thread under asyncio loop", r1)
    report("worker thread under asyncio loop (again)", r2)


if __name__ == "__main__":
    main()
