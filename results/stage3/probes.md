# Stage 3 probes (raw output)

Same host and model as stage 0. Run 2026-10-15.

## scripts/probe_core_steps.py --threads 32

Our own forward (GPT2Core) over the slot cache, against the Hugging Face
numbers in `results/stage0/probes.md` (bare batch-1 step 28.7 ms, batch-8
step 18.2 ms, 128-token prefill 38 ms).

```
model=gpt2 threads=32 context=160
decode step, all slots active (and with one slot active, same max_slots):
  max_slots  1: all active  26.70 ms (min 26.49 max 26.93)  = 26.70 ms per sequence-token;  one active  26.67 ms
  max_slots  2: all active  12.34 ms (min 12.24 max 12.51)  =  6.17 ms per sequence-token;  one active  12.34 ms
  max_slots  4: all active  12.52 ms (min 12.39 max 12.79)  =  3.13 ms per sequence-token;  one active  12.46 ms
  max_slots  8: all active  14.83 ms (min 14.70 max 14.98)  =  1.85 ms per sequence-token;  one active  14.89 ms
  max_slots 16: all active  19.69 ms (min 19.57 max 20.35)  =  1.23 ms per sequence-token;  one active  19.67 ms
  max_slots 32: all active  23.65 ms (min 23.36 max 24.11)  =  0.74 ms per sequence-token;  one active  23.55 ms
prefill, one sequence:
    16 tokens:    24.7 ms (min 24.3 max 24.8)
    64 tokens:    30.8 ms (min 30.5 max 30.9)
   128 tokens:    37.6 ms (min 37.0 max 38.1)
   256 tokens:    49.2 ms (min 48.8 max 49.4)
   512 tokens:    71.8 ms (min 71.6 max 72.3)
```

The step is 15-20% cheaper than Hugging Face's at every batch size. The
difference is the KV cache: ours is preallocated and written in place,
theirs is re-concatenated every step (`aten::cat` was 8-16% of their step
in the stage 0 and 2 profiles). "One active" equals "all active" by
construction: the engine computes every slot.

## scripts/probe_step_gaps.py --threads 32 --slots 8

Does idle time between steps (the asyncio loop doing bookkeeping while the
model thread waits) slow the next step?

```
main thread, idle gap before each step:
  gap   0.0 ms: step  14.51 ms
  gap   0.5 ms: step  14.56 ms
  gap   1.0 ms: step  14.52 ms
  gap   2.0 ms: step  14.53 ms
  gap   5.0 ms: step  14.53 ms
  gap  20.0 ms: step  14.64 ms
worker thread (ThreadPoolExecutor), main thread idle:
  gap   0.0 ms: step  14.65 ms
  ...
  gap  20.0 ms: step  14.69 ms
worker thread steps, main thread runs clone+argmax on the logits between steps:
  step  16.47 ms (measured around the worker call only)
  step  16.34 ms when argmax runs on the worker thread too
```

No: gaps up to 20 ms cost nothing, and the worker-thread hop is 0.1 ms.

## scripts/probe_engine_step.py --threads 32 --slots 8 (offline workload, real engine)

Before the fix (argmax over the [8, 50257] logits ran on the asyncio thread):

```
core.decode inside worker: median 16.56 ms  (min 15.46, max 35.70)
engine decode_time (executor round trip): mean 18.13 ms
_step wall (incl. pick + bookkeeping): median 21.24 ms
gap between steps (admit/prefill/loop): median 0.01 ms, max 345.5 ms, total 1.27 s
prefill: 32 x 39.2 ms
```

After moving token selection onto the model thread (the asyncio thread
handles only Python ints):

```
core.decode inside worker: median 15.81 ms  (min 15.41, max 16.22)
engine decode_time (executor round trip): mean 16.22 ms
_step wall (incl. pick + bookkeeping): median 16.23 ms
gap between steps (admit/prefill/loop): median 0.00 ms, max 298.3 ms, total 1.18 s
prefill: 32 x 36.7 ms
```

Per-step overhead outside the model went from 4.7 ms to 0.4 ms, and the
decode itself got 0.8 ms faster (16.6 to 15.8 ms). Two facts, one
hypothesis. Facts: the `clone()` + `argmax` over the [8, 50257] logits cost
about 3 ms when run on the asyncio thread, and about 1 ms (including the
hop) when run on the worker; and in the gap probe above, a main-thread
argmax between steps slowed the worker's step by 1.2 ms even though the
step was timed on its own. Hypothesis consistent with both: a torch op on
the main thread starts a second OpenMP team of 32 threads in the same
process, which is slow to spin up for one small op and then competes with
the worker's team for cores while it spins down. Not verified further; the
fix (never touch a tensor on the asyncio thread) removes it either way.
The `max 345 ms` gap is the sequential prefill of 8 admitted requests
between two decode steps.
