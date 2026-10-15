# Stage 0: naive serving

    request -> model.generate() -> complete -> next request

One request at a time. The whole request is handed to Hugging Face's
`generate()`. The only addition is a streamer that timestamps each emitted
token so the harness can report TTFT and TPOT.

## Setup

- Host: AMD EPYC 9R14, 64 cores, no GPU, 246 GiB RAM. torch 2.8.0+cpu (MKL),
  transformers 4.57.1. Everything runs in fp32 on the CPU.
- Model: `gpt2` (124M parameters, 497.8 MB of fp32 weights, 12 layers, 1024
  context). The host decides the model: GPT-2 small is the largest model
  where one request takes about 2 s here, which keeps a 32-request workload
  near one minute.
- torch threads: 32, chosen by `scripts/calibrate_threads.py` (prefill stops
  improving past 32 threads, decode never moves; raw numbers in
  `results/stage0/probes.md`).
- Workloads: the five presets in `miniserve/bench/workload.py`, 32 requests
  each, seed 0. Prompts are random token ids; output length is forced
  (`min_new_tokens == max_new_tokens`) so length is a controlled variable.
- Command: `scripts/run_stage.sh naive results/stage0 --threads 32`.
- Raw records: `results/stage0/naive_*.json`. Table below from
  `python -m miniserve.bench.compare results/stage0/*.json`.

## Results

| workload | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | queue p50 | queue p99 | e2e p50 | e2e p99 | busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline (32 at t=0) | 0.524 | 33.5 | 29.7 s | 58.6 s | 30 ms | 29.7 s | 58.6 s | 31.6 s | 60.5 s | 1.00 |
| closed8 | 0.522 | 33.4 | 13.4 s | 13.5 s | 30 ms | 13.4 s | 13.4 s | 15.3 s | 15.3 s | 1.00 |
| uniform (Poisson 0.4/s) | 0.434 | 27.8 | 1.66 s | 7.3 s | 30 ms | 1.62 s | 7.3 s | 3.5 s | 9.2 s | 0.83 |
| hetero (Poisson 0.2/s) | 0.210 | 26.9 | 5.3 s | 22.1 s | 30 ms | 5.2 s | 22.1 s | 7.1 s | 24.5 s | 0.81 |
| burst (4 x 8, 20 s apart) | 0.423 | 27.1 | 6.8 s | 13.6 s | 30 ms | 6.7 s | 13.6 s | 8.7 s | 15.5 s | 0.82 |

Head-of-line blocking, from the same records:

| workload | requests that waited longer than their own service | wait / own service, p50 | max | shortest-quartile requests: queue wait p50 | their service p50 |
|---|---:|---:|---:|---:|---:|
| offline | 31/32 | 15.7x | 30.3x | 29.7 s | 1.91 s |
| closed8 | 31/32 | 7.0x | 7.0x | 13.4 s | 1.91 s |
| uniform | 13/32 | 0.8x | 4.0x | 1.6 s | 1.92 s |
| hetero | 16/32 | 1.2x | 27.0x | 4.4 s | 0.55 s |
| burst | 28/32 | 3.5x | 7.0x | 6.7 s | 1.92 s |

Per-request service time for a 128-in/64-out request: 1.91 s on every
workload (min 1.89, max 2.01 s across all 128 such requests). That is 43 ms
of prefill plus 63 decode steps at 29.6 ms each.

## What the numbers say

Service time does not depend on the workload. Prefill is 43 ms and TPOT is
30 ms on all five workloads. Everything that differs between the rows is
queueing. On `offline`, 94% of median end-to-end latency is time spent in
the queue; on `closed8` it is 88%.

Throughput is capped at one request per service time. `offline` and
`closed8` both reach 0.52 req/s = 1 / 1.91 s and cannot go higher no matter
how many clients wait. Adding clients (closed8) only adds queue time: each
request waits for the seven others, 7.0x its own service time.

Head-of-line blocking is worst when lengths vary. On `hetero`, the request
with the largest wait-to-service ratio generated 9 tokens (0.35 s of work)
and waited 9.6 s, 27x its own service time, behind requests generating up
to 256 tokens. On `uniform`, with identical requests, the median request
waited less than its own service time (0.8x); heterogeneity is what turns a
queue into a lottery.

The server is 100% busy on `offline` while the machine is mostly idle. The
probes explain why. One decode step reads all 497.8 MB of weights once
(16.8 GB/s at 29.6 ms per step). The profiler splits the step into the LM
head matmul (12.0 ms, 154 MB, 12.9 GB/s), the 48 per-layer linears
(11.7 ms, 340 MB, 29 GB/s), and the KV-cache `cat` (2.1 ms); attention is
0.3 ms. The batch-1 step takes 28.4 ms with 1 thread and 28.7 ms with 32:
torch's M=1 matmul path (`linear`, `mm`, `mv`) does not use more than one
core on this build, so batch-1 decode streams half a gigabyte of weights
through one core's share of memory bandwidth, 31 cores idle.

A batch of 2 is cheaper per step than a batch of 1: 15.0 ms versus 28.7 ms.
With M=2 the matmuls go through the threaded GEMM path and read the same
weights 3x faster (the LM head: 3.6 ms versus 10.7 ms). From batch 2 to 32
the step grows slowly, 15.0 ms to 28.4 ms, about 0.45 ms per extra
sequence. At batch 32 a step costs the same wall time as a batch-1 step and
produces 32 tokens.

Prefill is cheap relative to decode here: a 512-token prompt costs 103 ms,
less than four decode steps. On this host the problem is decode, and decode
is a weight-bandwidth problem.

A caveat on CPU versus GPU. The jump from batch 1 to batch 2 being free is
an artifact of this torch build's single-threaded M=1 path. On a GPU, the
batch-1 step already uses the full memory system, so going to batch 2 costs
about the same step time rather than less. The shape beyond that is the
same: decode is bound by reading the weights, and a step's cost grows
slowly with batch size until compute saturates. The KV cache is not a
constraint yet: peak resident KV was 48.6 MiB (691 tokens, `hetero`) in a
246 GiB machine.

## The bottleneck

Serialization. The engine runs one sequence per step, so every step pays
the full weight read for one token, and every queued request waits for all
the tokens of every request ahead of it. Throughput is pinned at
1 / service time, and user-visible latency is dominated by queue time.

## Three candidate next steps

1. Static batching. Take up to B queued requests, pad them to a common
   length, run one `generate()` on the batch.
   - Prediction: on `offline`, `closed8` and `burst`, eight 128/64 requests
     cost 63 decode steps at 18.2 ms = 1.15 s plus one batched prefill,
     instead of 8 x 1.91 s = 15.3 s serially: about 10x the throughput and
     about 10x less queue wait. On `uniform` the queue is short (utilization
     0.77, median one request waiting), so batches will be small and the
     gain modest. On `hetero`, padding wastes work (short prompts padded to
     the longest, every sequence decoded to the longest output), and a
     short request still waits for the batch's longest member: HOL blocking
     moves inside the batch.
   - Risk: per-request TPOT becomes the batched step time, 15-18 ms at
     batch 2-8 but back to 28 ms at batch 32. Past batch 8, latency and
     throughput pull in opposite directions.

2. Asynchronous intake with a request queue, no batching. Run `generate()`
   on a worker thread so the event loop keeps accepting requests.
   - Expected: no change to any number in the table. The engine is still
     serial. This step is the plumbing that streaming, cancellation and a
     real scheduler need; measuring it should confirm that intake is not
     where the time goes.
   - Risk: none to performance; adds a thread hand-off. A result of "no
     change" is the point.

3. Make the batch-1 step cheaper. The step is 43% LM head, 42% layer
   linears, 8% KV `cat`. A preallocated cache would remove the `cat`
   (2.1 ms, 7%). A threaded GEMV would cut the LM head from 12 ms to about
   3.6 ms, but no torch M=1 path does that on this build (`linear`, `mm`
   and `mv` all run single-threaded), so this needs either a custom kernel
   or, in effect, batching.
   - Expected: at most 1.1x from the cache change alone; up to 1.7x if the
     LM head could be threaded.
   - Risk: this is a CPU backend quirk, not a serving-architecture lesson.
     Production CPU engines (llama.cpp) solve it with hand-written GEMV
     kernels; on a GPU the batch-1 GEMV is already bandwidth-bound at full
     speed and the only fix is batching.

## Decision

Do 2 first, then 1. Option 2 is small, predicted to change nothing, and the
prediction is worth confirming before anything else is built on top of the
intake path; every later stage needs it. Option 1 is the first real
throughput change and the measurement that shows padding waste and
in-batch HOL blocking, which is what motivates continuous batching next.
Option 3 is recorded as the explanation of the baseline's per-step cost,
not pursued: the project is about the serving architecture, and the one
part of it that generalizes (amortizing the weight read across sequences)
is exactly what batching does.
