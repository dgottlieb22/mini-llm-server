# Stage 2: static batching

    request -> queue -> [engine loop: take up to B queued] -> generate(batch) -> results

When the model is free, the engine loop takes everything queued, up to
`max_batch_size`, left-pads the prompts to the longest one and runs a single
`generate()` over the batch. It does not wait for the batch to fill. Every
member decodes in lockstep until the longest output is done, and all
results return together. `scripts/check_batched_equivalence.py` confirmed
the batched path produces token-for-token the same greedy output as the
one-at-a-time path on 8 heterogeneous requests.

## Prediction (from the stage 0 notes)

About 10x throughput and 10x less queue wait on `offline`, `closed8` and
`burst` at B=8; a modest gain on `uniform`; on `hetero`, padding waste and
head-of-line blocking that moves inside the batch.

## Setup changes

- Two higher-load workloads, `uniform_hi` (Poisson 3 req/s) and `hetero_hi`
  (Poisson 1 req/s). The original rates were sized for the serial engine
  and leave a batching engine idle most of the time. Stage 0 was re-run on
  both for comparison; it is overloaded there (utilization above 1).
- Warm-up. The harness now serves a dummy batch at the workload's largest
  shape before the clock starts. See "Failures" below for why.
- `occupancy` (fraction of the run with at least one request in service)
  replaces `busy_fraction` as the utilization column; for a batching engine
  the sum of per-request service times exceeds the makespan.

Records: `results/stage2/*.json`; probes: `results/stage2/probes.md`.

## Results at B=8, against stage 0

| workload | stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | queue p50 | queue p99 | e2e p50 | e2e p99 | occupancy | avg batch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline | 0 | 0.524 | 33.5 | 29.7 s | 58.6 s | 30 ms | 29.7 s | 58.6 s | 31.6 s | 60.5 s | 1.00 | 1.0 |
| offline | 2 | 6.02 | 386 | 2.1 s | 4.1 s | 19 ms | 2.0 s | 4.0 s | 3.3 s | 5.3 s | 1.00 | 8.0 |
| closed8 | 0 | 0.522 | 33.4 | 13.4 s | 13.5 s | 30 ms | 13.4 s | 13.4 s | 15.3 s | 15.3 s | 1.00 | 1.0 |
| closed8 | 2 | 5.83 | 373 | 112 ms | 112 ms | 20 ms | 1 ms | 1 ms | 1.37 s | 1.38 s | 1.00 | 8.0 |
| burst | 0 | 0.423 | 27.1 | 6.8 s | 13.6 s | 30 ms | 6.7 s | 13.6 s | 8.7 s | 15.5 s | 0.82 | 1.0 |
| burst | 2 | 0.522 | 33.4 | 125 ms | 131 ms | 18 ms | 15 ms | 20 ms | 1.24 s | 1.25 s | 0.08 | 8.0 |
| uniform | 0 | 0.434 | 27.8 | 1.66 s | 7.3 s | 30 ms | 1.62 s | 7.3 s | 3.5 s | 9.2 s | 0.83 | 1.0 |
| uniform | 2 | 0.488 | 31.2 | 0.86 s | 2.0 s | 30 ms | 0.81 s | 1.9 s | 2.2 s | 3.8 s | 0.69 | 1.1 |
| hetero | 0 | 0.210 | 26.9 | 5.3 s | 22.1 s | 30 ms | 5.2 s | 22.1 s | 7.1 s | 24.5 s | 0.81 | 1.0 |
| hetero | 2 | 0.244 | 31.2 | 1.7 s | 6.8 s | 25 ms | 1.5 s | 6.7 s | 6.0 s | 12.1 s | 0.65 | 1.6 |
| uniform_hi | 0 | 0.515 | 33.0 | 25.9 s | 51.1 s | 30 ms | 25.8 s | 51.1 s | 27.8 s | 53.0 s | 1.00 | 1.0 |
| uniform_hi | 2 | 3.00 | 192 | 0.88 s | 1.9 s | 20 ms | 0.80 s | 1.8 s | 2.2 s | 3.3 s | 1.00 | 4.2 |
| hetero_hi | 0 | 0.255 | 32.7 | 39.0 s | 97.1 s | 30 ms | 38.9 s | 97.0 s | 43.3 s | 99.1 s | 1.00 | 1.0 |
| hetero_hi | 2 | 0.887 | 114 | 3.5 s | 8.2 s | 26 ms | 3.0 s | 8.0 s | 9.9 s | 13.4 s | 1.00 | 5.8 |

Batch size sweep on `offline` (32 identical 128/64 requests, all present at t=0):

| B | req/s | out tok/s | TPOT | prefill p50 (B x 128 tokens, single run, noisy: see text) | queue p50 | e2e p99 | peak KV tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (stage 0) | 0.524 | 33.5 | 29.6 ms | 43 ms | 29.7 s | 60.5 s | 192 |
| 4 | 3.61 | 231 | 16.5 ms | 66 ms | 3.9 s | 8.9 s | 768 |
| 8 | 6.02 | 386 | 19.3 ms | 111 ms | 2.0 s | 5.3 s | 1536 |
| 16 | 8.91 | 570 | 24.9 ms | 227 ms | 0.9 s | 3.6 s | 3072 |
| 32 | 10.48 | 671 | 32.2 ms | 1000 ms | 1 ms | 3.1 s | 6144 |

Padding and lockstep waste, from the engine's own counters:

| workload (B=8) | batch sizes seen | prompt positions that were padding | decode slots spent on finished sequences | service time spent holding a finished result | longest hold |
|---|---|---:|---:|---:|---:|
| offline, closed8, burst, uniform_hi | 8,8,8,8 / mixed | 0% | 0% | 0% | 0 |
| uniform | 20 x 1, 6 x 2 | 0% | 0% | 0% | 0 |
| hetero | 14 x 1, 4 x 2, 2 x 3, 1 x 4 | 26% | 19% | 16% | 5.7 s |
| hetero_hi | 3 x 1, 6, 7, 8, 8 | 43% | 42% | 38% | 6.2 s |

## What the numbers say

The prediction held where it was specific. `offline` went from 0.524 to
6.02 req/s (11.5x) and `closed8` from 0.522 to 5.83 req/s (11.2x); median
queue wait on `offline` fell 14.9x. `burst` is now bounded by its own
arrival schedule: each burst of 8 is served as one batch in 1.24 s and the
engine idles 92% of the time, so throughput equals the arrival rate and
TTFT is 125 ms. On `uniform` at the stage 0 rate the queue rarely holds more
than one request (20 of 26 batches had size 1), so throughput moved 12%
and the gain is in the tail: queue p99 7.3 s to 1.9 s.

The two overloaded workloads show what batching is for. At 3 req/s of
identical requests, stage 0 cannot keep up (TTFT p50 25.9 s and growing
with every request); stage 2 keeps pace at an average batch of 4.2 with
TTFT p50 0.88 s. At 1 req/s of heterogeneous requests, stage 0 is at 39 s
median TTFT; stage 2 holds 3.5 s.

A sequence in a batch of 8 finishes sooner than a sequence alone: TPOT
19.3 ms against 29.6 ms, and a 128/64 request takes 1.33 s instead of
1.91 s. This is the CPU quirk from the stage 0 probes (the batch-1 matmul
path is single-threaded) and will not transfer to a GPU, where a batch of 8
costs slightly more per step than a batch of 1, not less.

Throughput keeps rising with B while per-token latency falls then rises.
From B=4 to B=32, output tokens/s goes 231, 386, 570, 671: each doubling of
the batch buys less (1.67x, 1.48x, 1.18x). TPOT goes the other way after
B=4: 16.5, 19.3, 24.9, 32.2 ms. At B=32 a token takes as long as it did
serially, and the engine produces 20x as many of them. The profile at B=32
(`results/stage2/probes.md`) shows why the step grows: the weight-streaming
matmuls are now 42% of the step and the elementwise GELU and residual ops
over [32, 3072] activations plus the KV `cat` are a third.

The prefill column in the sweep should not be read as a curve. In the
benchmark the 32 x 128 prefill took 1.0 s against 111 ms for 8 x 128, which
looks superlinear, but `scripts/profile_prefill.py` run on the idle host
gives 565-897 ms over 9 repeats for the 32 x 128 shape and 196-214 ms for
8 x 128: per-token prefill throughput is roughly flat at 5-6 tokens/ms
(1.2-1.5 TFLOP/s) from B=4 up, with run-to-run variance of up to 1.6x at
the larger shapes. The first draft of these notes blamed the LM head
(logits for every prompt position); that was wrong, `generate()` passes
`logits_to_keep=1` and GPT-2 honors it. Prefill cost is linear in tokens
here and noisy; it becomes important at the next stage, where prefills run
inside the decode loop, and will be measured there with repeats.

Heterogeneous lengths turn static batching into a waste machine. On
`hetero_hi` with batches of 6-8, 43% of prompt positions were padding and
42% of decode slots were spent on sequences that had already produced
their last token. 38% of all service time was results sitting in a
finished state waiting for the batch to end; one request waited 6.2 s after
its own last token. The batch that started at 13.28 s had members
generating 28, 29 and 36 tokens next to ones generating 235 and 237; the
three short ones were done 1.8 s into a 7.0 s batch and their slots then
ran empty for 5.2 s.

The batch boundary is the new head-of-line block. 31 of the 32 `hetero_hi`
requests arrived while a batch was running, and each waited for that whole
batch to end (median 3.1 s) because the engine admits only between
batches. During those waits the batch had free slots: the moment a short
member finished, its slot stayed idle until the longest member was done.
The HOL summary says the same thing from the request's side: on `hetero`
the worst wait is still 7.5x the request's own service time (down from
27x), and the shortest-quartile requests' service time went up from 0.55 s
to 0.81 s because some of them are now held in a batch with a long member.

KV footprint rose with concurrency as expected: peak resident tokens went
from 192 (one 128/64 sequence) to 1536 at B=8 and 6144 at B=32 on
`offline`, 108 MiB and 432 MiB of fp32 KV for GPT-2 small. The padded
allocation is larger than the resident count: on `hetero_hi` the engine's
padded peak was 6080 tokens where 3453 were real. Still nowhere near a
constraint on a 246 GiB host.

## Failures and corrections

- The first B=32 run reported 55.5 ms TPOT and a 1.35 s prefill, against
  the stage 0 probe's 28.4 ms for a bare batch-32 step. Two B=16/B=32 runs
  had been launched concurrently and were discarded, but a solo re-run
  still gave 55.5 ms. Three probes narrowed it: `generate()` adds nothing
  over a bare forward at B=32 (33.9 vs 34.1 ms); running on the engine's
  worker thread under an asyncio loop changes nothing (30-33 ms either way);
  the first `generate_batch` call in a fresh process is 1.7x slower than
  the second (80 ms median step vs 34 ms), and the `offline` B=32 run is
  exactly one call in a fresh process. The harness now runs a warm-up
  batch at the workload's largest shape before t0, the way a serving
  engine does at startup. Stage 0 and 1 records were kept: the first
  request in those runs was within 0.08 s of the median (1.93-1.99 s vs
  1.91 s), so the cold cost for a batch-1 engine is one request's worth
  of a few percent.
- The first `closed8` record at stage 0 was invalid: closed-loop workers
  did not yield after stamping an arrival, so with a blocking engine one
  worker drained the queue alone. Fixed in the harness and re-measured
  before the stage 0 notes were written.

## The bottleneck

Lockstep. The batch is a fixed set with one length. Everything the engine
wastes at this stage traces to that: padded prompt positions, decode slots
run for finished sequences, results held until the slowest member is
done, and new arrivals held at the door while slots inside sit idle. The
single-step cost is now reasonable (0.6-1.0 ms per sequence-token from B=8
to B=32); what is wrong is which sequences occupy those slots.

## Three candidate next steps

1. Continuous batching: schedule at the step level. Keep one running
   batch; at every decode step, drop sequences that finished and admit
   queued requests into the freed slots (prefill them, then they join the
   decode step). This means taking the loop out of `generate()` and
   owning it: our own prefill call, our own decode step, our own KV cache
   bookkeeping per sequence.
   - Prediction: on `hetero_hi`, decode-slot waste goes from 42% to near 0
     and the batch-boundary wait disappears, so queue p50 should drop from
     3.0 s toward the time to the next step (tens of ms) whenever a slot is
     free; throughput rises by roughly the reclaimed slot fraction. On the
     homogeneous workloads nothing changes (there was no waste to
     reclaim). TPOT may get worse by the cost of the prefills that now run
     inside the decode loop: a 512-token prefill is 100 ms, which every
     other sequence in the batch waits through.
   - Risk: the biggest code change in the project, and it is where the KV
     cache becomes our problem: sequences of different lengths in one
     batch need per-sequence cache management, which Hugging Face's padded
     `DynamicCache` does not give us cleanly.

2. Length-aware batch formation. Sort the queue by prompt length (and by
   requested output length, which this benchmark knows) and group similar
   requests, so padding and lockstep waste shrink without changing the
   execution model.
   - Prediction: on `hetero_hi` padding would fall from 43% toward a few
     percent and slot waste similarly when the queue is deep enough to
     choose from; no change on homogeneous workloads.
   - Risk: it works here only because forced output lengths make the
     output length known. A real server does not know how many tokens a
     request will generate, and prompt-length grouping alone leaves the
     decode-slot waste (42%) in place. Sorting also starves long requests
     unless an aging rule is added. This is why production engines did
     not stop here.

3. Fill-wait batching. Hold a batch open for up to T ms to collect more
   requests before launching it.
   - Prediction: larger average batches on `uniform` (1.1) and `uniform_hi`
     (4.2) at the cost of up to T ms of TTFT on every request; throughput
     improves only when the engine is the bottleneck, so `uniform_hi`
     would gain and `uniform` would only get slower.
   - Risk: a latency knob with no effect on the waste inside a batch; it
     trades TTFT for batch size and does nothing about lockstep.

## Decision

Option 1. The measurements point at lockstep, and options 2 and 3 work
around it rather than remove it; option 2 also depends on knowing output
lengths, which real serving does not. Continuous batching is the change
production engines made for exactly the `hetero_hi` numbers above, and it
is the step that puts KV-cache management into our code, which the next
stages (explicit KV lifecycle, paged allocation) need anyway. Build it
with the prediction above as the check, and measure the prefill-inside-
the-loop cost explicitly, since that is the tradeoff continuous batching
introduces.
