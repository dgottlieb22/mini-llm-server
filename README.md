# mini-llm-server

Starting with naive `model.generate()`, I progressively implemented the core
ideas behind modern LLM serving: asynchronous intake, static batching,
continuous batching over an explicit KV cache, paged (block) KV allocation,
batched prefill, prefix caching, and optimistic allocation with preemption.
Hugging Face runs the model's weights; scheduling, request state, batching,
cache allocation and the forward pass over the cache are in this repository.

Every stage was an experiment. Before changing code I wrote down the
bottleneck, the prediction, the measurement that would confirm it and the
tradeoff; after measuring I wrote what happened. Those records are
`notes/stage0.md` through `notes/stage7.md`; the raw benchmark records and
probe outputs they cite are committed under `results/`. Nothing in this
README was typed in by hand: the tables come from
`scripts/summary_table.py` over the committed records.

Everything ran on a CPU-only host (AMD EPYC 9R14, 64 cores, 32 torch
threads) with GPT-2 small (124M parameters) in fp32. GPU utilization and GPU
memory are recorded as "not measured" in every record. The serving problems
the stages address exist on both kinds of host; the per-step numbers do not
transfer, and the notes say where a CPU result would come out differently
on a GPU.

## The story in one table

Throughput and latency of each stage on `offline` (32 identical requests
present at t=0) and `hetero_4x` (64 requests of 16-512 prompt and 8-256
output tokens arriving at 4 per second, enough to saturate 8 concurrent
sequences). Full tables for all eleven workloads are below.

| stage | what changed | offline req/s | offline TTFT p99 | hetero_4x req/s | hetero_4x TTFT p50 |
|---|---|---:|---:|---:|---:|
| 0 | one request at a time through `generate()` | 0.52 | 58.6 s | | |
| 1 | async intake + FIFO queue, model still serial | 0.52 | 59.4 s | | |
| 2 | static batching, up to 8 padded requests per `generate()` | 6.02 | 4.1 s | | |
| 3 | continuous batching: our own forward over a slot KV cache, admit/retire per step | 6.07 | 4.2 s | 3.35 | 2.19 s |
| 4 | paged KV: block allocator, 2-4x the sequences from the same bytes | 12.02 | 1.2 s | 3.03 | 1.05 s |
| 5 | co-arriving requests prefill in one padded forward | 15.51 | 0.5 s | 3.11 | 0.73 s |
| 6 | prefix caching (content-hashed blocks, refcounts) | | | | |
| 7 | admit on the prompt alone, grow per step, preempt youngest when the pool is empty | | | 3.25 | 0.11 s |

Stage 6 is measured on its own workloads with a shared 256-token prefix
(below). The stage 4 regression on `hetero_4x` is real and explained in
[notes/stage4.md](notes/stage4.md): on this CPU the per-step gather of
non-contiguous blocks costs about as much as the attention it feeds, and
long contexts make the step KV-bound. vLLM's PagedAttention is a kernel for
exactly that reason.

## What I expected and what happened

Stage 0 (`naive`). Expected a serial baseline. Found that service time is
identical on every workload (43 ms prefill, 29.6 ms per token) and every
latency difference between workloads is queueing: on `offline` 94% of
median latency is time in the queue, and on heterogeneous arrivals a
9-token request waited 27x its own service time. Probes found why the step
costs what it does: one decode step streams all 498 MB of weights through
a single core, because this torch build's M=1 matmul path is
single-threaded (28.4 ms with 1 thread, 28.7 ms with 32). A batch of 2 is
cheaper per step than a batch of 1. [notes/stage0.md](notes/stage0.md)

Stage 1 (`async_serial`). Predicted no change; measured no change (within
1%). Accepting requests promptly does nothing when the model loop is still
serial. The gain is in a test: the event loop stays responsive while the
model runs, which every later stage needs.
[notes/stage1.md](notes/stage1.md)

Stage 2 (`static_batch`). Predicted about 10x on offline and closed-loop
workloads; measured 11.5x and 11.2x. On heterogeneous lengths the cost
showed up as predicted: 43% of prompt positions were padding, 42% of
decode slots ran for sequences that had already finished, and 38% of
service time was finished results waiting for the batch's longest member.
31 of 32 requests arrived while a batch ran and waited for the whole batch,
with idle slots inside it. [notes/stage2.md](notes/stage2.md)

Stage 3 (`continuous`). Replacing `generate()` with our own GPT-2 forward
(120 lines, checked against Hugging Face logits to 1e-4) over an explicit
per-slot KV cache, with admission and retirement at every decode step.
Predicted the batch-boundary wait would vanish; on `hetero_hi` median queue
wait went from 3.0 s to 8 ms and TTFT p99 from 8.2 s to 100 ms, with zero
requests waiting longer than their own service time. Our decode step is
15-20% cheaper than Hugging Face's (preallocated cache written in place
instead of re-concatenated). Two new costs: co-arriving requests were
prefilled one at a time with decode stalled (23-25% of the makespan when
saturated), and the one-slot-per-sequence reservation was 70-93% empty at
the busiest moment while being what capped concurrency at 8.
[notes/stage3.md](notes/stage3.md)

Stage 4 (`paged`). Fixed-size 16-token blocks, block tables, conservative
admission (prompt + requested output), a gather of each sequence's blocks
into a contiguous batch every step. Predicted 2-4x the sequences from the
same 8192-token budget; measured 32 sequences at once on `offline` (12.0
vs 6.1 req/s) and 15 on `uniform_8x` (TTFT p99 2.3 s to 114 ms). Measured
the risk too: on long contexts the gather doubles KV traffic and the paged
engine loses to slots (3.03 vs 3.35 req/s on `hetero_4x`).
[notes/stage4.md](notes/stage4.md)

Stage 5 (batched prefill). Predicted closed8/burst TTFT p99 from about
300 to about 110 ms; measured 106 and 125 ms. `offline` did better than
predicted (12.0 to 15.5 req/s): one 517 ms forward for 32 prompts instead
of 32 forwards, and every sequence entering the decode loop at the same
step. [notes/stage5.md](notes/stage5.md)

Stage 6 (`prefix_cached`). Predicted about 3x cheaper prefill with a
256-token shared prefix; measured 1.4x (63 to 44 ms). 63% of prompt tokens
were served from cache and physical KV at the peak halved, but a tail-only
prefill costs 26 ms whether the tail is 32 tokens or 256: that is one
forward pass through the model on this host, and caching removes tokens,
not forwards. [notes/stage6.md](notes/stage6.md)

Stage 7 (`preempting`). Admit on the prompt alone, grow by a block per
step, preempt the youngest sequence (recompute on readmission) when the
pool runs out. Predicted 20-30% more sequences at the same budget; measured
17% at a 4096-token budget, where conservative admission had held 33% of
the pool for outputs not yet generated. With no growth reserve the engine
thrashed (26 preemptions, 30% extra prefill work); one spare block per
running sequence cut that to 3. [notes/stage7.md](notes/stage7.md)

## Failed approaches and corrections

- The first B=32 static-batching run was 2x slower than the probes
  predicted. Three probes traced it to the first `generate()` call in a
  fresh process (80 ms steps against 34 ms warm). The harness now warms the
  engine before the clock starts; the cold records are kept under
  `results/stage2/cold/`.
- I first blamed the LM head for a superlinear prefill cost at batch 32
  (logits for every prompt position). Wrong: `generate()` already passes
  `logits_to_keep=1`. The cost was run-to-run variance in a 50 MB
  activation allocation. The notes keep the retraction.
- Stage 3's first version picked the next token on the asyncio thread: 4.7
  ms of overhead per 16 ms step. Moving token selection onto the model
  thread (the asyncio thread never touches a tensor) cut it to 0.4 ms.
- Stage 4's first version allocated the gathered KV fresh every call. Above
  glibc's 32 MB mmap threshold each call page-faults the whole buffer: 5 ms
  for a 38 MB gather, and a 75 ms decode step on `hetero_4x`. A buffer
  allocated once brought the step to 31 ms. Records under
  `results/stage4/alloc-churn/`.
- A counter leak: core counters were not reset after the warm-up, so stage
  6's first runs reported more cache hits than the workload could produce.
  Fixed; the stage 4 attention-padding ratio was corrected from 1.56 to
  1.93 by a re-run.
- Two bugs the tests caught while building stage 7: a request preempted
  twice lost its first segment, and the stage 6 cache could evict a hit
  block during that request's own admission. Both fixed before measuring.

## Performance bottlenecks, stage by stage

| stage | bound by |
|---|---|
| 0, 1 | serialization: one sequence per weight read; latency is queueing |
| 2 | lockstep: padding, finished sequences holding slots, arrivals held at the batch boundary |
| 3 | KV reservation (concurrency cap) and sequential prefill of co-arrivals |
| 4 | per-step KV traffic, doubled by the gather, on long contexts |
| 5 | same, with co-arrival prefill fixed; prefill is now compute, not serialization |
| 6 | a one-forward floor per prefill (26 ms) that caching cannot remove |
| 7 | TPOT tail under preemption; the gather remains |

## What production engines do differently

vLLM's PagedAttention is a kernel that reads the block table inside
attention, so a paged cache costs no gather; here the gather is a plain
`index_select` and on long contexts it costs as much as the attention it
feeds (stage 4). vLLM also caches decode blocks for prefix reuse, swaps
preempted sequences to host memory when available, and chunks long
prefills so a decode step is never stalled for a whole prompt; none of
those apply or pay off on a host where a decode step is 15-30 ms. The
scheduler shape (admit at the step boundary, grow by a block, watermark
before preempting, youngest-first recompute) is the same one this project
arrived at by measurement.

Two CPU-specific findings would not appear on a GPU. The batch-1 decode
step here is single-threaded weight streaming, so a batch of 2 is cheaper
than a batch of 1 and the slot core's always-8 batch beat the paged core's
active-only batch at low load; on a GPU the batch-1 step already uses the
whole memory system. And allocator page faults above the mmap threshold
cost milliseconds per call; CUDA's caching allocator hides the equivalent.

## What I would build next

In the order the measurements point: a paged-attention kernel (Triton, on
a GPU host), chunked prefill, then quantized weights and speculative
decoding, each as a stage with its prediction written down first.

## Results, all workloads

Generated by `scripts/summary_table.py`. Row labels name the configuration;
side experiments (live-token bounds, zero growth reserve, other block
sizes) are in the per-stage notes.

**offline**: All 32 requests present at t=0: pure throughput.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 naive | 0.52 | 34 | 29708 ms | 58622 ms | 30 ms | 60.5 s |
| 1 async intake | 0.52 | 33 | 30496 ms | 59372 ms | 30 ms | 61.2 s |
| 2 static batching, B=8 | 6.02 | 385 | 2105 ms | 4097 ms | 19 ms | 5.3 s |
| 3 continuous, 8 slots | 6.07 | 388 | 2150 ms | 4245 ms | 18 ms | 5.3 s |
| 4 paged, 8192 tokens | 12.02 | 770 | 611 ms | 1173 ms | 33 ms | 2.7 s |
| 5 + batched prefill | 15.51 | 992 | 517 ms | 517 ms | 25 ms | 2.1 s |

**closed8**: 8 closed-loop clients, each with one request in flight.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 naive | 0.52 | 33 | 13437 ms | 13464 ms | 30 ms | 15.3 s |
| 1 async intake | 0.51 | 33 | 13454 ms | 13928 ms | 30 ms | 15.9 s |
| 2 static batching, B=8 | 5.83 | 373 | 112 ms | 112 ms | 20 ms | 1.4 s |
| 3 continuous, 8 slots | 6.52 | 417 | 169 ms | 304 ms | 17 ms | 1.2 s |
| 4 paged, 8192 tokens | 5.81 | 372 | 164 ms | 293 ms | 19 ms | 1.4 s |
| 5 + batched prefill | 6.72 | 430 | 105 ms | 106 ms | 17 ms | 1.2 s |

**burst**: Four bursts of 8 identical requests, 20 s apart.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 naive | 0.42 | 27 | 6768 ms | 13634 ms | 30 ms | 15.5 s |
| 1 async intake | 0.42 | 27 | 6762 ms | 13501 ms | 30 ms | 15.4 s |
| 2 static batching, B=8 | 0.52 | 33 | 125 ms | 131 ms | 18 ms | 1.2 s |
| 3 continuous, 8 slots | 0.52 | 33 | 177 ms | 315 ms | 17 ms | 1.3 s |
| 4 paged, 8192 tokens | 0.52 | 33 | 172 ms | 308 ms | 19 ms | 1.4 s |
| 5 + batched prefill | 0.52 | 33 | 124 ms | 125 ms | 17 ms | 1.2 s |

**uniform**: Homogeneous requests (128 in, 64 out), Poisson arrivals at 0.4 req/s.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 naive | 0.43 | 28 | 1663 ms | 7316 ms | 30 ms | 9.2 s |
| 1 async intake | 0.43 | 28 | 1665 ms | 7494 ms | 30 ms | 9.4 s |
| 2 static batching, B=8 | 0.49 | 31 | 858 ms | 1992 ms | 30 ms | 3.8 s |
| 3 continuous, 8 slots | 0.49 | 31 | 41 ms | 66 ms | 16 ms | 1.1 s |
| 4 paged, 8192 tokens | 0.49 | 31 | 43 ms | 70 ms | 21 ms | 1.8 s |
| 5 + batched prefill | 0.49 | 31 | 44 ms | 68 ms | 21 ms | 1.8 s |

**hetero**: Heterogeneous lengths (prompt 16-512, output 8-256), Poisson at 0.2 req/s.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 naive | 0.21 | 27 | 5329 ms | 22108 ms | 30 ms | 24.5 s |
| 1 async intake | 0.21 | 27 | 5248 ms | 21675 ms | 30 ms | 24.0 s |
| 2 static batching, B=8 | 0.24 | 31 | 1710 ms | 6841 ms | 25 ms | 12.1 s |
| 3 continuous, 8 slots | 0.25 | 32 | 52 ms | 90 ms | 17 ms | 4.3 s |
| 4 paged, 8192 tokens | 0.25 | 32 | 65 ms | 100 ms | 25 ms | 6.3 s |
| 5 + batched prefill | 0.25 | 32 | 59 ms | 95 ms | 23 ms | 5.9 s |

**uniform_hi**: Homogeneous requests (128 in, 64 out), Poisson arrivals at 3 req/s.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 naive | 0.52 | 33 | 25861 ms | 51100 ms | 30 ms | 53.0 s |
| 2 static batching, B=8 | 3.00 | 192 | 880 ms | 1900 ms | 20 ms | 3.2 s |
| 3 continuous, 8 slots | 3.29 | 211 | 46 ms | 192 ms | 16 ms | 1.2 s |
| 4 paged, 8192 tokens | 3.29 | 211 | 46 ms | 89 ms | 18 ms | 1.5 s |
| 5 + batched prefill | 3.29 | 211 | 44 ms | 67 ms | 17 ms | 1.4 s |

**hetero_hi**: Heterogeneous lengths (prompt 16-512, output 8-256), Poisson at 1 req/s.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 naive | 0.26 | 33 | 38961 ms | 97084 ms | 30 ms | 99.1 s |
| 2 static batching, B=8 | 0.89 | 114 | 3513 ms | 8173 ms | 26 ms | 13.4 s |
| 3 continuous, 8 slots | 1.13 | 145 | 57 ms | 100 ms | 17 ms | 4.3 s |
| 4 paged, 8192 tokens | 1.08 | 139 | 65 ms | 117 ms | 19 ms | 5.4 s |
| 5 + batched prefill | 1.08 | 138 | 64 ms | 143 ms | 18 ms | 5.4 s |

**uniform_8x**: Homogeneous requests (128 in, 64 out), Poisson arrivals at 8 req/s, 64 requests.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 3 continuous, 8 slots | 6.12 | 392 | 1444 ms | 2272 ms | 19 ms | 3.4 s |
| 4 paged, 8192 tokens | 7.32 | 468 | 51 ms | 114 ms | 31 ms | 2.6 s |
| 5 + batched prefill | 7.37 | 472 | 49 ms | 68 ms | 28 ms | 2.3 s |
| 7 + preemption | 7.26 | 465 | 50 ms | 72 ms | 31 ms | 2.4 s |

**hetero_4x**: Heterogeneous lengths (prompt 16-512, output 8-256), Poisson at 4 req/s, 64 requests.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 3 continuous, 8 slots | 3.35 | 401 | 2185 ms | 4196 ms | 18 ms | 7.9 s |
| 4 paged, 8192 tokens | 3.03 | 363 | 1046 ms | 4518 ms | 43 ms | 12.4 s |
| 5 + batched prefill | 3.11 | 373 | 733 ms | 4199 ms | 41 ms | 11.8 s |
| 7 + preemption | 3.25 | 390 | 105 ms | 2350 ms | 42 ms | 11.2 s |

**prefix_2x**: 256-token shared prefix + 32-256 own tokens, output 16-128, Poisson at 2 req/s.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 5 + batched prefill | 2.08 | 138 | 74 ms | 153 ms | 20 ms | 3.7 s |
| 6 + prefix cache | 2.05 | 136 | 60 ms | 121 ms | 22 ms | 3.5 s |

**prefix_4x**: 256-token shared prefix + 32-256 own tokens, output 16-128, Poisson at 4 req/s.

| stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | e2e p99 |
|---|---:|---:|---:|---:|---:|---:|
| 5 + batched prefill | 3.89 | 258 | 77 ms | 354 ms | 28 ms | 4.2 s |
| 6 + prefix cache | 3.82 | 254 | 64 ms | 138 ms | 30 ms | 4.1 s |

## Architecture

```
client coroutine ──generate(request)──▶ FIFO queue
                                           │
                 engine loop (asyncio), every step:
                   1. release finished / cancelled sequences      ─┐
                   2. [stage 7] grow block tables; preempt youngest │ core.admit / free
                   3. admit queued requests that fit (FIFO)        ─┘
                      prefill them (one padded forward if co-arriving)
                   4. one decode step over all active sequences
                                           │
                       model thread (ThreadPoolExecutor, 1 worker)
                   core.prefill / prefill_batch / decode  → next-token ids
                                           │
                 KV cache (owned by the core)
                   stage 3: [layer, slot, head, max_seq_len, head_dim]
                   stage 4+: [layer, block, token, head, head_dim] + block tables
                   stage 6+: content hash → block index, refcounts, LRU eviction
```

The asyncio thread never touches a tensor; it moves Python ints and
bookkeeping. The engine does not know how the cache is organized: the core
answers `fits / can_admit / admit / free`, and stages 4-7 swap cores behind
those calls.

## Layout

```
miniserve/
  request.py            GenerationRequest / GenerationResult / RequestRecord
  model.py              HFModelRunner (generate() + per-token timestamps), fake runner and cores for tests
  gpt2.py               our own GPT-2 forward over a slot-indexed KV cache (stage 3+)
  paged.py              block allocator, block tables, paged GPT-2 core (stage 4+)
  prefix_cache.py       content-addressed prompt blocks on the paged core (stage 6+)
  engines/
    base.py             the Engine interface every stage implements
    naive.py            stage 0
    async_serial.py     stage 1
    static_batch.py     stage 2
    continuous.py       stage 3 (the loop stages 4-7 reuse)
    paged.py            stage 4 (stage 5 is its prefill_batch_tokens flag)
    prefix_cached.py    stage 6
    preempting.py       stage 7
  bench/
    workload.py         WorkloadSpec, length distributions, arrival processes, presets
    harness.py          asyncio load generator; replays a workload against any engine (with warm-up)
    metrics.py          TTFT / TPOT / queue wait / throughput / occupancy definitions
    run.py              CLI: one engine x one workload -> JSON record
    compare.py          tables across records, head-of-line blocking summary
    resummarize.py      recompute summaries of old records after a metric change
    envinfo.py          hardware and version capture
scripts/                probes and calibration cited in the notes; summary_table.py
notes/                  one experiment record per stage
results/<stage>/        committed JSON records and probe outputs the notes cite
tests/                  95 tests: scheduler, cancellation, allocation and freeing, starvation,
                        malformed requests, admission control, numerical equivalence to HF
```

## Running

```
make venv-cpu                                  # python3 -m venv .venv; pinned CPU deps
make test
.venv/bin/python -m miniserve.bench.run --engine paged --workload hetero_4x --threads 32 \
    --engine-kw kv_budget_tokens=8192 --engine-kw block_size=16 --out results/raw
scripts/run_stage.sh preempting results/raw --threads 32 --engine-kw prefill_batch_tokens=4096
.venv/bin/python -m miniserve.bench.compare results/raw/*.json
.venv/bin/python scripts/summary_table.py
```

`--model fake` runs the harness with a deterministic fake model (no weights,
optional simulated per-token cost); most tests use it, and the tests that
need a real forward use a 2-layer random GPT-2 built in memory.

## Workloads

All presets use random token ids, forced output length (so length is a
controlled variable), seed 0.

| name | arrivals | prompt | output |
|---|---|---|---|
| offline | all at t=0 | 128 | 64 |
| closed8 | 8 closed-loop clients | 128 | 64 |
| uniform | Poisson 0.4 req/s | 128 | 64 |
| hetero | Poisson 0.2 req/s | U[16, 512] | U[8, 256] |
| burst | 4 bursts of 8, 20 s apart | 128 | 64 |
| uniform_hi | Poisson 3 req/s | 128 | 64 |
| hetero_hi | Poisson 1 req/s | U[16, 512] | U[8, 256] |
| uniform_8x | Poisson 8 req/s, 64 requests | 128 | 64 |
| hetero_4x | Poisson 4 req/s, 64 requests | U[16, 512] | U[8, 256] |
| prefix_2x | Poisson 2 req/s, 64 requests | 256 shared + U[32, 256] | U[16, 128] |
| prefix_4x | Poisson 4 req/s, 64 requests | 256 shared + U[32, 256] | U[16, 128] |

The first five rates were sized to the stage 0 service time (1.91 s for a
128/64 request on this host) so the baseline runs at about 0.8 utilization;
the later ones were added when a stage made the earlier ones arrival-bound.
Every run warms the engine with dummy requests at the workload's largest
shape before the clock starts (`--no-warmup` to measure cold).

## Metrics

Per request: queue wait (service start minus arrival), TTFT (first token
minus arrival, so it includes queueing), prefill latency, TPOT, end-to-end
latency. Per run: requests/s, output tokens/s, occupancy (fraction of the
run with at least one request in service), mean sequences in service, peak
resident tokens. Engine counters per stage: padding and lockstep waste,
slot utilization, prefill forwards and padded tokens, KV allocated versus
resident, cache hits and evictions, preemptions and recomputed tokens.
Definitions in `miniserve/bench/metrics.py`.
