# Stage 7: optimistic allocation with preemption

Stages 4-6 admitted a request only when the pool had blocks for its prompt
plus its whole requested output. That can never run out of memory
mid-generation, but it holds memory the sequence has not used yet, and it
needs the output length, which a real server does not have (clients send
`max_tokens` as a ceiling, and most outputs stop well short of it).

Stage 7 admits on the prompt plus one position and grows each sequence by a
block as it decodes. When a step needs more blocks than the pool has, the
youngest running sequences are preempted: their blocks are freed and they
go back to the head of the queue, to be readmitted later with their prompt
plus everything generated so far re-prefilled, after which generation
resumes. The client sees one result with a gap in its token timestamps.

## Design decisions

Recompute, not swap. vLLM offers both; swapping moves a preempted
sequence's blocks to host memory and back. This host has no slower tier
below its one memory, so recompute is the only form. It costs one prefill
of prompt-so-far per preemption, which the engine counts
(`recomputed_tokens`).

Youngest first. The sequence admitted last is preempted first. It has the
least work to redo, and the sequences that have waited longest keep
running, so FIFO fairness is preserved; a preempted request re-enters at
the queue head, ahead of anything that arrived after it.

A growth reserve at admission. With the pool allowed to fill completely,
every block boundary a running sequence crosses afterwards is a preemption,
and a preempted sequence readmitted into a still-full pool is preempted
again after one token: thrash. Admission therefore leaves `growth_reserve`
free blocks per running sequence (default 1). vLLM's watermark plays the
same role. Both settings are measured below.

The prefix cache was expected to make recompute cheap, since a preempted
sequence's prompt blocks stay indexed after they are freed. Under memory
pressure it does not: those blocks are the only unreferenced ones in a pool
full of running sequences, so they are the first evicted when the others
grow, and the resumed prompt misses (`cached_prompt_tokens` is 0 in the
reserve-1 runs). Preemption happens exactly when there is no room to keep
the preempted state. The recompute is real work.

Two bugs found by the tests while building this. A request preempted twice
lost its first segment (the history record was overwritten); history is
now folded at preemption time. And the stage 6 cache could evict a hit
block during that request's own admission (the hit blocks were unreferenced
and oldest, and eviction ran before the refcount); hits are now pinned
first. The second bug existed in stage 6's measured runs and did not fire
there because the pool was never full enough while prefix blocks were
unreferenced.

## Prediction (from the stage 6 notes)

On `hetero_4x` 20-30% more sequences admitted at the same budget; some
preemptions under the heaviest load, each costing one recomputed prompt.
Risk: a thrash regime.

## Results, hetero_4x (64 heterogeneous requests, Poisson 4 req/s)

| budget | admission | req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | e2e p50 | e2e p99 | active per step | preemptions | recomputed tokens (share of prefill) | KV reserved but empty at peak |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | conservative (stage 6 engine) | 2.45 | 5.74 s | 8.41 s | 28.6 ms | 34.4 ms | 8.6 s | 15.5 s | 8.3 | n/a | 0 | 1348 (33%) |
| 4096 | optimistic, reserve 1 | 2.68 | 4.08 s | 6.81 s | 31.3 ms | 40.8 ms | 7.7 s | 13.8 s | 9.7 | 3 | 670 (4%) | 133 (3%) |
| 4096 | optimistic, reserve 0 | 2.60 | 4.20 s | 7.61 s | 34.1 ms | 53.2 ms | 8.3 s | 14.7 s | 9.8 | 26 | 6882 (30%) | 152 (4%) |
| 8192 | conservative (stage 5) | 3.11 | 0.73 s | 4.20 s | 40.5 ms | 45.4 ms | 5.6 s | 11.8 s | 12.8 | n/a | 0 | 2365 (29%) |
| 8192 | optimistic, reserve 1 | 3.25 | 0.11 s | 2.35 s | 42.0 ms | 65.4 ms | 5.2 s | 11.2 s | 13.6 | 3 | 958 (6%) | 198 (2%) |

`uniform_8x` at 8192, optimistic: 7.26 req/s, TTFT p99 72 ms, 0
preemptions, peak allocation 4272 against 5376 conservative. Same
throughput (arrival-bound), 20% less memory held.

## What the numbers say

The reservation was the waste. At the 4096 budget, conservative admission
held 1348 tokens (33% of the pool) for outputs that had not been generated
yet; optimistic admission holds 133 (3%, the partial last blocks). That
room became 17% more concurrent sequences (8.3 to 9.7 active per step),
9% more throughput (2.45 to 2.68 req/s), and 1.7 s less median TTFT, at
the price of 3 preemptions whose recompute added 4% to the prefill work.
At 8192 the same policy admitted earlier (median TTFT 0.73 to 0.11 s) and
raised throughput 5%, with 3 preemptions.

The prediction's 20-30% more sequences came out as 17%; the rest of the
reserved space is partial blocks and the growth reserve itself.

The thrash regime is real and the reserve prevents it. With `growth_reserve
= 0` the engine admitted until the pool was full, then preempted 26 times
and recomputed 6882 tokens, 30% on top of the real prefill work; TPOT p99
went from 41 to 53 ms (every recompute is a prefill the running sequences
wait through), and throughput fell below the reserve-1 run. One spare block
per running sequence cut preemptions to 3. This is why every production
scheduler has a watermark.

Preemption trades TPOT tail for admission. Every optimistic run has a worse
TPOT p99 than its conservative counterpart (40.8 vs 34.4 ms at 4096; 65 vs
45 ms at 8192): more sequences per step make the step longer, and each
recompute prefill stalls the batch. The preempted request itself pays most:
its e2e includes the recompute and its place back in the queue. For a
latency-sensitive deployment the conservative policy with a known
`max_tokens` is the safer one; for throughput under a memory cap the
optimistic one is better, and it is the only one available when output
lengths are unknown.

## Where the project stands

Seven stages, each measured against the one before on the same workloads.
The serving-architecture questions the project set out to answer have
measurements behind them:

- naive batching and head-of-line blocking: stage 0 (a 9-token request
  waited 27x its own service time) and stage 2 (42% of decode slots run for
  finished sequences);
- variable sequence length: stage 2's padding and lockstep waste, stage 3's
  removal of it;
- why KV-cache memory is the critical resource: stage 3's reservation (up
  to 93% empty), stage 4's 2x concurrency from the same bytes, stage 7's
  reservation removal;
- continuous batching's latency/throughput effect: stage 3's tables;
- prefill versus decode scheduling: stages 3 and 5 (sequential prefill cost
  23-44% of saturated makespans; batching co-arrivals recovered it);
- memory pressure and admission: stages 4 and 7;
- cancellation: stage 3 (freed at the next step boundary; tested for
  running, queued and preempted requests).

What remains for a GPU host, in the order the measurements suggest: a
paged-attention kernel (stage 4's gather costs as much as the attention it
feeds), chunked prefill (a GPU decode step is tens of times shorter, so a
prefill stall is tens of steps), and the stages that need hardware
features: quantized weights and speculative decoding.
