# Stage 3: continuous batching with an explicit KV cache

    request -> queue -> [every step: free finished/cancelled slots,
                         admit queued requests into free slots (prefill each),
                         one decode step over all active slots] -> tokens

The unit of scheduling is now one decode step. A sequence joins the running
batch the step after a slot frees up and leaves the step its output is done;
its result returns right then. Hugging Face's `generate()` is gone from this
path: `miniserve/gpt2.py` is our own GPT-2 forward (about 120 lines, weights
from the HF checkpoint, checked against HF logits to 1e-4 in
`tests/test_gpt2_core.py`) over an explicit KV cache.

## Design decisions

Why our own forward. HF's forward assumes a padded batch whose cache grows
by concatenation; it cannot drop one sequence from the middle of a running
batch or admit another without rebuilding the whole cache. Step-level
scheduling needs a model that can prefill one sequence into a cache slot
and run one decode step over slots of different lengths. Writing the forward
also makes the cache literally our code, which stages 4 and 5 need.

Cache layout: `k_cache[layer]` is `[max_slots, n_head, max_seq_len,
head_dim]`. Slot s owns a contiguous region of `max_seq_len` positions;
every sequence uses positions `0..L-1` of its own region, so there is no
padding anywhere: position embeddings and the attention mask are per slot.
This is the naive allocator (one maximum-length reservation per sequence),
kept deliberately: stage 4 replaces it and measures the difference.

Compute every slot, every step. Idle slots get a dummy token and a one-key
mask. The alternative, gathering only active slots into a contiguous batch,
copies the live KV every step (8 slots x 192 tokens x 72 KB = 113 MB for
GPT-2 small in fp32), which on a CPU costs about as much as the step and
would confound the measurement. The step cost is nearly flat in batch size
here, so computing idle slots is cheap; the engine records
`slot_utilization` so the waste is visible.

Admission is FIFO into free slots, one prefill per admitted request, all
admitted requests prefilled before the next decode step. Admission control
is the rule "prompt + output must fit a slot"; oversize requests are
rejected at admission with a `ValueError`. Cancellation (the awaiting task
is cancelled) frees the slot at the next step boundary, never mid-step.

The asyncio thread never touches a tensor. The first version picked the
argmax on the asyncio thread; `scripts/probe_engine_step.py` found that
cost 4.7 ms per 16 ms step (details in `results/stage3/probes.md`). Token
selection now runs on the model thread and returns Python ints; overhead
outside the model is 0.4 ms per step.

## Prediction (from the stage 2 notes)

On `hetero_hi`: decode-slot waste from 42% to near zero, the batch-boundary
wait gone, queue p50 from 3.0 s to tens of milliseconds, throughput up by
roughly the reclaimed slot fraction. On homogeneous workloads no change.
TPOT may worsen by the prefills that now run inside the loop.

## Results, 8 slots, against stage 2 at B=8

| workload | stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | queue p50 | queue p99 | e2e p50 | e2e p99 | avg in service |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline | 2 | 6.02 | 386 | 2.1 s | 4.1 s | 19 ms | 2.0 s | 4.0 s | 3.3 s | 5.3 s | 8.0 |
| offline | 3 | 6.07 | 388 | 2.2 s | 4.2 s | 18 ms | 2.1 s | 4.2 s | 3.3 s | 5.3 s | 7.2 |
| closed8 | 2 | 5.83 | 373 | 112 ms | 112 ms | 20 ms | 1 ms | 1 ms | 1.37 s | 1.38 s | 8.0 |
| closed8 | 3 | 6.52 | 417 | 169 ms | 304 ms | 17 ms | 132 ms | 266 ms | 1.23 s | 1.23 s | 7.1 |
| burst | 2 | 0.522 | 33.4 | 125 ms | 131 ms | 18 ms | 15 ms | 20 ms | 1.24 s | 1.25 s | 8.0 |
| burst | 3 | 0.523 | 33.4 | 177 ms | 315 ms | 17 ms | 140 ms | 279 ms | 1.28 s | 1.29 s | 7.2 |
| uniform | 2 | 0.488 | 31.2 | 858 ms | 2.0 s | 30 ms | 813 ms | 1.9 s | 2.2 s | 3.8 s | 1.1 |
| uniform | 3 | 0.491 | 31.4 | 41 ms | 66 ms | 16 ms | 4 ms | 29 ms | 1.0 s | 1.1 s | 1.3 |
| hetero | 2 | 0.244 | 31.2 | 1.7 s | 6.8 s | 25 ms | 1.5 s | 6.7 s | 6.0 s | 12.1 s | 1.6 |
| hetero | 3 | 0.247 | 31.7 | 52 ms | 90 ms | 17 ms | 3 ms | 22 ms | 2.4 s | 4.3 s | 1.3 |
| uniform_hi | 2 | 3.00 | 192 | 880 ms | 1.9 s | 20 ms | 797 ms | 1.8 s | 2.2 s | 3.3 s | 4.2 |
| uniform_hi | 3 | 3.29 | 211 | 46 ms | 192 ms | 16 ms | 10 ms | 155 ms | 1.1 s | 1.2 s | 3.5 |
| hetero_hi | 2 | 0.887 | 114 | 3.5 s | 8.2 s | 26 ms | 3.0 s | 8.0 s | 9.9 s | 13.4 s | 5.8 |
| hetero_hi | 3 | 1.133 | 145 | 57 ms | 100 ms | 17 ms | 8 ms | 64 ms | 2.6 s | 4.3 s | 2.6 |

Engine counters (stage 3):

| workload | decode steps | step mean | active per step | slot utilization | prefill total (share of makespan) | KV peak used / budget |
|---|---:|---:|---:|---:|---:|---:|
| offline | 252 | 16.1 ms | 8.0 | 1.00 | 1.21 s (23%) | 1528 / 8192 (19%) |
| closed8 | 252 | 14.7 ms | 8.0 | 1.00 | 1.21 s (25%) | 1528 / 8192 (19%) |
| burst | 252 | 15.2 ms | 8.0 | 1.00 | 1.18 s (2%) | 1528 / 8192 (19%) |
| uniform | 1510 | 15.3 ms | 1.34 | 0.17 | 1.18 s (2%) | 565 / 8192 (7%) |
| hetero | 3128 | 16.4 ms | 1.30 | 0.16 | 1.59 s (1%) | 1301 / 8192 (16%) |
| uniform_hi | 581 | 14.7 ms | 3.47 | 0.43 | 1.18 s (12%) | 1368 / 8192 (17%) |
| hetero_hi | 1606 | 16.4 ms | 2.53 | 0.32 | 1.60 s (6%) | 2462 / 8192 (30%) |

Slot sweep on `offline`, against static batching at the same B:

| slots | stage 2 req/s | stage 3 req/s | stage 2 TPOT | stage 3 TPOT | stage 3 TTFT p50 |
|---:|---:|---:|---:|---:|---:|
| 8 | 6.02 | 6.07 | 19.3 ms | 18.2 ms | 2.2 s |
| 16 | 8.91 | 8.98 | 24.9 ms | 23.4 ms | 1.2 s |
| 32 | 10.48 | 11.51 | 32.2 ms | 34.6 ms | 0.6 s |

## What the numbers say

The prediction held. On `hetero_hi` the median queue wait went from 3.0 s to
8 ms and TTFT p99 from 8.2 s to 100 ms. The HOL table in the compare output
has every stage 3 open-loop row at 0 requests waiting longer than their own
service time; at stage 2 the `hetero` worst case was 7.5x. Decode-slot
waste is zero by construction (a finished sequence leaves its slot the same
step), and nothing is held: `finish_time` equals the last token's time for
every request.

Throughput rose where the engine had been throughput-bound and is now
arrival-bound. `hetero_hi` went from 0.887 to 1.133 req/s, but mean
concurrency fell from 5.8 to 2.6: the engine keeps up with 1 req/s with
slots to spare, so the run ends when the arrivals do. The same on
`uniform_hi` (3.00 to 3.29 req/s, arrivals at 3.0). To see this engine's
capacity the load has to go up again, which the stage 4 experiment does.

TPOT fell from 19-30 ms to 15-17 ms on every workload: our decode step is
15-20% cheaper than HF's at the same batch size (preallocated cache written
in place instead of re-concatenated every step; `results/stage3/probes.md`),
and a sequence at low load no longer waits for a batch to fill.

The cost shows up in TTFT on the closed-loop and burst workloads. When 8
requests arrive together, stage 2 prefilled them as one padded batch
(111 ms); stage 3 prefills them one at a time (8 x 37 ms) before the next
decode step, so the 8th waits 266 ms (closed8 queue p99) and TTFT p99 went
from 112 to 304 ms. On `offline`, 23% of the makespan is these sequential
prefills; at 32 slots TPOT p50 is 34.6 ms against 23.6 ms for the bare step
because the first sequence's second token waits behind 31 prefills. This
is the prefill-inside-the-loop tradeoff the stage 2 notes predicted, made
worse by not batching co-arriving prefills.

The KV budget is barely used. With 8 slots of 1024 positions the engine
reserves 8192 tokens (576 MiB of fp32 KV for GPT-2 small) and never had
more than 30% of it occupied; on `uniform` the peak was 7%. The reservation
is what caps concurrency at 8, and 70-93% of it was empty at the peak. This
is the fragmentation argument for paging, measured rather than asserted.

## The bottleneck

Two, and they are different kinds.

1. Per byte of KV cache, concurrency. The naive allocator reserves
   `max_seq_len` positions for every sequence, so a budget that holds 8192
   tokens admits 8 sequences averaging 190-310 resident tokens. At the
   stage 2 sweep's rates, 16 or 32 concurrent sequences would give 1.5-1.9x
   the throughput, and the budget had room for them.

2. Prefill scheduling. Co-arriving requests are prefilled one by one with
   decode stalled, costing 23-25% of the makespan on the saturated
   workloads and tripling TTFT p99 on bursts.

## Three candidate next steps

1. Paged KV cache. Carve the budget into fixed-size blocks (say 16 tokens);
   a sequence holds a block table and takes one more block when its last
   one fills; a finished sequence returns its blocks. Admission needs
   enough free blocks for the prompt plus a margin, not a whole slot.
   - Prediction: at a fixed budget of 8192 tokens, the engine admits as
     many sequences as fit (20+ for `hetero`'s lengths), KV utilization at
     peak rises from about 30% to above 80%, and under a load the 8-slot
     engine cannot carry (`hetero` at 2 req/s) throughput rises toward the
     32-slot figure. New failure mode: a running sequence can run out of
     blocks mid-generation; the first version will admit conservatively
     (reserve the full output length) and measure how much that gives back.
   - Risk: on a CPU in framework ops, attention over non-contiguous blocks
     needs a gather of the live KV every step (the 113 MB copy the slot
     design avoided). That cost has to be measured and may eat part of the
     gain here; on a GPU it is why vLLM wrote a kernel.

2. Batched prefill of co-arriving requests. When several requests are
   admitted in one scheduling pass, prefill them as one padded forward.
   - Prediction: closed8/burst TTFT p99 back to about 110 ms; `offline`
     makespan down by about 15%; no change elsewhere.
   - Risk: reintroduces padding (the stage 2 waste) in the prefill only;
     a 16-token prompt padded to 512 wastes 97% of that forward.

3. Chunked prefill with a per-step token budget. Split long prompts into
   chunks and interleave them with decode steps, bounding how long running
   sequences stall.
   - Prediction: TPOT p99 on `hetero_hi` tightens (the 72 ms prefills of
     512-token prompts become several shorter stalls); TTFT for the long
     prompt gets worse by the interleaving.
   - Risk: more scheduler state; the gain is in tail TPOT, which at 18-19 ms
     p99 is not the problem these workloads show.

## Decision

Option 1. It addresses the concurrency cap, which is the bottleneck that
bounds throughput, and it is the project's central question about KV memory.
Option 2 is small and clearly worth doing; it goes in as the stage after
paging, measured on its own, so the two effects are not mixed. Option 3
waits until a workload shows tail TPOT as the problem. The stage 4
experiment fixes the KV budget at 8192 tokens for both allocators and
raises the heterogeneous load until the 8-slot engine saturates, so the
comparison is per byte of cache, which is the comparison that matters.
