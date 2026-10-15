# Stage 4: paged KV cache

Same loop as stage 3; the storage behind `can_admit()/admit()/free()` is now
a pool of fixed-size blocks (16 tokens) instead of one 1024-position slot
per sequence. A sequence holds a block table; its logical block i lives in
physical block `table[i]`. Concurrency is no longer a configured slot count
but however many sequences fit the pool at their actual lengths.
`miniserve/paged.py` has the allocator, the block table and the paged
forward; `tests/test_paged.py` checks it against Hugging Face logits across
block boundaries and checks that blocks come back on finish and on cancel.

## Design decisions

Conservative admission. A request is admitted only when the pool has blocks
for its prompt plus its whole requested output, so a running sequence can
never run out of blocks. This relies on knowing the output length, which
the benchmark fixes; a real server has only `max_tokens` as an upper bound,
which is why production engines allocate on demand and preempt instead.
That optimistic policy is a later stage; this one isolates the allocator.

Block size 16, as in vLLM. Internal fragmentation is at most 15 tokens per
sequence (one partial last block); smaller blocks mean more gather rows,
larger ones more waste. The probe at 64 and 256 showed no gain here.

The gather. Torch's attention wants `[B, heads, T, head_dim]` per batch and
a sequence's blocks are not contiguous, so every decode step copies each
active sequence's blocks into a contiguous padded batch (`index_select` into
a buffer allocated once), then runs the same masked attention as the slot
core. That copy is the live KV of the batch, every step. The cache layout
is token-major inside a block (`[block, token, head, head_dim]`) so the
gathered blocks view as attention's layout without a second copy. vLLM
avoids the copy with a kernel that reads the block table directly; here it
is a measured cost.

Compute only active sequences. The slot core computes every slot because
there is no gather to pay for; the paged core gathers anyway, so it batches
only the active sequences. On this host that is a disadvantage at low load:
one active sequence means an M=1 matmul, the single-threaded path from the
stage 0 probes (27 ms) where the slot core's always-8 batch costs 15 ms.
On a GPU it would be the other way round.

Reused gather buffer. The first measurement had the gather allocating its
result every call; above about 32 MB each call page-faults the whole buffer
(5 ms for 38 MB, `results/stage4/probes.md`). Writing into a preallocated
buffer took the `hetero_4x` step from 75 ms to 31 ms. The pre-fix records are
kept in `results/stage4/alloc-churn/`.

`max_live_tokens`. A new, optional admission bound on tokens resident across
running sequences, independent of the memory check. Memory says how many
sequences fit; this bounds the work of a decode step, which reads every
resident token's K/V. It is a soft bound (sequences grow after admission).

## Prediction (from the stage 3 notes)

At a fixed 8192-token budget, the paged engine admits as many sequences as
fit (20 or more at `hetero`'s lengths), peak KV utilization rises from about
30% to above 80%, and under a load the 8-slot engine cannot carry,
throughput rises toward the 32-slot figure. Risk: the per-step gather may
eat part of the gain on this CPU.

## Results: slots (8 x 1024) vs paged (512 x 16), same 8192-token budget

| workload | engine | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | queue p50 | queue p99 | e2e p50 | e2e p99 | avg in service |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline | slots | 6.07 | 388 | 2.2 s | 4.2 s | 18 ms | 2.1 s | 4.2 s | 3.3 s | 5.3 s | 7.2 |
| offline | paged | 12.02 | 770 | 0.6 s | 1.2 s | 33 ms | 0.6 s | 1.1 s | 2.7 s | 2.7 s | 25.1 |
| closed8 | slots | 6.52 | 417 | 169 ms | 304 ms | 17 ms | 132 ms | 266 ms | 1.23 s | 1.23 s | 7.1 |
| closed8 | paged | 5.81 | 372 | 164 ms | 293 ms | 19 ms | 127 ms | 257 ms | 1.37 s | 1.39 s | 7.3 |
| burst | slots | 0.523 | 33.4 | 177 ms | 315 ms | 17 ms | 140 ms | 279 ms | 1.28 s | 1.29 s | 7.2 |
| burst | paged | 0.521 | 33.4 | 172 ms | 308 ms | 19 ms | 136 ms | 272 ms | 1.39 s | 1.40 s | 7.3 |
| uniform | slots | 0.491 | 31.4 | 41 ms | 66 ms | 16 ms | 4 ms | 29 ms | 1.02 s | 1.10 s | 1.3 |
| uniform | paged | 0.489 | 31.3 | 43 ms | 70 ms | 21 ms | 6 ms | 33 ms | 1.35 s | 1.85 s | 1.3 |
| hetero | slots | 0.247 | 31.7 | 52 ms | 90 ms | 17 ms | 3 ms | 22 ms | 2.4 s | 4.3 s | 1.3 |
| hetero | paged | 0.247 | 31.6 | 65 ms | 100 ms | 25 ms | 7 ms | 28 ms | 3.5 s | 6.3 s | 1.4 |
| uniform_hi | slots | 3.29 | 211 | 46 ms | 192 ms | 16 ms | 10 ms | 155 ms | 1.08 s | 1.23 s | 3.5 |
| uniform_hi | paged | 3.29 | 211 | 46 ms | 89 ms | 18 ms | 11 ms | 54 ms | 1.16 s | 1.46 s | 3.9 |
| hetero_hi | slots | 1.133 | 145 | 57 ms | 100 ms | 17 ms | 8 ms | 64 ms | 2.6 s | 4.3 s | 2.6 |
| hetero_hi | paged | 1.081 | 139 | 65 ms | 117 ms | 19 ms | 11 ms | 80 ms | 3.0 s | 5.4 s | 2.8 |
| uniform_8x | slots | 6.12 | 392 | 1.44 s | 2.27 s | 19 ms | 1.41 s | 2.24 s | 2.65 s | 3.42 s | 7.5 |
| uniform_8x | paged | 7.32 | 468 | 51 ms | 114 ms | 31 ms | 15 ms | 78 ms | 1.99 s | 2.55 s | 15.0 |
| hetero_4x | slots | 3.35 | 401 | 2.19 s | 4.20 s | 18 ms | 2.14 s | 4.14 s | 4.1 s | 7.9 s | 7.1 |
| hetero_4x | paged | 3.00 | 359 | 1.15 s | 4.65 s | 43 ms | 1.11 s | 4.60 s | 6.0 s | 12.6 s | 14.8 |
| hetero_4x | paged, max_live_tokens=4096 | 2.80 | 336 | 3.36 s | 5.86 s | 33 ms | 3.32 s | 5.79 s | 7.2 s | 13.0 s | 10.9 |

Engine counters, paged:

| workload | step mean | active per step | attention positions computed / live | KV peak allocated of 8192 | reserved but empty at peak | admission waits |
|---|---:|---:|---:|---:|---:|---:|
| offline | 23.4 ms | 32.0 | 1.05 | 6144 (75%) | 2016 | 0 |
| uniform_8x | 21.9 ms | 13.8 | 1.13 | 5376 (66%) | 1156 | 0 |
| hetero_4x | 30.6 ms | 12.8 | 1.56 | 8192 (100%) | 2387 | 240 |
| hetero_4x, live 4096 | 26.3 ms | 10.2 | 1.54 | 5776 (71%) | 1685 | 453 |
| hetero_hi | 19.3 ms | 2.8 | 1.23 | 2704 (33%) | 768 | 0 |
| uniform | 23.7 ms | 1.5 | 1.06 | 576 (7%) | 185 | 0 |

For comparison the slot engine's peak allocation is 8192 on every saturated
workload with 1382-3568 resident (17-44%).

## What the numbers say

The allocator does what it is for. On `offline`, all 32 requests fit the
pool at once (6144 of 8192 tokens) where the slot engine ran 8 at a time:
12.0 req/s against 6.1, matching the 32-slot figure from the stage 3 sweep.
On `uniform_8x`, 15 sequences run where 8 did; throughput went from 6.1 to
7.3 req/s and the engine stopped falling behind the 8 req/s arrivals:
median queue wait 1.4 s to 15 ms, TTFT p99 2.3 s to 114 ms, e2e p99 3.4 s
to 2.6 s. Peak utilization of the budget went from 17% resident to 66%
allocated. Where the slot engine had slots free (`uniform`, `hetero`,
`uniform_hi`, `hetero_hi`, `burst`, `closed8`) paging changes nothing about
admission, as expected.

The gather does what the risk said. On `hetero_4x` the paged engine admits
13-15 sequences against 7 but the step costs 31-43 ms against 15-18, and
throughput is lower (3.0 vs 3.35 req/s) with worse TPOT and e2e tails,
though the median queue wait halved. With long contexts (up to 768) the
per-step copy of the live KV is 13 sequences x up to 48 blocks x 48 KB per
layer, about 30 MB per layer, 720 MB per step, and the padded batch
computes 1.56 positions per live one. The step is bound by KV traffic, and
the gather adds a full read and write of that KV on top of the read the
attention does anyway. Bounding live tokens to 4096 brought the step to
26 ms but admitted fewer sequences and lost throughput; the bound trades
TPOT for queue wait, it does not recover the gather.

Where the step is weight-bound (short contexts), more sequences per step
are nearly free and paging wins; where it is KV-bound (long contexts, many
sequences), the gather costs about as much as the attention itself and
paging loses on this host. vLLM's PagedAttention is a kernel for exactly
this reason: the allocator is the easy half.

Internal fragmentation is small; the output reservation is not. On
`hetero_4x`, 2387 of the 8192 reserved tokens held nothing at the busiest
moment: a few hundred are partial last blocks (at most 15 per sequence),
the rest is the conservative reservation of each sequence's full requested
output. That is the cost of admitting on `prompt + max_new_tokens`, and it
is what optimistic allocation with preemption would reclaim.

Two findings about the host rather than the design. The gather result must
not be allocated per call: above the allocator's mmap threshold every call
page-faults the buffer, 5 ms for 38 MB, and the first `hetero_4x` run spent
most of its 75 ms step there (details in `results/stage4/probes.md`; the
same mechanism explains the stage 2 cold-start and the large-prefill
variance). And computing only active sequences costs 8 ms per step at low
load here because a single sequence hits the single-threaded M=1 matmul
path; the slot core's always-8 batch avoided it by accident.

## The bottleneck

Per-step KV traffic at high concurrency with long contexts, doubled by the
gather. Memory capacity is no longer what caps concurrency (the paged pool
holds 2-4x more sequences than the slots); what caps useful concurrency is
how many resident tokens a step can read in the time budget, and on this
host without a paged-attention kernel every resident token is read twice.
Behind it, unchanged from stage 3: sequential prefill of co-arriving
requests (15-44% of the makespan on saturated workloads) and the TTFT it
costs on bursts.

## Three candidate next steps

1. Prefix caching on the paged pool. Hash each full block of prompt tokens
   by its content and the prefix before it; a new request whose prompt
   starts with blocks already resident reuses them (refcount, no copy) and
   prefills only the rest. Needs a workload with shared prefixes (the
   generator has `shared_prefix_len`).
   - Prediction: with a 256-token shared prefix on 384-token prompts,
     prefill cost per request drops about 3x, TTFT with it, and resident KV
     per sequence drops by the shared part; no effect on decode steps.
   - Risk: lookup and refcount bookkeeping per block; eviction policy when
     the pool is full of cached-but-unused blocks; only helps workloads
     that share prefixes, which the current ones do not.

2. Batched prefill of co-arriving requests (stage 3's option 2). Prefill
   the requests admitted in one pass as one padded forward.
   - Prediction: closed8/burst TTFT p99 from about 300 to about 110 ms;
     offline makespan down about 15%; nothing else moves.
   - Risk: padding in the prefill forward (a 16-token prompt next to a
     512-token one wastes 97% of its row); bounded by capping the batch's
     padded tokens.

3. A paged-attention kernel. Attention that reads the block table directly
   and never materializes the gathered batch.
   - Prediction: paged step cost falls to the slot core's at equal B and T
     (the 22.6 to 16.2 ms gap at 8 x 700), making paging a strict win.
   - Risk: this host has no CUDA device and no Python development headers
     for a C++ extension, so the kernel waits for the GPU stage (Triton).

## Decision

Option 2 first, then 1. Option 2 is small, its prediction is specific, and
the sequential-prefill cost has shown up in every stage since 3; doing it
before prefix caching keeps the prefill path simple when caching changes
what a prefill is. Option 1 follows as stage 6 with a shared-prefix
workload added for it; it is the capability the block design exists for
beyond fragmentation, and the one the slot design could not offer. Option 3
is recorded as the first GPU-stage item: the measurement above is the
case for it.
