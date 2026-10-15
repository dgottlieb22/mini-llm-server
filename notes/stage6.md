# Stage 6: prefix caching

Same engine and allocator as stage 5. The core (`miniserve/prefix_cache.py`)
names each full block of prompt tokens by a hash of (the previous block's
hash, the block's tokens); a new request whose prompt starts with blocks
already resident points its block table at them and prefills only the rest.
Shared blocks carry a reference count; a block nobody uses stays indexed as
a future hit until the pool runs low, then the least recently used cached
blocks are evicted. Nothing in the request marks the shared part: the match
is on content, as in vLLM's automatic prefix caching.

## Design decisions

Only full prompt blocks are cached. A partial block's K/V depend on tokens
the next request may not share, and blocks filled while decoding are the
sequence's own output, which no workload here would hit.

A prompt is never a complete hit. `prefill()` returns the logits at the last
position, so at least the last block is always computed; an identical
prompt reuses all but one block.

Admission is token-aware. `can_admit()` had taken lengths; with a cache the
blocks a request needs depend on which of its blocks are resident, so the
core offers `can_admit_prompt(tokens, ...)` and the engine uses it when
present. The hit is counted at admission and the blocks are refcounted
then, so an eviction between admission and prefill cannot take them.

Hits prefill alone. A batched prefill (stage 5) assumes every row starts at
position 0; a row with a cached prefix starts at position `cached`, with
attention over the shared blocks. Mixing the two in one padded forward is
possible with per-row masks but adds complexity for a case the stage 5
groups rarely contain; a request with a hit is prefilled on its own.

`allocated_tokens()` counts a shared block once, so the engine's "allocated"
stat is physical blocks held and "resident" is the logical sum over
sequences; their difference is what sharing saved.

Warm-up now also clears the cache and resets the core's counters. The
first stage 6 runs reported 1132 hit blocks for a workload that can produce
at most 1008 (63 requests x 16 prefix blocks); the extra were the 12
identical warm-up prompts hitting each other. The same leak had inflated
stage 4's attention-padding counter; its notes carry the correction.

## Prediction (from the stage 4 notes)

With a 256-token shared prefix on prompts of 288-512 tokens: prefill cost
per request drops about 3x, TTFT with it, and resident KV per sequence
drops by the shared part; no effect on decode steps.

## Results: paged + batched prefill (stage 5) vs prefix cached (stage 6)

Workloads: 64 requests, every prompt = the same 256 tokens + 32-256 own
tokens, output 16-128, Poisson at 2 req/s (`prefix_2x`) and 4 req/s
(`prefix_4x`). Records in `results/stage6/`.

| workload | engine | req/s | TTFT p50 | TTFT p99 | prefill p50 | queue p99 | TPOT p50 | TPOT p99 | e2e p99 | prefill forwards, total, share of makespan | KV peak: physical allocated / logical resident | admission waits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| prefix_2x | paged | 2.08 | 74 ms | 153 ms | 67 ms | 57 ms | 20.4 ms | 37.8 ms | 3.69 s | 60, 4.03 s, 13% | 5520 / 4930 | 0 |
| prefix_2x | prefix_cached | 2.05 | 60 ms | 121 ms | 48 ms | 47 ms | 21.7 ms | 34.3 ms | 3.55 s | 60, 2.83 s, 9% | 2960 / 4935 | 0 |
| prefix_4x | paged | 3.89 | 77 ms | 354 ms | 66 ms | 293 ms | 28.4 ms | 50.4 ms | 4.21 s | 58, 3.88 s, 24% | 8144 / 7389 | 31 |
| prefix_4x | prefix_cached | 3.82 | 64 ms | 138 ms | 49 ms | 50 ms | 29.6 ms | 47.9 ms | 4.10 s | 59, 2.87 s, 17% | 4672 / 8136 | 0 |

Cache counters (both workloads): 1008 block hits, 563 misses; 16128 of
25634 prompt tokens (63%) served from cache; 79 and 108 evictions.

Prefill cost in isolation (`scripts/probe_prefix_prefill.py`, one request
at a time, median of 7):

| prompt | own tail | paged, full prefill | cached, cold (miss) | cached, prefix resident (hit) |
|---:|---:|---:|---:|---:|
| 288 | 32 | 51.1 ms | 51.2 ms | 26.1 ms |
| 320 | 64 | 54.8 ms | 54.4 ms | 26.3 ms |
| 384 | 128 | 62.4 ms | 61.8 ms | 26.4 ms |
| 448 | 192 | 71.7 ms | 68.4 ms | 26.7 ms |
| 512 | 256 | 79.1 ms | 72.7 ms | 27.0 ms |

## What the numbers say

The cache hit what it could: 63 of 64 requests found all 16 prefix blocks
(1008 hits), and 63% of all prompt tokens were never recomputed. Physical
KV at the peak halved: 2960 against 5520 blocks' worth of tokens on
`prefix_2x` and 4672 against 8144 on `prefix_4x`, with the same logical
footprint, because 16 blocks of the prefix exist once instead of once per
running sequence. On `prefix_4x` the paged engine filled its pool (31
admission waits, queue p99 293 ms); the cached engine never did (0 waits,
queue p99 50 ms). That is the memory half of the prediction, met.

The compute half fell short of 3x. Prefill time per request went from 63 to
44 ms (1.4x), and the TTFT median from 74-77 to 60-64 ms. The probe says
why: a tail-only prefill costs 26 ms whether the tail is 32 tokens or 256,
against 51-79 ms for the full prompt. 26 ms is the fixed cost of one forward
pass through GPT-2 on this host (a 16-token prefill took 24 ms at stage 0:
weight streaming plus per-layer overhead), and prefix caching removes
tokens, not forwards. The 3x prediction assumed prefill time was
proportional to tokens; it is proportional above a 25 ms floor.

Decode did not change (TPOT p50 20-29 ms both ways; the 1.2-1.3 ms
difference is within what the stage 4 runs showed between repeats). The
gather cost per step is the same: a shared block is read once per sequence
that holds it, the same as an unshared one. A paged-attention kernel that
reads the block table directly would share that read too; the gather
cannot.

Throughput is unchanged because both workloads are arrival-bound at these
rates once the engine stops queueing; the gain shows up as latency and as
the pool having room, which at a higher arrival rate would become
throughput. The experiment was sized to show the mechanism, not the
capacity; `prefix_4x` at 8 req/s would show the capacity.

Evictions happened (79 and 108) with the index holding 455-484 blocks at
the end: the workload's own tails get indexed too (they are full prompt
blocks) and are never hit, so they fill the index until LRU pushes them
out. Indexing only blocks that are likely to be shared would need a hint
the request does not carry; vLLM indexes everything too and relies on the
same eviction.

## The bottleneck

Prefill now has a floor of one forward per request (26 ms) that no caching
removes, and above the floor it is sequential per hit. The decode step is
still the gather-bound step from stage 4 on long contexts. For the
co-arrival case the batched prefill of stage 5 does not apply to hits.

## Three candidate next steps

1. Batched tail prefill: let a padded forward carry rows that start at
   different positions (per-row position offsets and attention over each
   row's own prefix blocks), so hits can share one forward the way misses
   do.
   - Prediction: on `prefix_4x`, co-arriving hits (the queue p99 moments)
     prefill in one 26-30 ms forward instead of 26 ms each; TTFT p99 from
     138 toward about 100 ms. Nothing else moves.
   - Risk: per-row masks over block-gathered K/V in the prefill path;
     modest complexity for a tail-only gain.

2. Optimistic allocation with preemption. Admit on the prompt alone, grow a
   sequence block by block as it decodes, and when the pool is empty
   preempt the youngest sequence (free its blocks, requeue it, recompute
   its prefix on readmission, which the cache then serves). Removes the
   output-length reservation, which held 2387 empty tokens at the peak on
   `hetero_4x` and which a real server cannot make because it does not know
   the output length.
   - Prediction: on `hetero_4x` 20-30% more sequences admitted at the same
     budget; some preemptions under the heaviest load, each costing one
     recomputed prompt.
   - Risk: a thrashing regime where preemption and readmission chase each
     other; the scheduler needs a rule for it.

3. Chunked prefill. Split long prompts into fixed-size chunks and interleave
   them with decode steps, so a 512-token prompt does not stall every
   running sequence for 70 ms.
   - Prediction: TPOT p99 on the heterogeneous workloads tightens from
     about 50 toward about 35 ms; TTFT of the long prompt gets worse by
     the interleaving.
   - Risk: the stall it removes is one or two decode steps' worth here; on
     a GPU with a 100x faster decode step the same prefill is tens of
     steps, which is where chunked prefill matters.

## Decision

Option 2. It is the piece of the serving story the project has not touched
(what happens when memory runs out while a sequence is still generating),
it removes the one assumption in the allocator that a real server cannot
make (knowing the output length), and prefix caching makes its cost
measurable: a preempted sequence's prompt comes back from the cache.
Options 1 and 3 are recorded as GPU-stage items; both address a prefill
cost that on this host is dominated by a fixed floor the GPU does not
have.
