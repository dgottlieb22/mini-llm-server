# Stage 5: batched prefill of co-arriving requests

Same engine as stage 4 (`paged`, 8192-token budget, 16-token blocks). The
change is one flag, `prefill_batch_tokens`: the requests admitted in one
scheduling pass are prefilled together in a single right-padded forward, as
long as rows x longest prompt stays within the cap (4096 here). With the
flag at 0 the engine prefills one request at a time, which is what stages 3
and 4 measured.

## Design decisions

Right padding, no mask. Rows are padded at the end to the longest prompt.
Under causal attention a pad token after a row's real tokens cannot
influence any real position, so the forward needs no attention mask; the
padded positions' K/V are computed and thrown away (not written to the
cache), and the logits are read at each row's own last real position.
Checked against Hugging Face logits for three different-length prompts on
both cores, including a decode step afterwards to confirm each row's cache
holds only its real tokens (`tests/test_batched_prefill.py`).

The cap is on padded tokens, not rows. A group of 8 prompts of 16 tokens
next to one of 512 would compute 8 x 512 positions for 8 x 16 + 512 of real
work (16% useful); the cap bounds that waste by bounding the whole padded
forward, and `prefill_padded_tokens / prefill_tokens` reports what it cost.
Length-aware grouping (sorting the queue by prompt length) would cut the
padding further but reorders admission; this stage keeps FIFO.

Groups form only from what is admitted in one pass. Nothing waits for a
batch to fill; a request that arrives alone is prefilled alone. The gain is
confined to co-arrivals, which is where the stage 3 and 4 notes measured
the cost.

## Prediction (from the stage 4 notes)

closed8 and burst TTFT p99 from about 300 ms to about 110 ms; offline
makespan down about 15%; nothing else moves.

## Results, paged engine, prefill one at a time (stage 4) vs batched (stage 5)

| workload | prefill | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | queue p50 | queue p99 | e2e p50 | e2e p99 | prefill share of makespan |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline | one at a time | 12.02 | 770 | 611 ms | 1173 ms | 33 ms | 573 ms | 1137 ms | 2.66 s | 2.66 s | 44% |
| offline | batched | 15.51 | 992 | 517 ms | 517 ms | 25 ms | 1 ms | 1 ms | 2.06 s | 2.06 s | 25% |
| closed8 | one at a time | 5.81 | 372 | 164 ms | 293 ms | 19 ms | 127 ms | 257 ms | 1.37 s | 1.39 s | 21% |
| closed8 | batched | 6.72 | 430 | 105 ms | 106 ms | 17 ms | 0 ms | 0 ms | 1.19 s | 1.19 s | 9% |
| burst | one at a time | 0.52 | 33 | 172 ms | 308 ms | 19 ms | 136 ms | 272 ms | 1.39 s | 1.40 s | 2% |
| burst | batched | 0.52 | 33 | 124 ms | 125 ms | 17 ms | 18 ms | 20 ms | 1.17 s | 1.17 s | 1% |
| uniform | one at a time | 0.49 | 31 | 43 ms | 70 ms | 21 ms | 6 ms | 33 ms | 1.35 s | 1.85 s | 2% |
| uniform | batched | 0.49 | 31 | 44 ms | 68 ms | 21 ms | 6 ms | 32 ms | 1.35 s | 1.82 s | 2% |
| hetero | one at a time | 0.25 | 32 | 65 ms | 100 ms | 25 ms | 7 ms | 28 ms | 3.50 s | 6.26 s | 1% |
| hetero | batched | 0.25 | 32 | 59 ms | 95 ms | 23 ms | 6 ms | 27 ms | 3.23 s | 5.86 s | 1% |
| uniform_hi | one at a time | 3.29 | 211 | 46 ms | 89 ms | 18 ms | 11 ms | 54 ms | 1.16 s | 1.46 s | 11% |
| uniform_hi | batched | 3.29 | 211 | 44 ms | 67 ms | 17 ms | 8 ms | 31 ms | 1.13 s | 1.44 s | 11% |
| hetero_hi | one at a time | 1.08 | 139 | 65 ms | 117 ms | 19 ms | 11 ms | 80 ms | 2.98 s | 5.40 s | 5% |
| hetero_hi | batched | 1.08 | 138 | 64 ms | 143 ms | 18 ms | 10 ms | 55 ms | 2.98 s | 5.41 s | 6% |
| uniform_8x | one at a time | 7.32 | 468 | 51 ms | 114 ms | 31 ms | 15 ms | 78 ms | 1.99 s | 2.55 s | 27% |
| uniform_8x | batched | 7.37 | 472 | 49 ms | 68 ms | 28 ms | 14 ms | 25 ms | 1.81 s | 2.25 s | 24% |
| hetero_4x | one at a time | 3.00 | 359 | 1153 ms | 4649 ms | 43 ms | 1111 ms | 4603 ms | 6.01 s | 12.64 s | 15% |
| hetero_4x | batched | 3.11 | 373 | 733 ms | 4199 ms | 41 ms | 677 ms | 4143 ms | 5.57 s | 11.81 s | 14% |

Prefill grouping that happened (stage 5 counters):

| workload | prefill forwards for N requests | padded / real tokens |
|---|---:|---:|
| offline | 1 for 32 | 1.00 |
| closed8, burst | 4 for 32 | 1.00 |
| uniform_hi | 29 for 32 | 1.00 |
| hetero_hi | 30 for 32 | 1.10 |
| uniform_8x | 57 for 64 | 1.00 |
| hetero_4x | 47 for 64 | 1.23 |
| uniform, hetero | 32 for 32 | 1.00 |

## What the numbers say

The prediction held on the co-arrival workloads. closed8 TTFT p99 went from
293 to 106 ms and burst from 308 to 125 ms; the 8 identical prompts are one
forward of 105 ms instead of eight of 37 ms (8 x 37 = 296 ms, which is the
stage 4 p99). The queue wait that remained at stage 4 (p99 257-272 ms) was
this serialization and is gone (p99 0-20 ms).

offline did better than the prediction: makespan down 22% (2.66 to 2.06 s)
and throughput 12.0 to 15.5 req/s. All 32 prompts prefill in one 517 ms
forward (4096 tokens at 7.9 tokens/ms) where stage 4 spent 1.17 s on 32
single prefills; and with every sequence entering the decode loop at the
same step, TPOT fell from 33 to 25 ms because no decode step was interrupted
by a late prefill. The slot sweep at stage 3 had 32 slots at 11.5 req/s;
the same batch of 32 now runs at 15.5.

On the Poisson workloads the groups are small (1-2 requests) and the medians
do not move, which is the "nothing else moves" half of the prediction. The
tails do move: uniform_8x TTFT p99 114 to 68 ms and queue p99 78 to 25 ms,
because the moments when two or three requests arrive within one step are
exactly the moments that set the tail. hetero_4x, which is saturated, gets
the same effect on its queue (p50 1.1 s to 0.68 s) with 47 forwards for 64
requests and 23% padding.

Padding stayed cheap. The worst case was hetero_4x at 1.23 padded tokens per
real token; the cap never had to split a group on the homogeneous workloads
(1.00). With a 1-second Poisson gap between arrivals the queue rarely holds
a 16-token and a 512-token prompt at the same moment.

Prefill is still 24-25% of the makespan on offline and uniform_8x. That is
now the prefill compute itself (4096 prompt tokens at about 8 tokens/ms),
not serialization overhead; the remaining lever there is chunked prefill
(interleaving prefill chunks with decode steps so TPOT tails do not pay for
it) rather than batching.

## The bottleneck

Unchanged from stage 4 for the saturated heterogeneous case: per-step KV
traffic with the gather, which needs a paged-attention kernel. For the
co-arrival case this stage closed the gap: closed8 and burst TTFT are back
to the static-batching figures (105 and 124 ms against 112 and 125 ms at
stage 2) with none of stage 2's lockstep waste.

Next in the plan is prefix caching (stage 4's option 1). Nothing in the
current workloads shares a prefix, so the stage 6 experiment needs a
workload with `shared_prefix_len` set, and the prediction from stage 4
stands: prefill per request drops by the shared fraction, TTFT with it, and
resident KV per sequence drops by the shared blocks.
