# Stage 4 probes (raw output)

Same host and model as stage 0. Run 2026-10-15.

## scripts/probe_paged_step.py, three versions of the gather

Decode step at context 160, all sequences active, slot core versus paged
core (block size 16), ms per step.

```
version                                                B=1    B=4    B=8    B=16   B=32
slot core (no gather)                                  25.7   13.3   14.7   18.7   22.7
paged v1: [block, head, token] layout, index + permute+reshape (2 copies)
                                                       27.7   15.4   17.5   29.3   52.7
paged v2: [block, token, head] layout, index_select (1 copy)
                                                       27.4   14.8   17.2   25.9   40.3
paged v3: [block, token, head] layout, advanced index (1 copy)
                                                       26.2   15.1   16.3   20.6   24.2
paged v4: v3 + index_select into a reused buffer (1 copy, no allocation)
                                                       27.2   14.0   17.1   22.5   23.6
```

v2 and v3 use different CPU kernels for the same gather; v2's index_select
parallelizes over rows and is slow when there are few large rows (block
size 64 and 256 were 2-4x worse than block size 16), v3's `index` kernel
parallelizes over elements and has a higher fixed cost per call (150 us at
B=2, see profile below). v4 is v2's kernel writing into a preallocated
buffer: see the allocator cliff below for why that matters more than the
kernel choice.

At context 700 (the long end of the heterogeneous workloads), v4:

```
  B    slots ms   paged bs=16 ms
  2     12.94        15.86
  8     16.17        22.58
 16     23.60        28.10
```

Prefill is unchanged by paging (the scatter into blocks is one extra copy
of the prompt's K/V): 23.6/36.4/72.3 ms for 16/128/512 tokens on both.

## scripts/profile_paged_step.py --batch 2 --context 160 (v3)

```
slots: 12.3 ms self time per step; aten::addmm 7.8 ms, aten::mm 2.0 ms, attention 0.4 ms
paged: 16.5 ms self time per step; aten::addmm 8.0 ms, aten::index 3.9 ms (26 calls, 150 us each),
       aten::mm 2.0 ms, attention 0.7 ms
```

## scripts/probe_gather_kernels.py: the allocator cliff

Gathering B*nb blocks of 48 KB (block size 16, GPT-2 small fp32) from a
513-block cache, median of 50 calls, microseconds. Default glibc malloc:

```
  B x nb  rows     MB  index us  index_select us  2D index_select us  embedding us
  1x11      11    0.5        16               13                  13            13
  2x11      22    1.1        18               14                  14            14
  8x11      88    4.3        32               16                  15            16
 16x11     176    8.7        51               18                  18            18
 32x11     352   17.3        90               28                  27            28
  8x48     384   18.9        97               29                  28            29
 16x48     768   37.7      5268             5242                5263          5263
```

Under `LD_PRELOAD=/usr/lib64/libjemalloc.so.2`:

```
 16x11     176    8.7      1244             1246                1234          1220
 32x11     352   17.3      2398             2382                2329          2317
  8x48     384   18.9      2586             2532                2544          2531
 16x48     768   37.7      5128             5069                5040          4942
```

A 38 MB result costs 5 ms whichever kernel produces it, 54x the cost of a
19 MB result: the output tensor is allocated fresh per call, and above
glibc's mmap threshold (32 MB maximum on 64-bit) each allocation is a new
mapping whose pages fault in on first touch. jemalloc returns large
allocations to the OS too, with a lower threshold here (the cliff moves to
9 MB). In the first stage 4 runs the per-layer gather on `hetero_4x`
(13 sequences x up to 48 blocks x 48 KB = 30 MB) sat on this cliff and the
decode step took 75 ms; with the gather writing into a buffer allocated
once (v4) the same step is 31 ms. Those first runs are kept in
`results/stage4/alloc-churn/`.

The same mechanism is the likeliest explanation for two earlier
observations that were left open: the stage 2 cold-first-call slowdown at
batch 32 (Hugging Face's `cat` allocates a fresh, growing KV tensor per
step, 19 MB per layer at batch 32) and the run-to-run variance of large
prefills (the [4096, 3072] fp32 MLP activations are 50 MB each).
