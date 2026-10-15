# Stage 2 probes (raw output)

Same host and model as stage 0 (`results/stage0/probes.md`). Run 2026-10-15.

## scripts/check_batched_equivalence.py --threads 32

Batched (left-padded, per-sequence stopping) versus one-at-a-time greedy
decoding, 8 heterogeneous requests:

```
r0: prompt 114 gen 33  same  (batched 33 tokens, single 33)
r1: prompt 196 gen 39  same  (batched 39 tokens, single 39)
r2: prompt  99 gen 31  same  (batched 31 tokens, single 31)
r3: prompt 117 gen 22  same  (batched 22 tokens, single 22)
r4: prompt  18 gen 13  same  (batched 13 tokens, single 13)
r5: prompt 148 gen 40  same  (batched 40 tokens, single 40)
r6: prompt 194 gen 25  same  (batched 25 tokens, single 25)
r7: prompt 177 gen 26  same  (batched 26 tokens, single 26)
8/8 identical. batch 1.09 s vs singles 6.84 s (6.3x), 40 lockstep steps
```

## scripts/probe_cold_warm.py --batch 32 --repeats 4

The exact benchmark call path (`HFModelRunner.generate_batch`), 32 x (128 in,
64 out), four times in one process:

```
call 1: prefill   1071 ms   step median  80.1 ms  (first 5 steps: 39 44 38 35 32 ms)   total 5.39 s
call 2: prefill   1036 ms   step median  34.0 ms  (first 5 steps: 32 35 32 32 37 ms)   total 3.23 s
call 3: prefill   1033 ms   step median  33.9 ms  (first 5 steps: 32 31 36 40 32 ms)   total 3.21 s
call 4: prefill   1008 ms   step median  33.7 ms  (first 5 steps: 31 31 44 32 32 ms)   total 3.19 s
```

The first call in a process is 1.7x slower end to end, and the slowdown
sits in the later decode steps (the first five are only slightly slower,
so the 63-step median of 80 ms means the late steps ran well over 100 ms).
A plausible mechanism, not verified here: each step's `torch.cat` on the
growing KV cache allocates a larger block than the step before, and on the
first pass those pages have never been touched, while later calls get
memory the allocator already owns. The warm-up removes the effect, which
is what the benchmark needs; the mechanism is left open. At batch 8 the
effect is below the noise (TPOT p99 was 19.3 ms against a 19.2 ms median
in the cold B=8 run).

## scripts/probe_thread_context.py --batch 32

Ruling out the engine's worker thread as the cause:

```
main thread                              prefill     771 ms   decode step median   32.4 ms   torch threads 32
main thread (again)                      prefill    1006 ms   decode step median   32.5 ms   torch threads 32
worker thread                            prefill    1003 ms   decode step median   31.3 ms   torch threads 32
worker thread (again)                    prefill     793 ms   decode step median   33.1 ms   torch threads 32
worker thread under asyncio loop         prefill     850 ms   decode step median   31.8 ms   torch threads 32
worker thread under asyncio loop (again) prefill     901 ms   decode step median   30.4 ms   torch threads 32
```

## scripts/profile_generate.py --batch 32 --prompt-len 128 --gen-len 9 --top 10

```
batch=32 prompt=128 gen=9 threads=32
generate(): total 1093 ms, prefill 820 ms, decode step median 33.9 ms (min 31.6, max 36.4)
bare forward: prefill 986 ms (all logits), 1022 ms (logits_to_keep=1); decode step median 34.1 ms
                                                 Name    Self CPU %      Self CPU   # of Calls
                                          aten::addmm        41.99%     340.861ms          432
                                            aten::cat        12.42%     100.826ms          234
                                            aten::mul        11.14%      90.400ms          450
                                            aten::add         8.05%      65.325ms          459
                                             aten::mm         6.59%      53.498ms            9
                                          aten::copy_         5.99%      48.662ms          930
                                           aten::tanh         3.90%      31.669ms          108
    aten::_scaled_dot_product_flash_attention_for_cpu         3.63%      29.452ms          108
                                            aten::pow         2.83%      22.970ms          108
                              aten::native_layer_norm         1.18%       9.582ms          225
Self CPU time total: 811.773ms
```

At batch 32 the per-step cost is no longer dominated by weight streaming:
the elementwise ops (`mul`, `add`, `tanh`, `pow`: the GELU and residual
adds over [32, 3072] activations) and the KV `cat` are a third of the time,
and `generate()` adds nothing measurable over a bare forward.

## scripts/profile_prefill.py --batches 1 4 8 16 32 --repeats 9 (host idle: vmstat 99-100% idle, 0 steal)

```
batch  1 x 128 =  128 tokens:  42 ms median, min  41, max  49  (3.0 tok/ms, 0.75 TFLOP/s)
batch  4 x 128 =  512 tokens: 105 ms median, min  63, max 109  (4.9 tok/ms, 1.21 TFLOP/s)
batch  8 x 128 = 1024 tokens: 204 ms median, min 196, max 214  (5.0 tok/ms, 1.25 TFLOP/s)
batch 16 x 128 = 2048 tokens: 360 ms median, min 299, max 372  (5.7 tok/ms, 1.41 TFLOP/s)
batch 32 x 128 = 4096 tokens: 670 ms median, min 565, max 897  (6.1 tok/ms, 1.52 TFLOP/s)
```

With `--profile` at 8 and 32: `aten::addmm` is 57-61% of the forward at
both shapes, the GELU/residual elementwise ops 20-27%, attention 4%; the
per-operator split does not change between the shapes, only the variance.
`generate()` sets `logits_to_keep=1` for GPT-2 (transformers 4.57,
`generation/utils.py` line 2512), so the LM head runs on one position per
sequence in prefill.

## Cold records

`results/stage2/cold/` holds the first stage 2 runs, taken before the warm-up
was added. The B=8 numbers match the warm runs; the B=16 and B=32 offline
records there are the ones that exposed the problem (TPOT 24.6 and 55.5 ms
versus 24.9 and 32.2 ms warm). Two earlier B=16/B=32 records that ran
concurrently with each other were discarded.
