# Stage 0 probes (raw output)

Host: AMD EPYC 9R14, 64 cores, no GPU, 246 GiB RAM. torch 2.8.0+cpu (MKL),
transformers 4.57.1, Python 3.9. Model: gpt2 (124,439,808 params, fp32,
497.8 MB of weights). Run 2026-10-15.

## scripts/calibrate_threads.py --model gpt2 --threads 4 8 16 32 64

```
model=gpt2 prompt=128 gen=32 repeats=3
threads  prefill_ms  decode_ms_per_tok
      4       119.2              28.84
      8        71.0              29.40
     16        54.4              29.44
     32        42.4              29.52
     64        43.2              31.77
```

Decision: 32 threads for every benchmark in this project on this host.
Prefill stops improving past 32; decode does not move at all.

## scripts/probe_decode_step.py --model gpt2 --threads 32 --thread-sweep 1 2 4 8 16 32

```
model=gpt2 threads=32 context=160
(a) bare forward, one new token, KV cache populated:
    batch  1:   28.66 ms/step  =  28.66 ms per sequence-token
    batch  2:   15.00 ms/step  =   7.50 ms per sequence-token
    batch  4:   15.15 ms/step  =   3.79 ms per sequence-token
    batch  8:   18.18 ms/step  =   2.27 ms per sequence-token
    batch 16:   23.04 ms/step  =   1.44 ms per sequence-token
    batch 32:   28.37 ms/step  =   0.89 ms per sequence-token
(a') batch-1 and batch-2 step vs thread count:
    threads  1: batch 1   28.41 ms   batch 2   41.99 ms
    threads  2: batch 1   28.03 ms   batch 2   41.37 ms
    threads  4: batch 1   27.68 ms   batch 2   23.52 ms
    threads  8: batch 1   28.46 ms   batch 2   20.88 ms
    threads 16: batch 1   28.53 ms   batch 2   19.09 ms
    threads 32: batch 1   28.66 ms   batch 2   15.03 ms
(b) model.generate(), batch 1, per generated token:
      29.86 ms per token (median of 20 steps)
(c) prefill cost vs prompt length, batch 1:
      16 tokens:    21.2 ms
      64 tokens:    29.0 ms
     128 tokens:    38.2 ms
     256 tokens:    55.5 ms
     512 tokens:   103.1 ms
```

## scripts/probe_gemv.py

```
BLAS: MKL | mkldnn: True
W is 9.0 MiB; GPT-2 small reads ~473 MiB of weights per decode step
threads    M=1 us    M=2 us    M=4 us  M=1 GB/s
      1       200       417       563    47.3
      4       151       176       247    62.5
     16       150       123       169    62.8
     32       151       118       133    62.6

LM-head sized W [50257x768] = 147 MiB
threads  linear(M=1) us     mv us  linear(M=2) us
      1           10683     10590           23904   (M=1 linear: 14.5 GB/s)
     32           10733     10577            3562   (M=1 linear: 14.4 GB/s)
```

The 9 MiB matrix fits in L3 (256 MiB), so its 62 GB/s is cache bandwidth.
The 147 MiB LM-head matrix streams from DRAM: 14.4 GB/s single-threaded,
and the M=1 path (`linear` or `mv`) ignores the thread count. M=2 at 32
threads reads the same matrix 3x faster.

## scripts/profile_decode_step.py --threads 32 --top 10 (self CPU time per step, 10 steps)

```
=== batch 1: 27.5 ms per step ===
aten::mm        119.5 ms / 10 steps  (1 call per step: the LM head, 154 MB)
aten::addmm     116.6 ms / 10 steps  (48 calls per step: 4 linears x 12 layers, 340 MB)
aten::cat        21.3 ms / 10 steps  (24 calls per step: DynamicCache append)
aten::_scaled_dot_product_flash_attention_for_cpu  3.4 ms / 10 steps
everything else  < 15 ms / 10 steps

=== batch 2: 16.6 ms per step ===
aten::addmm      92.2 ms / 10 steps
aten::cat        26.5 ms / 10 steps
aten::mm         21.9 ms / 10 steps  (5.5x faster than at batch 1)
aten::_scaled_dot_product_flash_attention_for_cpu  4.3 ms / 10 steps
```
