# mini-llm-server

A miniature LLM serving engine, built up from `model.generate()` one measured
architectural change at a time. Hugging Face runs the model; scheduling,
request state, batching and (later) KV-cache allocation are in this repo
where you can read them.

This is a case study, not a vLLM replacement. Each stage answers four
questions before any code changes: what is the bottleneck, why should this
change help, what measurement will show it, and what does it cost. The
answers are in `notes/stage*.md`; the raw numbers behind every table are in
`results/`. Every performance number in the notes comes from a committed
result file or a probe whose raw output is committed next to it.

## Status

| stage | engine | what changed | notes |
|---|---|---|---|
| 0 | `naive` | one request at a time, whole request to `generate()` | [notes/stage0.md](notes/stage0.md) |
| 1 | `async_serial` | async intake + FIFO queue; model still serial on a worker thread. Predicted and measured: no change | [notes/stage1.md](notes/stage1.md) |
| 2 | `static_batch` | take up to B queued requests, pad, one `generate()` per batch. 11x throughput on offline/closed-loop; 42% of decode slots wasted on heterogeneous lengths | [notes/stage2.md](notes/stage2.md) |
| 3 | `continuous` | our own GPT-2 forward over an explicit slot KV cache; admit and retire sequences at every decode step. Queue wait on hetero_hi 3.0 s to 8 ms; KV reservation 70-93% empty at peak | [notes/stage3.md](notes/stage3.md) |
| 4 | `paged` | block allocator + block tables, gather per step. Same 8192-token budget holds 2-4x the sequences: 12.0 vs 6.1 req/s offline, uniform_8x TTFT p99 2.3 s to 114 ms; loses on long contexts, where the gather doubles KV traffic | [notes/stage4.md](notes/stage4.md) |
| 5 | `paged --engine-kw prefill_batch_tokens=4096` | co-arriving requests prefill in one padded forward. closed8/burst TTFT p99 300 to 106/125 ms; offline 12.0 to 15.5 req/s | [notes/stage5.md](notes/stage5.md) |

Everything so far ran on a CPU-only host (AMD EPYC 9R14, 64 cores) with
GPT-2 small in fp32. GPU utilization and GPU memory are recorded as "not
measured" in every result file. The serving problems the stages address
(queueing, head-of-line blocking, amortizing the weight read, KV-cache
growth) exist on both; the per-step numbers do not transfer.

## Layout

```
miniserve/
  request.py            GenerationRequest / GenerationResult / RequestRecord
  model.py              HFModelRunner (generate() + per-token timestamps), FakeModelRunner
  gpt2.py               our own GPT-2 forward over a slot-indexed KV cache (stage 3+)
  paged.py              block allocator, block tables, paged GPT-2 core (stage 4+)
  engines/
    base.py             the Engine interface every stage implements
    naive.py            stage 0
    async_serial.py     stage 1
    static_batch.py     stage 2
    continuous.py       stage 3
    paged.py            stage 4
  bench/
    workload.py         WorkloadSpec, length distributions, arrival processes, presets
    harness.py          asyncio load generator; replays a workload against any engine (with warm-up)
    metrics.py          TTFT / TPOT / queue wait / throughput / occupancy definitions
    run.py              CLI: one engine x one workload -> JSON record
    compare.py          tables across records, head-of-line blocking summary
    resummarize.py      recompute summaries of old records after a metric change
    envinfo.py          hardware and version capture
scripts/                probes and calibration used in the notes
notes/                  one experiment record per stage
results/<stage>/        committed JSON records the notes cite
tests/                  scheduler, harness, metrics and request validation
```

## Running

```
make venv-cpu                                  # python3 -m venv .venv; pinned CPU deps
make test
.venv/bin/python -m miniserve.bench.run --engine naive --workload uniform --threads 32 --out results/raw
scripts/run_stage.sh naive results/stage0 --threads 32    # the five standard workloads
.venv/bin/python -m miniserve.bench.compare results/stage0/*.json
```

`--model fake` runs the harness with a deterministic fake model (no weights,
optional simulated per-token cost); the tests use it.

## Workloads

All presets use 32 requests, random token ids, forced output length, seed 0.

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

The first five rates were sized to the stage 0 service time (1.91 s for a
128/64 request on this host) so the baseline runs at about 0.8 utilization.
The two `_hi` variants were added at stage 2 for the batching engines; the
serial engine is overloaded on them. The `_8x`/`_4x` variants were added at
stage 4 to saturate 8 concurrent sequences. Every run warms the engine with
one dummy batch at the workload's largest shape before the clock starts
(`--no-warmup` to measure cold).

## Metrics

Per request: queue wait (service start minus arrival), TTFT (first token
minus arrival, so it includes queueing), prefill latency (first token minus
service start), TPOT, end-to-end latency. Per run: requests/s, output
tokens/s, total tokens/s, busy fraction, peak resident tokens (the KV
footprint the engine needed). Definitions in `miniserve/bench/metrics.py`.
