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
  engines/
    base.py             the Engine interface every stage implements
    naive.py            stage 0
    async_serial.py     stage 1
  bench/
    workload.py         WorkloadSpec, length distributions, arrival processes, presets
    harness.py          asyncio load generator; replays a workload against any engine
    metrics.py          TTFT / TPOT / queue wait / throughput definitions
    run.py              CLI: one engine x one workload -> JSON record
    compare.py          tables across records, head-of-line blocking summary
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

The rates were sized to the stage 0 service time (1.91 s for a 128/64
request on this host) so the baseline runs at about 0.8 utilization.

## Metrics

Per request: queue wait (service start minus arrival), TTFT (first token
minus arrival, so it includes queueing), prefill latency (first token minus
service start), TPOT, end-to-end latency. Per run: requests/s, output
tokens/s, total tokens/s, busy fraction, peak resident tokens (the KV
footprint the engine needed). Definitions in `miniserve/bench/metrics.py`.
