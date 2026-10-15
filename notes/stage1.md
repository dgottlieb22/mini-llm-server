# Stage 1: asynchronous intake and a request queue

    request -> queue -> [engine loop] -> generate() on a worker thread -> result

Requests are accepted the moment they arrive and sit in a FIFO queue; one
engine loop pulls them in order and runs `generate()` on a single worker
thread, so the event loop is never blocked. The model does exactly the same
work as in stage 0.

## Prediction (written before the run)

No throughput or latency number changes. The engine is still serial, so
queue wait, TTFT and throughput are set by the same 1.91 s service time.
The change is plumbing: the engine loop is where a scheduler, batching,
cancellation and streaming will live.

## Results

Same host, model, threads and workloads as stage 0. Records in
`results/stage1/`. Table from `python -m miniserve.bench.compare results/stage0/*.json results/stage1/*.json`.

| workload | stage | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | queue p50 | queue p99 | e2e p50 | e2e p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline | 0 naive | 0.524 | 33.5 | 29.7 s | 58.6 s | 30 ms | 29.7 s | 58.6 s | 31.6 s | 60.5 s |
| offline | 1 async_serial | 0.518 | 33.1 | 30.5 s | 59.4 s | 30 ms | 30.5 s | 59.3 s | 32.4 s | 61.2 s |
| closed8 | 0 naive | 0.522 | 33.4 | 13.4 s | 13.5 s | 30 ms | 13.4 s | 13.4 s | 15.3 s | 15.3 s |
| closed8 | 1 async_serial | 0.514 | 32.9 | 13.5 s | 13.9 s | 30 ms | 13.4 s | 13.9 s | 15.3 s | 15.9 s |
| uniform | 0 naive | 0.434 | 27.8 | 1.66 s | 7.3 s | 30 ms | 1.62 s | 7.3 s | 3.5 s | 9.2 s |
| uniform | 1 async_serial | 0.433 | 27.7 | 1.67 s | 7.5 s | 30 ms | 1.62 s | 7.4 s | 3.5 s | 9.4 s |
| hetero | 0 naive | 0.210 | 26.9 | 5.3 s | 22.1 s | 30 ms | 5.2 s | 22.1 s | 7.1 s | 24.5 s |
| hetero | 1 async_serial | 0.210 | 26.9 | 5.2 s | 21.7 s | 30 ms | 5.2 s | 21.6 s | 7.0 s | 24.1 s |
| burst | 0 naive | 0.423 | 27.1 | 6.8 s | 13.6 s | 30 ms | 6.7 s | 13.6 s | 8.7 s | 15.5 s |
| burst | 1 async_serial | 0.424 | 27.2 | 6.8 s | 13.5 s | 30 ms | 6.7 s | 13.5 s | 8.6 s | 15.4 s |

## What the numbers say

The prediction held. Every metric is within about 1% of stage 0 (offline
throughput 0.518 versus 0.524 req/s; the largest TTFT p99 difference is
0.8 s on a 59 s value, which is one request's worth of the run-to-run
variation in when the 32 requests finish). Accepting requests promptly
does nothing for the user when the thing they are waiting for is still
one serial model loop. The queue made the waiting visible to the engine
(`max_queue_depth` in the records: 32 on `offline`, 8 on `closed8` and
`burst`, 5 on `hetero`, 4 on `uniform`) without shortening it.

What changed is observable only in the tests:
`test_event_loop_stays_responsive_while_model_runs` shows a side coroutine
running about 20 times during one request on stage 1 and at most twice on
stage 0. That is the property the next stages need: a scheduler cannot
look at the queue if the loop that would run it is stuck inside
`generate()`.

## Decision

Proceed to static batching (stage 0 decision, option 1), with the stage 0
prediction as the thing to check: about 10x throughput on `offline`,
`closed8` and `burst`; a small gain on `uniform`; padding waste and
in-batch head-of-line blocking on `hetero`.
