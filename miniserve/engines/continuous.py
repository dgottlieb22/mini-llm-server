"""Stage 3: continuous batching.

    request -> queue -> [engine loop, every step:
                           release the storage of finished or cancelled sequences,
                           admit queued requests the cache can hold (prefill each),
                           run one decode step over every active sequence]
                     -> token per active sequence per step

The batch is no longer a fixed set. A sequence joins the running batch the
step after storage frees up and leaves the moment its own output is done;
its result returns right then, not when the slowest member finishes. The
unit of scheduling is one decode step.

What this costs: a prefill runs inside the loop, so every other sequence
waits through it (the stats record prefill time separately so the effect on
TPOT is attributable).

The KV cache is explicit from here on, and the engine does not know how it
is organized. The core answers can_admit()/admit()/free(): at stage 3 the
storage unit is a slot of max_seq_len positions (one reservation per
sequence, so there is no mid-generation out-of-memory and the price is the
reservation itself); stage 4 swaps in a block-granular core behind the same
three calls. Requests that could never fit are rejected at admission.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..gpt2 import argmax_tokens
from ..model import ModelRunner
from ..request import GenerationRequest, GenerationResult, dummy_request

_QueueItem = Tuple[GenerationRequest, "asyncio.Future[GenerationResult]"]


@dataclass
class _Seq:
    request: GenerationRequest
    fut: "asyncio.Future[GenerationResult]"
    handle: Any                 # the core's storage handle (slot id, or block table)
    service_start: float
    output: List[int]
    token_times: List[float]
    cache_len: int  # tokens whose K/V are in the cache; the last output token is not yet among them

    @property
    def done(self) -> bool:
        return len(self.output) >= self.request.max_new_tokens


class ContinuousBatchEngine:
    name = "continuous"
    stage = 3

    def __init__(self, runner: ModelRunner, max_slots: int = 8, max_seq_len: int = 1024,
                 max_live_tokens: Optional[int] = None) -> None:
        if max_slots < 1 or max_seq_len < 2:
            raise ValueError("need max_slots >= 1 and max_seq_len >= 2")
        if max_live_tokens is not None and max_live_tokens < max_seq_len:
            raise ValueError("max_live_tokens must be at least max_seq_len or no request could run")
        self.runner = runner
        self.max_slots, self.max_seq_len = max_slots, max_seq_len
        # Bound on tokens resident across running sequences. Memory is what
        # the core's can_admit() checks; this bounds the work of a decode step,
        # which reads every resident token's K/V. None = memory only.
        self.max_live_tokens = max_live_tokens
        self.core = self._make_core()
        self._pending: Deque[_QueueItem] = deque()
        self._running: Dict[str, _Seq] = {}   # by request id, in admission order
        self._wake: Optional[asyncio.Event] = None
        self._loop_task: Optional["asyncio.Task[None]"] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._reset_stats()

    def _make_core(self):
        return self.runner.make_core(self.max_slots, self.max_seq_len)

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        self._wake = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
        self._loop_task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._loop_task is None:
            return
        while self._pending or self._running:  # let in-flight work finish
            await asyncio.sleep(0.005)
        self._loop_task.cancel()
        try:
            await self._loop_task
        except asyncio.CancelledError:
            pass
        assert self._executor is not None
        self._executor.shutdown(wait=True)
        self._loop_task = None

    def _warmup_count(self, prompt_len: int, gen_len: int) -> int:
        return self.max_slots

    async def warmup(self, prompt_len: int, gen_len: int) -> None:
        """Fill the cache once at the largest shape, through the normal path."""
        if self._loop_task is None:
            await self.start()
        prompt_len = min(prompt_len, self.max_seq_len - gen_len)
        n = self._warmup_count(prompt_len, gen_len)
        reqs = [dummy_request(f"warmup-{i}", prompt_len, gen_len) for i in range(n)]
        await asyncio.gather(*(self.generate(r) for r in reqs))
        self._reset_stats()

    # ---- client API ------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._loop_task is None:
            await self.start()
        assert self._wake is not None
        fut: "asyncio.Future[GenerationResult]" = asyncio.get_running_loop().create_future()
        self._pending.append((request, fut))
        self._max_queue_depth = max(self._max_queue_depth, len(self._pending))
        self._wake.set()
        try:
            return await fut
        except asyncio.CancelledError:
            fut.cancel()  # the loop sees this at the next step boundary and frees the storage
            raise

    # ---- model calls (run on the worker thread) --------------------------
    #
    # Both return plain Python ints. The asyncio thread never touches a
    # tensor: scripts/probe_engine_step.py measured 3 ms per step of
    # main-thread work when the argmax ran there, on a 16.6 ms decode.

    def _prefill_and_pick(self, handle: Any, request: GenerationRequest) -> int:
        logits = self.core.prefill(handle, request.prompt_ids)
        return self._pick(logits, [request])[0]

    def _decode_and_pick(self, handles: List[Any], tokens: List[int], positions: List[int],
                         requests: List[GenerationRequest]) -> List[int]:
        logits = self.core.decode(handles, tokens, positions)
        return self._pick(logits, requests)

    def _pick(self, logits, requests: List[GenerationRequest]) -> List[int]:
        """Greedy token per row; EOS is never chosen for forced-length requests."""
        eos = self.core.eos_token_id
        forced = [i for i, r in enumerate(requests) if r.force_length]
        if forced and eos is not None:
            logits = logits.clone()
            logits[forced, eos] = -math.inf
        return argmax_tokens(logits)

    # ---- the loop --------------------------------------------------------

    async def _run(self) -> None:
        assert self._wake is not None
        while True:
            if not self._running and not self._pending:
                self._wake.clear()
                await self._wake.wait()
            await self._admit()
            if self._running:
                await self._step()

    def _reject(self, request: GenerationRequest, fut: "asyncio.Future[GenerationResult]", why: str) -> None:
        self._rejected += 1
        if not fut.done():
            fut.set_exception(ValueError(f"{request.request_id}: {why}"))

    async def _admit(self) -> None:
        """Move queued requests into the cache, FIFO, prefilling each one.

        Stops at the first request that does not fit right now: admitting a
        later, smaller request ahead of it would starve the big one.
        """
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        while self._pending:
            request, fut = self._pending[0]
            if fut.cancelled():
                self._pending.popleft()
                self._cancelled += 1
                continue
            if not self.core.fits(request.prompt_len, request.max_new_tokens):
                self._pending.popleft()
                self._reject(request, fut, f"prompt {request.prompt_len} + output {request.max_new_tokens} "
                                           f"cannot fit in the cache")
                continue
            if not self.core.can_admit(request.prompt_len, request.max_new_tokens) or \
                    not self._within_live_budget(request):
                self._admission_waits += 1
                break
            self._pending.popleft()
            handle = self.core.admit(request.prompt_len, request.max_new_tokens)
            t0 = time.perf_counter()
            try:
                tok = await loop.run_in_executor(self._executor, self._prefill_and_pick, handle, request)
            except Exception as exc:  # noqa: BLE001 (handed to the caller)
                self.core.free(handle)
                if not fut.done():
                    fut.set_exception(exc)
                continue
            t1 = time.perf_counter()
            self._prefill_count += 1
            self._prefill_time += t1 - t0
            self._prefill_tokens += request.prompt_len
            seq = _Seq(request, fut, handle, service_start=t0, output=[tok], token_times=[t1],
                       cache_len=request.prompt_len)
            self._running[request.request_id] = seq
            self._note_kv_usage()
            if seq.done or self._stopped_on_eos(seq, tok):
                self._finish(seq, t1)

    def _stopped_on_eos(self, seq: _Seq, tok: int) -> bool:
        return (not seq.request.force_length) and tok == self.core.eos_token_id

    def _within_live_budget(self, request: GenerationRequest) -> bool:
        """Would admitting this request keep resident tokens under max_live_tokens?

        Counts what is resident now plus the new prompt. Running sequences
        keep growing by one token per step, so this is a soft bound on the
        step's work, not a hard one; an empty engine always admits.
        """
        if self.max_live_tokens is None or not self._running:
            return True
        resident = sum(s.cache_len + 1 for s in self._running.values())
        return resident + request.prompt_len <= self.max_live_tokens

    async def _step(self) -> None:
        """One decode step over every active sequence."""
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        for seq in list(self._running.values()):
            if seq.fut.cancelled():
                self._release(seq)
                self._cancelled += 1
        if not self._running:
            return
        seqs = list(self._running.values())
        handles = [s.handle for s in seqs]
        tokens = [s.output[-1] for s in seqs]
        positions = [s.cache_len for s in seqs]
        t0 = time.perf_counter()
        picks = await loop.run_in_executor(
            self._executor, self._decode_and_pick, handles, tokens, positions, [s.request for s in seqs]
        )
        t1 = time.perf_counter()
        self._decode_steps += 1
        self._decode_time += t1 - t0
        self._active_sum += len(seqs)
        for seq, tok in zip(seqs, picks):
            seq.cache_len += 1
            seq.output.append(tok)
            seq.token_times.append(t1)
            if seq.done or self._stopped_on_eos(seq, tok):
                self._finish(seq, t1)
        self._note_kv_usage()

    def _release(self, seq: _Seq) -> None:
        del self._running[seq.request.request_id]
        self.core.free(seq.handle)

    def _finish(self, seq: _Seq, finish_time: float) -> None:
        self._release(seq)
        self._served += 1
        reason = "length" if len(seq.output) >= seq.request.max_new_tokens else "stop"
        if not seq.fut.done():
            seq.fut.set_result(GenerationResult(
                request_id=seq.request.request_id,
                output_ids=seq.output,
                service_start=seq.service_start,
                token_times=seq.token_times,
                finish_time=finish_time,
                finish_reason=reason,
            ))

    # ---- stats -----------------------------------------------------------

    def _reset_stats(self) -> None:
        self._served = self._rejected = self._cancelled = 0
        self._admission_waits = 0
        self._max_queue_depth = 0
        self._decode_steps = 0
        self._decode_time = 0.0
        self._active_sum = 0
        self._prefill_count = 0
        self._prefill_time = 0.0
        self._prefill_tokens = 0
        self._kv_peak_resident_tokens = 0
        self._kv_peak_allocated_tokens = 0

    def _note_kv_usage(self) -> None:
        """Resident = tokens whose K/V are stored; allocated = positions reserved for them."""
        resident = sum(s.cache_len + 1 for s in self._running.values())
        allocated = self.core.allocated_tokens([s.handle for s in self._running.values()])
        self._kv_peak_resident_tokens = max(self._kv_peak_resident_tokens, resident)
        self._kv_peak_allocated_tokens = max(self._kv_peak_allocated_tokens, allocated)

    def _concurrency_limit(self) -> int:
        return self.max_slots

    def stats(self) -> Dict[str, object]:
        steps = self._decode_steps
        s: Dict[str, object] = {
            "served": self._served,
            "rejected": self._rejected,
            "cancelled": self._cancelled,
            "admission_waits": self._admission_waits,
            "max_queue_depth": self._max_queue_depth,
            "max_seq_len": self.max_seq_len,
            "decode_steps": steps,
            "decode_time_s": self._decode_time,
            "decode_step_mean_ms": 1000 * self._decode_time / steps if steps else None,
            "mean_active_per_step": self._active_sum / steps if steps else None,
            "prefill_count": self._prefill_count,
            "prefill_time_s": self._prefill_time,
            "prefill_tokens": self._prefill_tokens,
            "kv_budget_tokens": self.core.budget_tokens,
            "kv_peak_resident_tokens": self._kv_peak_resident_tokens,
            "kv_peak_allocated_tokens": self._kv_peak_allocated_tokens,
            "kv_cache_bytes": self.core.cache_bytes,
            "max_live_tokens": self.max_live_tokens,
            "attn_positions_computed": getattr(self.core, "attn_positions_computed", None),
            "attn_positions_live": getattr(self.core, "attn_positions_live", None),
        }
        s.update(self._extra_stats())
        return s

    def _extra_stats(self) -> Dict[str, object]:
        steps = self._decode_steps
        return {
            "max_slots": self.max_slots,
            "slot_utilization": self._active_sum / (steps * self.max_slots) if steps else None,
        }
