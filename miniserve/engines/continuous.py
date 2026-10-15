"""Stage 3: continuous batching.

    request -> queue -> [engine loop, every step:
                           free the slots of finished or cancelled sequences,
                           admit queued requests into free slots (prefill each),
                           run one decode step over every active slot]
                     -> token per active sequence per step

The batch is no longer a fixed set. A sequence joins the running batch the
step after a slot frees up and leaves the moment its own output is done;
its result returns right then, not when the slowest member finishes. The
unit of scheduling is one decode step.

What this costs: a prefill runs inside the loop, so every other sequence
waits through it (the stats record prefill time separately so the effect on
TPOT is attributable), and the engine computes all max_slots slots every
step whether or not they are occupied (see miniserve/gpt2.py for why; the
stats record slot utilization).

The KV cache is explicit from here on: GPT2Core reserves max_seq_len
positions per slot, so the engine's admission rule is "a request needs a
free slot and prompt_len + max_new_tokens <= max_seq_len". There is no
mid-generation out-of-memory: a slot always has room for the sequence it
admitted. The price is the reservation itself, which stage 4 measures.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from ..gpt2 import argmax_tokens
from ..model import ModelRunner
from ..request import GenerationRequest, GenerationResult, dummy_request

_QueueItem = Tuple[GenerationRequest, "asyncio.Future[GenerationResult]"]


@dataclass
class _Seq:
    request: GenerationRequest
    fut: "asyncio.Future[GenerationResult]"
    slot: int
    service_start: float
    output: List[int]
    token_times: List[float]
    cache_len: int  # tokens whose K/V are in the slot; the last output token is not yet among them

    @property
    def done(self) -> bool:
        return len(self.output) >= self.request.max_new_tokens


class ContinuousBatchEngine:
    name = "continuous"
    stage = 3

    def __init__(self, runner: ModelRunner, max_slots: int = 8, max_seq_len: int = 1024) -> None:
        if max_slots < 1 or max_seq_len < 2:
            raise ValueError("need max_slots >= 1 and max_seq_len >= 2")
        self.runner = runner
        self.max_slots, self.max_seq_len = max_slots, max_seq_len
        self.core = runner.make_core(max_slots, max_seq_len)
        self._pending: Deque[_QueueItem] = deque()
        self._running: Dict[int, _Seq] = {}
        self._free_slots: List[int] = list(range(max_slots - 1, -1, -1))
        self._wake: Optional[asyncio.Event] = None
        self._loop_task: Optional["asyncio.Task[None]"] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._reset_stats()

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

    async def warmup(self, prompt_len: int, gen_len: int) -> None:
        """Fill every slot once at the largest shape, through the normal path."""
        if self._loop_task is None:
            await self.start()
        prompt_len = min(prompt_len, self.max_seq_len - gen_len)
        reqs = [dummy_request(f"warmup-{i}", prompt_len, gen_len) for i in range(self.max_slots)]
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
            fut.cancel()  # the loop sees this at the next step boundary and frees the slot
            raise

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
        """Move queued requests into free slots, prefilling each one."""
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        while self._pending and self._free_slots:
            request, fut = self._pending.popleft()
            if fut.cancelled():
                self._cancelled += 1
                continue
            if request.prompt_len + request.max_new_tokens > self.max_seq_len:
                self._reject(request, fut, f"prompt {request.prompt_len} + output {request.max_new_tokens} "
                                           f"exceeds max_seq_len {self.max_seq_len}")
                continue
            slot = self._free_slots.pop()
            t0 = time.perf_counter()
            try:
                logits = await loop.run_in_executor(self._executor, self.core.prefill, slot, request.prompt_ids)
            except Exception as exc:  # noqa: BLE001 (handed to the caller)
                self._free_slots.append(slot)
                if not fut.done():
                    fut.set_exception(exc)
                continue
            t1 = time.perf_counter()
            self._prefill_count += 1
            self._prefill_time += t1 - t0
            self._prefill_tokens += request.prompt_len
            tok = self._pick(logits, [request])[0]
            seq = _Seq(request, fut, slot, service_start=t0, output=[tok], token_times=[t1],
                       cache_len=request.prompt_len)
            self._running[slot] = seq
            self._note_kv_usage()
            if seq.done or self._stopped_on_eos(seq, tok):
                self._finish(seq, t1)

    def _pick(self, logits, requests: List[GenerationRequest]) -> List[int]:
        """Greedy token per row; EOS is never chosen for forced-length requests."""
        eos = self.core.eos_token_id
        forced = [i for i, r in enumerate(requests) if r.force_length]
        if forced and eos is not None:
            logits = logits.clone()
            logits[forced, eos] = -math.inf
        return argmax_tokens(logits)

    def _stopped_on_eos(self, seq: _Seq, tok: int) -> bool:
        return (not seq.request.force_length) and tok == self.core.eos_token_id

    async def _step(self) -> None:
        """One decode step over every active slot."""
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        for slot, seq in list(self._running.items()):
            if seq.fut.cancelled():
                self._release(seq)
                self._cancelled += 1
        if not self._running:
            return
        seqs = list(self._running.values())
        slots = [s.slot for s in seqs]
        tokens = [s.output[-1] for s in seqs]
        positions = [s.cache_len for s in seqs]
        t0 = time.perf_counter()
        logits = await loop.run_in_executor(self._executor, self.core.decode, slots, tokens, positions)
        t1 = time.perf_counter()
        self._decode_steps += 1
        self._decode_time += t1 - t0
        self._active_sum += len(seqs)
        picks = self._pick(logits, [s.request for s in seqs])
        for seq, tok in zip(seqs, picks):
            seq.cache_len += 1
            seq.output.append(tok)
            seq.token_times.append(t1)
            if seq.done or self._stopped_on_eos(seq, tok):
                self._finish(seq, t1)
        self._note_kv_usage()

    def _release(self, seq: _Seq) -> None:
        del self._running[seq.slot]
        self._free_slots.append(seq.slot)

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
        self._max_queue_depth = 0
        self._decode_steps = 0
        self._decode_time = 0.0
        self._active_sum = 0
        self._prefill_count = 0
        self._prefill_time = 0.0
        self._prefill_tokens = 0
        self._kv_peak_used_tokens = 0

    def _note_kv_usage(self) -> None:
        used = sum(s.cache_len + 1 for s in self._running.values())
        self._kv_peak_used_tokens = max(self._kv_peak_used_tokens, used)

    def stats(self) -> Dict[str, object]:
        steps = self._decode_steps
        return {
            "served": self._served,
            "rejected": self._rejected,
            "cancelled": self._cancelled,
            "max_queue_depth": self._max_queue_depth,
            "max_slots": self.max_slots,
            "max_seq_len": self.max_seq_len,
            "decode_steps": steps,
            "decode_time_s": self._decode_time,
            "decode_step_mean_ms": 1000 * self._decode_time / steps if steps else None,
            "mean_active_per_step": self._active_sum / steps if steps else None,
            "slot_utilization": self._active_sum / (steps * self.max_slots) if steps else None,
            "prefill_count": self._prefill_count,
            "prefill_time_s": self._prefill_time,
            "prefill_tokens": self._prefill_tokens,
            "kv_budget_tokens": self.max_slots * self.max_seq_len,
            "kv_peak_used_tokens": self._kv_peak_used_tokens,
            "kv_cache_bytes": self.core.cache_bytes,
        }
