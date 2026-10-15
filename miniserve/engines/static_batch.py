"""Stage 2: static batching.

    request -> queue -> [engine loop: take up to B queued] -> generate(batch) -> results

The engine loop is stage 1's, with one change in the body: when the model
is free, it takes everything queued, up to max_batch_size, and runs one
generate() over the whole batch. It does not wait for the batch to fill.

Static means the batch is fixed for its whole lifetime: prompts are padded
to the longest prompt, every member decodes in lockstep until the longest
output is done, and all results return together. The stats count what that
costs: prompt positions spent on padding, and decode slots spent on
sequences that had already finished.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from typing import Dict, List

from ..model import ModelRunner
from ..request import GenerationRequest
from .async_serial import AsyncSerialEngine, _QueueItem


class StaticBatchEngine(AsyncSerialEngine):
    name = "static_batch"
    stage = 2

    def __init__(self, runner: ModelRunner, max_batch_size: int = 8) -> None:
        super().__init__(runner)
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.max_batch_size = max_batch_size
        self._batch_sizes: Counter = Counter()
        self._prompt_tokens_real = 0
        self._prompt_tokens_padded = 0
        self._decode_slots_real = 0
        self._decode_slots_padded = 0

    def _take_batch(self, first: _QueueItem) -> List[_QueueItem]:
        assert self._queue is not None
        batch = [first]
        while len(batch) < self.max_batch_size and not self._queue.empty():
            batch.append(self._queue.get_nowait())
        return batch

    def _account(self, requests: List[GenerationRequest]) -> None:
        max_prompt = max(r.prompt_len for r in requests)
        max_gen = max(r.max_new_tokens for r in requests)
        self._batch_sizes[len(requests)] += 1
        self._prompt_tokens_real += sum(r.prompt_len for r in requests)
        self._prompt_tokens_padded += max_prompt * len(requests)
        self._decode_slots_real += sum(r.max_new_tokens for r in requests)
        self._decode_slots_padded += max_gen * len(requests)

    async def _run(self) -> None:
        assert self._queue is not None and self._executor is not None
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch = self._take_batch(first)
            try:
                live = [(req, fut) for req, fut in batch if not fut.cancelled()]
                if not live:
                    continue
                requests = [req for req, _ in live]
                self._account(requests)
                try:
                    results = await loop.run_in_executor(self._executor, self.runner.generate_batch, requests)
                except Exception as exc:  # noqa: BLE001 (handed to every caller in the batch)
                    for _, fut in live:
                        if not fut.done():
                            fut.set_exception(exc)
                else:
                    self._served += len(results)
                    for (_, fut), res in zip(live, results):
                        if not fut.done():
                            fut.set_result(res)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def stats(self) -> Dict[str, object]:
        s = super().stats()
        s.update(
            max_batch_size=self.max_batch_size,
            batch_sizes={str(k): v for k, v in sorted(self._batch_sizes.items())},
            prompt_tokens_real=self._prompt_tokens_real,
            prompt_tokens_padded=self._prompt_tokens_padded,
            decode_slots_real=self._decode_slots_real,
            decode_slots_padded=self._decode_slots_padded,
        )
        if self._prompt_tokens_padded:
            s["prompt_pad_fraction"] = 1 - self._prompt_tokens_real / self._prompt_tokens_padded
        if self._decode_slots_padded:
            s["decode_slot_waste_fraction"] = 1 - self._decode_slots_real / self._decode_slots_padded
        return s
