"""Stage 1: asynchronous intake and a request queue. Still one request at a time.

    request -> queue -> [engine loop] -> generate() on a worker thread -> result

The event loop no longer blocks while the model runs: requests are accepted
the moment they arrive and sit in a FIFO queue; one engine loop pulls them
in order and hands each to the model on a single worker thread.

Nothing about the model's work changed, so the prediction is that no
throughput or latency number moves. What this buys is the plumbing every
later stage needs: the engine loop is the place where a scheduler, batching,
cancellation and streaming can live.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

from ..model import ModelRunner
from ..request import GenerationRequest, GenerationResult

_QueueItem = Tuple[GenerationRequest, "asyncio.Future[GenerationResult]"]


class AsyncSerialEngine:
    name = "async_serial"
    stage = 1

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self._queue: Optional["asyncio.Queue[_QueueItem]"] = None
        self._loop_task: Optional["asyncio.Task[None]"] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._served = 0
        self._max_queue_depth = 0

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        self._queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
        self._loop_task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._loop_task is None:
            return
        assert self._queue is not None and self._executor is not None
        await self._queue.join()  # let queued work finish
        self._loop_task.cancel()
        try:
            await self._loop_task
        except asyncio.CancelledError:
            pass
        self._executor.shutdown(wait=True)
        self._loop_task = None

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._loop_task is None:
            await self.start()
        assert self._queue is not None
        fut: "asyncio.Future[GenerationResult]" = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((request, fut))
        self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
        return await fut

    async def _run(self) -> None:
        assert self._queue is not None and self._executor is not None
        loop = asyncio.get_running_loop()
        while True:
            request, fut = await self._queue.get()
            try:
                if fut.cancelled():
                    continue
                try:
                    result = await loop.run_in_executor(self._executor, self.runner.generate_one, request)
                except Exception as exc:  # noqa: BLE001 (handed to the caller)
                    if not fut.done():
                        fut.set_exception(exc)
                else:
                    self._served += 1
                    if not fut.done():
                        fut.set_result(result)
            finally:
                self._queue.task_done()

    def stats(self) -> Dict[str, object]:
        return {"served": self._served, "max_queue_depth": self._max_queue_depth}
