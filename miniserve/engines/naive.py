"""Stage 0: the simplest possible server.

    request -> model.generate() -> complete -> next request

One asyncio.Lock serializes requests in arrival order. generate() runs on the
event-loop thread, so while a request is being served nothing else happens:
no new request is even accepted. That is deliberate; it is the baseline every
later stage is compared against.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Optional

from ..model import ModelRunner
from ..request import GenerationRequest, GenerationResult


class NaiveEngine:
    name = "naive"
    stage = 0

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self._lock: Optional[asyncio.Lock] = None
        self._served = 0

    async def start(self) -> None:
        # Created here, not in __init__: on Python 3.9 a Lock binds to the
        # event loop that is current when it is constructed.
        self._lock = asyncio.Lock()

    async def stop(self) -> None:
        pass

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._lock is None:
            await self.start()
        assert self._lock is not None
        async with self._lock:
            result = self.runner.generate_one(request)
            self._served += 1
            return result

    def stats(self) -> Dict[str, object]:
        return {"served": self._served}
