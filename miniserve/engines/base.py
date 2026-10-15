"""The interface every serving stage implements.

The benchmark harness only knows this interface, so every stage is measured
by exactly the same client code.
"""
from __future__ import annotations

from typing import Dict, Protocol

from ..request import GenerationRequest, GenerationResult


class Engine(Protocol):
    name: str
    stage: int

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stats(self) -> Dict[str, object]:
        """Engine-internal counters for the benchmark record (may be empty)."""
        ...
