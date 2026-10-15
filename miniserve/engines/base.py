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

    async def warmup(self, prompt_len: int, gen_len: int) -> None:
        """Run a dummy workload at the engine's largest shape before measuring.

        Production engines do this at startup (vLLM's profile run): the first
        call in a fresh process pays for page-faulting every allocation the
        model makes, which the benchmark should not attribute to the
        architecture. The harness records whether it ran.
        """
        ...

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stats(self) -> Dict[str, object]:
        """Engine-internal counters for the benchmark record (may be empty)."""
        ...
