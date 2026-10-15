"""Request and result records shared by every engine stage and by the benchmark.

A request carries token ids, not text. The benchmark generates synthetic token
ids directly (see miniserve.bench.workload), so tokenization is not part of
what we measure. Timestamps are time.perf_counter() values taken on the
serving side; the harness subtracts its own run start to report offsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationRequest:
    request_id: str
    prompt_ids: List[int]
    max_new_tokens: int
    # When True the engine must emit exactly max_new_tokens tokens and ignore
    # EOS. The benchmark uses this so output length is a controlled variable
    # rather than whatever the model decides.
    force_length: bool = True

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.prompt_ids:
            raise ValueError(f"{self.request_id}: prompt_ids must be non-empty")
        if any((not isinstance(t, int)) or t < 0 for t in self.prompt_ids):
            raise ValueError(f"{self.request_id}: prompt_ids must be non-negative ints")
        if self.max_new_tokens < 1:
            raise ValueError(f"{self.request_id}: max_new_tokens must be >= 1")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


@dataclass
class GenerationResult:
    request_id: str
    output_ids: List[int]
    # perf_counter timestamps on the serving side.
    service_start: float          # engine began working on this request
    token_times: List[float]      # one entry per generated token, in order
    finish_time: float
    finish_reason: str            # "length" or "stop"
    error: Optional[str] = None

    @property
    def first_token_time(self) -> float:
        return self.token_times[0]

    @property
    def num_generated(self) -> int:
        return len(self.output_ids)


@dataclass
class RequestRecord:
    """What the benchmark keeps per request: the spec, when it arrived, and the result."""
    request: GenerationRequest
    arrival_time: float
    result: Optional[GenerationResult] = None
    extra: dict = field(default_factory=dict)
