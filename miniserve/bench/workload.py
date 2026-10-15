"""Synthetic workload generation.

A WorkloadSpec says how many requests, how long their prompts and outputs
are, and how they arrive. Everything is drawn from one seeded RNG so a
workload replays identically against every engine stage.

Arrival modes
  poisson   open loop, exponential inter-arrival times at `rate` req/s
  burst     open loop, `burst_size` requests arrive together every
            `burst_interval` seconds
  instant   open loop, every request arrives at t=0 (offline throughput)
  closed    `concurrency` clients each keep exactly one request in flight;
            arrival times are decided at run time by the harness
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..request import GenerationRequest


@dataclass(frozen=True)
class LengthSpec:
    """A distribution over token counts.

    fixed:     always `value`
    uniform:   integer uniform on [low, high]
    lognormal: round(exp(N(mu, sigma))) clipped to [low, high]; mu/sigma are
               the parameters of the underlying normal.
    """

    kind: str = "fixed"
    value: int = 0
    low: int = 1
    high: int = 1
    mu: float = 0.0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "uniform", "lognormal"):
            raise ValueError(f"unknown length kind {self.kind!r}")
        if self.kind == "fixed" and self.value < 1:
            raise ValueError("fixed length must be >= 1")
        if self.kind != "fixed" and not (1 <= self.low <= self.high):
            raise ValueError("need 1 <= low <= high")

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return self.value
        if self.kind == "uniform":
            return rng.randint(self.low, self.high)
        x = int(round(math.exp(rng.gauss(self.mu, self.sigma))))
        return min(self.high, max(self.low, x))

    @property
    def max(self) -> int:
        return self.value if self.kind == "fixed" else self.high

    def describe(self) -> str:
        if self.kind == "fixed":
            return str(self.value)
        if self.kind == "uniform":
            return f"U[{self.low},{self.high}]"
        return f"LogN(mu={self.mu},sigma={self.sigma})[{self.low},{self.high}]"


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    num_requests: int
    prompt_len: LengthSpec
    gen_len: LengthSpec
    arrival: str = "poisson"
    rate: float = 1.0
    burst_size: int = 1
    burst_interval: float = 1.0
    concurrency: int = 1
    shared_prefix_len: int = 0
    vocab_size: int = 50000
    seed: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if self.arrival not in ("poisson", "burst", "instant", "closed"):
            raise ValueError(f"unknown arrival mode {self.arrival!r}")
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if self.arrival == "poisson" and self.rate <= 0:
            raise ValueError("poisson arrivals need rate > 0")
        if self.arrival == "burst" and (self.burst_size < 1 or self.burst_interval < 0):
            raise ValueError("burst arrivals need burst_size >= 1 and burst_interval >= 0")
        if self.arrival == "closed" and self.concurrency < 1:
            raise ValueError("closed loop needs concurrency >= 1")
        if self.shared_prefix_len < 0 or self.shared_prefix_len > self.prompt_len.max:
            raise ValueError("shared_prefix_len must be within [0, max prompt length]")

    @property
    def open_loop(self) -> bool:
        return self.arrival != "closed"

    def to_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["prompt_len"] = self.prompt_len.describe()
        d["gen_len"] = self.gen_len.describe()
        return d


def generate_requests(spec: WorkloadSpec) -> List[GenerationRequest]:
    """Token ids and lengths for every request, deterministic in spec.seed."""
    rng = random.Random(spec.seed)
    prefix = [rng.randrange(spec.vocab_size) for _ in range(spec.shared_prefix_len)]
    requests = []
    for i in range(spec.num_requests):
        plen = max(spec.prompt_len.sample(rng), spec.shared_prefix_len or 1)
        body = [rng.randrange(spec.vocab_size) for _ in range(plen - len(prefix))]
        requests.append(
            GenerationRequest(
                request_id=f"{spec.name}-{i:04d}",
                prompt_ids=prefix + body,
                max_new_tokens=spec.gen_len.sample(rng),
            )
        )
    return requests


def arrival_offsets(spec: WorkloadSpec) -> Optional[List[float]]:
    """Seconds after run start at which each request arrives (open loop only).

    Uses a separate RNG stream (seed+1) so changing the arrival process does
    not change the request lengths.
    """
    if not spec.open_loop:
        return None
    rng = random.Random(spec.seed + 1)
    if spec.arrival == "instant":
        return [0.0] * spec.num_requests
    if spec.arrival == "poisson":
        t, out = 0.0, []
        for _ in range(spec.num_requests):
            out.append(t)
            t += rng.expovariate(spec.rate)
        return out
    # burst
    return [(i // spec.burst_size) * spec.burst_interval for i in range(spec.num_requests)]


def _fixed(n: int) -> LengthSpec:
    return LengthSpec("fixed", value=n)


def _uniform(lo: int, hi: int) -> LengthSpec:
    return LengthSpec("uniform", low=lo, high=hi)


# Named workloads. Rates were sized to the measured stage-0 service time on
# the CPU host (GPT-2 small, 32 threads: a 128-token prompt plus 64 generated
# tokens takes about 1.9 s), so the naive engine runs at roughly 0.8
# utilization on the open-loop workloads. A GPU host can scale num_requests
# and rate up; the point is that every stage sees the same workloads.
WORKLOADS: Dict[str, WorkloadSpec] = {
    "smoke": WorkloadSpec(
        name="smoke", num_requests=6, prompt_len=_fixed(32), gen_len=_fixed(8),
        arrival="poisson", rate=4.0, description="Tiny run to check the pipeline works.",
    ),
    "uniform": WorkloadSpec(
        name="uniform", num_requests=32, prompt_len=_fixed(128), gen_len=_fixed(64),
        arrival="poisson", rate=0.4,
        description="Homogeneous requests (128 in, 64 out), Poisson arrivals at 0.4 req/s.",
    ),
    "hetero": WorkloadSpec(
        name="hetero", num_requests=32, prompt_len=_uniform(16, 512), gen_len=_uniform(8, 256),
        arrival="poisson", rate=0.2,
        description="Heterogeneous lengths (prompt 16-512, output 8-256), Poisson at 0.2 req/s.",
    ),
    "burst": WorkloadSpec(
        name="burst", num_requests=32, prompt_len=_fixed(128), gen_len=_fixed(64),
        arrival="burst", burst_size=8, burst_interval=20.0,
        description="Four bursts of 8 identical requests, 20 s apart.",
    ),
    "offline": WorkloadSpec(
        name="offline", num_requests=32, prompt_len=_fixed(128), gen_len=_fixed(64),
        arrival="instant",
        description="All 32 requests present at t=0: pure throughput.",
    ),
    "closed8": WorkloadSpec(
        name="closed8", num_requests=32, prompt_len=_fixed(128), gen_len=_fixed(64),
        arrival="closed", concurrency=8,
        description="8 closed-loop clients, each with one request in flight.",
    ),
    # Higher-load variants, added at stage 2: the original rates were sized for
    # the serial engine and leave a batching engine idle most of the time.
    # At these rates the stage 0 engine is overloaded (utilization > 1).
    "uniform_hi": WorkloadSpec(
        name="uniform_hi", num_requests=32, prompt_len=_fixed(128), gen_len=_fixed(64),
        arrival="poisson", rate=3.0,
        description="Homogeneous requests (128 in, 64 out), Poisson arrivals at 3 req/s.",
    ),
    "hetero_hi": WorkloadSpec(
        name="hetero_hi", num_requests=32, prompt_len=_uniform(16, 512), gen_len=_uniform(8, 256),
        arrival="poisson", rate=1.0,
        description="Heterogeneous lengths (prompt 16-512, output 8-256), Poisson at 1 req/s.",
    ),
    # Added at stage 4: loads that saturate 8 concurrent sequences, so an
    # allocator that fits more sequences into the same KV budget can show it.
    "uniform_8x": WorkloadSpec(
        name="uniform_8x", num_requests=64, prompt_len=_fixed(128), gen_len=_fixed(64),
        arrival="poisson", rate=8.0,
        description="Homogeneous requests (128 in, 64 out), Poisson arrivals at 8 req/s, 64 requests.",
    ),
    "hetero_4x": WorkloadSpec(
        name="hetero_4x", num_requests=64, prompt_len=_uniform(16, 512), gen_len=_uniform(8, 256),
        arrival="poisson", rate=4.0,
        description="Heterogeneous lengths (prompt 16-512, output 8-256), Poisson at 4 req/s, 64 requests.",
    ),
    # Added at stage 6: every prompt starts with the same 256 tokens (a system
    # prompt), followed by its own 32-256 tokens. Two rates: one the paged
    # engine carries with room, one that saturates it.
    "prefix_2x": WorkloadSpec(
        name="prefix_2x", num_requests=64, prompt_len=_uniform(288, 512), gen_len=_uniform(16, 128),
        arrival="poisson", rate=2.0, shared_prefix_len=256,
        description="256-token shared prefix + 32-256 own tokens, output 16-128, Poisson at 2 req/s.",
    ),
    "prefix_4x": WorkloadSpec(
        name="prefix_4x", num_requests=64, prompt_len=_uniform(288, 512), gen_len=_uniform(16, 128),
        arrival="poisson", rate=4.0, shared_prefix_len=256,
        description="256-token shared prefix + 32-256 own tokens, output 16-128, Poisson at 4 req/s.",
    ),
}
