import random

import pytest

from miniserve.bench.workload import (
    WORKLOADS,
    LengthSpec,
    WorkloadSpec,
    arrival_offsets,
    generate_requests,
)


def _spec(**over):
    base = dict(
        name="t", num_requests=20, prompt_len=LengthSpec("uniform", low=4, high=40),
        gen_len=LengthSpec("uniform", low=2, high=9), arrival="poisson", rate=5.0, seed=3,
    )
    base.update(over)
    return WorkloadSpec(**base)


def test_same_seed_same_requests_and_arrivals():
    a, b = _spec(), _spec()
    assert [r.prompt_ids for r in generate_requests(a)] == [r.prompt_ids for r in generate_requests(b)]
    assert arrival_offsets(a) == arrival_offsets(b)


def test_different_seed_differs():
    assert generate_requests(_spec(seed=1))[0].prompt_ids != generate_requests(_spec(seed=2))[0].prompt_ids


def test_lengths_within_bounds_and_ids_in_vocab():
    spec = _spec(vocab_size=100)
    for r in generate_requests(spec):
        assert 4 <= r.prompt_len <= 40
        assert 2 <= r.max_new_tokens <= 9
        assert all(0 <= t < 100 for t in r.prompt_ids)


def test_shared_prefix_is_shared_and_prompt_never_shorter_than_prefix():
    spec = _spec(shared_prefix_len=10)
    reqs = generate_requests(spec)
    first = reqs[0].prompt_ids[:10]
    assert all(r.prompt_ids[:10] == first for r in reqs)
    assert all(r.prompt_len >= 10 for r in reqs)


def test_changing_arrival_mode_keeps_request_lengths():
    p = generate_requests(_spec(arrival="poisson"))
    b = generate_requests(_spec(arrival="burst", burst_size=4, burst_interval=1.0))
    assert [r.prompt_ids for r in p] == [r.prompt_ids for r in b]


def test_poisson_offsets_monotonic_start_at_zero_and_have_right_mean():
    spec = _spec(num_requests=2000, rate=10.0)
    offs = arrival_offsets(spec)
    assert offs[0] == 0.0
    assert all(b >= a for a, b in zip(offs, offs[1:]))
    mean_gap = offs[-1] / (len(offs) - 1)
    assert abs(mean_gap - 0.1) < 0.01


def test_burst_offsets():
    spec = _spec(num_requests=10, arrival="burst", burst_size=4, burst_interval=2.0)
    assert arrival_offsets(spec) == [0, 0, 0, 0, 2, 2, 2, 2, 4, 4]


def test_instant_and_closed():
    assert arrival_offsets(_spec(arrival="instant")) == [0.0] * 20
    assert arrival_offsets(_spec(arrival="closed", concurrency=3)) is None


def test_lognormal_clipped():
    ls = LengthSpec("lognormal", low=5, high=50, mu=3.0, sigma=1.5)
    rng = random.Random(0)
    xs = [ls.sample(rng) for _ in range(500)]
    assert min(xs) >= 5 and max(xs) <= 50
    assert min(xs) == 5 and max(xs) == 50  # clipping actually happened at both ends


@pytest.mark.parametrize(
    "bad",
    [
        dict(arrival="weird"),
        dict(num_requests=0),
        dict(rate=0.0),
        dict(arrival="closed", concurrency=0),
        dict(arrival="burst", burst_size=0),
        dict(shared_prefix_len=41),
    ],
)
def test_invalid_specs_rejected(bad):
    with pytest.raises(ValueError):
        _spec(**bad)


def test_presets_generate():
    for name, spec in WORKLOADS.items():
        reqs = generate_requests(spec)
        assert len(reqs) == spec.num_requests, name
        if spec.open_loop:
            assert len(arrival_offsets(spec)) == spec.num_requests
