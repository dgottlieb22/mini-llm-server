import pytest

from miniserve.bench import metrics
from miniserve.bench.harness import peak_live_tokens
from miniserve.request import GenerationRequest, GenerationResult, RequestRecord


def _rec(rid, arrival, start, token_times, finish, prompt_len=4):
    req = GenerationRequest(rid, list(range(prompt_len)), max_new_tokens=len(token_times))
    res = GenerationResult(rid, [1] * len(token_times), start, list(token_times), finish, "length")
    return RequestRecord(req, arrival, res)


def test_per_request_definitions():
    t0 = 100.0
    rec = _rec("a", arrival=101.0, start=101.5, token_times=[102.0, 102.1, 102.3], finish=102.35)
    (m,) = metrics.per_request([rec], t0)
    assert m.arrival == pytest.approx(1.0)
    assert m.queue_wait == pytest.approx(0.5)
    assert m.ttft == pytest.approx(1.0)
    assert m.prefill_latency == pytest.approx(0.5)
    assert m.tpot == pytest.approx(0.15)  # (102.3-102.0)/2
    assert m.e2e_latency == pytest.approx(1.35)
    assert m.gen_len == 3


def test_single_token_has_no_tpot():
    rec = _rec("a", 0.0, 0.1, [0.2], 0.2)
    (m,) = metrics.per_request([rec], 0.0)
    assert m.tpot is None
    s = metrics.summarize([m], 1)
    assert s["tpot_s"] == {}


def test_errored_requests_are_excluded():
    bad = RequestRecord(
        GenerationRequest("b", [1], 1), 0.0,
        GenerationResult("b", [], 0.0, [], 0.1, "error", error="boom"),
    )
    assert metrics.per_request([bad], 0.0) == []


def test_summary_throughput_and_busy_fraction():
    # Two back-to-back requests on a serial server: busy fraction 1.
    recs = [
        _rec("a", 0.0, 0.0, [0.5, 1.0], 1.0),
        _rec("b", 0.0, 1.0, [1.5, 2.0], 2.0),
    ]
    ms = metrics.per_request(recs, 0.0)
    s = metrics.summarize(ms, num_submitted=2)
    assert s["makespan_s"] == pytest.approx(2.0)
    assert s["requests_per_s"] == pytest.approx(1.0)
    assert s["output_tok_per_s"] == pytest.approx(2.0)
    assert s["total_tok_per_s"] == pytest.approx((8 + 4) / 2.0)
    assert s["busy_fraction"] == pytest.approx(1.0)
    assert s["occupancy"] == pytest.approx(1.0)
    assert s["mean_in_service"] == pytest.approx(1.0)
    assert s["queue_wait_s"]["p50"] == pytest.approx(0.5)
    assert s["queue_wait_s"]["max"] == pytest.approx(1.0)


def test_occupancy_with_overlap_and_gaps():
    # a and b fully overlap on [0,1]; c runs alone on [3,4]; idle on (1,3).
    recs = [
        _rec("a", 0.0, 0.0, [0.5, 1.0], 1.0),
        _rec("b", 0.0, 0.0, [0.5, 1.0], 1.0),
        _rec("c", 0.0, 3.0, [3.5, 4.0], 4.0),
    ]
    ms = metrics.per_request(recs, 0.0)
    s = metrics.summarize(ms, num_submitted=3)
    assert s["busy_fraction"] == pytest.approx(3.0 / 4.0)
    assert s["occupancy"] == pytest.approx(2.0 / 4.0)
    assert s["mean_in_service"] == pytest.approx(1.5)
    assert metrics.occupancy([], 1.0) == 0.0


def test_percentiles_shape():
    p = metrics.percentiles([1.0, 2.0, 3.0, 4.0])
    assert set(p) == {"p50", "p95", "p99", "mean", "max", "n"}
    assert p["p50"] == pytest.approx(2.5)
    assert p["max"] == 4.0 and p["n"] == 4
    assert metrics.percentiles([]) == {}


def test_peak_live_tokens_serial_vs_overlapping():
    serial = [
        _rec("a", 0.0, 0.0, [0.5, 0.9], 1.0, prompt_len=10),
        _rec("b", 0.0, 1.0, [1.5, 1.9], 2.0, prompt_len=10),
    ]
    assert peak_live_tokens(serial) == 12  # 10 prompt + 2 generated, never both resident
    overlapping = [
        _rec("a", 0.0, 0.0, [0.5, 1.0], 1.5, prompt_len=10),
        _rec("b", 0.0, 0.2, [0.7, 1.2], 1.6, prompt_len=10),
    ]
    assert peak_live_tokens(overlapping) == 24


def test_peak_live_tokens_token_at_finish_instant_is_not_resident():
    # Release sorts before acquire at a tie, so a token stamped at exactly the
    # finish time occupied no time and is not counted.
    rec = _rec("a", 0.0, 0.0, [0.5, 1.0], 1.0, prompt_len=10)
    assert peak_live_tokens([rec]) == 11
