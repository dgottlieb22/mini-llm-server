import pytest

from miniserve.request import GenerationRequest


def test_valid_request():
    r = GenerationRequest("r1", [1, 2, 3], max_new_tokens=4)
    assert r.prompt_len == 3
    assert r.force_length is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(request_id="", prompt_ids=[1], max_new_tokens=1),
        dict(request_id="r", prompt_ids=[], max_new_tokens=1),
        dict(request_id="r", prompt_ids=[1, -5], max_new_tokens=1),
        dict(request_id="r", prompt_ids=[1, 2.5], max_new_tokens=1),
        dict(request_id="r", prompt_ids=[1], max_new_tokens=0),
    ],
)
def test_malformed_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GenerationRequest(**kwargs)
