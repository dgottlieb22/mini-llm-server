import asyncio

import pytest
import torch

from miniserve.bench.harness import run_workload
from miniserve.bench.workload import LengthSpec, WorkloadSpec
from miniserve.engines.continuous import ContinuousBatchEngine
from miniserve.engines.paged import PagedEngine
from miniserve.gpt2 import GPT2Core
from miniserve.model import FakeModelRunner
from miniserve.paged import PagedGPT2Core
from miniserve.request import GenerationRequest


@pytest.fixture(scope="module")
def tiny():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=97, n_positions=64, n_embd=32, n_layer=2, n_head=4,
                     bos_token_id=96, eos_token_id=96)
    return GPT2LMHeadModel(cfg).eval()


def hf_logits(model, ids):
    with torch.inference_mode():
        return model(torch.tensor([ids])).logits[0]


PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [40, 41, 42], [60, 61, 62, 63, 64, 65]]


def test_slot_core_batched_prefill_matches_hf_and_leaves_usable_caches(tiny):
    core = GPT2Core(tiny, max_slots=3, max_seq_len=64)
    logits = core.prefill_batch([0, 1, 2], PROMPTS)
    for i, p in enumerate(PROMPTS):
        assert torch.allclose(logits[i], hf_logits(tiny, p)[-1], atol=1e-4)
    # The padded rows must have written only their real positions: decode on each.
    out = core.decode([0, 1, 2], [11, 12, 13], [len(p) for p in PROMPTS])
    for i, p in enumerate(PROMPTS):
        assert torch.allclose(out[i], hf_logits(tiny, p + [11 + i])[-1], atol=1e-4)


def test_paged_core_batched_prefill_matches_hf(tiny):
    core = PagedGPT2Core(tiny, budget_tokens=64, block_size=4, max_seq_len=64)
    handles = [core.admit(len(p), 3) for p in PROMPTS]
    logits = core.prefill_batch(handles, PROMPTS)
    for i, p in enumerate(PROMPTS):
        assert torch.allclose(logits[i], hf_logits(tiny, p)[-1], atol=1e-4)
    out = core.decode(handles, [11, 12, 13], [len(p) for p in PROMPTS])
    for i, p in enumerate(PROMPTS):
        assert torch.allclose(out[i], hf_logits(tiny, p + [11 + i])[-1], atol=1e-4)


def _spec(**over):
    base = dict(
        name="t", num_requests=8, prompt_len=LengthSpec("uniform", low=3, high=9),
        gen_len=LengthSpec("uniform", low=2, high=6), arrival="instant", seed=3,
    )
    base.update(over)
    return WorkloadSpec(**base)


def test_co_arriving_requests_are_prefilled_together_within_the_cap():
    runner = FakeModelRunner()
    engine = ContinuousBatchEngine(runner, max_slots=8, prefill_batch_tokens=4 * 9)  # 4 rows of up to 9
    spec = _spec(num_requests=8)
    asyncio.run(run_workload(engine, spec, warmup=False))
    assert [len(b) for b in engine.core.prefill_batches] == [4, 4]
    s = engine.stats()
    assert s["prefill_batches"] == 2 and s["prefill_count"] == 8
    assert s["prefill_padded_tokens"] >= s["prefill_tokens"]


def test_zero_cap_means_one_at_a_time():
    engine = ContinuousBatchEngine(FakeModelRunner(), max_slots=8, prefill_batch_tokens=0)
    asyncio.run(run_workload(engine, _spec(num_requests=5), warmup=False))
    assert engine.core.prefill_batches == []
    assert engine.stats()["prefill_batches"] == 5


def test_batched_prefill_outputs_match_one_at_a_time_for_both_engines():
    spec = _spec(num_requests=10)
    _, a = asyncio.run(run_workload(ContinuousBatchEngine(FakeModelRunner(), max_slots=4), spec, warmup=False))
    _, b = asyncio.run(run_workload(ContinuousBatchEngine(FakeModelRunner(), max_slots=4,
                                                          prefill_batch_tokens=64), spec, warmup=False))
    _, c = asyncio.run(run_workload(PagedEngine(FakeModelRunner(), kv_budget_tokens=128, block_size=4,
                                                prefill_batch_tokens=64), spec, warmup=False))
    assert [r.result.output_ids for r in a] == [r.result.output_ids for r in b] == [r.result.output_ids for r in c]


def test_group_respects_storage_and_live_bounds():
    # 2 slots: a group can never hold more than 2 even with a generous prefill cap.
    runner = FakeModelRunner()
    engine = ContinuousBatchEngine(runner, max_slots=2, prefill_batch_tokens=1000)
    asyncio.run(run_workload(engine, _spec(num_requests=6), warmup=False))
    assert max(len(b) for b in engine.core.prefill_batches) <= 2
    with pytest.raises(ValueError):
        ContinuousBatchEngine(FakeModelRunner(), prefill_batch_tokens=-1)
