import pytest
import torch

from conftest import hf_logits
from miniserve.gpt2 import GPT2Core, argmax_tokens




def test_prefill_matches_hf_last_position(tiny):
    core = GPT2Core(tiny, max_slots=2, max_seq_len=64)
    ids = [5, 17, 3, 44, 9, 60, 2]
    ours = core.prefill(slot=1, token_ids=ids)[0]
    assert torch.allclose(ours, hf_logits(tiny, ids)[-1], atol=1e-4, rtol=1e-4)


def test_decode_matches_hf_for_two_sequences_of_different_length(tiny):
    core = GPT2Core(tiny, max_slots=3, max_seq_len=64)
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [40, 41, 42]
    core.prefill(0, a)
    core.prefill(2, b)
    # Feed two more tokens to each and compare every step with a full HF forward.
    nxt = {0: 11, 2: 12}
    seqs = {0: list(a), 2: list(b)}
    for step in range(3):
        slots = [0, 2]
        tokens = [nxt[0], nxt[2]]
        positions = [len(seqs[0]), len(seqs[2])]
        logits = core.decode(slots, tokens, positions)
        for i, s in enumerate(slots):
            seqs[s].append(tokens[i])
            ref = hf_logits(tiny, seqs[s])[-1]
            assert torch.allclose(logits[i], ref, atol=1e-4, rtol=1e-4), f"step {step} slot {s}"
        nxt = {0: 20 + step, 2: 50 + step}


def test_slot_is_reusable_and_slots_are_independent(tiny):
    core = GPT2Core(tiny, max_slots=2, max_seq_len=64)
    long_ids = list(range(1, 30))
    core.prefill(0, long_ids)             # fills slot 0 deep
    short = [7, 8, 9]
    out1 = core.prefill(0, short)         # reuse slot 0 with a shorter prompt
    assert torch.allclose(out1[0], hf_logits(tiny, short)[-1], atol=1e-4)
    # A decode step on slot 0 only must not be influenced by stale positions.
    logits = core.decode([0], [10], [3])
    assert torch.allclose(logits[0], hf_logits(tiny, short + [10])[-1], atol=1e-4)


def test_greedy_matches_hf_generate(tiny):
    core = GPT2Core(tiny, max_slots=1, max_seq_len=64)
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    with torch.inference_mode():
        ref = tiny.generate(torch.tensor([ids]), max_new_tokens=6, min_new_tokens=6, do_sample=False,
                            pad_token_id=tiny.config.eos_token_id)[0, len(ids):].tolist()
    out = []
    tok = argmax_tokens(core.prefill(0, ids), suppress=[tiny.config.eos_token_id])[0]
    out.append(tok)
    pos = len(ids)
    while len(out) < 6:
        tok = argmax_tokens(core.decode([0], [tok], [pos]), suppress=[tiny.config.eos_token_id])[0]
        out.append(tok)
        pos += 1
    assert out == ref


def test_bounds(tiny):
    core = GPT2Core(tiny, max_slots=1, max_seq_len=8)
    with pytest.raises(ValueError):
        core.prefill(0, list(range(9)))
    core.prefill(0, [1, 2, 3])
    with pytest.raises(ValueError):
        core.decode([0], [4], [8])
    with pytest.raises(ValueError):
        GPT2Core(tiny, max_slots=1, max_seq_len=65)  # beyond n_positions


def test_sizes(tiny):
    core = GPT2Core(tiny, max_slots=4, max_seq_len=16)
    assert core.kv_bytes_per_token == 2 * 2 * 4 * 8 * 4  # layers * (k,v) * heads * head_dim * fp32
    assert core.cache_bytes == core.kv_bytes_per_token * 4 * 16
