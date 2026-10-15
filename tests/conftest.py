import pytest
import torch


@pytest.fixture(scope="session")
def tiny():
    """A 2-layer random GPT-2 (vocab 97, 64 positions): real cores, no weights download."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=97, n_positions=64, n_embd=32, n_layer=2, n_head=4,
                     bos_token_id=96, eos_token_id=96)
    return GPT2LMHeadModel(cfg).eval()


@pytest.fixture(scope="session")
def tiny_runner(tiny):
    """An HFModelRunner wrapping the tiny model, without from_pretrained."""
    from miniserve.model import HFModelRunner

    runner = HFModelRunner.__new__(HFModelRunner)
    runner.model, runner.config, runner.name, runner.device = tiny, tiny.config, "tiny", "cpu"
    runner.eos_token_id, runner.vocab_size, runner.dtype = tiny.config.eos_token_id, 97, torch.float32
    runner.torch_threads = torch.get_num_threads()
    return runner


def hf_logits(model, ids):
    with torch.inference_mode():
        return model(torch.tensor([ids])).logits[0]
