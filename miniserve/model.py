"""Model execution.

Stage 0 hands the whole request to Hugging Face's model.generate(). We only
add a streamer that records a timestamp per emitted token, which is what lets
the benchmark report TTFT and TPOT without changing the architecture.

Later stages replace generate() with our own prefill/decode loop; the
HFModelRunner then becomes the thing that runs one forward pass.
"""
from __future__ import annotations

import time
from typing import List, Optional, Protocol

from .request import GenerationRequest, GenerationResult


class ModelRunner(Protocol):
    name: str

    @property
    def kv_bytes_per_token(self) -> int: ...

    def generate_one(self, request: GenerationRequest) -> GenerationResult: ...


class _TimestampStreamer:
    """Minimal BaseStreamer: generate() calls put() once with the prompt, then
    once per generated token, then end()."""

    def __init__(self) -> None:
        self.times: List[float] = []

    def put(self, value) -> None:  # noqa: ANN001 (torch tensor)
        self.times.append(time.perf_counter())

    def end(self) -> None:
        pass


class HFModelRunner:
    """A causal LM loaded through transformers, run one request at a time."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        dtype: str = "float32",
        torch_threads: Optional[int] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        self.name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype)
        self.model.to(device).eval()
        self.config = self.model.config
        self.eos_token_id = self.config.eos_token_id
        self.vocab_size = self.config.vocab_size
        self.torch_threads = torch.get_num_threads()

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of key+value cache one token occupies across all layers."""
        c = self.config
        n_layers = c.num_hidden_layers
        n_heads = c.num_attention_heads
        n_kv_heads = getattr(c, "num_key_value_heads", None) or n_heads
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // n_heads
        import torch

        elem = torch.tensor([], dtype=self.dtype).element_size()
        return n_layers * 2 * n_kv_heads * head_dim * elem

    def generate_one(self, request: GenerationRequest) -> GenerationResult:
        import torch

        input_ids = torch.tensor([request.prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        streamer = _TimestampStreamer()
        kwargs = dict(
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=self.eos_token_id,
            attention_mask=attention_mask,
        )
        if request.force_length:
            kwargs["min_new_tokens"] = request.max_new_tokens
        service_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(input_ids, **kwargs)
        finish = time.perf_counter()
        output_ids = out[0, input_ids.shape[1]:].tolist()
        token_times = streamer.times[1:]  # drop the prompt put()
        if len(token_times) != len(output_ids):
            raise RuntimeError(
                f"{request.request_id}: {len(token_times)} token timestamps for "
                f"{len(output_ids)} tokens"
            )
        reason = "length" if len(output_ids) == request.max_new_tokens else "stop"
        return GenerationResult(
            request_id=request.request_id,
            output_ids=output_ids,
            service_start=service_start,
            token_times=token_times,
            finish_time=finish,
            finish_reason=reason,
        )


class FakeModelRunner:
    """Deterministic stand-in for tests: no weights, optional simulated cost.

    prefill_cost_per_token and decode_cost_per_token are seconds of
    time.sleep(), so tests can create requests with very different service
    times and check what the engine does with them.
    """

    name = "fake"

    def __init__(
        self,
        vocab_size: int = 1000,
        prefill_cost_per_token: float = 0.0,
        decode_cost_per_token: float = 0.0,
        kv_bytes_per_token: int = 1024,
    ) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1
        self.prefill_cost_per_token = prefill_cost_per_token
        self.decode_cost_per_token = decode_cost_per_token
        self._kv_bytes_per_token = kv_bytes_per_token
        self.calls: List[str] = []

    @property
    def kv_bytes_per_token(self) -> int:
        return self._kv_bytes_per_token

    def next_token(self, prompt_ids: List[int], generated: List[int]) -> int:
        # Avoid the EOS id so forced-length runs and natural runs agree.
        return (sum(prompt_ids) + len(generated) * 7919) % (self.vocab_size - 1)

    def generate_one(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.request_id)
        service_start = time.perf_counter()
        if self.prefill_cost_per_token:
            time.sleep(self.prefill_cost_per_token * len(request.prompt_ids))
        output: List[int] = []
        times: List[float] = []
        for _ in range(request.max_new_tokens):
            if self.decode_cost_per_token and output:
                time.sleep(self.decode_cost_per_token)
            output.append(self.next_token(request.prompt_ids, output))
            times.append(time.perf_counter())
        return GenerationResult(
            request_id=request.request_id,
            output_ids=output,
            service_start=service_start,
            token_times=times,
            finish_time=time.perf_counter(),
            finish_reason="length",
        )
