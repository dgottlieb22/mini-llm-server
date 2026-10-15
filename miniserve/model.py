"""Model execution.

Stage 0 hands the whole request to Hugging Face's model.generate(). We only
add a streamer that records a timestamp per emitted token, which is what lets
the benchmark report TTFT and TPOT without changing the architecture.

Later stages replace generate() with our own prefill/decode loop; the
HFModelRunner then becomes the thing that runs one forward pass.
"""
from __future__ import annotations

import time
from typing import List, Optional, Protocol, Sequence

from .request import GenerationRequest, GenerationResult


class ModelRunner(Protocol):
    name: str

    @property
    def kv_bytes_per_token(self) -> int: ...

    def generate_one(self, request: GenerationRequest) -> GenerationResult: ...

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> List[GenerationResult]: ...


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

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> List[GenerationResult]:
        """Static batching: one generate() over a left-padded batch.

        Prompts are left-padded to the longest prompt; every sequence then
        decodes in lockstep. A sequence that reaches its own length stops
        producing real tokens but stays in the batch (HF writes pad tokens
        for it) until the longest member finishes, and its result is only
        returned when the whole batch returns. finish_time is therefore the
        batch end for every member; token_times are the per-step times.
        """
        import torch
        from transformers import LogitsProcessorList, StoppingCriteriaList, SuppressTokensLogitsProcessor

        batch = len(requests)
        max_prompt = max(r.prompt_len for r in requests)
        gen_lens = [r.max_new_tokens for r in requests]
        pad = self.eos_token_id
        input_ids = torch.full((batch, max_prompt), pad, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch, max_prompt), dtype=torch.long, device=self.device)
        for i, r in enumerate(requests):
            input_ids[i, max_prompt - r.prompt_len:] = torch.tensor(r.prompt_ids, dtype=torch.long)
            attention_mask[i, max_prompt - r.prompt_len:] = 1
        streamer = _TimestampStreamer()
        kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max(gen_lens),
            do_sample=False,
            pad_token_id=pad,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_PerSequenceLength(gen_lens, max_prompt)]),
        )
        if all(r.force_length for r in requests):
            kwargs["logits_processor"] = LogitsProcessorList([SuppressTokensLogitsProcessor([pad])])
        service_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(input_ids, **kwargs)
        finish = time.perf_counter()
        step_times = streamer.times[1:]
        results = []
        for i, r in enumerate(requests):
            n = gen_lens[i]
            ids = out[i, max_prompt:max_prompt + n].tolist()
            if len(ids) != n or len(step_times) < n:
                raise RuntimeError(f"{r.request_id}: got {len(ids)} tokens / {len(step_times)} steps for {n}")
            results.append(
                GenerationResult(
                    request_id=r.request_id,
                    output_ids=ids,
                    service_start=service_start,
                    token_times=step_times[:n],
                    finish_time=finish,
                    finish_reason="length",
                )
            )
        return results


class _PerSequenceLength:
    """StoppingCriteria: sequence i is done once it has produced gen_lens[i] tokens."""

    def __init__(self, gen_lens: Sequence[int], prompt_len: int) -> None:
        import torch

        self.targets = torch.tensor([prompt_len + g for g in gen_lens])

    def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
        return input_ids.shape[1] >= self.targets.to(input_ids.device)


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
        self.batches: List[List[str]] = []

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

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> List[GenerationResult]:
        """Lockstep decode like the HF batch: the batch runs to its longest
        member; one step costs decode_cost_per_token regardless of batch size
        (the idealized 'batching is free' model). Padded prompt positions
        cost prefill time like real ones."""
        self.calls.append("+".join(r.request_id for r in requests))
        self.batches.append([r.request_id for r in requests])
        service_start = time.perf_counter()
        max_prompt = max(r.prompt_len for r in requests)
        if self.prefill_cost_per_token:
            time.sleep(self.prefill_cost_per_token * max_prompt * len(requests))
        outputs: List[List[int]] = [[] for _ in requests]
        step_times: List[float] = []
        for step in range(max(r.max_new_tokens for r in requests)):
            if self.decode_cost_per_token and step:
                time.sleep(self.decode_cost_per_token)
            for i, r in enumerate(requests):
                if step < r.max_new_tokens:
                    outputs[i].append(self.next_token(r.prompt_ids, outputs[i]))
            step_times.append(time.perf_counter())
        finish = time.perf_counter()
        return [
            GenerationResult(
                request_id=r.request_id,
                output_ids=outputs[i],
                service_start=service_start,
                token_times=step_times[: r.max_new_tokens],
                finish_time=finish,
                finish_reason="length",
            )
            for i, r in enumerate(requests)
        ]
