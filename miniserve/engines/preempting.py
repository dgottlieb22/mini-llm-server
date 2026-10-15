"""Stage 7: optimistic allocation with preemption.

Stages 4-6 admitted a request only if the pool had blocks for its prompt
plus its whole requested output. That cannot run out of memory mid-stream,
but it reserves memory the sequence may never use (2387 of 8192 tokens
empty at the peak on hetero_4x, notes/stage4.md) and it needs the output
length, which a real server does not know.

Here a request is admitted with blocks for its prompt plus one position;
every step, each sequence whose next token needs a new block gets one. When
the pool cannot supply them all, the youngest running sequences are
preempted: their blocks are freed and they go back to the head of the queue
to be readmitted later with their prompt plus everything generated so far
re-prefilled (the prefix cache serves the prompt's shared part), after which
generation resumes where it stopped. The client sees one result; the
preemption shows in its token timestamps as a gap, and in the engine stats.

This is vLLM's "recompute" preemption. The alternative, swapping the
sequence's blocks to a slower tier and back, needs a slower tier; a CPU
host has none. Youngest-first is the policy that preserves FIFO fairness:
the sequence that has waited longest keeps running, and the one preempted
is the one with the least work to redo.

Because the queue is FIFO and admission is on the prompt alone, a long
prompt at the head no longer blocks shorter ones behind it for its whole
output's worth of blocks; it blocks only until its prompt fits.
"""
from __future__ import annotations

from typing import Any, Dict

from ..request import GenerationRequest
from .continuous import _Seq
from .prefix_cached import PrefixCachedEngine


class PreemptingEngine(PrefixCachedEngine):
    name = "preempting"
    stage = 7

    def __init__(self, *args, growth_reserve: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if growth_reserve < 0:
            raise ValueError("growth_reserve must be >= 0")
        # Admission leaves this many free blocks per running sequence, so the
        # sequences already running can grow for a while before anyone has to
        # be preempted. With 0, admission fills the pool and every block
        # boundary that follows is a preemption (thrash); vLLM's watermark
        # plays the same role.
        self.growth_reserve = growth_reserve
        self._preemptions = 0
        self._recomputed_tokens = 0
        self._resumed: Dict[str, _Seq] = {}  # request id -> the preempted state to stitch into

    # ---- admission: prompt only ---------------------------------------------

    def _can_admit(self, request: GenerationRequest) -> bool:
        # Blocks for the prompt plus one position (minus cache hits), plus the
        # growth reserve for what is already running.
        reserve = self.growth_reserve * len(self._running)
        return self.core.can_admit_prompt_only(request.prompt_ids, reserve)

    def _do_admit(self, request: GenerationRequest) -> Any:
        handle = self.core.admit_prompt_only(request.prompt_ids)
        assert handle is not None, "admit_prompt_only failed right after can_admit_prompt_only"
        self._cached_prompt_tokens += getattr(handle, "cached_tokens", 0)
        return handle

    # ---- per-step growth and preemption -------------------------------------

    async def _step(self) -> None:
        self._grow_or_preempt()
        await super()._step()

    def _grow_or_preempt(self) -> None:
        """Give every sequence that needs a block for this step one, preempting the youngest if short."""
        needing = [s for s in self._running.values() if self.core.needs_block(s.handle, s.cache_len)]
        # Youngest = admitted last. _running preserves insertion order.
        while len(needing) > self.core.free_blocks:
            victim = list(self._running.values())[-1]
            self._preempt(victim)
            needing = [s for s in self._running.values() if self.core.needs_block(s.handle, s.cache_len)]
            if not self._running:
                break
        for s in needing:
            if not self.core.grow(s.handle):
                raise RuntimeError("grow failed after the free-block check")  # the check above guarantees room

    def _preempt(self, seq: _Seq) -> None:
        """Free the sequence's blocks and requeue it at the head to resume later."""
        self._preemptions += 1
        self._recomputed_tokens += seq.cache_len
        del self._running[seq.request.request_id]
        self.core.free(seq.handle)
        self._absorb_history(seq)
        # Resume later with the original prompt plus everything generated so far
        # as the new prompt; the remaining output length is what is left.
        done = len(seq.output)
        resumed = GenerationRequest(
            request_id=seq.request.request_id,
            prompt_ids=list(seq.request.prompt_ids) + list(seq.output),
            max_new_tokens=seq.request.max_new_tokens - done,
            force_length=seq.request.force_length,
        )
        self._resumed[seq.request.request_id] = seq
        self._pending.appendleft((resumed, seq.fut))
        self._admission_waits += 1

    def _absorb_history(self, seq: _Seq) -> None:
        """If seq is itself a resumed segment, fold the earlier segments into it
        so one record carries the client's request and the full output."""
        prior = self._resumed.pop(seq.request.request_id, None)
        if prior is not None:
            seq.request = prior.request
            seq.output = prior.output + seq.output
            seq.token_times = prior.token_times + seq.token_times
            seq.service_start = prior.service_start

    # ---- stitch a resumed sequence's result onto its history -----------------

    def _finish(self, seq: _Seq, finish_time: float) -> None:
        self._absorb_history(seq)
        super()._finish(seq, finish_time)

    def _release(self, seq: _Seq) -> None:
        # A cancelled resumed sequence drops its history too.
        self._resumed.pop(seq.request.request_id, None)
        super()._release(seq)

    def _dropped_while_queued(self, request: GenerationRequest) -> None:
        # A resumed request cancelled before readmission: its history would otherwise leak.
        self._resumed.pop(request.request_id, None)

    # ---- stats -----------------------------------------------------------------

    def _reset_stats(self) -> None:
        super()._reset_stats()
        self._preemptions = 0
        self._recomputed_tokens = 0

    def _extra_stats(self) -> Dict[str, object]:
        s = super()._extra_stats()
        s["preemptions"] = self._preemptions
        s["recomputed_tokens"] = self._recomputed_tokens
        return s
