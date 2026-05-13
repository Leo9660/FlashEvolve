from abc import ABC, abstractmethod
from typing import Any, Callable

from .llm.base import ChatRequest, LLMClient


class Feedback(ABC):
    """Per-sample textual feedback over a minibatch.

    Plugged into ``Rollout`` to enrich a ``Trajectory`` with per-sample
    diagnostic strings consumed by downstream Reflect / Propose. Two
    common modes ship in the framework:

    - ``RuleBasedFeedback``: sync function, no LLM (GEPA paper style).
    - ``BatchLLMFeedback``: single batched LLM call across the minibatch.

    The shape is fixed: ``generate`` returns one string per sample,
    aligned with ``samples``.
    """

    @abstractmethod
    async def generate(
        self,
        samples: list[Any],
        outputs: list[Any],
        scores: list[float],
    ) -> list[str]: ...


class RuleBasedFeedback(Feedback):
    """Per-sample sync function. No LLM, no I/O — fast path for tasks with
    deterministic feedback rules (GEPA's IFBench / HoVer / HotpotQA /
    MATH all use this).
    """

    def __init__(self, fn: Callable[[Any, Any, float], str]) -> None:
        self.fn = fn

    async def generate(
        self,
        samples: list[Any],
        outputs: list[Any],
        scores: list[float],
    ) -> list[str]:
        return [self.fn(s, o, sc) for s, o, sc in zip(samples, outputs, scores)]


class BatchLLMFeedback(Feedback):
    """One LLM call per minibatch; the LLM sees all (sample, output, score)
    triples and emits per-sample feedback in a single response. ``parse``
    extracts the per-sample list from the raw text.

    Used when no rule-based feedback exists (open-ended tasks, LLM-as-judge
    setups). Token-cheaper than per-sample LLM calls because the model
    amortizes context across the batch.
    """

    def __init__(
        self,
        render: Callable[[list[Any], list[Any], list[float]], ChatRequest],
        parse: Callable[[str, list[Any]], list[str]],
        llm: LLMClient,
    ) -> None:
        self.render = render
        self.parse = parse
        self.llm = llm

    async def generate(
        self,
        samples: list[Any],
        outputs: list[Any],
        scores: list[float],
    ) -> list[str]:
        response = await self.llm.chat(self.render(samples, outputs, scores))
        feedback = self.parse(response.content, samples)
        if len(feedback) != len(samples):
            raise ValueError(
                f"BatchLLMFeedback.parse returned {len(feedback)} items, "
                f"expected {len(samples)} (one per sample)"
            )
        return feedback
