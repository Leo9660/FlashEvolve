"""GEPA on FlashEvolve.

Reference: Khatri et al., "GEPA: Reflective Prompt Evolution Can Outperform
Reinforcement Learning" (2025).

Layout:
    ``GEPAReflect``: no-LLM Stage that formats per-sample (input, output,
    feedback) tuples into a single ``<inputs_outputs_feedback>`` block as
    ``Critique.text``. GEPA's "reflection" is rule-based feedback assembly,
    not an LLM call.

    ``gepa_propose_render`` / ``gepa_propose_parse``: format and parse for
    the single GEPA LLM call (``PROPOSE_PROMPT`` in ``prompts.py``).

    ``make_gepa_propose`` / ``make_gepa_rollout`` / etc.: thin builders
    that wire the right defaults so a 5-line caller can construct a
    GEPA pipeline.

Task-specific pieces — Agent, metric, feedback_fn, sampler, valset — are
the user's responsibility. See ``run.py`` for a toy end-to-end demo.
"""

import re
from typing import Any, Callable

from flashevolve.artifacts import PromptArtifact
from flashevolve.feedback import RuleBasedFeedback
from flashevolve.llm import ChatRequest, LLMClient
from flashevolve.samplers import Sampler
from flashevolve.stages import (
    Critique,
    Evaluate,
    Metric,
    Propose,
    Rollout,
    Stage,
    Trajectory,
)
from flashevolve.agents.base import Agent

from .prompts import PROPOSE_PROMPT


# ─── Reflect: no-LLM, just format feedback into a block ──────────────────


class GEPAReflect(Stage[Trajectory, Critique]):
    """No-LLM Reflect that formats per-sample (input, output, feedback)
    tuples into the ``<inputs_outputs_feedback>`` block GEPA's proposal
    LLM expects.

    GEPA's "reflection" is rule-based: it's just feedback string assembly.
    The real LLM work happens at Propose. ``Stage = LLM-heavy`` is a
    default expectation, not a hard rule; this is a sanctioned exception
    per DESIGN.md §0 (reflective async stale-patch worker also reuses
    Reflect as a Stage).
    """

    DEFAULT_TEMPLATE = (
        "### Example {idx}\n"
        "Input: {sample}\n"
        "Response: {output}\n"
        "Feedback: {feedback}\n"
    )

    def __init__(
        self,
        *,
        per_sample_template: str = DEFAULT_TEMPLATE,
        sample_renderer: Callable[[Any], str] = repr,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.per_sample_template = per_sample_template
        self.sample_renderer = sample_renderer

    async def process(self, item: Trajectory) -> Critique:
        feedback = item.signals.get("feedback") or [""] * len(item.samples)
        blocks = [
            self.per_sample_template.format(
                idx=i + 1,
                sample=self.sample_renderer(s),
                output=o,
                feedback=f,
            )
            for i, (s, o, f) in enumerate(zip(item.samples, item.outputs, feedback))
        ]
        return Critique(trajectory=item, text="\n".join(blocks))


# ─── Propose render / parse for GEPA's single LLM call ────────────────────


def gepa_propose_render(
    critique: Critique,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    model: str | None = None,
    extra: dict | None = None,
) -> ChatRequest:
    """Substitute ``<curr_instructions>`` (parent prompt) and
    ``<inputs_outputs_feedback>`` (Reflect's formatted block) into
    GEPA's ``PROPOSE_PROMPT``.

    ``extra`` is merged into the ChatRequest's ``extra`` dict, after the
    built-in ``max_tokens`` / ``temperature`` (so the caller can override
    or add provider-specific keys like ``chat_template_kwargs``).
    """
    parent = critique.trajectory.artifact
    if not isinstance(parent, PromptArtifact):
        raise TypeError(
            f"GEPA's Propose expects PromptArtifact parent, got {type(parent).__name__}"
        )
    user_msg = PROPOSE_PROMPT.replace("<curr_instructions>", parent.text).replace(
        "<inputs_outputs_feedback>", critique.text
    )
    merged_extra: dict = {"max_tokens": max_tokens, "temperature": temperature}
    if extra:
        merged_extra.update(extra)
    return ChatRequest(
        messages=[{"role": "user", "content": user_msg}],
        model=model,
        extra=merged_extra,
    )


_CODEBLOCK_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)


def gepa_propose_parse(text: str, critique: Critique) -> PromptArtifact:
    """Extract the new instruction between the LAST pair of triple
    backticks (GEPA's convention: the model first restates its reasoning,
    then puts the final instruction in ``` blocks at the end).

    Falls back to the entire response if no code block is found — better
    than crashing during an evolution run.
    """
    matches = _CODEBLOCK_RE.findall(text)
    body = matches[-1].strip() if matches else text.strip()
    return PromptArtifact(text=body)


# ─── Thin builders ────────────────────────────────────────────────────────


def make_gepa_rollout(
    *,
    agent: Agent,
    metric: Metric,
    feedback_fn: Callable[[Any, Any, float], str],
    llm: LLMClient,
    workers: int = 1,
) -> Rollout:
    """GEPA's Rollout: agent + rule-based feedback (sync function), no
    Feedback-LLM call. Per-sample feedback is the textual diagnostic
    that gets fed back into Propose."""
    return Rollout(
        agent=agent,
        metric=metric,
        llm=llm,
        feedback=RuleBasedFeedback(feedback_fn),
        workers=workers,
    )


def make_gepa_propose(
    *,
    llm: LLMClient,
    workers: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    model: str | None = None,
    extra: dict | None = None,
) -> Propose:
    """GEPA's single proposal LLM call (PROPOSE_PROMPT).

    ``extra`` is forwarded to ``ChatRequest.extra`` (use for
    provider-specific keys like Qwen's ``chat_template_kwargs``).
    """
    def _render(c: Critique) -> ChatRequest:
        return gepa_propose_render(
            c,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            extra=extra,
        )

    return Propose(render=_render, parse=gepa_propose_parse, llm=llm, workers=workers)


def make_gepa_evaluate(
    *,
    agent: Agent,
    metric: Metric,
    llm: LLMClient,
    sampler: Sampler,
    workers: int = 1,
) -> Evaluate:
    """GEPA's Evaluate runs the candidate on a held-out set (typically
    the full valset for paper-faithful Pareto-front tracking).
    """
    return Evaluate(
        agent=agent, metric=metric, llm=llm, sampler=sampler, workers=workers
    )


__all__ = [
    "GEPAReflect",
    "gepa_propose_render",
    "gepa_propose_parse",
    "make_gepa_rollout",
    "make_gepa_propose",
    "make_gepa_evaluate",
    "PROPOSE_PROMPT",
]
