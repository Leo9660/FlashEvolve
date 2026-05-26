from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from flashevolve.artifacts.base import Artifact
from flashevolve.llm import ChatRequest, LLMClient
from flashevolve.stages import Candidate, Critique, Stage, Trajectory

from .prompts import SYSTEM_PROMPT, USER_PROMPT

_CODEBLOCK_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class OpenEvolvePromptArtifact(Artifact):
    text: str
    program_id: str
    parent_id: str | None = None
    changes_description: str = ""


@dataclass(frozen=True)
class OpenEvolveContext:
    version: int
    parent: OpenEvolvePromptArtifact
    island: int
    parent_score: float
    parent_metrics: dict[str, Any]
    top_programs: list[dict[str, Any]]
    inspirations: list[dict[str, Any]]
    global_best: dict[str, Any] | None
    recent_history: list[dict[str, Any]]
    parent_artifacts: dict[str, Any]


def _format_programs(title: str, programs: list[dict[str, Any]]) -> str:
    if not programs:
        return f"{title}\n(none)"
    lines = [title]
    for idx, program in enumerate(programs, start=1):
        changes = program.get("changes_description") or "(no change summary yet)"
        prompt = program.get("prompt", "").strip()
        failed_examples = program.get("failed_examples", [])
        lines.append(
            f"{idx}. score={program.get('score', 0.0):.3f} | changes={changes}\n"
            f"Prompt:\n```text\n{prompt}\n```"
        )
        if failed_examples:
            first_failure = failed_examples[0]
            lines.append(
                "Recent failure:\n"
                f"- instruction: {first_failure.get('instruction', '')}\n"
                f"- feedback: {first_failure.get('feedback', '')}"
            )
    return "\n\n".join(lines)


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "- combined_score: 0.0000"
    lines: list[str] = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {name}: {value:.4f}")
        else:
            lines.append(f"- {name}: {value}")
    return "\n".join(lines)


def _identify_improvement_areas(
    parent_score: float,
    global_best: dict[str, Any] | None,
    parent_artifacts: dict[str, Any],
) -> str:
    messages: list[str] = []
    if global_best is not None:
        best_score = float(global_best.get("score", 0.0))
        delta = best_score - parent_score
        if delta > 1e-9:
            messages.append(
                f"Close the gap to the current global best ({delta:.3f} score difference)."
            )
    failed_examples = list(parent_artifacts.get("failed_examples", []))
    if failed_examples:
        messages.append(
            "Address the recent failed examples and tighten instructions around missed constraints."
        )
    if not messages:
        messages.append("Preserve the current strengths while improving clarity and consistency.")
    return " ".join(messages)


def _format_artifacts(parent_artifacts: dict[str, Any]) -> str:
    failed_examples = list(parent_artifacts.get("failed_examples", []))
    if not failed_examples:
        return "Artifacts: none available."
    lines = ["Artifacts from recent failed examples:"]
    for idx, failure in enumerate(failed_examples[:3], start=1):
        lines.append(
            f"{idx}. instruction={failure.get('instruction', '')!r} | "
            f"score={failure.get('score', 0.0):.3f} | "
            f"feedback={failure.get('feedback', '')}"
        )
    return "\n".join(lines)


def _format_evolution_history(
    recent_history: list[dict[str, Any]],
    top_programs: list[dict[str, Any]],
    inspirations: list[dict[str, Any]],
) -> str:
    sections: list[str] = []
    if recent_history:
        lines = ["Recent lineage:"]
        for idx, record in enumerate(recent_history, start=1):
            lines.append(
                f"{idx}. score={record.get('score', 0.0):.3f} | "
                f"changes={record.get('changes_description') or '(initial prompt)'}"
            )
        sections.append("\n".join(lines))
    sections.append(_format_programs("Top prompts", top_programs))
    sections.append(_format_programs("Inspiration prompts", inspirations))
    return "\n\n".join(sections)


def make_openevolve_propose(
    *,
    llm: LLMClient,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    extra: dict[str, Any] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt_template: str = USER_PROMPT,
    workers: int = 1,
) -> Stage[OpenEvolveContext, Candidate]:
    request_extra = extra or {}

    class OpenEvolvePropose(Stage[OpenEvolveContext, Candidate]):
        def __init__(self) -> None:
            super().__init__(workers=workers)

        async def process(self, item: OpenEvolveContext) -> Candidate:
            user_message = user_prompt_template.format(
                metrics=_format_metrics(item.parent_metrics),
                improvement_areas=_identify_improvement_areas(
                    item.parent_score,
                    item.global_best,
                    item.parent_artifacts,
                ),
                artifacts=_format_artifacts(item.parent_artifacts),
                evolution_history=_format_evolution_history(
                    item.recent_history,
                    item.top_programs,
                    item.inspirations,
                ),
                current_prompt=item.parent.text,
                current_program=item.parent.text,
            )
            response = await llm.chat(
                ChatRequest(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    model=model,
                    extra={
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        **request_extra,
                    },
                )
            )
            response_content = response.content or ""
            matches = _CODEBLOCK_RE.findall(response_content)
            body = matches[-1].strip() if matches else response_content.strip()
            body = body or item.parent.text
            change_summary = _summarize_prompt_change(item.parent.text, body)
            artifact = OpenEvolvePromptArtifact(
                text=body,
                program_id=str(uuid.uuid4()),
                parent_id=item.parent.program_id,
                changes_description=change_summary,
            )
            bootstrap_traj = Trajectory(
                version=item.version,
                artifact=item.parent,
                samples=[],
                outputs=[],
                signals={},
            )
            bootstrap_critique = Critique(
                trajectory=bootstrap_traj,
                text=json.dumps(
                    {
                        "island": item.island,
                        "parent_score": item.parent_score,
                        "top_programs": item.top_programs,
                        "inspirations": item.inspirations,
                    },
                    indent=2,
                ),
            )
            return Candidate(
                parent_version=item.version,
                parent_artifact=item.parent,
                artifact=artifact,
                critique=bootstrap_critique,
            )

    return OpenEvolvePropose()


def _summarize_prompt_change(previous: str, current: str) -> str:
    if previous == current:
        return "no textual change"
    prev_words = set(previous.lower().split())
    curr_words = set(current.lower().split())
    added = sorted(curr_words - prev_words)
    if not added:
        return "prompt rewritten with structural edits"
    return "added emphasis on " + ", ".join(added[:6])
