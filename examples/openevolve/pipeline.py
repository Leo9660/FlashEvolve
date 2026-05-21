from __future__ import annotations

import asyncio
import json
import random
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from flashevolve.agents.base import Agent
from flashevolve.artifacts.base import Artifact
from flashevolve.llm import ChatRequest, LLMClient
from flashevolve.solvers import Solver, SolverResult
from flashevolve.stages import Candidate, Critique, ScoredCandidate, Stage, Trajectory

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
    top_programs: list[dict[str, Any]]
    inspirations: list[dict[str, Any]]
    global_best: dict[str, Any] | None


class IFBenchPromptSolver(Solver[OpenEvolvePromptArtifact]):
    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        user_template: str = "Instruction:\n{instruction}\n\nResponse:",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.user_template = user_template
        self.extra = extra or {}

    def render(self, artifact: OpenEvolvePromptArtifact, sample: Any) -> ChatRequest:
        user_message = self.user_template.format(instruction=sample["prompt"])
        return ChatRequest(
            messages=[
                {"role": "system", "content": artifact.text},
                {"role": "user", "content": user_message},
            ],
            model=self.model,
            extra={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **self.extra,
            },
        )

    def parse(self, response, sample: Any) -> SolverResult:
        return SolverResult(output=(response.content or "").strip(), signals={})


class HotpotQAPromptSolver(Solver[OpenEvolvePromptArtifact]):
    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = extra or {}

    def render(self, artifact: OpenEvolvePromptArtifact, sample: Any) -> ChatRequest:
        user_message = (
            f"Context:\n{sample['context']}\n\n"
            f"Question: {sample['question']}\n\n"
            "Provide a clear, concise answer based only on the context."
        )
        return ChatRequest(
            messages=[
                {"role": "system", "content": artifact.text},
                {"role": "user", "content": user_message},
            ],
            model=self.model,
            extra={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **self.extra,
            },
        )

    def parse(self, response, sample: Any) -> SolverResult:
        return SolverResult(output=(response.content or "").strip(), signals={})


def calculate_prompt_features(prompt: str) -> tuple[int, float]:
    prompt_length = len(prompt)
    prompt_lower = prompt.lower()
    sophistication = 0.0

    if len(prompt) >= 100:
        sophistication += 0.1
    if "step by step" in prompt_lower or "step-by-step" in prompt_lower:
        sophistication += 0.25
    if "exactly" in prompt_lower or "must" in prompt_lower:
        sophistication += 0.15
    if "format" in prompt_lower or "structure" in prompt_lower:
        sophistication += 0.15
    if "constraint" in prompt_lower or "requirements" in prompt_lower:
        sophistication += 0.15
    if "example" in prompt_lower:
        sophistication += 0.2

    return prompt_length, min(1.0, sophistication)


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


def make_openevolve_propose(
    *,
    llm: LLMClient,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    extra: dict[str, Any] | None = None,
    workers: int = 1,
) -> Stage[OpenEvolveContext, Candidate]:
    request_extra = extra or {}

    class OpenEvolvePropose(Stage[OpenEvolveContext, Candidate]):
        def __init__(self) -> None:
            super().__init__(workers=workers)

        async def process(self, item: OpenEvolveContext) -> Candidate:
            global_best = item.global_best
            global_best_score = 0.0 if global_best is None else global_best.get("score", 0.0)
            user_message = USER_PROMPT.format(
                parent_score=item.parent_score,
                island=item.island,
                global_best_score=global_best_score,
                current_prompt=item.parent.text,
                top_programs=_format_programs("Top prompts", item.top_programs),
                inspirations=_format_programs("Inspiration prompts", item.inspirations),
            )
            response = await llm.chat(
                ChatRequest(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
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
            matches = _CODEBLOCK_RE.findall(response.content)
            body = matches[-1].strip() if matches else response.content.strip()
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


class OpenEvolveDatasetEvaluate(Stage[Candidate, ScoredCandidate]):
    def __init__(
        self,
        *,
        agent: Agent,
        dataset: list[dict[str, Any]],
        metric: Callable[[Any, Any], float],
        feedback_fn: Callable[[Any, Any, float], str],
        llm: LLMClient,
        stage1_samples: int = 10,
        stage2_samples: int = 40,
        cascade_threshold: float = 0.9,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.agent = agent
        self.dataset = list(dataset)
        self.metric = metric
        self.feedback_fn = feedback_fn
        self.llm = llm
        self.stage1_samples = stage1_samples
        self.stage2_samples = stage2_samples
        self.cascade_threshold = cascade_threshold
        self.rng = random.Random(seed)

    async def process(self, item: Candidate) -> ScoredCandidate:
        stage1 = await self._evaluate_batch(item.artifact, self._sample_batch(self.stage1_samples))
        final = stage1
        if stage1["score"] >= self.cascade_threshold:
            final = await self._evaluate_batch(
                item.artifact, self._sample_batch(self.stage2_samples)
            )

        score = final["score"]
        per_sample_scores = final["per_sample_scores"]
        feedback = final["feedback"]
        outputs = final["outputs"]
        prompt_length, reasoning_strategy = calculate_prompt_features(item.artifact.text)
        metrics = {
            "combined_score": score,
            "prompt_length": float(prompt_length),
            "reasoning_strategy": reasoning_strategy,
        }
        return ScoredCandidate(
            candidate=item,
            score=score,
            signals={
                "metrics": metrics,
                "per_sample_scores": per_sample_scores,
                "num_samples": len(per_sample_scores),
                "outputs": outputs,
                "feedback": feedback,
                "agent_signals": final["agent_signals"],
                "artifacts": final["artifacts"],
            },
        )

    def _sample_batch(self, k: int) -> list[dict[str, Any]]:
        if not self.dataset:
            return []
        return self.rng.sample(self.dataset, min(k, len(self.dataset)))

    async def _evaluate_batch(
        self,
        artifact: OpenEvolvePromptArtifact,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        results = await self.agent.run(artifact, samples, llm=self.llm)
        outputs = [result.output for result in results]
        per_sample_scores = [self.metric(sample, output) for sample, output in zip(samples, outputs)]
        feedback = [
            self.feedback_fn(sample, output, score)
            for sample, output, score in zip(samples, outputs, per_sample_scores)
        ]
        failed_examples = [
            {
                "instruction": sample["prompt"],
                "output": output,
                "score": sample_score,
                "feedback": diag,
            }
            for sample, output, sample_score, diag in zip(
                samples, outputs, per_sample_scores, feedback
            )
            if sample_score < 1.0
        ]
        return {
            "score": sum(per_sample_scores) / len(per_sample_scores) if per_sample_scores else 0.0,
            "per_sample_scores": per_sample_scores,
            "feedback": feedback,
            "outputs": outputs,
            "agent_signals": [result.signals for result in results],
            "artifacts": {"failed_examples": failed_examples[:3]},
        }


class OpenEvolveIFBenchEvaluate(OpenEvolveDatasetEvaluate):
    pass


async def evaluate_dataset(
    *,
    agent: Agent,
    artifact: OpenEvolvePromptArtifact,
    dataset: list[dict[str, Any]],
    metric: Callable[[Any, Any], float],
    feedback_fn: Callable[[Any, Any, float], str],
    llm: LLMClient,
    batch_size: int = 20,
) -> dict[str, Any]:
    all_scores: list[float] = []
    failed_examples: list[dict[str, Any]] = []
    for offset in range(0, len(dataset), batch_size):
        batch = dataset[offset : offset + batch_size]
        results = await agent.run(artifact, batch, llm=llm)
        for sample, result in zip(batch, results):
            output = result.output
            score = metric(sample, output)
            all_scores.append(score)
            if score < 1.0:
                failed_examples.append(
                    {
                        "instruction": sample["prompt"],
                        "output": output,
                        "score": score,
                        "feedback": feedback_fn(sample, output, score),
                    }
                )
    accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "combined_score": accuracy,
        "num_samples": len(dataset),
        "num_failures": len(failed_examples),
        "failed_examples": failed_examples[:10],
    }


def ifbench_metric(sample: dict[str, Any], output: str) -> float:
    import dspy
    from gepa_artifact.benchmarks.IFBench.ifbench_metric import metric_with_feedback

    return float(
        metric_with_feedback(
            dspy.Example(**sample),
            dspy.Prediction(response=output),
        ).score
    )


def ifbench_feedback(sample: dict[str, Any], output: str, score: float) -> str:
    import dspy
    from gepa_artifact.benchmarks.IFBench.ifbench_metric import metric_with_feedback

    return metric_with_feedback(
        dspy.Example(**sample),
        dspy.Prediction(response=output),
    ).feedback


def normalize_hotpot_answer(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"[\"'`]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[.,!?;:]+$", "", normalized)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def hotpotqa_metric(sample: dict[str, Any], output: str) -> float:
    prediction = normalize_hotpot_answer(output)
    expected = normalize_hotpot_answer(str(sample["answer"]))
    if prediction == expected:
        return 1.0
    if expected and expected in prediction:
        return 1.0
    return 0.0


def hotpotqa_feedback(sample: dict[str, Any], output: str, score: float) -> str:
    if score >= 1.0:
        return "Answer matches the expected HotpotQA answer."
    return (
        f"Expected answer {sample['answer']!r}, but the model responded with {output!r}. "
        "The answer should be concise and grounded in the provided context."
    )
