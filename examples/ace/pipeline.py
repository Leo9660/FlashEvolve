import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

from flashevolve.artifacts import PlaybookArtifact
from flashevolve.llm import ChatRequest, LLMClient
from flashevolve.solvers import Solver, SolverResult
from flashevolve.stages import Critique, Propose, Stage, Trajectory

from .playbook import (
    apply_curator_operations,
    extract_json_from_text,
    extract_playbook_bullets,
    get_next_global_id,
    get_playbook_stats,
    update_bullet_counts,
)
from .prompts import (
    CURATOR_PROMPT,
    CURATOR_PROMPT_NO_GT,
    GENERATOR_PROMPT,
    REFLECTOR_PROMPT,
    REFLECTOR_PROMPT_NO_GT,
)


def _json_mode_extra(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {}
    return {"response_format": {"type": "json_object"}}


class ACEGeneratorSolver(Solver[PlaybookArtifact]):
    """ACE generator prompt preserved inside a FlashEvolve solver."""

    def __init__(
        self,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: str | None = None,
        use_json_mode: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.use_json_mode = use_json_mode
        self.extra = extra or {}

    def render(self, artifact: PlaybookArtifact, sample: Any) -> ChatRequest:
        question = sample["question"]
        context = sample.get("context", "")
        reflection = sample.get("reflection", "(empty)")
        prompt = GENERATOR_PROMPT.format(artifact.text, reflection, question, context)
        merged_extra = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **_json_mode_extra(self.use_json_mode),
            **self.extra,
        }
        return ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            extra=merged_extra,
        )

    def parse(self, response, sample: Any) -> SolverResult:
        parsed = extract_json_from_text(response.content) or {}
        bullet_ids = parsed.get("bullet_ids", [])
        if not isinstance(bullet_ids, list):
            bullet_ids = []
        final_answer = str(parsed.get("final_answer", "")).strip()
        reasoning = str(parsed.get("reasoning", response.content)).strip()
        return SolverResult(
            output=final_answer,
            signals={
                "raw_response": response.content,
                "generator_json": parsed,
                "reasoning_trace": reasoning,
                "bullet_ids": [str(x) for x in bullet_ids],
                "final_answer": final_answer,
            },
        )


class ACEExactMatchMetric:
    """Small default metric for the example package."""

    def __call__(self, sample: dict[str, Any], output: str) -> float:
        gold = str(sample.get("target", "")).strip().lower()
        pred = str(output).strip().lower()
        return 1.0 if gold == pred else 0.0


@dataclass(frozen=True)
class ACEReflectionRecord:
    sample: dict[str, Any]
    output: str
    score: float
    bullet_ids: list[str]
    bullet_tags: list[dict[str, str]]
    reflection_json: dict[str, Any]
    reflection_text: str
    used_bullets: str


class ACEReflect(Stage[Trajectory, Critique]):
    """ACE reflector over a minibatch of generator trajectories."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        use_ground_truth: bool = True,
        use_json_mode: bool = True,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: str | None = None,
        feedback_fn: Callable[[dict[str, Any], str, float], str] | None = None,
        extra: dict[str, Any] | None = None,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.llm = llm
        self.use_ground_truth = use_ground_truth
        self.use_json_mode = use_json_mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.feedback_fn = feedback_fn or self._default_feedback
        self.extra = extra or {}

    def _default_feedback(self, sample: dict[str, Any], output: str, score: float) -> str:
        target = sample.get("target", "")
        if score >= 1.0:
            return "Predicted answer matches ground truth"
        return f"Predicted answer does not match ground truth: expected {target!r}"

    async def _reflect_one(
        self,
        item: Trajectory,
        idx: int,
    ) -> ACEReflectionRecord:
        sample = item.samples[idx]
        output = item.outputs[idx]
        score = item.signals["scores"][idx]
        agent_signal = item.signals["agent_signals"][idx]
        bullet_ids = list(agent_signal.get("bullet_ids", []))
        used_bullets = extract_playbook_bullets(item.artifact.text, bullet_ids)
        environment_feedback = self.feedback_fn(sample, output, score)

        if self.use_ground_truth and sample.get("target"):
            prompt = REFLECTOR_PROMPT.format(
                sample["question"],
                agent_signal.get("reasoning_trace", agent_signal.get("raw_response", "")),
                output,
                sample.get("target", ""),
                environment_feedback,
                used_bullets,
            )
        else:
            prompt = REFLECTOR_PROMPT_NO_GT.format(
                sample["question"],
                agent_signal.get("reasoning_trace", agent_signal.get("raw_response", "")),
                output,
                environment_feedback,
                used_bullets,
            )

        response = await self.llm.chat(
            ChatRequest(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                extra={
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    **_json_mode_extra(self.use_json_mode),
                    **self.extra,
                },
            )
        )
        parsed = extract_json_from_text(response.content) or {}
        bullet_tags = parsed.get("bullet_tags", [])
        if not isinstance(bullet_tags, list):
            bullet_tags = []

        return ACEReflectionRecord(
            sample=sample,
            output=output,
            score=score,
            bullet_ids=bullet_ids,
            bullet_tags=[
                {"id": str(tag.get("id", "")), "tag": str(tag.get("tag", "neutral"))}
                for tag in bullet_tags
                if isinstance(tag, dict)
            ],
            reflection_json=parsed,
            reflection_text=response.content,
            used_bullets=used_bullets,
        )

    async def process(self, item: Trajectory) -> Critique:
        records = await asyncio.gather(
            *(self._reflect_one(item, idx) for idx in range(len(item.samples)))
        )
        critique_payload = {
            "records": [
                {
                    "question": record.sample.get("question", ""),
                    "context": record.sample.get("context", ""),
                    "target": record.sample.get("target", ""),
                    "predicted_answer": record.output,
                    "score": record.score,
                    "bullet_ids": record.bullet_ids,
                    "bullet_tags": record.bullet_tags,
                    "used_bullets": record.used_bullets,
                    "reflection": record.reflection_json or record.reflection_text,
                }
                for record in records
            ]
        }
        return Critique(
            trajectory=item,
            text=json.dumps(critique_payload, indent=2),
            signals={"records": records},
        )


class ACEProposeParse:
    """Callable parser object so proposal state can stay algorithm-specific."""

    def __init__(
        self,
        *,
        curator_frequency: int,
        total_samples: int,
        token_budget: int,
        use_ground_truth: bool = True,
    ) -> None:
        if curator_frequency < 1:
            raise ValueError("curator_frequency must be >= 1")
        self.curator_frequency = curator_frequency
        self.total_samples = total_samples
        self.token_budget = token_budget
        self.use_ground_truth = use_ground_truth
        self.next_global_id = 1
        self.processed_batches = 0
        self.processed_samples = 0

    def __call__(self, text: str, critique: Critique) -> PlaybookArtifact:
        parent = critique.trajectory.artifact
        if not isinstance(parent, PlaybookArtifact):
            raise TypeError(
                f"ACE propose expects PlaybookArtifact parent, got {type(parent).__name__}"
            )

        updated_text = parent.text
        records: list[ACEReflectionRecord] = critique.signals.get("records", [])
        for record in records:
            updated_text = update_bullet_counts(updated_text, record.bullet_tags)

        if self.next_global_id == 1:
            self.next_global_id = get_next_global_id(updated_text)

        self.processed_batches += 1
        self.processed_samples += len(critique.trajectory.samples)
        should_curate = self.processed_batches % self.curator_frequency == 0

        if should_curate:
            parsed = extract_json_from_text(text) or {}
            operations = parsed.get("operations", [])
            if isinstance(operations, list):
                updated_text, self.next_global_id = apply_curator_operations(
                    updated_text, operations, self.next_global_id
                )

        return PlaybookArtifact(text=updated_text)


def make_ace_propose(
    *,
    llm: LLMClient,
    curator_frequency: int,
    total_samples: int,
    token_budget: int = 80_000,
    use_ground_truth: bool = True,
    use_json_mode: bool = True,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
    workers: int = 1,
) -> Propose:
    state = ACEProposeParse(
        curator_frequency=curator_frequency,
        total_samples=total_samples,
        token_budget=token_budget,
        use_ground_truth=use_ground_truth,
    )
    extra = extra or {}

    def _render(critique: Critique) -> ChatRequest:
        parent = critique.trajectory.artifact
        if not isinstance(parent, PlaybookArtifact):
            raise TypeError(
                f"ACE propose expects PlaybookArtifact parent, got {type(parent).__name__}"
            )
        records: list[ACEReflectionRecord] = critique.signals.get("records", [])
        current_playbook = parent.text
        for record in records:
            current_playbook = update_bullet_counts(current_playbook, record.bullet_tags)

        recent_reflection = json.dumps(
            [
                {
                    "question": record.sample.get("question", ""),
                    "predicted_answer": record.output,
                    "target": record.sample.get("target", ""),
                    "score": record.score,
                    "reflection": record.reflection_json or record.reflection_text,
                }
                for record in records
            ],
            indent=2,
        )
        question_context = "\n\n".join(
            [
                f"Question: {record.sample.get('question', '')}\n"
                f"Context: {record.sample.get('context', '')}"
                for record in records
            ]
        )
        playbook_stats = json.dumps(get_playbook_stats(current_playbook), indent=2)
        current_step = min(total_samples, state.processed_samples + len(records))
        prompt_template = CURATOR_PROMPT if use_ground_truth else CURATOR_PROMPT_NO_GT
        prompt = prompt_template.format(
            token_budget=token_budget,
            current_step=current_step,
            total_samples=total_samples,
            playbook_stats=playbook_stats,
            recent_reflection=recent_reflection,
            current_playbook=current_playbook,
            question_context=question_context,
        )
        return ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            extra={
                "max_tokens": max_tokens,
                "temperature": temperature,
                **_json_mode_extra(use_json_mode),
                **extra,
            },
        )

    return Propose(render=_render, parse=state, llm=llm, workers=workers)
