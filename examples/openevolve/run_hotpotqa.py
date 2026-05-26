"""Run a faithful OpenEvolve-style prompt optimization example on HotpotQA.

This mirrors the original OpenEvolve llm_prompt_optimization example while
executing through FlashEvolve's runtime:

    select parent/context -> propose -> evaluate -> admit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from flashevolve.agents import SingleCallAgent
from flashevolve.llm import ChatRequest, OpenAIChatClient
from flashevolve.runtime import OpenEvolveRuntime
from flashevolve.solvers import Solver, SolverResult
from flashevolve.stages import Candidate, ScoredCandidate, Stage

from examples.openevolve import (
    OpenEvolvePromptArtifact,
    OpenEvolvePromptPool,
    make_openevolve_propose,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_EXAMPLE_ROOT = REPO_ROOT  / "examples" / "openevolve"/ "llm_prompt_optimization"
INITIAL_PROMPT_PATH = ORIGINAL_EXAMPLE_ROOT / "hotpotqa_prompt.txt"
DATASET_CONFIG_PATH = ORIGINAL_EXAMPLE_ROOT / "hotpotqa_prompt_dataset.yaml"
EVOLUTION_CONFIG_PATH = ORIGINAL_EXAMPLE_ROOT / "config_qwen3_evolution.yaml"
REWRITE_TEMPLATE_PATH = ORIGINAL_EXAMPLE_ROOT / "templates" / "full_rewrite_user.txt"
EVALUATOR_SYSTEM_PATH = ORIGINAL_EXAMPLE_ROOT / "templates" / "evaluator_system_message.txt"
EVALUATION_TEMPLATE_PATH = ORIGINAL_EXAMPLE_ROOT / "templates" / "evaluation.txt"

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3-8B"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
NO_THINK_EXTRA = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve-style HotpotQA prompt evolution."
    )
    parser.add_argument("--provider", choices=("vllm", "openai"), default="vllm")
    parser.add_argument(
        "--model",
        default=None,
        help="Default model for propose, task execution, and prompt-quality feedback.",
    )
    parser.add_argument("--task-model", default=None, help="Override the task execution model.")
    parser.add_argument("--propose-model", default=None, help="Override the propose model.")
    parser.add_argument(
        "--feedback-model",
        default=None,
        help="Override the prompt-quality feedback model.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Number of evolution iterations.",
    )
    parser.add_argument(
        "--stage1-samples",
        type=int,
        default=10,
        help="Cascade stage 1 sample count.",
    )
    parser.add_argument(
        "--stage2-samples",
        type=int,
        default=40,
        help="Cascade stage 2 sample count.",
    )
    parser.add_argument(
        "--cascade-threshold",
        type=float,
        default=0.9,
        help="Score threshold to trigger stage 2.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=200,
        help="Maximum number of validation examples to sample from during evolution.",
    )
    parser.add_argument(
        "--full-eval-samples",
        type=int,
        default=5447,
        help="Number of validation examples to score for final reporting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-dir",
        default=str(REPO_ROOT / "examples" / "openevolve" / "runs"),
        help="Directory for run outputs.",
    )
    return parser.parse_args()


def build_client(
    *,
    provider: str,
    model: str,
    request_timeout: float = 180.0,
) -> tuple[OpenAIChatClient, dict[str, object] | None]:
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAIChatClient(
            base_url=OPENAI_BASE_URL,
            model=model,
            api_key=api_key,
            timeout=request_timeout,
        )
        return client, None

    client = OpenAIChatClient(
        base_url=VLLM_BASE_URL,
        model=model,
        timeout=request_timeout,
    )
    return client, NO_THINK_EXTRA


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_hotpotqa_examples(limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,
        streaming=False,
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(example) for example in dataset]


def format_hotpot_context(example: dict[str, Any]) -> str:
    context_items = example.get("context", {})
    context_text = []
    if "title" in context_items and "sentences" in context_items:
        for index, (title, sentences) in enumerate(
            zip(context_items["title"], context_items["sentences"]),
            start=1,
        ):
            context_text.append(f"Paragraph {index} ({title}):")
            context_text.append(" ".join(sentences))
            context_text.append("")
    return "\n".join(context_text).strip()


class HotpotQAPromptSolver(Solver[OpenEvolvePromptArtifact]):
    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 16000,
        temperature: float = 0.1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = extra or {}

    def render(self, artifact: OpenEvolvePromptArtifact, sample: Any) -> ChatRequest:
        user_message = artifact.text.format(
            context=format_hotpot_context(sample),
            question=sample["question"],
        )
        return ChatRequest(
            messages=[{"role": "user", "content": user_message}],
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
    sophistication_score = 0.0

    if len(prompt) >= 100:
        sophistication_score += 0.1

    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )
    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(
            phrase in prompt_lower
            for phrase in ["think through", "reasoning", "explain your"]
        )
        or bool(re.search(r"(first|then|next|finally)", prompt_lower))
    )
    has_directive = "solve" in prompt_lower or "calculate" in prompt_lower
    has_strict = "must" in prompt_lower or "exactly" in prompt_lower

    if has_example:
        sophistication_score += 0.6
        if has_cot:
            sophistication_score += 0.3
        elif len(prompt) > 1500:
            sophistication_score += 0.2
        else:
            sophistication_score += 0.1
    elif has_cot:
        sophistication_score += 0.4
        if has_strict:
            sophistication_score += 0.2
        elif len(prompt) > 500:
            sophistication_score += 0.15
        else:
            sophistication_score += 0.1
    else:
        if has_directive:
            sophistication_score += 0.2
        else:
            sophistication_score += 0.1

    return prompt_length, min(1.0, max(0.0, sophistication_score))


def hotpotqa_metric(sample: dict[str, Any], output: str) -> float:
    output_lower = output.lower().strip().rstrip(".,!?;:")
    expected_lower = str(sample["answer"]).lower().strip().rstrip(".,!?;:")
    if output_lower == expected_lower:
        return 1.0
    if expected_lower and expected_lower in output_lower:
        return 1.0
    return 0.0


def hotpotqa_feedback(sample: dict[str, Any], output: str, score: float) -> str:
    if score >= 1.0:
        return "Matched the expected answer."
    return (
        f"Expected an exact answer matching '{sample['answer']}', "
        f"but received '{output.strip()}'."
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(match.strip() for match in _CODEBLOCK_RE.findall(text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError("No JSON object found in evaluator response")


class OpenEvolveHotpotQAEvaluate(Stage[Candidate, ScoredCandidate]):
    def __init__(
        self,
        *,
        agent: SingleCallAgent,
        dataset: list[dict[str, Any]],
        task_llm: OpenAIChatClient,
        feedback_llm: OpenAIChatClient,
        feedback_extra: dict[str, Any] | None,
        feedback_system_prompt: str,
        feedback_user_template: str,
        stage1_samples: int = 10,
        stage2_samples: int = 40,
        cascade_threshold: float = 0.9,
        llm_feedback_weight: float = 0.2,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.agent = agent
        self.dataset = list(dataset)
        self.task_llm = task_llm
        self.feedback_llm = feedback_llm
        self.feedback_extra = feedback_extra or {}
        self.feedback_system_prompt = feedback_system_prompt
        self.feedback_user_template = feedback_user_template
        self.stage1_samples = stage1_samples
        self.stage2_samples = stage2_samples
        self.cascade_threshold = cascade_threshold
        self.llm_feedback_weight = llm_feedback_weight

    async def process(self, item: Candidate) -> ScoredCandidate:
        stage1 = await self._evaluate_batch(item.artifact, self._sample_batch(self.stage1_samples))
        final = stage1
        print(
            f"[OpenEvolve eval] stage1 score={stage1['accuracy']:.3f} "
            f"(n={len(stage1['per_sample_scores'])})"
        )
        if stage1["accuracy"] >= self.cascade_threshold:
            final = await self._evaluate_batch(
                item.artifact,
                self._sample_batch(self.stage2_samples),
            )
            print(
                f"[OpenEvolve eval] stage2 score={final['accuracy']:.3f} "
                f"(n={len(final['per_sample_scores'])})"
            )
        else:
            print(
                f"[OpenEvolve eval] stage2 skipped "
                f"(threshold={self.cascade_threshold:.3f})"
            )

        llm_eval = await self._evaluate_prompt_quality(item.artifact.text)
        llm_average = (
            llm_eval["clarity"]
            + llm_eval["specificity"]
            + llm_eval["robustness"]
            + llm_eval["format_specification"]
        ) / 4.0
        combined_score = final["accuracy"] * 0.7 + llm_average * 0.3

        prompt_length, reasoning_strategy = calculate_prompt_features(item.artifact.text)
        metrics = {
            "combined_score": combined_score,
            "prompt_length": float(prompt_length),
            "reasoning_strategy": reasoning_strategy,
            "accuracy": final["accuracy"],
            "llm_clarity": llm_eval["clarity"] * self.llm_feedback_weight,
            "llm_specificity": llm_eval["specificity"] * self.llm_feedback_weight,
            "llm_robustness": llm_eval["robustness"] * self.llm_feedback_weight,
            "llm_format_specification": llm_eval["format_specification"]
            * self.llm_feedback_weight,
            "llm_average": llm_average * self.llm_feedback_weight,
        }

        return ScoredCandidate(
            candidate=item,
            score=combined_score,
            signals={
                "metrics": metrics,
                "per_sample_scores": final["per_sample_scores"],
                "num_samples": len(final["per_sample_scores"]),
                "outputs": final["outputs"],
                "feedback": final["feedback"],
                "llm_feedback": llm_eval,
                "agent_signals": final["agent_signals"],
                "artifacts": {
                    "failed_examples": final["failed_examples"][:3],
                    "llm_evaluation": llm_eval,
                },
            },
        )

    def _sample_batch(self, k: int) -> list[dict[str, Any]]:
        if not self.dataset:
            return []
        return self.dataset[: min(k, len(self.dataset))]

    async def _evaluate_batch(
        self,
        artifact: OpenEvolvePromptArtifact,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        results = await self.agent.run(artifact, samples, llm=self.task_llm)
        outputs = [result.output for result in results]
        per_sample_scores = [
            hotpotqa_metric(sample, output)
            for sample, output in zip(samples, outputs)
        ]
        feedback = [
            hotpotqa_feedback(sample, output, sample_score)
            for sample, output, sample_score in zip(samples, outputs, per_sample_scores)
        ]
        failed_examples = [
            {
                "instruction": sample["question"],
                "output": output,
                "score": sample_score,
                "feedback": diag,
            }
            for sample, output, sample_score, diag in zip(
                samples, outputs, per_sample_scores, feedback
            )
            if sample_score < 1.0
        ]
        accuracy = sum(per_sample_scores) / len(per_sample_scores) if per_sample_scores else 0.0
        return {
            "accuracy": accuracy,
            "per_sample_scores": per_sample_scores,
            "feedback": feedback,
            "outputs": outputs,
            "agent_signals": [result.signals for result in results],
            "failed_examples": failed_examples,
        }

    async def _evaluate_prompt_quality(self, prompt_text: str) -> dict[str, Any]:
        response = await self.feedback_llm.chat(
            ChatRequest(
                messages=[
                    {"role": "system", "content": self.feedback_system_prompt},
                    {
                        "role": "user",
                        "content": self.feedback_user_template.format(
                            current_program=prompt_text
                        ),
                    },
                ],
                extra={
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    **self.feedback_extra,
                },
            )
        )
        raw = _extract_json_object(response.content or "{}")
        return {
            "clarity": float(raw.get("clarity", 0.0)),
            "specificity": float(raw.get("specificity", 0.0)),
            "robustness": float(raw.get("robustness", 0.0)),
            "format_specification": float(raw.get("format_specification", 0.0)),
            "reasoning": str(raw.get("reasoning", "")),
        }


async def evaluate_dataset(
    *,
    agent: SingleCallAgent,
    artifact: OpenEvolvePromptArtifact,
    dataset: list[dict[str, Any]],
    task_llm: OpenAIChatClient,
    feedback_llm: OpenAIChatClient,
    feedback_extra: dict[str, Any] | None,
    feedback_system_prompt: str,
    feedback_user_template: str,
    llm_feedback_weight: float = 0.2,
    batch_size: int = 20,
) -> dict[str, Any]:
    all_scores: list[float] = []
    failed_examples: list[dict[str, Any]] = []
    for offset in range(0, len(dataset), batch_size):
        batch = dataset[offset : offset + batch_size]
        results = await agent.run(artifact, batch, llm=task_llm)
        for sample, result in zip(batch, results):
            output = result.output
            score = hotpotqa_metric(sample, output)
            all_scores.append(score)
            if score < 1.0:
                failed_examples.append(
                    {
                        "instruction": sample["question"],
                        "output": output,
                        "score": score,
                        "feedback": hotpotqa_feedback(sample, output, score),
                    }
                )

    accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
    evaluator = OpenEvolveHotpotQAEvaluate(
        agent=agent,
        dataset=[],
        task_llm=task_llm,
        feedback_llm=feedback_llm,
        feedback_extra=feedback_extra,
        feedback_system_prompt=feedback_system_prompt,
        feedback_user_template=feedback_user_template,
        llm_feedback_weight=llm_feedback_weight,
    )
    llm_eval = await evaluator._evaluate_prompt_quality(artifact.text)
    llm_average = (
        llm_eval["clarity"]
        + llm_eval["specificity"]
        + llm_eval["robustness"]
        + llm_eval["format_specification"]
    ) / 4.0
    return {
        "combined_score": accuracy * 0.7 + llm_average * 0.3,
        "accuracy": accuracy,
        "llm_average": llm_average * llm_feedback_weight,
        "num_samples": len(dataset),
        "num_failures": len(failed_examples),
        "failed_examples": failed_examples[:10],
        "llm_feedback": llm_eval,
    }


async def main() -> None:
    args = parse_args()
    default_model = args.model or (OPENAI_MODEL if args.provider == "openai" else VLLM_MODEL)
    task_model = args.task_model or default_model
    propose_model = args.propose_model or default_model
    feedback_model = args.feedback_model or default_model

    task_client, task_extra = build_client(provider=args.provider, model=task_model)
    propose_client, propose_extra = build_client(provider=args.provider, model=propose_model)
    if feedback_model == task_model:
        feedback_client = task_client
        feedback_extra = task_extra
    elif feedback_model == propose_model:
        feedback_client = propose_client
        feedback_extra = propose_extra
    else:
        feedback_client, feedback_extra = build_client(
            provider=args.provider,
            model=feedback_model,
        )

    print(f"Pinging task model {task_model!r} via {args.provider}...")
    pong = await task_client.chat(
        ChatRequest(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            extra={"max_tokens": 8, "temperature": 0.0},
        )
    )
    print(f"Task model responded: {pong.content!r}")

    initial_prompt = read_text(INITIAL_PROMPT_PATH)
    rewrite_user_template = read_text(REWRITE_TEMPLATE_PATH)
    evaluator_system_prompt = read_text(EVALUATOR_SYSTEM_PATH)
    evaluator_user_template = read_text(EVALUATION_TEMPLATE_PATH)

    import yaml

    dataset_config = yaml.safe_load(DATASET_CONFIG_PATH.read_text(encoding="utf-8"))
    evolution_config = yaml.safe_load(EVOLUTION_CONFIG_PATH.read_text(encoding="utf-8"))

    system_prompt = evolution_config["prompt"]["system_message"].strip()
    llm_feedback_weight = float(
        evolution_config["evaluator"].get("llm_feedback_weight", 0.2)
    )
    initial_artifact = OpenEvolvePromptArtifact(text=initial_prompt, program_id="seed")

    train = load_hotpotqa_examples(limit=args.train_samples)
    full_eval = load_hotpotqa_examples(limit=args.full_eval_samples)
    print(
        f"Loaded HotpotQA validation samples for evolution={len(train)} "
        f"full_eval={len(full_eval)}"
    )

    agent = SingleCallAgent(
        HotpotQAPromptSolver(
            model=task_model,
            max_tokens=16000,
            temperature=0.1,
            extra=task_extra or {},
        )
    )
    pool = OpenEvolvePromptPool(
        seed=args.seed,
        feature_bins=10,
        population_size=50,
        archive_size=500,
        num_islands=4,
        num_top_programs=5,
        num_diverse_programs=3,
        num_inspirations=5,
        exploration_ratio=0.3,
        exploitation_ratio=0.3,
        migration_interval=20,
        migration_rate=0.1,
    )
    propose = make_openevolve_propose(
        llm=propose_client,
        model=propose_model,
        max_tokens=4096,
        temperature=0.8,
        extra=propose_extra or {},
        system_prompt=system_prompt,
        user_prompt_template=rewrite_user_template,
    )
    evaluate = OpenEvolveHotpotQAEvaluate(
        agent=agent,
        dataset=train,
        task_llm=task_client,
        feedback_llm=feedback_client,
        feedback_extra=feedback_extra,
        feedback_system_prompt=evaluator_system_prompt,
        feedback_user_template=evaluator_user_template,
        stage1_samples=args.stage1_samples,
        stage2_samples=args.stage2_samples,
        cascade_threshold=args.cascade_threshold,
        llm_feedback_weight=llm_feedback_weight,
        seed=args.seed,
    )

    runtime = OpenEvolveRuntime(
        pool=pool,
        build_context=pool.build_context,
        propose=propose,
        evaluate=evaluate,
        initial_artifact=initial_artifact,
        budget=args.budget,
        selection_strategy="openevolve",
    )

    initial_full_result = await evaluate_dataset(
        agent=agent,
        artifact=initial_artifact,
        dataset=full_eval,
        task_llm=task_client,
        feedback_llm=feedback_client,
        feedback_extra=feedback_extra,
        feedback_system_prompt=evaluator_system_prompt,
        feedback_user_template=evaluator_user_template,
        llm_feedback_weight=llm_feedback_weight,
        batch_size=20,
    )

    print(
        f"Running faithful OpenEvolve-style HotpotQA prompt evolution for {args.budget} iterations"
    )
    await runtime.run()

    best_record = pool.best_record
    if best_record is None:
        raise RuntimeError("No best record found after evolution")

    full_result = await evaluate_dataset(
        agent=agent,
        artifact=best_record.artifact,
        dataset=full_eval,
        task_llm=task_client,
        feedback_llm=feedback_client,
        feedback_extra=feedback_extra,
        feedback_system_prompt=evaluator_system_prompt,
        feedback_user_template=evaluator_user_template,
        llm_feedback_weight=llm_feedback_weight,
        batch_size=20,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir / f"hotpotqa_budget{args.budget}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_prompt.txt").write_text(best_record.artifact.text, encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "provider": args.provider,
                "task_model": task_model,
                "propose_model": propose_model,
                "feedback_model": feedback_model,
                "budget": args.budget,
                "dataset_config": dataset_config,
                "stage1_samples": args.stage1_samples,
                "stage2_samples": args.stage2_samples,
                "cascade_threshold": args.cascade_threshold,
                "train_samples": args.train_samples,
                "full_eval_samples": args.full_eval_samples,
                "initial_full_result": initial_full_result,
                "best_score": best_record.score,
                "best_metrics": best_record.metrics,
                "full_result": full_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best in-loop score: {best_record.score:.3f}")
    print(
        f"Initial full validation accuracy: "
        f"{initial_full_result['accuracy']:.3f}"
    )
    print(f"Full validation accuracy: {full_result['accuracy']:.3f}")
    print(
        f"Accuracy improvement: "
        f"{full_result['accuracy'] - initial_full_result['accuracy']:+.3f}"
    )
    print(f"Saved outputs to {run_dir}")

    await task_client.aclose()
    if propose_client is not task_client:
        await propose_client.aclose()
    if feedback_client not in {task_client, propose_client}:
        await feedback_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
