"""Finance ACE runner on FlashEvolve.

Implements ACE-style offline / online / eval-only workflows for the
finance datasets while preserving the ACE generator / reflector /
curator stage prompts and playbook artifact structure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from flashevolve.agents import SingleCallAgent
from flashevolve.artifacts import PlaybookArtifact
from flashevolve.llm import OpenAIChatClient
from flashevolve.pools import SingletonPool
from flashevolve.runtime import ACERuntime
from flashevolve.samplers.base import Sampler
from flashevolve.stages import Candidate, Rollout, ScoredCandidate, Stage

from . import ACEExactMatchMetric, ACEGeneratorSolver, ACEReflect, make_ace_propose
from .finance_data import FinanceDataProcessor, load_jsonl
from .playbook import make_empty_playbook

ACE_ROOT = Path("/workspace/FlashEvolve/ace")
FINANCE_CONFIG_PATH = ACE_ROOT / "eval" / "finance" / "data" / "sample_config.json"


class SequentialBatchSampler(Sampler):
    """Sequential finite sampler over a dataset for one training run."""

    def __init__(self, dataset: list[Any], batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = list(dataset)
        self.batch_size = batch_size
        self.cursor = 0
        self.last_batch_size = 0

    async def sample(self) -> list[Any]:
        if self.cursor >= len(self.dataset):
            self.last_batch_size = 0
            return []
        end = min(self.cursor + self.batch_size, len(self.dataset))
        batch = self.dataset[self.cursor:end]
        self.cursor = end
        self.last_batch_size = len(batch)
        return batch

    @property
    def num_batches(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACE on finance datasets.")
    parser.add_argument("--task_name", required=True, choices=("finer", "formula"))
    parser.add_argument(
        "--mode",
        default="offline",
        choices=("offline", "online", "eval_only"),
        help="ACE workflow mode.",
    )
    parser.add_argument("--initial_playbook_path", default=None)
    parser.add_argument(
        "--provider",
        default="openai",
        choices=("openai", "vllm"),
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--generator_model", default=None)
    parser.add_argument("--reflector_model", default=None)
    parser.add_argument("--curator_model", default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def provider_config(provider: str) -> tuple[str, str]:
    if provider == "openai":
        return "https://api.openai.com/v1", os.environ.get("OPENAI_API_KEY", "")
    return "http://localhost:8000/v1", os.environ.get("VLLM_API_KEY", "EMPTY")


def build_client(args: argparse.Namespace) -> OpenAIChatClient:
    base_url, api_key = provider_config(args.provider)
    if args.provider != "vllm" and not api_key:
        raise RuntimeError(f"Missing API key for provider {args.provider!r}")
    return OpenAIChatClient(
        base_url=base_url,
        model=args.model,
        api_key=api_key or "EMPTY",
        timeout=180.0,
    )


def load_task_splits(
    task_name: str, mode: str
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, list[dict[str, Any]] | None, FinanceDataProcessor]:
    with FINANCE_CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)[task_name]

    processor = FinanceDataProcessor(task_name)
    def _ace_data_path(raw: str) -> Path:
        normalized = raw[2:] if raw.startswith("./") else raw
        return ACE_ROOT / normalized

    if mode in {"online", "eval_only"}:
        test_samples = processor.process_task_data(
            load_jsonl(_ace_data_path(config["test_data"]))
        )
        return None, None, test_samples, processor

    train_samples = processor.process_task_data(
        load_jsonl(_ace_data_path(config["train_data"]))
    )
    val_samples = processor.process_task_data(
        load_jsonl(_ace_data_path(config["val_data"]))
    )
    test_samples = processor.process_task_data(
        load_jsonl(_ace_data_path(config["test_data"]))
    )
    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path: str | None) -> PlaybookArtifact:
    if path:
        playbook_path = Path(path)
        if not playbook_path.exists():
            raise FileNotFoundError(f"Initial playbook not found: {playbook_path}")
        return PlaybookArtifact(text=playbook_path.read_text(encoding="utf-8"))
    return PlaybookArtifact(text=make_empty_playbook())


def create_run_dir(save_path: str, task_name: str, mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_path) / f"ace_run_{timestamp}_{task_name}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def evaluate_dataset(
    *,
    agent: SingleCallAgent,
    artifact: PlaybookArtifact,
    dataset: list[dict[str, Any]],
    processor: FinanceDataProcessor,
    client: OpenAIChatClient,
) -> dict[str, Any]:
    results = await agent.run(artifact, dataset, llm=client)
    predictions = [str(result.output) for result in results]
    targets = [str(sample["target"]) for sample in dataset]
    accuracy = processor.evaluate_accuracy(predictions, targets)
    errors = [
        {
            "index": idx,
            "prediction": prediction,
            "ground_truth": target,
        }
        for idx, (prediction, target) in enumerate(zip(predictions, targets))
        if not processor.answer_is_correct(prediction, target)
    ]
    print(f"evaluated accuracy:{accuracy}")
    return {
        "accuracy": accuracy,
        "correct": len(dataset) - len(errors),
        "total": len(dataset),
        "errors": errors,
        "predictions": predictions,
        "targets": targets,
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FinanceEvaluate(Stage[Candidate, ScoredCandidate]):
    """Held-out evaluation using the finance task's native accuracy logic."""

    def __init__(
        self,
        *,
        agent: SingleCallAgent,
        llm: OpenAIChatClient,
        dataset: list[dict[str, Any]],
        processor: FinanceDataProcessor,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.llm = llm
        self.dataset = list(dataset)
        self.processor = processor

    async def process(self, item: Candidate) -> ScoredCandidate:
        results = await self.agent.run(item.artifact, self.dataset, llm=self.llm)
        outputs = [str(result.output) for result in results]
        targets = [str(sample["target"]) for sample in self.dataset]
        score = self.processor.evaluate_accuracy(outputs, targets)
        per_sample_scores = [
            1.0 if self.processor.answer_is_correct(output, target) else 0.0
            for output, target in zip(outputs, targets)
        ]
        return ScoredCandidate(
            candidate=item,
            score=score,
            signals={
                "per_sample_scores": per_sample_scores,
                "num_samples": len(self.dataset),
                "predictions": outputs,
                "targets": targets,
                "agent_signals": [result.signals for result in results],
            },
        )


async def run_offline(
    *,
    args: argparse.Namespace,
    client: OpenAIChatClient,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]] | None,
    processor: FinanceDataProcessor,
    run_dir: Path,
) -> None:
    generator_model = args.generator_model or args.model
    reflector_model = args.reflector_model or args.model
    curator_model = args.curator_model or args.model

    solver = ACEGeneratorSolver(
        max_tokens=args.max_tokens,
        temperature=0.0,
        model=generator_model,
        use_json_mode=args.json_mode,
    )
    agent = SingleCallAgent(solver)
    rollout = Rollout(agent=agent, metric=ACEExactMatchMetric(), llm=client)
    reflect = ACEReflect(
        llm=client,
        use_ground_truth=not args.no_ground_truth,
        use_json_mode=args.json_mode,
        max_tokens=args.max_tokens,
        temperature=0.0,
        model=reflector_model,
    )
    total_training_samples = len(train_samples) * args.num_epochs
    propose = make_ace_propose(
        llm=client,
        curator_frequency=args.curator_frequency,
        total_samples=total_training_samples,
        token_budget=args.playbook_token_budget,
        use_ground_truth=not args.no_ground_truth,
        use_json_mode=args.json_mode,
        max_tokens=args.max_tokens,
        temperature=0.0,
        model=curator_model,
    )

    pool = SingletonPool()
    initial_artifact = load_initial_playbook(args.initial_playbook_path)
    training_samples = train_samples * args.num_epochs
    sampler = SequentialBatchSampler(training_samples, args.minibatch)
    evaluate = FinanceEvaluate(
        agent=agent,
        llm=client,
        dataset=val_samples,
        processor=processor,
    )
    runtime = ACERuntime(
        pool=pool,
        sampler=sampler,
        rollout=rollout,
        reflect=reflect,
        propose=propose,
        evaluate=evaluate,
        initial_artifact=initial_artifact,
        budget=sampler.num_batches,
        eval_every=args.eval_steps,
        selection_strategy="latest",
    )
    await runtime.run()

    final_artifact = (
        pool.artifact if isinstance(pool.artifact, PlaybookArtifact) else initial_artifact
    )
    final_val_results = await evaluate_dataset(
        agent=agent,
        artifact=final_artifact,
        dataset=val_samples,
        processor=processor,
        client=client,
    )

    save_json(
        run_dir / "train_results.json",
        {
            "processed_samples": len(training_samples),
            "iterations_run": runtime.iterations_run,
            "num_batches": sampler.num_batches,
            "periodic_eval_history": runtime.evaluation_history,
            "final_val_result": final_val_results,
        },
    )

    (run_dir / "final_playbook.txt").write_text(final_artifact.text, encoding="utf-8")
    (run_dir / "best_playbook.txt").write_text(final_artifact.text, encoding="utf-8")

    if test_samples:
        final_test_results = await evaluate_dataset(
            agent=agent,
            artifact=final_artifact,
            dataset=test_samples,
            processor=processor,
            client=client,
        )
        save_json(run_dir / "final_test_results.json", final_test_results)


async def run_online(
    *,
    args: argparse.Namespace,
    client: OpenAIChatClient,
    test_samples: list[dict[str, Any]],
    processor: FinanceDataProcessor,
    run_dir: Path,
) -> None:
    generator_model = args.generator_model or args.model
    reflector_model = args.reflector_model or args.model
    curator_model = args.curator_model or args.model

    solver = ACEGeneratorSolver(
        max_tokens=args.max_tokens,
        temperature=0.0,
        model=generator_model,
        use_json_mode=args.json_mode,
    )
    agent = SingleCallAgent(solver)
    rollout = Rollout(agent=agent, metric=ACEExactMatchMetric(), llm=client)
    reflect = ACEReflect(
        llm=client,
        use_ground_truth=not args.no_ground_truth,
        use_json_mode=args.json_mode,
        max_tokens=args.max_tokens,
        temperature=0.0,
        model=reflector_model,
    )
    propose = make_ace_propose(
        llm=client,
        curator_frequency=args.curator_frequency,
        total_samples=len(test_samples),
        token_budget=args.playbook_token_budget,
        use_ground_truth=not args.no_ground_truth,
        use_json_mode=args.json_mode,
        max_tokens=args.max_tokens,
        temperature=0.0,
        model=curator_model,
    )

    pool = SingletonPool()
    initial_artifact = load_initial_playbook(args.initial_playbook_path)
    sampler = SequentialBatchSampler(test_samples, args.minibatch)
    evaluate = FinanceEvaluate(
        agent=agent,
        llm=client,
        dataset=test_samples,
        processor=processor,
    )
    runtime = ACERuntime(
        pool=pool,
        sampler=sampler,
        rollout=rollout,
        reflect=reflect,
        propose=propose,
        evaluate=evaluate,
        initial_artifact=initial_artifact,
        budget=sampler.num_batches,
        eval_every=args.online_eval_frequency,
        selection_strategy="latest",
    )
    await runtime.run()

    final_artifact = (
        pool.artifact if isinstance(pool.artifact, PlaybookArtifact) else initial_artifact
    )
    final_results = await evaluate_dataset(
        agent=agent,
        artifact=final_artifact,
        dataset=test_samples,
        processor=processor,
        client=client,
    )
    save_json(
        run_dir / "test_results.json",
        {
            "iterations_run": runtime.iterations_run,
            "num_batches": sampler.num_batches,
            "periodic_eval_history": runtime.evaluation_history,
            "final": final_results,
        },
    )
    (run_dir / "final_playbook.txt").write_text(final_artifact.text, encoding="utf-8")


async def run_eval_only(
    *,
    args: argparse.Namespace,
    client: OpenAIChatClient,
    test_samples: list[dict[str, Any]],
    processor: FinanceDataProcessor,
    run_dir: Path,
) -> None:
    solver = ACEGeneratorSolver(
        max_tokens=args.max_tokens,
        temperature=0.0,
        model=args.generator_model or args.model,
        use_json_mode=args.json_mode,
    )
    agent = SingleCallAgent(solver)
    artifact = load_initial_playbook(args.initial_playbook_path)
    test_results = await evaluate_dataset(
        agent=agent,
        artifact=artifact,
        dataset=test_samples,
        processor=processor,
        client=client,
    )
    save_json(run_dir / "test_results.json", test_results)


async def main() -> None:
    args = parse_args()
    if args.eval_steps < 1:
        raise ValueError(f"--eval_steps must be >= 1, got {args.eval_steps}")
    if args.minibatch < 1:
        raise ValueError(f"--minibatch must be >= 1, got {args.minibatch}")
    if args.online_eval_frequency < 1:
        raise ValueError(
            f"--online_eval_frequency must be >= 1, got {args.online_eval_frequency}"
        )
    run_dir = create_run_dir(args.save_path, args.task_name, args.mode)
    save_json(
        run_dir / "run_config.json",
        {
            "task_name": args.task_name,
            "mode": args.mode,
            "provider": args.provider,
            "model": args.model,
            "generator_model": args.generator_model or args.model,
            "reflector_model": args.reflector_model or args.model,
            "curator_model": args.curator_model or args.model,
            "minibatch": args.minibatch,
            "num_epochs": args.num_epochs,
            "curator_frequency": args.curator_frequency,
            "eval_steps": args.eval_steps,
            "online_eval_frequency": args.online_eval_frequency,
            "max_tokens": args.max_tokens,
            "playbook_token_budget": args.playbook_token_budget,
            "json_mode": args.json_mode,
            "no_ground_truth": args.no_ground_truth,
        },
    )

    train_samples, val_samples, test_samples, processor = load_task_splits(
        args.task_name, args.mode
    )
    client = build_client(args)

    try:
        if args.mode == "offline":
            assert train_samples is not None and val_samples is not None
            await run_offline(
                args=args,
                client=client,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                processor=processor,
                run_dir=run_dir,
            )
        elif args.mode == "online":
            assert test_samples is not None
            await run_online(
                args=args,
                client=client,
                test_samples=test_samples,
                processor=processor,
                run_dir=run_dir,
            )
        else:
            assert test_samples is not None
            await run_eval_only(
                args=args,
                client=client,
                test_samples=test_samples,
                processor=processor,
                run_dir=run_dir,
            )
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
