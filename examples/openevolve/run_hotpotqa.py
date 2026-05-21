"""Run a faithful OpenEvolve-style prompt optimization example on HotpotQA.

Loop:
    select parent/context -> propose -> evaluate -> admit

This example uses HotpotQA train samples for in-loop evolution and reports
final performance on a held-out slice of the validation split.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from flashevolve.agents import SingleCallAgent
from flashevolve.llm import ChatRequest, OpenAIChatClient
from flashevolve.runtime import OpenEvolveRuntime

from examples.openevolve import (
    HotpotQAPromptSolver,
    OpenEvolveDatasetEvaluate,
    OpenEvolvePromptArtifact,
    OpenEvolvePromptPool,
    evaluate_dataset,
    hotpotqa_feedback,
    hotpotqa_metric,
    make_openevolve_propose,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INITIAL_PROMPT_PATH = Path(__file__).with_name("hotpotqa_prompt.txt")

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3-8B"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
NO_THINK_EXTRA = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEvolve-style HotpotQA prompt evolution.")
    parser.add_argument("--provider", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--model", default=None, help="Default model for both propose and task calls.")
    parser.add_argument("--task-model", default=None, help="Override the task execution model.")
    parser.add_argument("--propose-model", default=None, help="Override the propose model.")
    parser.add_argument("--budget", type=int, default=50, help="Number of evolution iterations.")
    parser.add_argument("--stage1-samples", type=int, default=10, help="Cascade stage 1 sample count.")
    parser.add_argument("--stage2-samples", type=int, default=40, help="Cascade stage 2 sample count.")
    parser.add_argument("--cascade-threshold", type=float, default=0.6, help="Score threshold to trigger stage 2.")
    parser.add_argument("--train-size", type=int, default=200, help="Number of HotpotQA train examples used for evolution.")
    parser.add_argument("--val-size", type=int, default=200, help="Number of validation examples used for final reporting.")
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


def _format_hotpot_context(example: dict[str, Any]) -> str:
    context = example["context"]
    titles = context.get("title", [])
    sentence_groups = context.get("sentences", [])
    blocks: list[str] = []
    for idx, (title, sentences) in enumerate(zip(titles, sentence_groups), start=1):
        blocks.append(f"Paragraph {idx} ({title}): {' '.join(sentences)}")
    return "\n\n".join(blocks)


def load_hotpotqa_splits(*, train_size: int, val_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_raw = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
        trust_remote_code=True,
    )
    val_raw = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,
    )

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": example["question"],
            "answer": example["answer"],
            "context": _format_hotpot_context(example),
            "id": example.get("id", ""),
        }

    train = [_convert(train_raw[i]) for i in range(min(train_size, len(train_raw)))]
    val = [_convert(val_raw[i]) for i in range(min(val_size, len(val_raw)))]
    return train, val


async def main() -> None:
    args = parse_args()
    default_model = args.model or (OPENAI_MODEL if args.provider == "openai" else VLLM_MODEL)
    task_model = args.task_model or default_model
    propose_model = args.propose_model or default_model

    task_client, task_extra = build_client(provider=args.provider, model=task_model)
    propose_client, propose_extra = build_client(provider=args.provider, model=propose_model)

    print(f"Pinging task model {task_model!r} via {args.provider}...")
    pong = await task_client.chat(
        ChatRequest(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            extra={"max_tokens": 8, "temperature": 0.0},
        )
    )
    print(f"Task model responded: {pong.content!r}")

    train, val = load_hotpotqa_splits(train_size=args.train_size, val_size=args.val_size)
    print(f"Loaded HotpotQA train={len(train)} val={len(val)}")

    initial_prompt = INITIAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    initial_artifact = OpenEvolvePromptArtifact(text=initial_prompt, program_id="seed")

    agent = SingleCallAgent(
        HotpotQAPromptSolver(
            model=task_model,
            max_tokens=256,
            temperature=0.0,
            extra=task_extra or {},
        )
    )
    pool = OpenEvolvePromptPool(
        seed=args.seed,
        num_islands=4,
        feature_bins=10,
        archive_size=64,
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
        max_tokens=1024,
        temperature=0.8,
        extra=propose_extra or {},
    )
    evaluate = OpenEvolveDatasetEvaluate(
        agent=agent,
        dataset=train,
        metric=hotpotqa_metric,
        feedback_fn=hotpotqa_feedback,
        llm=task_client,
        stage1_samples=args.stage1_samples,
        stage2_samples=args.stage2_samples,
        cascade_threshold=args.cascade_threshold,
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

    print(
        f"Running faithful OpenEvolve-style HotpotQA prompt evolution for {args.budget} iterations"
    )
    await runtime.run()

    best_record = pool.best_record
    if best_record is None:
        raise RuntimeError("No best record found after evolution")

    val_result = await evaluate_dataset(
        agent=agent,
        artifact=best_record.artifact,
        dataset=val,
        metric=hotpotqa_metric,
        feedback_fn=hotpotqa_feedback,
        llm=task_client,
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
                "budget": args.budget,
                "stage1_samples": args.stage1_samples,
                "stage2_samples": args.stage2_samples,
                "cascade_threshold": args.cascade_threshold,
                "train_size": len(train),
                "val_size": len(val),
                "best_score": best_record.score,
                "best_metrics": best_record.metrics,
                "val_result": val_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best in-loop score: {best_record.score:.3f}")
    print(f"Validation score: {val_result['combined_score']:.3f}")
    print(f"Saved outputs to {run_dir}")

    await task_client.aclose()
    if propose_client is not task_client:
        await propose_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
