"""Run a faithful OpenEvolve-style prompt optimization example on IFBench.

Loop:
    select parent/context -> propose -> evaluate -> admit

The evaluate step performs its own cascade sampling over IFBench, which keeps
the example aligned with the official OpenEvolve control flow.

Data source:
    ./gepa-artifact/gepa_artifact/benchmarks/IFBench/data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from flashevolve.agents import SingleCallAgent
from flashevolve.llm import ChatRequest, OpenAIChatClient
from flashevolve.runtime import OpenEvolveRuntime

from examples.gepa.datasets import load_ifbench
from examples.openevolve import (
    IFBenchPromptSolver,
    OpenEvolveIFBenchEvaluate,
    OpenEvolvePromptArtifact,
    OpenEvolvePromptPool,
    evaluate_dataset,
    ifbench_feedback,
    ifbench_metric,
    make_openevolve_propose,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GEPA_ARTIFACT_ROOT = REPO_ROOT / "gepa-artifact"
GEPA_ARTIFACT_DATA = GEPA_ARTIFACT_ROOT / "gepa_artifact" / "benchmarks" / "IFBench" / "data"
INITIAL_PROMPT_PATH = Path(__file__).with_name("ifeval_prompt.txt")

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3-8B"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
NO_THINK_EXTRA = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

if str(GEPA_ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEPA_ARTIFACT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEvolve-style IFBench prompt evolution.")
    parser.add_argument("--provider", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--model", default=None, help="Default model for both propose and task calls.")
    parser.add_argument("--task-model", default=None, help="Override the task execution model.")
    parser.add_argument("--propose-model", default=None, help="Override the propose model.")
    parser.add_argument("--budget", type=int, default=50, help="Number of evolution iterations.")
    parser.add_argument("--stage1-samples", type=int, default=10, help="Cascade stage 1 sample count.")
    parser.add_argument("--stage2-samples", type=int, default=40, help="Cascade stage 2 sample count.")
    parser.add_argument("--cascade-threshold", type=float, default=0.9, help="Score threshold to trigger stage 2.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        default=str(GEPA_ARTIFACT_DATA),
        help="Directory containing IFBench_train.jsonl and IFBench_test.jsonl.",
    )
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

    data_dir = Path(args.data_dir)
    train = load_ifbench(data_dir, "train")
    val = load_ifbench(data_dir, "val")
    test = load_ifbench(data_dir, "test")
    print(f"Loaded IFBench train={len(train)} val={len(val)} test={len(test)}")

    initial_prompt = INITIAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    initial_artifact = OpenEvolvePromptArtifact(text=initial_prompt, program_id="seed")

    agent = SingleCallAgent(
        IFBenchPromptSolver(
            model=task_model,
            max_tokens=1024,
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
    evaluate = OpenEvolveIFBenchEvaluate(
        agent=agent,
        dataset=train,
        metric=ifbench_metric,
        feedback_fn=ifbench_feedback,
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
        f"Running faithful OpenEvolve-style IFBench prompt evolution for {args.budget} iterations"
    )
    await runtime.run()

    best_record = pool.best_record
    if best_record is None:
        raise RuntimeError("No best record found after evolution")

    val_result = await evaluate_dataset(
        agent=agent,
        artifact=best_record.artifact,
        dataset=val,
        metric=ifbench_metric,
        feedback_fn=ifbench_feedback,
        llm=task_client,
        batch_size=20,
    )
    test_result = await evaluate_dataset(
        agent=agent,
        artifact=best_record.artifact,
        dataset=test,
        metric=ifbench_metric,
        feedback_fn=ifbench_feedback,
        llm=task_client,
        batch_size=20,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir / f"ifbench_budget{args.budget}_seed{args.seed}"
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
                "best_score": best_record.score,
                "best_metrics": best_record.metrics,
                "val_result": val_result,
                "test_result": test_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best in-loop score: {best_record.score:.3f}")
    print(f"Validation score: {val_result['combined_score']:.3f}")
    print(f"Test score: {test_result['combined_score']:.3f}")
    print(f"Saved outputs to {run_dir}")

    await task_client.aclose()
    if propose_client is not task_client:
        await propose_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
