"""Smoke test: GEPA on IFBench (3 iter, val_subsample=10).

Validates that ``examples/gepa/datasets.py::load_ifbench`` wires
correctly into the GEPA pipeline. Single-predictor evolution (the main
``generate_response`` prompt); ``EnsureCorrectResponse`` second stage
omitted for the smoke run - adding it is a straightforward extension
of ``IFBenchAgent.run`` (two chat calls instead of one).

Prereqs:
    1. vLLM @ :8000  (see ``run.py`` docstring for launch command)
    2. Upstream GEPA artifact checked out at
       ``/workspace/flash-evolve/gepa-artifact/`` so we can import the
       IFBench metric (reused verbatim, no reimplementation).

Run:
    python -m examples.gepa.run_ifbench                       # local vLLM
    python -m examples.gepa.run_ifbench --provider openai     # OpenAI API
"""

import argparse
import asyncio
import os
import sys

# IFBench metric lives in the upstream GEPA artifact - make it importable.
sys.path.insert(0, "/workspace/flash-evolve/gepa-artifact")

import dspy  # noqa: E402
from gepa_artifact.benchmarks.IFBench.ifbench_metric import metric_with_feedback  # noqa: E402

from flashevolve.agents import Agent, AgentResult  # noqa: E402
from flashevolve.artifacts import PromptArtifact  # noqa: E402
from flashevolve.llm import ChatRequest, OpenAIChatClient  # noqa: E402
from flashevolve.pools import AppendOnlyPool  # noqa: E402
from flashevolve.runtime import GEPARuntime  # noqa: E402
from flashevolve.samplers import FullSampler, RandomSampler  # noqa: E402

from examples.gepa import (  # noqa: E402
    GEPAReflect,
    make_gepa_evaluate,
    make_gepa_propose,
    make_gepa_rollout,
)
from examples.gepa.datasets import load_ifbench  # noqa: E402

# Config

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3-8B"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
DATA_DIR = "/workspace/FlashEvolve/gepa-artifact/gepa_artifact/benchmarks/IFBench/data"

BUDGET = 3
MINIBATCH = 3
VAL_SUBSAMPLE: int | None = None  # None -> full 300 (paper-faithful); int -> subsample
SEED = 42
INITIAL_PROMPT = "Respond to the query."

NO_THINK_EXTRA = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


# Task adapters


class IFBenchAgent(Agent):
    """Single LLM call per sample: system = artifact.text, user = sample['prompt'].

    Mirrors ``GenerateResponse`` in the upstream IFBench program (one
    predictor only - A-tier integration; B-tier would chain a second
    ``EnsureCorrectResponse`` call).
    """

    def __init__(
        self, llm: OpenAIChatClient, *, request_extra: dict | None = None
    ) -> None:
        self.llm = llm
        self.request_extra = request_extra or {}

    async def run(self, artifact, samples, *, llm):
        client = llm or self.llm

        async def one(s):
            r = await client.chat(
                ChatRequest(
                    messages=[
                        {"role": "system", "content": artifact.text},
                        {"role": "user", "content": s["prompt"]},
                    ],
                    extra={
                        "max_tokens": 1024,
                        "temperature": 0.0,
                        **self.request_extra,
                    },
                )
            )
            return AgentResult(output=r.content.strip(), signals={})

        return await asyncio.gather(*(one(s) for s in samples))


def ifbench_metric(sample: dict, output: str) -> float:
    """Fraction of instructions in ``sample['instruction_id_list']`` that
    the response satisfies. Reuses the upstream rule checker verbatim."""
    return float(
        metric_with_feedback(
            dspy.Example(**sample),
            dspy.Prediction(response=output),
        ).score
    )


def ifbench_feedback(sample: dict, output: str, score: float) -> str:
    """Per-sample diagnostic string - exactly what GEPA's Reflect needs."""
    return metric_with_feedback(
        dspy.Example(**sample),
        dspy.Prediction(response=output),
    ).feedback


# Run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA on IFBench.")
    parser.add_argument(
        "--provider",
        choices=("vllm", "openai"),
        default="vllm",
        help="LLM provider preset. Defaults to local vLLM.",
    )
    parser.add_argument("--model", default=None, help="Override provider model name.")
    parser.add_argument(
        "--budget",
        type=int,
        default=BUDGET,
        help="Number of GEPA evolution iterations.",
    )
    return parser.parse_args()


def build_client_and_agent(
    args: argparse.Namespace,
) -> tuple[OpenAIChatClient, IFBenchAgent, str, dict | None]:
    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; export it before using "
                "--provider openai"
            )
        model = args.model or OPENAI_MODEL
        client = OpenAIChatClient(
            base_url=OPENAI_BASE_URL,
            model=model,
            api_key=api_key,
            timeout=180.0,
        )
        agent = IFBenchAgent(client)
        return client, agent, model, None

    model = args.model or VLLM_MODEL
    client = OpenAIChatClient(
        base_url=VLLM_BASE_URL,
        model=model,
        timeout=180.0,
    )
    agent = IFBenchAgent(client, request_extra=NO_THINK_EXTRA)
    return client, agent, model, NO_THINK_EXTRA


async def main() -> None:
    args = parse_args()
    client, agent, model, propose_extra = build_client_and_agent(args)

    print(f"Ping {args.provider} endpoint ({model})...")
    pong = await client.chat(
        ChatRequest(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            extra={"max_tokens": 8, "temperature": 0.0},
        )
    )
    print(f"  {args.provider} responded: {pong.content!r}")

    train = load_ifbench(DATA_DIR, "train")
    val_full = load_ifbench(DATA_DIR, "val")
    val = val_full if VAL_SUBSAMPLE is None else val_full[:VAL_SUBSAMPLE]
    print(f"Loaded train={len(train)} val={len(val)} (full={len(val_full)})")

    rollout = make_gepa_rollout(
        agent=agent, metric=ifbench_metric, feedback_fn=ifbench_feedback, llm=client
    )
    reflect = GEPAReflect()
    propose = make_gepa_propose(
        llm=client,
        max_tokens=1024,
        temperature=0.7,
        extra=propose_extra,
    )
    evaluate = make_gepa_evaluate(
        agent=agent, metric=ifbench_metric, llm=client, sampler=FullSampler(val)
    )

    pool = AppendOnlyPool(seed=SEED)
    runtime = GEPARuntime(
        pool=pool,
        sampler=RandomSampler(train, k=MINIBATCH, seed=SEED),
        rollout=rollout,
        reflect=reflect,
        propose=propose,
        evaluate=evaluate,
        initial_artifact=PromptArtifact(text=INITIAL_PROMPT),
        budget=args.budget,
        selection_strategy="pareto_front",
    )

    print(
        f"Running GEPA for {args.budget} iters "
        f"(1 bootstrap eval + accept-on-improvement evolution)"
    )
    await runtime.run()

    print(
        f"Accepted={runtime.accepted_iterations} "
        f"Rejected={runtime.rejected_iterations}"
    )

    print(f"\n=== Final pool (version={pool.version}) ===")
    for i, sc in enumerate(pool._scores):
        label = "bootstrap" if i == 0 else f"iter {i}"
        head = pool._artifacts[i].text.splitlines()[0][:80]
        print(f"  v{i + 1} [{label}]  score={sc:.3f}  artifact[0]: {head!r}")

    best = max(range(len(pool._scores)), key=lambda i: pool._scores[i])
    print(
        f"\n=== Best: v{best + 1} score={pool._scores[best]:.3f} "
        f"vs bootstrap={pool._scores[0]:.3f} ==="
    )

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
