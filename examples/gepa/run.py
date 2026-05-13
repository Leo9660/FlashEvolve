"""Toy GEPA run on FlashEvolve.

End-to-end demo: GEPA pipeline (Rollout → GEPAReflect → Propose → Evaluate)
on a 5-sample trivia task. Used as a smoke test to verify framework
abstractions hold under a real (vLLM-served) LLM.

The trivia dataset is hard-coded in this file (``TRAINSET``) so the demo
is self-contained. For the framework's dataset conventions and a
real-task example (IFBench), see ``datasets.py`` in this directory.

Prereqs:
    bash scripts/start_server.sh   # vLLM @ :8000 (Qwen3-8B by default)

Run:
    python -m examples.gepa.run    # from the repo root
"""

import asyncio

from flashevolve.agents import Agent, AgentResult
from flashevolve.artifacts import PromptArtifact
from flashevolve.llm import ChatRequest, OpenAIChatClient
from flashevolve.pools import AppendOnlyPool
from flashevolve.runtime import SyncRuntime
from flashevolve.samplers import FullSampler, RandomSampler

from examples.gepa import (
    GEPAReflect,
    make_gepa_evaluate,
    make_gepa_propose,
    make_gepa_rollout,
)

# ─── Config ───────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-8B"

BUDGET = 3
MINIBATCH = 3
SELECTION_STRATEGY = "pareto_front"
SEED = 42

TRAINSET = [
    {"q": "What color is the sky on a clear day?", "gold": "blue"},
    {"q": "What is 2+2?", "gold": "4"},
    {"q": "What is the capital of France?", "gold": "Paris"},
    {"q": "How many continents are there?", "gold": "7"},
    {"q": "What is H2O commonly known as?", "gold": "water"},
]
VALSET = TRAINSET  # toy: train = val (paper-faithful eval uses a held-out set)

INITIAL_PROMPT = "Answer."


# ─── Task adapters ────────────────────────────────────────────────────────


class TriviaAgent(Agent):
    """Single LLM call per sample: system = artifact.text, user = sample['q']."""

    def __init__(self, llm: OpenAIChatClient) -> None:
        self.llm = llm

    async def run(self, artifact, samples, *, llm):
        client = llm or self.llm

        async def one(s):
            r = await client.chat(
                ChatRequest(
                    messages=[
                        {"role": "system", "content": artifact.text},
                        {"role": "user", "content": s["q"]},
                    ],
                    extra={
                        "max_tokens": 64,
                        "temperature": 0.0,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    },
                )
            )
            return AgentResult(output=r.content.strip(), signals={})

        return await asyncio.gather(*(one(s) for s in samples))


def trivia_metric(sample: dict, output: str) -> float:
    return 1.0 if sample["gold"].lower() in output.lower() else 0.0


def trivia_feedback(sample: dict, output: str, score: float) -> str:
    if score >= 1.0:
        return f"Q: {sample['q']!r} → answer {output[:80]!r} matched gold {sample['gold']!r}."
    return (
        f"Q: {sample['q']!r} → answer {output[:80]!r} did NOT contain gold "
        f"{sample['gold']!r}. Be more direct."
    )


# ─── Run ──────────────────────────────────────────────────────────────────


async def main() -> None:
    client = OpenAIChatClient(
        base_url=BASE_URL, model=MODEL, api_key="EMPTY", timeout=180.0
    )

    # Quick ping
    print("• Ping vLLM...")
    pong = await client.chat(
        ChatRequest(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            extra={"max_tokens": 8, "temperature": 0.0},
        )
    )
    print(f"  vLLM responded: {pong.content!r}")

    # Wire GEPA pipeline
    agent = TriviaAgent(client)
    rollout = make_gepa_rollout(
        agent=agent, metric=trivia_metric, feedback_fn=trivia_feedback, llm=client
    )
    reflect = GEPAReflect()
    propose = make_gepa_propose(
        llm=client,
        max_tokens=1024,
        temperature=0.7,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
    evaluate = make_gepa_evaluate(
        agent=agent, metric=trivia_metric, llm=client, sampler=FullSampler(VALSET)
    )

    pool = AppendOnlyPool(seed=SEED)
    runtime = SyncRuntime(
        pool=pool,
        sampler=RandomSampler(TRAINSET, k=MINIBATCH, seed=SEED),
        rollout=rollout,
        reflect=reflect,
        propose=propose,
        evaluate=evaluate,
        initial_artifact=PromptArtifact(text=INITIAL_PROMPT),
        budget=BUDGET,
        selection_strategy=SELECTION_STRATEGY,
    )

    print(f"• Running GEPA for {BUDGET} iters (1 bootstrap eval + {BUDGET} evolve)")
    await runtime.run()

    print(f"\n=== Final pool (version={pool.version}) ===")
    for i, (a, sc, ps) in enumerate(
        zip(pool._artifacts, pool._scores, pool._per_sample_scores)
    ):
        label = "bootstrap" if i == 0 else f"iter {i}"
        print(f"\nv{i + 1} [{label}]  score={sc:.2f}  per_sample={ps}")
        print(f"  artifact:")
        for line in a.text.splitlines()[:6]:
            print(f"    | {line}")
        if len(a.text.splitlines()) > 6:
            print(f"    | ... ({len(a.text.splitlines()) - 6} more lines)")

    best_idx = max(range(len(pool._scores)), key=lambda i: pool._scores[i])
    print(
        f"\n=== Best: v{best_idx + 1} with score={pool._scores[best_idx]:.2f} "
        f"vs bootstrap={pool._scores[0]:.2f} ==="
    )

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
