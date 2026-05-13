"""End-to-end smoke test: OpenAIChatClient → SyncRuntime against vLLM.

Prereqs:
    bash scripts/start_server.sh   # vLLM @ :8000 (Qwen3-8B by default)

Run:
    python -m tests.smoke_vllm     # from the repo root
"""

import asyncio
import re

from flashevolve.agents import Agent, AgentResult
from flashevolve.artifacts import PromptArtifact
from flashevolve.feedback import RuleBasedFeedback
from flashevolve.llm import ChatRequest, OpenAIChatClient
from flashevolve.pools import AppendOnlyPool
from flashevolve.runtime import SyncRuntime
from flashevolve.samplers import FullSampler, RandomSampler
from flashevolve.stages import (
    Critique,
    Evaluate,
    Propose,
    Rollout,
    Stage,
    Trajectory,
)

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-8B"


# ─── Step 1: bare connectivity check ──────────────────────────────────────


async def verify_connection(client: OpenAIChatClient) -> None:
    resp = await client.chat(
        ChatRequest(
            messages=[{"role": "user", "content": "Reply with only the word PONG."}],
            extra={"max_tokens": 16, "temperature": 0.0},
        )
    )
    print(f"  → response: {resp.content!r}")
    assert "PONG" in resp.content.upper(), f"expected PONG, got {resp.content!r}"
    print("  ✓ connection + chat completion works")


# ─── Step 2: toy GEPA loop on the real LLM ────────────────────────────────


class LLMAgent(Agent):
    """Use artifact.text as system prompt; sample['q'] as user msg."""

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
                    extra={"max_tokens": 64, "temperature": 0.0},
                )
            )
            return AgentResult(output=r.content.strip(), signals={})

        return await asyncio.gather(*(one(s) for s in samples))


def metric(s, o) -> float:
    return 1.0 if s["gold"].lower() in o.lower() else 0.0


def feedback_fn(s, o, sc) -> str:
    return (
        f"Q: {s['q']!r}\n  answer: {o!r}\n  gold: {s['gold']!r}  score={sc:.1f}"
    )


class FormatReflect(Stage[Trajectory, Critique]):
    """No-LLM Reflect — joins per-sample feedback into Critique.text.
    GEPA-style pass-through."""

    def __init__(self, *, workers: int = 1) -> None:
        super().__init__(workers=workers)

    async def process(self, item: Trajectory) -> Critique:
        text = "\n".join(item.signals["feedback"])
        return Critique(trajectory=item, text=text)


def propose_render(c: Critique) -> ChatRequest:
    return ChatRequest(
        messages=[
            {
                "role": "system",
                "content": "You improve system prompts based on test-case feedback.",
            },
            {
                "role": "user",
                "content": (
                    f"Current system prompt:\n```\n{c.trajectory.artifact.text}\n```\n\n"
                    f"Feedback from test cases:\n{c.text}\n\n"
                    "Write a single improved system prompt that would score better. "
                    "Put the new prompt between triple backticks."
                ),
            },
        ],
        extra={"max_tokens": 256, "temperature": 0.7},
    )


def propose_parse(text: str, c: Critique) -> PromptArtifact:
    m = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    body = m.group(1).strip() if m else text.strip()
    return PromptArtifact(text=body)


async def main() -> None:
    print("[1/3] vLLM connectivity")
    client = OpenAIChatClient(
        base_url=BASE_URL, model=MODEL, api_key="EMPTY", timeout=120.0
    )
    await verify_connection(client)

    print("\n[2/3] Wire SyncRuntime")
    trainset = [
        {"q": "What color is the sky on a clear day?", "gold": "blue"},
        {"q": "What is 2+2?", "gold": "4"},
        {"q": "What is the capital of France?", "gold": "Paris"},
    ]
    valset = trainset

    agent = LLMAgent(client)
    rollout = Rollout(
        agent=agent,
        metric=metric,
        llm=client,
        feedback=RuleBasedFeedback(feedback_fn),
    )
    reflect = FormatReflect()
    propose = Propose(render=propose_render, parse=propose_parse, llm=client)
    evaluate = Evaluate(
        agent=agent, metric=metric, llm=client, sampler=FullSampler(valset)
    )

    initial = PromptArtifact(text="Reply with one word.")
    pool = AppendOnlyPool(seed=42)
    runtime = SyncRuntime(
        pool=pool,
        sampler=RandomSampler(trainset, k=3, seed=42),
        rollout=rollout,
        reflect=reflect,
        propose=propose,
        evaluate=evaluate,
        initial_artifact=initial,
        budget=2,
        selection_strategy="pareto_front",
    )

    print("[3/3] Run 2 iters of SyncRuntime")
    await runtime.run()

    print(f"\n=== Final pool (version={pool.version}) ===")
    for i, (a, sc, ps) in enumerate(
        zip(pool._artifacts, pool._scores, pool._per_sample_scores)
    ):
        print(f"v{i + 1}: score={sc:.2f}  per_sample={ps}")
        print(f"      artifact: {a.text[:140]!r}")

    await client.aclose()
    print("\nALL GREEN")


if __name__ == "__main__":
    asyncio.run(main())
