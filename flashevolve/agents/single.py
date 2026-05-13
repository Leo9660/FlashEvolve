import asyncio
from typing import Any, Generic

from ..llm.base import LLMClient
from ..solvers.base import Solver
from .base import A, Agent, AgentResult


class SingleCallAgent(Agent[A], Generic[A]):
    """Agent that performs one LLM call per sample, dispatched in parallel."""

    def __init__(self, solver: Solver[A]):
        self.solver = solver

    async def run(
        self,
        artifact: A,
        samples: list[Any],
        *,
        llm: LLMClient,
    ) -> list[AgentResult]:
        async def _one(sample: Any) -> AgentResult:
            request = self.solver.render(artifact, sample)
            response = await llm.chat(request)
            parsed = self.solver.parse(response, sample)
            return AgentResult(output=parsed.output, signals=parsed.signals)

        return await asyncio.gather(*(_one(s) for s in samples))
