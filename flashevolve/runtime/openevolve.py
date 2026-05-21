from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..pools.base import Pool
from ..stages.base import Candidate, Critique, ScoredCandidate, Stage, Trajectory


class OpenEvolveRuntime:
    """OpenEvolve-style synchronous runtime.

    Faithful to the official algorithm shape:

        seed initial artifact via evaluate -> admit
        for i in range(budget):
            version, parent = pool.select_parent(...)
            context = build_context(version, parent)
            candidate = propose(context)
            scored = evaluate(candidate)
            pool.admit(scored)

    This runtime intentionally skips FlashEvolve's rollout/reflect stages because
    upstream OpenEvolve proposes directly from population context and evaluates the
    child afterward.
    """

    def __init__(
        self,
        *,
        pool: Pool,
        build_context: Callable[[int, Any], Any],
        propose: Stage[Any, Candidate],
        evaluate: Stage[Candidate, ScoredCandidate],
        initial_artifact: Any,
        budget: int,
        selection_strategy: str = "openevolve",
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        self.pool = pool
        self.build_context = build_context
        self.propose = propose
        self.evaluate = evaluate
        self.initial_artifact = initial_artifact
        self.budget = budget
        self.selection_strategy = selection_strategy

    async def _seed_pool(self) -> None:
        bootstrap_traj = Trajectory(
            version=0,
            artifact=self.initial_artifact,
            samples=[],
            outputs=[],
            signals={},
        )
        bootstrap_critique = Critique(trajectory=bootstrap_traj, text="<bootstrap>")
        bootstrap_candidate = Candidate(
            parent_version=0,
            parent_artifact=self.initial_artifact,
            artifact=self.initial_artifact,
            critique=bootstrap_critique,
        )
        scored = await self.evaluate.process(bootstrap_candidate)
        await self.pool.admit(scored)

    async def run(self) -> Pool:
        if self.pool.version == 0:
            await self._seed_pool()

        for iteration in range(1, self.budget + 1):
            version, artifact = await self.pool.select_parent(
                strategy=self.selection_strategy
            )
            context = self.build_context(version, artifact)
            candidate = await self.propose.process(context)
            scored = await self.evaluate.process(candidate)
            await self.pool.admit(scored)
            print(
                f"[OpenEvolve iter {iteration}] evaluate score={scored.score:.3f}"
            )

        return self.pool
