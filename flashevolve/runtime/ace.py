from __future__ import annotations

from ..pools.base import Pool
from ..samplers.base import Sampler
from ..stages.base import Candidate, Critique, Sampled, Stage, Trajectory
from ..stages.evaluate import Evaluate
from ..stages.propose import Propose
from ..stages.rollout import Rollout


class ACERuntime:
    """ACE-specific synchronous runtime.

    ACE seeds the pool without held-out evaluation, always admits the latest
    proposed artifact, and only runs held-out evaluation every ``eval_every``
    iterations (plus an optional final evaluation).
    """

    def __init__(
        self,
        *,
        pool: Pool,
        sampler: Sampler,
        rollout: Rollout,
        reflect: Stage,
        propose: Propose,
        evaluate: Evaluate | None,
        initial_artifact,
        budget: int,
        eval_every: int = 1,
        selection_strategy: str = "latest",
        final_evaluate: bool = True,
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        if eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {eval_every}")
        self.pool = pool
        self.sampler = sampler
        self.rollout = rollout
        self.reflect = reflect
        self.propose = propose
        self.evaluate = evaluate
        self.initial_artifact = initial_artifact
        self.budget = budget
        self.eval_every = eval_every
        self.selection_strategy = selection_strategy
        self.final_evaluate = final_evaluate
        self.evaluation_history: list[dict[str, object]] = []
        self.iterations_run = 0

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
        await self.pool.admit(bootstrap_candidate)

    async def _evaluate_candidate(
        self,
        candidate: Candidate,
        *,
        iteration: int,
        reason: str,
    ) -> None:
        if self.evaluate is None:
            return
        scored = await self.evaluate.process(candidate)
        self.evaluation_history.append(
            {
                "iteration": iteration,
                "reason": reason,
                "score": scored.score,
                "num_samples": scored.signals.get("num_samples", 0),
            }
        )

    async def run(self) -> Pool:
        if self.pool.version == 0:
            await self._seed_pool()

        last_candidate: Candidate | None = None
        for iteration in range(1, self.budget + 1):
            version, artifact = await self.pool.select_parent(
                strategy=self.selection_strategy
            )
            samples = await self.sampler.sample()
            if not samples:
                break

            sampled = Sampled(version=version, artifact=artifact, samples=samples)
            trajectory = await self.rollout.process(sampled)
            critique = await self.reflect.process(trajectory)
            candidate = await self.propose.process(critique)
            await self.pool.admit(candidate)

            last_candidate = candidate
            self.iterations_run = iteration
            if iteration % self.eval_every == 0:
                await self._evaluate_candidate(
                    candidate, iteration=iteration, reason="periodic"
                )

        if (
            self.final_evaluate
            and self.evaluate is not None
            and last_candidate is not None
            and (
                not self.evaluation_history
                or self.evaluation_history[-1]["iteration"] != self.iterations_run
            )
        ):
            await self._evaluate_candidate(
                last_candidate,
                iteration=self.iterations_run,
                reason="final",
            )

        return self.pool
