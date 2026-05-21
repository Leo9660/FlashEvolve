from __future__ import annotations

from ..pools.base import Pool
from ..samplers.base import Sampler
from ..stages.base import Candidate, Critique, Sampled, ScoredCandidate, Stage, Trajectory
from ..stages.evaluate import Evaluate
from ..stages.propose import Propose
from ..stages.rollout import Rollout


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


class GEPARuntime:
    """GEPA-specific synchronous runtime.

    GEPA evaluates the bootstrap artifact once to seed the Pareto-front pool.
    Each iteration proposes a new artifact, reruns rollout on the same sampled
    minibatch, and only evaluates/admits the candidate if its minibatch score
    improves over the parent.
    """

    def __init__(
        self,
        *,
        pool: Pool,
        sampler: Sampler,
        rollout: Rollout,
        reflect: Stage,
        propose: Propose,
        evaluate: Evaluate,
        initial_artifact,
        budget: int,
        selection_strategy: str = "pareto_front",
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        self.pool = pool
        self.sampler = sampler
        self.rollout = rollout
        self.reflect = reflect
        self.propose = propose
        self.evaluate = evaluate
        self.initial_artifact = initial_artifact
        self.budget = budget
        self.selection_strategy = selection_strategy
        self.accepted_iterations = 0
        self.rejected_iterations = 0
        self.acceptance_history: list[dict[str, object]] = []

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

    async def _evaluate_and_admit(self, candidate: Candidate) -> ScoredCandidate:
        scored = await self.evaluate.process(candidate)
        await self.pool.admit(scored)
        return scored

    async def run(self) -> Pool:
        if self.pool.version == 0:
            await self._seed_pool()

        for iteration in range(1, self.budget + 1):
            version, artifact = await self.pool.select_parent(
                strategy=self.selection_strategy
            )
            samples = await self.sampler.sample()
            if not samples:
                break

            parent_sampled = Sampled(version=version, artifact=artifact, samples=samples)
            parent_trajectory = await self.rollout.process(parent_sampled)
            critique = await self.reflect.process(parent_trajectory)
            candidate = await self.propose.process(critique)
            candidate_sampled = Sampled(
                version=version,
                artifact=candidate.artifact,
                samples=samples,
            )
            candidate_trajectory = await self.rollout.process(candidate_sampled)

            parent_score = _mean(parent_trajectory.signals.get("scores", []))
            candidate_score = _mean(candidate_trajectory.signals.get("scores", []))
            improved = candidate_score > parent_score
            print(
                f"[GEPA iter {iteration}] rollout parent={parent_score:.3f} "
                f"candidate={candidate_score:.3f}; "
                f"the score is {'improved' if improved else 'not improved'}; "
                f"{'run evaluate' if improved else 'skip evaluate'}"
            )

            history_item = {
                "iteration": iteration,
                "parent_rollout_score": parent_score,
                "candidate_rollout_score": candidate_score,
                "accepted": improved,
            }

            if improved:
                scored = await self._evaluate_and_admit(candidate)
                history_item["eval_score"] = scored.score
                self.accepted_iterations += 1
                print(f"[GEPA iter {iteration}] evaluate score={scored.score:.3f}")
            else:
                self.rejected_iterations += 1
                print(f"[GEPA iter {iteration}] evaluate skipped (no rollout improvement)")

            self.acceptance_history.append(history_item)

        return self.pool
