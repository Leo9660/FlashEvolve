from ..pools.base import Pool
from ..samplers.base import Sampler
from ..stages.base import Sampled, Stage
from ..stages.evaluate import Evaluate
from ..stages.propose import Propose
from ..stages.rollout import Rollout


class SyncRuntime:
    """One-iteration-at-a-time runtime. No pipeline parallelism, no
    queues, no staleness handling — strictly:

        for i in range(budget):
            v, art    = pool.select_parent(strategy)
            samples   = sampler.sample()
            traj      = rollout(Sampled(v, art, samples))
            critique  = reflect(traj)
            cand      = propose(critique)
            scored    = evaluate(cand)
            pool.admit(scored)

    Used as the correctness baseline for any later async runtime, and as
    the simplest path to a runnable end-to-end smoke test. Inside-stage
    parallelism (``stage.workers``) is ignored here — each ``stage.process``
    is awaited sequentially.

    The base artifact is evaluated once and seeded into the pool before
    the loop starts, so ``pool.select_parent`` always has something to
    return on iteration 0.
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

    async def _seed_pool(self) -> None:
        from ..stages.base import Candidate, Critique, Trajectory

        # Evaluate the initial artifact and admit it through the standard
        # ``Pool.admit`` path (the ABC method). We construct a minimal
        # Candidate where parent and child are both the initial artifact,
        # so the seed entry is structurally identical to later admissions.
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

        for _ in range(self.budget):
            version, artifact = await self.pool.select_parent(
                strategy=self.selection_strategy
            )
            samples = await self.sampler.sample()
            sampled = Sampled(version=version, artifact=artifact, samples=samples)

            trajectory = await self.rollout.process(sampled)
            critique = await self.reflect.process(trajectory)
            candidate = await self.propose.process(critique)
            scored = await self.evaluate.process(candidate)
            await self.pool.admit(scored)

        return self.pool
