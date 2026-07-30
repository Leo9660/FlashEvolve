import asyncio

from ..pools.base import Pool
from ..samplers.base import Sampler
from ..stages.base import Candidate, Sampled, Stage
from ..stages.evaluate import Evaluate
from ..stages.propose import Propose
from ..stages.reflect_patch import ReflectivePatch, StaleBatch
from ..stages.rollout import Rollout
from .queues import FIFOQueue, QueueClosed


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class AsyncGEPARuntime:
    """Asynchronous, pipelined counterpart of GEPARuntime.

    Same GEPA loop as GEPARuntime — roll out the parent and the proposed
    candidate on the same minibatch and only full-evaluate a candidate whose
    minibatch score beats the parent (the gate) — but stages run concurrently
    with ``pipeline_depth`` proposals in flight and a pool of ``eval_workers``
    running full validations in parallel (the throughput win).

    Because proposals overlap, a candidate may be stale at admission (the
    frontier advanced since it was sampled). ``staleness_mode`` sets the policy:
        none  = admit as-is
        hard  = discard
        reflective = rebase each stale proposal onto the current frontier
        meta  = buffer stale proposals, then on a genuine accept distill the
                batch's transferable principles onto the frontier.

    Frontier versioning is tracked here (bumped only when an admission advances
    the best score), so the pool is used exactly as GEPARuntime uses it.
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
        max_evals: int | None = None,
        eval_count=None,
        pipeline_depth: int = 16,
        staleness_mode: str = "none",
        repair: ReflectivePatch | None = None,
        eval_workers: int = 8,
        meta_stale_n: int = 4,
        meta_mode: str = "serial",
        selection_strategy: str = "pareto_front",
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        if pipeline_depth < 1:
            raise ValueError(f"pipeline_depth must be >= 1, got {pipeline_depth}")
        if staleness_mode not in ("none", "hard", "reflective", "meta"):
            raise ValueError(f"unknown staleness_mode {staleness_mode!r}")
        if staleness_mode in ("reflective", "meta") and repair is None:
            raise ValueError(f"staleness_mode={staleness_mode!r} requires a repair stage")
        if meta_mode not in ("serial", "parallel"):
            raise ValueError(f"unknown meta_mode {meta_mode!r}")
        self.pool = pool
        self.sampler = sampler
        self.rollout = rollout
        self.reflect = reflect
        self.propose = propose
        self.evaluate = evaluate
        self.initial_artifact = initial_artifact
        self.budget = budget
        self.max_evals = max_evals
        self.eval_count = eval_count or (lambda: 0)
        self.pipeline_depth = pipeline_depth
        self.staleness_mode = staleness_mode
        self.repair = repair
        self.eval_workers = eval_workers
        self.meta_stale_n = meta_stale_n
        self.meta_mode = meta_mode
        self.selection_strategy = selection_strategy
        # stats
        self.stale_seen = 0
        self.stale_discarded = 0
        self.stale_repaired = 0
        self.gated_out = 0
        self.meta_syntheses = 0
        self.meta_buffer_max = 0
        # frontier state (bumped only on an improving admit) + meta buffer
        self._frontier_version = 0
        self._best_score = float("-inf")
        self._best_artifact = initial_artifact
        self._meta_buffer: list[Candidate] = []
        self._n_admit = 0

    # ── seeding ──────────────────────────────────────────────────────────
    async def _seed_pool(self) -> None:
        from ..stages.base import Critique, Trajectory

        traj = Trajectory(
            version=0, artifact=self.initial_artifact, samples=[], outputs=[], signals={}
        )
        cand = Candidate(
            parent_version=0,
            parent_artifact=self.initial_artifact,
            artifact=self.initial_artifact,
            critique=Critique(trajectory=traj, text="<bootstrap>"),
        )
        scored = await self.evaluate.process(cand)
        self._best_score = scored.score
        self._best_artifact = self.initial_artifact
        await self.pool.admit(scored)

    # ── generic stage pump ───────────────────────────────────────────────
    async def _pump(self, stage: Stage, qin: FIFOQueue, qout: FIFOQueue) -> None:
        while True:
            try:
                item = await qin.get()
            except QueueClosed:
                return
            out = await stage.process(item)
            if out is not None:
                await qout.put(out)

    async def _run_stage(self, stage: Stage, qin: FIFOQueue, qout: FIFOQueue) -> None:
        await asyncio.gather(*(self._pump(stage, qin, qout) for _ in range(stage.workers)))
        await qout.close()

    async def _producer(self, q_sampled: FIFOQueue) -> None:
        """Dispatch proposals until the budget (max_evals or `budget` proposals);
        each Sampled is tagged with the CURRENT frontier version for staleness."""
        i = 0
        while True:
            if i >= self.budget:
                break
            if self.max_evals is not None and self.eval_count() >= self.max_evals:
                break
            _, artifact = await self.pool.select_parent(strategy=self.selection_strategy)
            samples = await self.sampler.sample()
            if not samples:
                break
            await q_sampled.put(
                Sampled(version=self._frontier_version, artifact=artifact, samples=samples)
            )
            i += 1
        await q_sampled.close()

    # ── candidate-rollout gate (GEPA's minibatch gate) ───────────────────
    async def _gate(self, q_cand: FIFOQueue, q_scored: FIFOQueue) -> None:
        """Roll out each candidate on its parent's minibatch and ATTACH its score
        as ``(candidate, candidate_score)``. The gate DECISION is applied in
        ``_tail`` AFTER staleness routing (matching async_gepa.py, where staleness
        handling precedes the score gate), so stale proposals are buffered (meta)
        or repaired (reflective) regardless of whether they beat the parent."""
        async def one() -> None:
            while True:
                try:
                    cand = await q_cand.get()
                except QueueClosed:
                    return
                traj = cand.critique.trajectory
                cand_traj = await self.rollout.process(
                    Sampled(version=cand.parent_version, artifact=cand.artifact, samples=traj.samples)
                )
                cand_score = _mean(cand_traj.signals.get("scores", []))
                await q_scored.put((cand, cand_score))
        await asyncio.gather(*(one() for _ in range(self.rollout.workers)))
        await q_scored.close()

    # ── admit helpers ────────────────────────────────────────────────────
    async def _admit(self, candidate: Candidate) -> bool:
        scored = await self.evaluate.process(candidate)
        improved = scored.score > self._best_score
        if improved:
            self._best_score = scored.score
            self._best_artifact = candidate.artifact
            self._frontier_version += 1
        await self.pool.admit(scored)
        self._n_admit += 1
        if improved or self._n_admit % 20 == 0:
            print(
                f"[async {self.staleness_mode}] admit#{self._n_admit} fv{self._frontier_version} "
                f"best={self._best_score:.4f} stale_seen={self.stale_seen} gated_out={self.gated_out} "
                f"repaired={self.stale_repaired} meta_synth={self.meta_syntheses}",
                flush=True,
            )
        return improved

    # ── tail: staleness routing FIRST, then the score gate ───────────────
    async def _tail(self, q_scored: FIFOQueue, q_stale: FIFOQueue,
                    q_meta: FIFOQueue, q_eval: FIFOQueue) -> None:
        while True:
            try:
                cand, cand_score = await q_scored.get()
            except QueueClosed:
                if self.staleness_mode == "meta" and self._meta_buffer:
                    await q_meta.put(self._meta_buffer)
                    self._meta_buffer = []
                await q_stale.close()
                await q_eval.close()
                return
            staleness = self._frontier_version - cand.parent_version
            if staleness > 0:
                self.stale_seen += 1
                if self.staleness_mode == "hard":
                    self.stale_discarded += 1
                    continue
                if self.staleness_mode == "reflective":
                    await q_stale.put(cand)
                    continue
                if self.staleness_mode == "meta":
                    self._meta_buffer.append(cand)
                    self.meta_buffer_max = max(self.meta_buffer_max, len(self._meta_buffer))
                    continue
                # none: fall through to the gate
            # GEPA minibatch gate, applied AFTER staleness (async_gepa.py order):
            # only full-validate a fresh proposal that beats its parent.
            parent_score = _mean(cand.critique.trajectory.signals.get("scores", []))
            if cand_score <= parent_score:
                self.gated_out += 1
                continue
            await q_eval.put(cand)

    async def _eval_worker(self, q_eval: FIFOQueue, q_meta: FIFOQueue) -> None:
        while True:
            try:
                cand = await q_eval.get()
            except QueueClosed:
                return
            improved = await self._admit(cand)
            if self.staleness_mode == "meta" and improved and self._meta_buffer:
                snapshot = self._meta_buffer
                self._meta_buffer = []
                try:
                    await q_meta.put(snapshot)
                except (QueueClosed, RuntimeError):
                    self._meta_buffer = snapshot

    async def _repair_worker(self, q_stale: FIFOQueue) -> None:
        while True:
            try:
                cand = await q_stale.get()
            except QueueClosed:
                return
            repaired = await self.repair.process(  # type: ignore[union-attr]
                StaleBatch(
                    current_version=self._frontier_version,
                    current_artifact=self._best_artifact,
                    stale=[cand],
                )
            )
            await self._admit(repaired)
            self.stale_repaired += 1

    async def _meta_worker(self, q_meta: FIFOQueue) -> None:
        while True:
            try:
                snapshot = await q_meta.get()
            except QueueClosed:
                return
            if not snapshot:
                continue
            cur_v, cur_a = self._frontier_version, self._best_artifact
            n = self.meta_stale_n
            batches = [snapshot[i:i + n] for i in range(0, len(snapshot), n)]
            if self.meta_mode == "serial":
                evolving = cur_a
                for batch in batches:
                    rep = await self.repair.process(  # type: ignore[union-attr]
                        StaleBatch(current_version=cur_v, current_artifact=evolving, stale=batch)
                    )
                    evolving = rep.artifact
                final = Candidate(
                    parent_version=cur_v, parent_artifact=cur_a,
                    artifact=evolving, critique=snapshot[-1].critique,
                )
                await self._admit(final)
                self.meta_syntheses += 1
            else:
                for batch in batches:
                    rep = await self.repair.process(  # type: ignore[union-attr]
                        StaleBatch(current_version=cur_v, current_artifact=cur_a, stale=batch)
                    )
                    await self._admit(rep)
                    self.meta_syntheses += 1

    # ── run ──────────────────────────────────────────────────────────────
    async def run(self) -> Pool:
        if self.pool.version == 0:
            await self._seed_pool()

        q_sampled: FIFOQueue = FIFOQueue(maxsize=self.pipeline_depth)
        q_ptraj: FIFOQueue = FIFOQueue()
        q_crit: FIFOQueue = FIFOQueue()
        q_cand: FIFOQueue = FIFOQueue()
        q_gated: FIFOQueue = FIFOQueue()
        q_eval: FIFOQueue = FIFOQueue()
        q_stale: FIFOQueue = FIFOQueue()
        q_meta: FIFOQueue = FIFOQueue()

        n_repair = self.repair.workers if self.staleness_mode == "reflective" else 0
        n_meta = self.repair.workers if self.staleness_mode == "meta" else 0

        async def pipeline_and_eval() -> None:
            await asyncio.gather(
                self._producer(q_sampled),
                self._run_stage(self.rollout, q_sampled, q_ptraj),
                self._run_stage(self.reflect, q_ptraj, q_crit),
                self._run_stage(self.propose, q_crit, q_cand),
                self._gate(q_cand, q_gated),
                self._tail(q_gated, q_stale, q_meta, q_eval),
                *(self._eval_worker(q_eval, q_meta) for _ in range(self.eval_workers)),
                *(self._repair_worker(q_stale) for _ in range(n_repair)),
            )
            await q_meta.close()

        await asyncio.gather(
            pipeline_and_eval(),
            *(self._meta_worker(q_meta) for _ in range(n_meta)),
        )
        return self.pool
