from typing import Any, Callable

from ..agents.base import Agent
from ..llm.base import LLMClient
from ..samplers.base import Sampler
from .base import Candidate, ScoredCandidate, Stage
from .rollout import Metric


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


Aggregator = Callable[[list[float]], float]


class Evaluate(Stage[Candidate, ScoredCandidate]):
    """Run a candidate artifact on a holdout set and aggregate per-sample scores.

    Each ``process(candidate)`` call pulls a fresh batch from ``sampler``
    (so different candidates may see different val samples if the sampler
    is randomized; use ``FullSampler`` for deterministic full-valset
    evaluation, ``RandomSampler`` for cheap stochastic checks).

    Output ``ScoredCandidate.score`` is ``aggregator(per_sample_scores)``;
    default aggregator is mean. ``signals`` carries ``per_sample_scores``,
    ``num_samples``, and ``agent_signals`` for downstream telemetry /
    admission gating.
    """

    def __init__(
        self,
        agent: Agent,
        metric: Metric,
        llm: LLMClient,
        sampler: Sampler,
        *,
        aggregator: Aggregator = _mean,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.agent = agent
        self.metric = metric
        self.llm = llm
        self.sampler = sampler
        self.aggregator = aggregator

    async def process(self, item: Candidate) -> ScoredCandidate:
        samples = await self.sampler.sample()
        results = await self.agent.run(item.artifact, samples, llm=self.llm)
        outputs = [r.output for r in results]
        per_sample = [self.metric(s, o) for s, o in zip(samples, outputs)]
        score = self.aggregator(per_sample)
        return ScoredCandidate(
            candidate=item,
            score=score,
            signals={
                "per_sample_scores": per_sample,
                "num_samples": len(samples),
                "agent_signals": [r.signals for r in results],
            },
        )
