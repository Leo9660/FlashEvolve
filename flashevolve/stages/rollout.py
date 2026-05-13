from typing import Any, Callable

from ..agents.base import Agent
from ..feedback import Feedback
from ..llm.base import LLMClient
from .base import Sampled, Stage, Trajectory

Metric = Callable[[Any, Any], float]


class Rollout(Stage[Sampled, Trajectory]):
    """Run an agent on a minibatch, score each sample, and (optionally)
    attach per-sample feedback.

    Composition: ``agent.run(artifact, samples, llm=...)`` produces one
    ``AgentResult`` per sample; ``metric(sample, output)`` produces one
    float score per sample. If a ``Feedback`` is supplied, it is invoked
    after metric to add per-sample diagnostic strings.

    The resulting ``Trajectory.signals`` carries fixed keys:
        - ``"scores"`` : list[float], one per sample
        - ``"feedback"`` : list[str] | None, one per sample (None if no
          ``Feedback`` configured)
        - ``"agent_signals"`` : list[dict], per-sample agent metadata
    """

    def __init__(
        self,
        agent: Agent,
        metric: Metric,
        llm: LLMClient,
        *,
        feedback: Feedback | None = None,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.agent = agent
        self.metric = metric
        self.llm = llm
        self.feedback = feedback

    async def process(self, item: Sampled) -> Trajectory:
        results = await self.agent.run(item.artifact, item.samples, llm=self.llm)
        outputs = [r.output for r in results]
        scores = [self.metric(s, o) for s, o in zip(item.samples, outputs)]
        feedback = (
            await self.feedback.generate(item.samples, outputs, scores)
            if self.feedback is not None
            else None
        )
        agent_signals = [r.signals for r in results]
        return Trajectory(
            version=item.version,
            artifact=item.artifact,
            samples=item.samples,
            outputs=outputs,
            signals={
                "scores": scores,
                "feedback": feedback,
                "agent_signals": agent_signals,
            },
        )
