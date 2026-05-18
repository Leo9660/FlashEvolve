from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..artifacts.base import Artifact

In = TypeVar("In")
Out = TypeVar("Out")


# --- Stage IO payloads ---------------------------------------------------
#
# Only LLM-heavy work units are Stage subclasses (4 of them). Pool reads
# and writes (the paper's Select / Update boxes) are pool methods invoked
# by the runtime/scheduler at iteration boundaries; ``Sampled`` and
# ``ScoredCandidate`` are still IO types but live at the pipeline head
# and tail, produced/consumed by the runtime rather than by Stage workers.
#
#   runtime  : pool.select_parent + sampler  -> Sampled (queue head)
#   Rollout  : Sampled                        -> Trajectory
#   Reflect  : Trajectory                     -> Critique
#   Propose  : Critique                       -> Candidate
#   Evaluate : Candidate                      -> ScoredCandidate
#   runtime  : ScoredCandidate -> pool.admit  (queue tail)


@dataclass(frozen=True)
class Sampled:
    """Select -> Rollout. Parent artifact bound to a pool version, plus a minibatch."""

    version: int
    artifact: Artifact
    samples: list[Any]


@dataclass(frozen=True)
class Trajectory:
    """Rollout -> Reflect. Agent outputs on the minibatch plus per-sample signals."""

    version: int
    artifact: Artifact
    samples: list[Any]
    outputs: list[Any]
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Critique:
    """Reflect -> Propose. LLM-written critique distilled from a trajectory."""

    trajectory: Trajectory
    text: str
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """Propose -> Evaluate. A proposed new artifact derived from a parent."""

    parent_version: int
    parent_artifact: Artifact
    artifact: Artifact
    critique: Critique


@dataclass(frozen=True)
class ScoredCandidate:
    """Evaluate -> Update. Candidate plus its held-out score."""

    candidate: Candidate
    score: float
    signals: dict[str, Any] = field(default_factory=dict)


Admission = Candidate | ScoredCandidate


# --- Stage base ----------------------------------------------------------


class Stage(ABC, Generic[In, Out]):
    """One stage in the evolution pipeline.

    A stage is a declaration: an IO type pair plus a per-item transform.
    It does NOT own queues, spawn workers, or know about asyncio. The
    runtime reads ``workers`` to decide concurrency and reads ``process``
    to do the work; how items are routed between stages is the runtime's
    business.

    ``workers=1`` declares serial execution; ``workers>1`` declares that
    the runtime may invoke ``process`` concurrently up to N times.
    """

    def __init__(self, *, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        self.workers = workers

    @abstractmethod
    async def process(self, item: In) -> Out:
        """Transform one input item into one output item."""
