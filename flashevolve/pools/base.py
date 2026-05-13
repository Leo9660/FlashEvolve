from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from ..artifacts.base import Artifact
from ..stages.base import ScoredCandidate

A = TypeVar("A", bound=Artifact)


class Pool(ABC, Generic[A]):
    """Storage for evolved artifacts plus the rules for picking and admitting.

    Pool is the framework's primary extension point (DESIGN.md §7). The
    runtime calls ``select_parent`` at iteration start and ``admit`` at
    iteration end; both are inline (LLM-free) operations and do not run
    in stage workers. New algorithms with custom selection / admission
    logic subclass Pool — not Stage.

    ``compatible_staleness`` declares which staleness modes the pool
    semantics support (DESIGN.md §7 Finding B). Append-only pools cannot
    support reflective stale-patch; Pareto pools can support all three.
    The runtime auto-warns / falls back when a user composes an
    incompatible (pool, staleness) pair.

    Methods are ``async`` for uniformity with the rest of the framework
    even though default implementations are synchronous; subclasses that
    do real I/O (distributed pool, durable store) can use the seam.
    """

    compatible_staleness: ClassVar[frozenset[str]] = frozenset()

    @property
    @abstractmethod
    def version(self) -> int:
        """Monotonically increasing version, bumped on every state-mutating ``admit``."""

    @abstractmethod
    async def select_parent(self, *, strategy: str | None = None) -> tuple[int, A]:
        """Return ``(version, artifact)`` for the next iteration's parent.

        ``strategy`` is pool-specific; pools document supported strategies
        and may raise ``ValueError`` on unsupported values.
        """

    @abstractmethod
    async def admit(self, scored: ScoredCandidate) -> bool:
        """Try to admit a scored candidate. Return True iff state changed
        (and ``version`` was bumped). Admission policy is pool-specific:
        overwrite always admits, Pareto admits if non-dominated,
        append-only always admits, tournament admits with capacity
        eviction.
        """
