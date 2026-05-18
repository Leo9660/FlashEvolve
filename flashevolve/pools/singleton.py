from ..artifacts.base import Artifact
from ..stages.base import Admission, Candidate, ScoredCandidate
from .base import Pool


class SingletonPool(Pool[Artifact]):
    """Single-artifact pool with overwrite admission semantics.

    Keeps only the latest admitted artifact. Useful for algorithms like
    ACE where there is one evolving playbook rather than a frontier of
    competing candidates.
    """

    compatible_staleness = frozenset({"full"})

    def __init__(self) -> None:
        self._artifact: Artifact | None = None
        self._version = 0
        self._last_score: float | None = None

    @property
    def version(self) -> int:
        return self._version

    @property
    def artifact(self) -> Artifact | None:
        return self._artifact

    @property
    def last_score(self) -> float | None:
        return self._last_score

    async def admit(self, admitted: Admission) -> bool:
        if isinstance(admitted, ScoredCandidate):
            artifact = admitted.candidate.artifact
            self._last_score = admitted.score
        elif isinstance(admitted, Candidate):
            artifact = admitted.artifact
            self._last_score = None
        else:
            raise TypeError(
                "SingletonPool.admit expects Candidate or ScoredCandidate; "
                f"got {type(admitted).__name__}"
            )
        self._artifact = artifact
        self._version += 1
        return True

    async def select_parent(
        self, *, strategy: str | None = None
    ) -> tuple[int, Artifact]:
        if self._artifact is None:
            raise RuntimeError(
                "SingletonPool is empty; seed the runtime before select_parent()"
            )
        if strategy not in (None, "latest", "only"):
            raise ValueError(
                f"unknown strategy {strategy!r}; supported: latest, only"
            )
        return self._version, self._artifact
