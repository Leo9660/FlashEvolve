import random
from typing import ClassVar

from ..artifacts.base import Artifact
from ..stages.base import ScoredCandidate
from .base import Pool


def _pareto_front(scores_matrix: list[list[float]]) -> list[int]:
    """Indices of programs not dominated by any other on per-sample scores.

    Program i is dominated by j iff scores[j] >= scores[i] element-wise and
    strict on at least one element. Programs with mismatched score-vector
    lengths are skipped (treated as un-rankable).
    """
    n = len(scores_matrix)
    if n == 0:
        return []
    m = len(scores_matrix[0])
    front: list[int] = []
    for i in range(n):
        if len(scores_matrix[i]) != m:
            continue
        dominated = False
        for j in range(n):
            if i == j or len(scores_matrix[j]) != m:
                continue
            sj, si = scores_matrix[j], scores_matrix[i]
            if all(sj[k] >= si[k] for k in range(m)) and any(sj[k] > si[k] for k in range(m)):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


class AppendOnlyPool(Pool[Artifact]):
    """Append-only storage of admitted artifacts. Every ``admit`` succeeds
    and bumps ``version``; nothing is ever evicted. Selection strategy is
    configurable.

    Tracks ``per_sample_scores`` for each admitted artifact (read from
    ``ScoredCandidate.signals['per_sample_scores']``) so that
    ``strategy="pareto_front"`` can derive the front on demand. Programs
    admitted without per-sample scores get an empty list and are excluded
    from Pareto-front consideration but still selectable by other strategies.

    Supported strategies:
        - ``"latest"``: most recently admitted
        - ``"greedy"``: argmax over aggregate score (``ScoredCandidate.score``)
        - ``"pareto_front"``: uniform pick from the Pareto front over
          per-sample scores (GEPA's default)
        - ``"random"``: uniform over all members

    Note: ``"pareto_front"`` requires all admitted candidates to have
    been evaluated on the same set of samples (same length, same order
    of ``per_sample_scores``). Use a deterministic sampler in Evaluate
    (e.g., ``FullSampler``) to satisfy this.
    """

    compatible_staleness: ClassVar[frozenset[str]] = frozenset({"full"})

    def __init__(self, *, seed: int | None = None) -> None:
        self._artifacts: list[Artifact] = []
        self._scores: list[float] = []
        self._per_sample_scores: list[list[float]] = []
        self._version: int = 0
        self._rng = random.Random(seed)

    @property
    def version(self) -> int:
        return self._version

    async def admit(self, scored: ScoredCandidate) -> bool:
        self._artifacts.append(scored.candidate.artifact)
        self._scores.append(scored.score)
        per_sample = scored.signals.get("per_sample_scores", [])
        self._per_sample_scores.append(list(per_sample))
        self._version += 1
        return True

    async def select_parent(
        self, *, strategy: str | None = None
    ) -> tuple[int, Artifact]:
        if not self._artifacts:
            raise RuntimeError(
                "AppendOnlyPool is empty; call seed() before select_parent()"
            )
        s = strategy or "latest"
        if s == "latest":
            idx = len(self._artifacts) - 1
        elif s == "greedy":
            idx = max(range(len(self._scores)), key=lambda i: self._scores[i])
        elif s == "random":
            idx = self._rng.randrange(len(self._artifacts))
        elif s == "pareto_front":
            front = _pareto_front(self._per_sample_scores)
            if not front:
                # fallback to greedy if no per-sample data
                idx = max(range(len(self._scores)), key=lambda i: self._scores[i])
            else:
                idx = self._rng.choice(front)
        else:
            raise ValueError(
                f"unknown strategy {s!r}; supported: "
                "latest, greedy, pareto_front, random"
            )
        return self._version, self._artifacts[idx]

    def __len__(self) -> int:
        return len(self._artifacts)
