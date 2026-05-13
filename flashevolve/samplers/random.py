import random
from typing import Any

from .base import Sampler


class RandomSampler(Sampler):
    """Independent random sample of ``k`` distinct items per call.

    Reproducible given a ``seed``. ``with_replacement`` controls within-call
    repetition; across calls samples are always independent (so the same
    item may appear in multiple batches).
    """

    def __init__(
        self,
        dataset: list[Any],
        k: int,
        *,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not with_replacement and k > len(dataset):
            raise ValueError(
                f"k={k} exceeds dataset size {len(dataset)} without replacement"
            )
        self.dataset = dataset
        self.k = k
        self.with_replacement = with_replacement
        self.rng = random.Random(seed)

    async def sample(self) -> list[Any]:
        if self.with_replacement:
            return [self.rng.choice(self.dataset) for _ in range(self.k)]
        return self.rng.sample(self.dataset, self.k)
