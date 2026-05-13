from typing import Any

from .base import Sampler


class FullSampler(Sampler):
    """Returns the entire dataset on every call.

    Used by Evaluate when scoring a candidate on a full valset / testset
    (GEPA's S5-tier full validation). For partial holdout evaluation,
    use ``RandomSampler`` with k < len(dataset).
    """

    def __init__(self, dataset: list[Any]) -> None:
        self.dataset = dataset

    async def sample(self) -> list[Any]:
        return list(self.dataset)
