from abc import ABC, abstractmethod
from typing import Any


class Sampler(ABC):
    """Source of samples for a stage.

    Called by the runtime (for Rollout's training minibatch) or by the
    Evaluate stage (for its holdout set). Stateful when ordering matters
    (sequential cursor, random-without-replacement seen set). Concrete
    subclasses document state semantics and seed handling.

    All sampling is LLM-free; ``async`` is for uniformity with stage IO,
    so future implementations doing remote fetches / async I/O fit the
    same interface.
    """

    @abstractmethod
    async def sample(self) -> list[Any]:
        """Return one batch of samples. Batch size is determined by the
        subclass's construction params (typically a fixed ``k``).
        """
