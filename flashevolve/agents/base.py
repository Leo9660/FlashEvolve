from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..artifacts.base import Artifact
from ..llm.base import LLMClient

A = TypeVar("A", bound=Artifact)


@dataclass(frozen=True)
class AgentResult:
    output: Any
    signals: dict[str, Any] = field(default_factory=dict)


class Agent(ABC, Generic[A]):
    """End-to-end mapping from a batch of samples to a batch of results.

    Agents may perform multiple LLM calls per sample (tool loops, aggregation,
    retrieval, etc.) and may parallelize internally. Unlike Solver (pure
    transform), Agent is the I/O boundary — `run` is async and the runtime
    supplies the LLMClient.
    """

    @abstractmethod
    async def run(
        self,
        artifact: A,
        samples: list[Any],
        *,
        llm: LLMClient,
    ) -> list[AgentResult]: ...
