from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..artifacts.base import Artifact
from ..llm.base import ChatRequest, ChatResponse

A = TypeVar("A", bound=Artifact)


@dataclass(frozen=True)
class SolverResult:
    output: Any
    signals: dict[str, Any] = field(default_factory=dict)


class Solver(ABC, Generic[A]):
    """Pure transform between (artifact, sample) and LLM request/response.

    Solvers must not perform I/O. The runtime owns LLM invocation; the solver
    declares request shape (`render`) and response parsing (`parse`) only.
    """

    @abstractmethod
    def render(self, artifact: A, sample: Any) -> ChatRequest: ...

    @abstractmethod
    def parse(self, response: ChatResponse, sample: Any) -> SolverResult: ...
