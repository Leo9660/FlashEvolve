from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, Any]]
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    content: str
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    async def chat(self, request: ChatRequest) -> ChatResponse: ...
