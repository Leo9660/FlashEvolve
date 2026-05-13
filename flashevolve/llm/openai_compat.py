from typing import Any

from openai import AsyncOpenAI

from .base import ChatRequest, ChatResponse


class OpenAIChatClient:
    """OpenAI-compatible ``LLMClient`` for any OpenAI Chat Completions API
    endpoint — works with the official OpenAI API, vLLM's
    ``vllm.entrypoints.openai.api_server``, llama.cpp's server,
    LM Studio, Together, Anyscale, etc.

    The framework's ``ChatRequest`` maps 1:1 onto the OpenAI request body:
    ``messages`` → ``messages``, ``model`` (optional, falls back to
    instance default) → ``model``, and any keys in ``extra`` are forwarded
    as kwargs (temperature, top_p, max_tokens, response_format, ...).

    Example:
        client = OpenAIChatClient(
            base_url="http://localhost:8000/v1",  # vLLM default
            model="Qwen/Qwen3-8B",
            api_key="EMPTY",                     # vLLM ignores this
            timeout=120.0,
        )
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.default_model = model

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model = request.model or self.default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            **request.extra,
        }
        resp = await self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage else {}
        return ChatResponse(
            content=content,
            raw={
                "usage": usage,
                "model": resp.model,
                "finish_reason": resp.choices[0].finish_reason,
                "id": resp.id,
            },
        )

    async def aclose(self) -> None:
        await self._client.close()
