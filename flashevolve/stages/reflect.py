from typing import Callable

from ..llm.base import ChatRequest, LLMClient
from .base import Critique, Stage, Trajectory

ReflectRender = Callable[[Trajectory], ChatRequest]
ReflectParse = Callable[[str, Trajectory], Critique]


def _default_parse(text: str, trajectory: Trajectory) -> Critique:
    return Critique(trajectory=trajectory, text=text)


class Reflect(Stage[Trajectory, Critique]):
    """LLM critique of a trajectory. Does not produce a new artifact.

    User supplies a ``render`` function that turns a ``Trajectory`` into
    a full ``ChatRequest`` (so model / extra params can be set per call),
    and an optional ``parse`` function that extracts a structured
    ``Critique`` from the LLM response. Default parse wraps the raw text.

    Reflective Async stale-patch workers are also Reflect nodes — same
    abstraction handles framework-level rebase and algorithm-level
    critique (DESIGN.md §0 / §2).
    """

    def __init__(
        self,
        render: ReflectRender,
        llm: LLMClient,
        *,
        parse: ReflectParse | None = None,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.render = render
        self.llm = llm
        self.parse = parse or _default_parse

    async def process(self, item: Trajectory) -> Critique:
        response = await self.llm.chat(self.render(item))
        return self.parse(response.content, item)
