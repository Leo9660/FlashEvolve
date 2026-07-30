from dataclasses import dataclass
from typing import Callable

from ..artifacts.base import Artifact
from ..artifacts.prompt import PromptArtifact
from ..llm.base import ChatRequest, LLMClient
from .base import Candidate, Stage

# Reflective-repair prompt — verbatim from the rebuttal; do not edit.
REFLECTIVE_REPAIR_PROMPT = (
    "You are given a batch of stale instruction proposals that were superseded before full "
    "validation. Extract only general and task agnostic principles from them and integrate these "
    "principles into the current best instruction. Ignore proposals containing specific entities, "
    "topics, numbers, domain details, incoherent content, or duplicated information. Do not copy "
    "or concatenate proposals. Retain only transferable principles, such as output rules, "
    "reasoning structure, constraint handling, and style. Output only the revised instruction."
)


@dataclass(frozen=True)
class StaleBatch:
    """The current frontier + stale candidates to rebase onto it."""

    current_version: int
    current_artifact: Artifact
    stale: list[Candidate]


StaleRender = Callable[[StaleBatch], ChatRequest]
StaleParse = Callable[[str, StaleBatch], Artifact]


class ReflectivePatch(Stage[StaleBatch, Candidate]):
    """Reflective staleness repair: one LLM call distills stale proposals onto
    the current frontier, yielding a candidate anchored to the current version
    that re-enters Evaluate -> admit. Prompt is REFLECTIVE_REPAIR_PROMPT;
    render/parse default to the prompt-artifact encoding."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        prompt: str = REFLECTIVE_REPAIR_PROMPT,
        model: str | None = None,
        render: StaleRender | None = None,
        parse: StaleParse | None = None,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.llm = llm
        self.prompt = prompt
        self.model = model
        self.render = render or self._default_render
        self.parse = parse or self._default_parse

    def _default_render(self, batch: StaleBatch) -> ChatRequest:
        proposals = "\n".join(
            f"{i + 1}. {c.artifact.text}" for i, c in enumerate(batch.stale)  # type: ignore[attr-defined]
        )
        return ChatRequest(
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": (
                        f"current_instruction:\n{batch.current_artifact.text}\n\n"  # type: ignore[attr-defined]
                        f"stale_proposals:\n{proposals}"
                    ),
                },
            ],
            model=self.model,
        )

    def _default_parse(self, text: str, batch: StaleBatch) -> Artifact:
        return PromptArtifact(text=text.strip())

    async def process(self, item: StaleBatch) -> Candidate:
        response = await self.llm.chat(self.render(item))
        new_artifact = self.parse(response.content, item)
        # Anchor the repaired candidate to the CURRENT frontier version, so
        # downstream staleness is 0 and it competes as a fresh proposal.
        return Candidate(
            parent_version=item.current_version,
            parent_artifact=item.current_artifact,
            artifact=new_artifact,
            critique=item.stale[-1].critique,
        )
