from typing import Callable

from ..artifacts.base import Artifact
from ..llm.base import ChatRequest, LLMClient
from .base import Candidate, Critique, Stage

ProposeRender = Callable[[Critique], ChatRequest]
ProposeParse = Callable[[str, Critique], Artifact]


class Propose(Stage[Critique, Candidate]):
    """LLM-driven generation of a new artifact from a critique.

    User supplies a ``render`` function turning the ``Critique`` into a
    full ``ChatRequest`` (so model / extra params can be set per call)
    and a ``parse`` function extracting the new ``Artifact`` from the LLM
    response. Both are required — there is no sensible default parse
    because the encoding of the artifact in the LLM output is
    task-specific (raw text, ``` blocks, JSON, structured edits, etc.).

    Provenance: the framework wraps the new artifact in a ``Candidate``
    carrying the parent version, the parent artifact, and the originating
    critique. Downstream Evaluate / Pool.admit can trace the lineage.
    """

    def __init__(
        self,
        render: ProposeRender,
        parse: ProposeParse,
        llm: LLMClient,
        *,
        workers: int = 1,
    ) -> None:
        super().__init__(workers=workers)
        self.render = render
        self.parse = parse
        self.llm = llm

    async def process(self, item: Critique) -> Candidate:
        response = await self.llm.chat(self.render(item))
        new_artifact = self.parse(response.content, item)
        return Candidate(
            parent_version=item.trajectory.version,
            parent_artifact=item.trajectory.artifact,
            artifact=new_artifact,
            critique=item,
        )
