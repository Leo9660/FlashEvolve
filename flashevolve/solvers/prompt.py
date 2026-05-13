from typing import Any

from ..artifacts.prompt import PromptArtifact
from ..llm.base import ChatRequest, ChatResponse
from .base import Solver, SolverResult


class PromptSolver(Solver[PromptArtifact]):
    def __init__(self, user_template: str = "{input}"):
        self.user_template = user_template

    def render(self, artifact: PromptArtifact, sample: Any) -> ChatRequest:
        if isinstance(sample, dict):
            user_text = self.user_template.format(**sample)
        else:
            user_text = self.user_template.format(sample)
        return ChatRequest(
            messages=[
                {"role": "system", "content": artifact.text},
                {"role": "user", "content": user_text},
            ]
        )

    def parse(self, response: ChatResponse, sample: Any) -> SolverResult:
        return SolverResult(output=response.content, signals={"raw": response.raw})
