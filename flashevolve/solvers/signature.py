from typing import Any

from ..artifacts.prompt import PromptArtifact
from ..llm.base import ChatRequest, ChatResponse
from .base import Solver, SolverResult

try:
    import dspy as _dspy
    from dspy.adapters.chat_adapter import ChatAdapter as _ChatAdapter

    _DSPY_AVAILABLE = True
except ImportError:
    _dspy = None
    _ChatAdapter = None
    _DSPY_AVAILABLE = False


class SignatureSolver(Solver[PromptArtifact]):
    """DSPy-style structured solver. Matches GEPA's rollout exactly.

    The `signature` is a `dspy.Signature` subclass declaring named input/output
    fields; `artifact.text` is injected as the signature's instructions. Uses
    DSPy's adapter to format messages and parse structured outputs.
    """

    def __init__(self, signature, demos: list[dict[str, Any]] | None = None, adapter=None):
        if not _DSPY_AVAILABLE:
            raise RuntimeError(
                "SignatureSolver requires the optional `dspy` dependency. "
                "Install with `pip install dspy` (tested with 2.6.x)."
            )
        self.signature = signature
        self.demos = demos or []
        self.adapter = adapter or _ChatAdapter()

    def render(self, artifact: PromptArtifact, sample: Any) -> ChatRequest:
        if not isinstance(sample, dict):
            raise TypeError(
                f"SignatureSolver expects sample to be a dict of input-field values, "
                f"got {type(sample).__name__}"
            )
        sig = self.signature.with_instructions(artifact.text)
        messages = self.adapter.format(sig, self.demos, sample)
        return ChatRequest(messages=messages)

    def parse(self, response: ChatResponse, sample: Any) -> SolverResult:
        parsed = self.adapter.parse(self.signature, response.content)
        return SolverResult(output=parsed, signals={"raw": response.raw})
