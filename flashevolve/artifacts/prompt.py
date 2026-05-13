from dataclasses import dataclass

from .base import Artifact


@dataclass(frozen=True)
class PromptArtifact(Artifact):
    text: str
