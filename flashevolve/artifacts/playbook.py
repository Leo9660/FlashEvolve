from dataclasses import dataclass

from .base import Artifact


@dataclass(frozen=True)
class PlaybookArtifact(Artifact):
    text: str
