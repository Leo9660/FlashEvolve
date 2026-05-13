from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
    """Marker base for evolvable artifacts. Subclasses are frozen dataclasses."""
