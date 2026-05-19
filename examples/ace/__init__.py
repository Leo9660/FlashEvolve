"""ACE on FlashEvolve.

This package preserves ACE's playbook-centric artifact, stage-specific
prompts, and playbook update mechanics while expressing the algorithm
through FlashEvolve's Artifact / Pool / Stage abstractions.
"""

from .pipeline import (
    ACEExactMatchMetric,
    ACEGeneratorSolver,
    ACEReflect,
    make_ace_propose,
)
from .playbook import make_empty_playbook

__all__ = [
    "ACEExactMatchMetric",
    "ACEGeneratorSolver",
    "ACEReflect",
    "make_ace_propose",
    "make_empty_playbook",
]
