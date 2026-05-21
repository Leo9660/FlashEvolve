from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import islice
from typing import Any

from flashevolve.pools.base import Pool
from flashevolve.stages.base import Admission, ScoredCandidate

from .pipeline import OpenEvolveContext, OpenEvolvePromptArtifact


@dataclass
class PromptRecord:
    artifact: OpenEvolvePromptArtifact
    score: float
    metrics: dict[str, float]
    per_sample_scores: list[float]
    island: int
    version: int
    parent_id: str | None
    artifacts: dict[str, Any]


class OpenEvolvePromptPool(Pool[OpenEvolvePromptArtifact]):
    """Small OpenEvolve-style population for prompt optimization.

    This is intentionally lighter than upstream OpenEvolve, but it preserves:
    - multiple islands
    - MAP-Elites-style cell elites over prompt features
    - elite/archive-biased parent selection
    - migration between neighboring islands
    """

    compatible_staleness = frozenset({"full"})

    def __init__(
        self,
        *,
        feature_bins: int = 10,
        num_islands: int = 4,
        archive_size: int = 64,
        num_top_programs: int = 5,
        num_diverse_programs: int = 3,
        num_inspirations: int = 5,
        exploration_ratio: float = 0.3,
        exploitation_ratio: float = 0.3,
        migration_interval: int = 20,
        migration_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.feature_bins = feature_bins
        self.num_islands = num_islands
        self.archive_size = archive_size
        self.num_top_programs = num_top_programs
        self.num_diverse_programs = num_diverse_programs
        self.num_inspirations = num_inspirations
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self._rng = random.Random(seed)

        self._records: dict[str, PromptRecord] = {}
        self._islands: list[list[str]] = [[] for _ in range(num_islands)]
        self._feature_maps: list[dict[tuple[int, int], str]] = [
            {} for _ in range(num_islands)
        ]
        self._archive: list[str] = []
        self._best_id: str | None = None
        self._version = 0
        self._current_island = 0
        self._last_selected_island = 0
        self._island_generations = [0 for _ in range(num_islands)]
        self._last_migration_generation = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def best_record(self) -> PromptRecord | None:
        if self._best_id is None:
            return None
        return self._records.get(self._best_id)

    async def select_parent(
        self, *, strategy: str | None = None
    ) -> tuple[int, OpenEvolvePromptArtifact]:
        if not self._records:
            raise RuntimeError("OpenEvolvePromptPool is empty")

        island = self._current_island
        self._current_island = (self._current_island + 1) % self.num_islands
        self._last_selected_island = island
        if not self._islands[island] and self._best_id is not None:
            self._islands[island].append(self._best_id)

        if strategy not in (None, "openevolve"):
            raise ValueError(f"unsupported strategy {strategy!r}; expected 'openevolve'")

        candidates = self._records_from_ids(self._islands[island]) or list(self._records.values())
        roll = self._rng.random()
        if roll < self.exploration_ratio:
            chosen = self._rng.choice(candidates)
        elif roll < self.exploration_ratio + self.exploitation_ratio:
            archive = self._archive_candidates(island) or candidates
            chosen = self._rng.choice(archive)
        else:
            weights = [max(self._score(record), 0.001) for record in candidates]
            chosen = self._rng.choices(candidates, weights=weights, k=1)[0]

        return self._version, chosen.artifact

    async def admit(self, admitted: Admission) -> bool:
        if not isinstance(admitted, ScoredCandidate):
            raise TypeError(
                "OpenEvolvePromptPool.admit expects ScoredCandidate; "
                f"got {type(admitted).__name__}"
            )

        scored = admitted
        artifact = scored.candidate.artifact
        island = self._last_selected_island
        metrics = dict(scored.signals.get("metrics", {}))
        if "combined_score" not in metrics:
            metrics["combined_score"] = scored.score
        record = PromptRecord(
            artifact=artifact,
            score=scored.score,
            metrics=metrics,
            per_sample_scores=list(scored.signals.get("per_sample_scores", [])),
            island=island,
            version=self._version + 1,
            parent_id=scored.candidate.parent_artifact.program_id,
            artifacts=dict(scored.signals.get("artifacts", {})),
        )

        self._records[artifact.program_id] = record
        self._islands[island].append(artifact.program_id)
        self._update_feature_elite(record)
        self._update_archive()
        self._update_best(record)

        self._version += 1
        self._island_generations[island] += 1
        if self._should_migrate():
            self._migrate()
        return True

    def build_context(self, version: int, parent: OpenEvolvePromptArtifact) -> OpenEvolveContext:
        parent_record = self._records[parent.program_id]
        island = self._last_selected_island
        island_records = self._records_from_ids(self._islands[island])
        island_records = [
            record for record in island_records if record.artifact.program_id != parent.program_id
        ]
        island_records.sort(key=self._score, reverse=True)

        top_records = island_records[: self.num_top_programs]
        top_ids = {record.artifact.program_id for record in top_records}
        used_ids = set(top_ids)
        diverse_records: list[PromptRecord] = []
        for cell_id in self._feature_maps[island].values():
            if cell_id == parent.program_id or cell_id in used_ids:
                continue
            diverse_records.append(self._records[cell_id])
            used_ids.add(cell_id)
            if len(diverse_records) >= self.num_diverse_programs:
                break

        inspiration_records = list(top_records)
        for record in diverse_records:
            if record.artifact.program_id not in top_ids:
                inspiration_records.append(record)
        if len(inspiration_records) < self.num_inspirations:
            remainder = [
                record
                for record in island_records
                if record.artifact.program_id not in used_ids
            ]
            self._rng.shuffle(remainder)
            inspiration_records.extend(
                islice(remainder, self.num_inspirations - len(inspiration_records))
            )

        best = self.best_record
        return OpenEvolveContext(
            version=version,
            parent=parent,
            island=island,
            parent_score=self._score(parent_record),
            top_programs=[self._summarize_record(record) for record in top_records],
            inspirations=[
                self._summarize_record(record)
                for record in inspiration_records[: self.num_inspirations]
            ],
            global_best=self._summarize_record(best) if best is not None else None,
        )

    def _score(self, record: PromptRecord) -> float:
        return float(record.metrics.get("combined_score", record.score))

    def _records_from_ids(self, ids: list[str]) -> list[PromptRecord]:
        seen: set[str] = set()
        records: list[PromptRecord] = []
        for program_id in ids:
            if program_id in seen:
                continue
            record = self._records.get(program_id)
            if record is None:
                continue
            seen.add(program_id)
            records.append(record)
        return records

    def _archive_candidates(self, island: int) -> list[PromptRecord]:
        local_ids = set(self._islands[island])
        local_archive = [
            self._records[program_id]
            for program_id in self._archive
            if program_id in self._records and program_id in local_ids
        ]
        if local_archive:
            return local_archive
        return [
            self._records[program_id]
            for program_id in self._archive
            if program_id in self._records
        ]

    def _update_best(self, record: PromptRecord) -> None:
        if self._best_id is None:
            self._best_id = record.artifact.program_id
            return
        best = self._records[self._best_id]
        if self._score(record) > self._score(best):
            self._best_id = record.artifact.program_id

    def _update_archive(self) -> None:
        ranked = sorted(self._records.values(), key=self._score, reverse=True)
        self._archive = [record.artifact.program_id for record in ranked[: self.archive_size]]

    def _feature_coords(self, record: PromptRecord) -> tuple[int, int]:
        prompt_length = int(record.metrics.get("prompt_length", len(record.artifact.text)))
        reasoning = float(record.metrics.get("reasoning_strategy", 0.0))
        length_bin = min(prompt_length // 250, self.feature_bins - 1)
        reasoning_bin = min(int(reasoning * self.feature_bins), self.feature_bins - 1)
        return (length_bin, reasoning_bin)

    def _update_feature_elite(self, record: PromptRecord) -> None:
        island_map = self._feature_maps[record.island]
        coords = self._feature_coords(record)
        incumbent_id = island_map.get(coords)
        if incumbent_id is None:
            island_map[coords] = record.artifact.program_id
            return
        incumbent = self._records[incumbent_id]
        if self._score(record) > self._score(incumbent):
            island_map[coords] = record.artifact.program_id

    def _should_migrate(self) -> bool:
        return (
            max(self._island_generations) - self._last_migration_generation
        ) >= self.migration_interval

    def _migrate(self) -> None:
        if self.num_islands < 2:
            return
        for island, ids in enumerate(self._islands):
            records = self._records_from_ids(ids)
            if not records:
                continue
            records.sort(key=self._score, reverse=True)
            migrants = records[: max(1, int(len(records) * self.migration_rate))]
            for neighbor in ((island + 1) % self.num_islands, (island - 1) % self.num_islands):
                for migrant in migrants:
                    program_id = migrant.artifact.program_id
                    if program_id not in self._islands[neighbor]:
                        self._islands[neighbor].append(program_id)
        self._last_migration_generation = max(self._island_generations)

    def _summarize_record(self, record: PromptRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        failed_examples = list(record.artifacts.get("failed_examples", []))[:2]
        return {
            "program_id": record.artifact.program_id,
            "score": self._score(record),
            "metrics": dict(record.metrics),
            "prompt": record.artifact.text,
            "changes_description": record.artifact.changes_description,
            "failed_examples": failed_examples,
        }
