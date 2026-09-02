"""Identity-preserving access to the frozen V3 candidate domain."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
V3_ARTIFACT_DIR = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_v3_trajectory_parameterization_design_v1"
)
V3_TABLE = V3_ARTIFACT_DIR / "V3_KINEMATIC_CANDIDATE_TABLE.csv"
V3_MANIFEST = (
    V3_ARTIFACT_DIR / "MYOLEG_V3_KINEMATIC_CANDIDATE_DOMAIN_V1_MANIFEST.json"
)
V3_PARAMETERIZATION_ID = "P4_BRANCH_AWARE_COORDINATION_FUNCTION_V3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class Candidate:
    """One unchanged V3 beta-grid member."""

    candidate_id: str
    beta_flex: float
    beta_extend: float
    candidate_index: int

    @property
    def beta(self) -> tuple[float, float]:
        return (self.beta_flex, self.beta_extend)

    def as_dict(self) -> dict[str, str | float | int]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "beta_flex": self.beta_flex,
            "beta_extend": self.beta_extend,
        }


class V3CandidateDomain:
    """Read-only view of the frozen 625-point V3 candidate artifact."""

    def __init__(self, candidates: Iterable[Candidate]) -> None:
        ordered = tuple(candidates)
        if not ordered:
            raise ValueError("candidate domain cannot be empty")
        ids = [candidate.candidate_id for candidate in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        self._candidates = ordered
        self._by_id = {candidate.candidate_id: candidate for candidate in ordered}
        references = [
            candidate
            for candidate in ordered
            if candidate.beta_flex == 0.0 and candidate.beta_extend == 0.0
        ]
        if len(references) != 1:
            raise ValueError("V3 domain must contain exactly one [0,0] reference")
        self._reference = references[0]

    @classmethod
    def from_frozen_artifact(cls) -> "V3CandidateDomain":
        manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
        actual_sha = _sha256(V3_TABLE)
        if actual_sha != manifest["candidate_table_sha256"]:
            raise RuntimeError("frozen V3 candidate table SHA-256 mismatch")
        with V3_TABLE.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        candidates = [
            Candidate(
                candidate_id=row["candidate_id"],
                candidate_index=int(row["candidate_index"]),
                beta_flex=float(row["beta_flex"]),
                beta_extend=float(row["beta_extend"]),
            )
            for row in rows
            if row["included"].lower() == "true"
            and row["kinematic_gate_pass"].lower() == "true"
        ]
        if len(candidates) != manifest["included_candidate_count"]:
            raise RuntimeError("frozen V3 candidate count mismatch")
        if [item.candidate_id for item in candidates] != manifest["ordered_candidate_ids"]:
            raise RuntimeError("frozen V3 candidate ordering mismatch")
        return cls(candidates)

    @classmethod
    def regular_grid(
        cls,
        values: Iterable[float],
        *,
        id_prefix: str = "V3_TEST",
    ) -> "V3CandidateDomain":
        """Small identity-compatible domain for unit tests only."""

        axis = tuple(float(value) for value in values)
        candidates = []
        for beta_flex in axis:
            for beta_extend in axis:
                index = len(candidates)
                candidates.append(
                    Candidate(
                        f"{id_prefix}_{index:04d}",
                        beta_flex,
                        beta_extend,
                        index,
                    )
                )
        return cls(candidates)

    @property
    def reference(self) -> Candidate:
        return self._reference

    def by_id(self, candidate_id: str) -> Candidate:
        return self._by_id[candidate_id]

    def nearest(self, beta_flex: float, beta_extend: float) -> Candidate:
        return min(
            self._candidates,
            key=lambda item: (
                (item.beta_flex - beta_flex) ** 2
                + (item.beta_extend - beta_extend) ** 2,
                item.candidate_index,
            ),
        )

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self._candidates)

    def __len__(self) -> int:
        return len(self._candidates)

    def as_tuple(self) -> tuple[Candidate, ...]:
        return self._candidates
