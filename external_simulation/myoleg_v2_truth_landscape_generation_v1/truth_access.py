"""Explicit information-reveal boundary for the frozen offline truth artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .replay_api import replay_subject_candidate


ROOT = Path(__file__).resolve().parents[2]
FINAL_MANIFEST = ROOT / "external_simulation_audits/myoleg_v2_truth_landscape_generation_v1/MYOLEG_V2_TRUTH_LANDSCAPE_V1_MANIFEST.json"


def query(subject_id: str, candidate_id: str) -> dict[str, Any]:
    """Reveal one executed candidate by deterministic prescribed replay only."""

    return replay_subject_candidate(subject_id, candidate_id)


class OracleLandscapeAccess:
    """Post-freeze evaluator access; never use inside a learner or selector."""

    _ALLOWED_PURPOSES = {"post_hoc_evaluation", "oracle", "regret", "personalization_analysis"}

    def __init__(self, purpose: str):
        if purpose not in self._ALLOWED_PURPOSES:
            raise PermissionError(f"oracle landscape purpose is not allowed: {purpose}")
        if not FINAL_MANIFEST.is_file():
            raise RuntimeError("truth landscape is not formally frozen")
        self.manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
        if not self.manifest.get("landscape_frozen_before_oracle_reveal", False):
            raise RuntimeError("oracle access attempted before landscape freeze")

    def subject_rows(self, subject_id: str) -> dict[str, np.ndarray]:
        chunks = [row for row in self.manifest["chunks"] if row["subject_id"] == subject_id]
        if not chunks:
            raise KeyError(subject_id)
        columns: dict[str, list[np.ndarray]] = {}
        for chunk in sorted(chunks, key=lambda row: row["candidate_start_rank"]):
            path = ROOT / chunk["path"]
            with np.load(path, allow_pickle=False) as shard:
                for key in shard.files:
                    columns.setdefault(key, []).append(np.asarray(shard[key]))
        return {key: np.concatenate(values) for key, values in columns.items()}

