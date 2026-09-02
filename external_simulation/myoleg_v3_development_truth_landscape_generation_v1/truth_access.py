"""Explicit query-only access boundary for MyoLeg-V3 development truth."""

from __future__ import annotations

from typing import Any

from external_simulation.myoleg_v3_development_truth_landscape_generation_v1.replay_api import replay_v3_subject_candidate


_ALLOWED_PURPOSES = frozenset({"executed_candidate_query", "deterministic_replay_validation"})


def query(subject_id: str, candidate_id: str, *, purpose: str) -> dict[str, Any]:
    """Reveal only a specifically executed development candidate replay."""

    if purpose not in _ALLOWED_PURPOSES:
        raise PermissionError(f"purpose is not query-authorized: {purpose}")
    return replay_v3_subject_candidate(subject_id, candidate_id)

