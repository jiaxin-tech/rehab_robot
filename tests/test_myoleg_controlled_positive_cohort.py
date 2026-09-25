from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest

from lower_limb_sim.myoleg_benchmark.controlled_positive_cohort import (
    COMMON_OPTIMUM,
    DIVERGENT_OPTIMA,
    ControlledCohortBackend,
    ControlledProfile,
    build_profiles,
    load_profiles,
    manifest,
    validate_profiles,
    write_manifest,
)


class _Point:
    def __init__(self, candidate_id: str, features: tuple[float, float, float]):
        self.candidate_id = candidate_id
        self.features = np.asarray(features, dtype=float)


class _Backend:
    def requested(self, point: _Point) -> np.ndarray:
        return np.ones((4, 2), dtype=float)


def test_manifest_round_trip_and_split_isolation() -> None:
    directory = Path(".cache") / f"controlled-cohort-test-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        path = write_manifest(directory / "cohort.json")
        assert len(load_profiles("DEVELOPMENT", path=path)) == 12
        assert len(load_profiles("CONFIRMATORY", path=path)) == 12
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["base_truth"]["v1_cohort_unchanged"] is True
        assert document["base_truth"]["learner_oracle_access"] is False
        assert document["task_scope"].startswith("controlled synthetic")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_common_null_and_divergent_positive_fields_are_predeclared() -> None:
    profiles = build_profiles(17)
    validate_profiles(profiles)
    common = [p for p in profiles if p.scenario == COMMON_OPTIMUM]
    positive = [p for p in profiles if p.scenario == DIVERGENT_OPTIMA]
    assert len(common) == 8
    assert len(positive) == 16
    assert all(p.gain((1.0, -1.0, 1.0)) == 1.0 for p in common)
    assert len({tuple(p.center_features) for p in positive}) == 16
    assert {p.arm for p in positive} == {"A", "B"}
    assert all(p.gain(p.center_features) < 1.0 for p in positive)


def test_wrapper_changes_only_requested_positive_response_and_preserves_reference() -> None:
    positive = next(p for p in build_profiles() if p.scenario == DIVERGENT_OPTIMA)
    backend = ControlledCohortBackend(_Backend(), positive)
    reference = backend.requested(_Point("reference", (0.0, 0.0, 0.0)))
    preferred = backend.requested(_Point("preferred", positive.center_features))
    assert np.array_equal(reference, np.ones((4, 2)))
    assert np.all(preferred < reference)
    assert backend.metadata()["oracle_exposed"] is False
    assert backend.metadata()["candidate_landscape_exposed"] is False


def test_wrapper_allows_evaluator_rereads_and_rejects_nonfinite_base_trace() -> None:
    common = next(p for p in build_profiles() if p.scenario == COMMON_OPTIMUM)
    backend = ControlledCohortBackend(_Backend(), common)
    point = _Point("same", (0.0, 0.0, 0.0))
    first = backend.requested(point)
    second = backend.requested(point)
    assert np.array_equal(first, second)

    class BadBackend:
        def requested(self, point: _Point) -> np.ndarray:
            return np.full((4, 2), np.nan)

    with pytest.raises(ValueError, match="INVALID_BASE_MYOLEG_RESPONSE"):
        ControlledCohortBackend(BadBackend(), common).requested(point)


def test_manifest_fingerprint_detects_tampering() -> None:
    directory = Path(".cache") / f"controlled-cohort-tamper-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        path = write_manifest(directory / "cohort.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        document["profiles"][0]["arm"] = "TAMPERED"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="MANIFEST_FINGERPRINT_MISMATCH"):
            load_profiles(path=path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
