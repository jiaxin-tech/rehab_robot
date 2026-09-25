"""Controlled positive/negative cohorts for the MyoLeg personalization study.

This module is deliberately separate from the frozen V1 virtual-patient
cohort.  It does not alter the MuJoCo model, V1 manifests, or the learner's
observation interface.  A profile wraps the native MyoLeg response with a
declared synthetic response field so that the gate can be tested in two
conditions:

``COMMON_OPTIMUM``
    Every profile has the same native response landscape.  This is the null
    condition used to measure false personalization.

``DIVERGENT_OPTIMA``
    Profiles in two predeclared arms apply distinct, smooth candidate-space
    response fields.  Development and confirmatory centers are different. The
    field is a software stress test, not a claim about patient physiology or a
    MyoLeg parameter identified from data.

The wrapper exposes only ``requested(point)`` and therefore can be passed to
the existing benchmark ``Environment``/``run_sequence`` code.  It never
exposes a full candidate landscape or an oracle to the learner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_controlled_positive_cohort_v2"
    / "MYOLEG_CONTROLLED_POSITIVE_COHORT_V2_MANIFEST.json"
)
COHORT_ID = "MYOLEG_CONTROLLED_POSITIVE_COHORT_V2"
SCHEMA_VERSION = 2
SCENARIOS = ("COMMON_OPTIMUM", "DIVERGENT_OPTIMA")
SPLITS = ("DEVELOPMENT", "CONFIRMATORY")
COMMON_OPTIMUM = "COMMON_OPTIMUM"
DIVERGENT_OPTIMA = "DIVERGENT_OPTIMA"
FEATURE_BOUNDS = ((-4.0, 4.0), (-4.0, 4.0), (-2.0, 2.0))
REFERENCE_FEATURES = np.zeros(3, dtype=float)
DEFAULT_ROOT_SEED = 20260923


class RequestedBackend(Protocol):
    """Minimal private simulator protocol required by the wrapper."""

    def requested(self, point: Any) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ControlledProfile:
    """One predeclared synthetic profile.

    ``center_features`` uses the same normalized candidate coordinates as
    ``Point.features``: beta flex/extend divided by 0.03 and share shift
    divided by 0.05.  ``field_strength`` is dimensionless and is frozen in
    the manifest.  The gain is one for the reference candidate, so all
    profile comparisons retain the existing reference normalization.
    """

    profile_id: str
    scenario: str
    split: str
    arm: str
    seed: int
    center_features: tuple[float, float, float]
    field_strength: float
    response_model: str = "multiplicative_log_gain_relative_to_reference"
    truth_scope: str = "CONTROLLED_SYNTHETIC_STRESS_TEST"

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown controlled scenario: {self.scenario}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown controlled split: {self.split}")
        if not self.profile_id.startswith("CTRL_"):
            raise ValueError("controlled profile IDs must start with CTRL_")
        center = np.asarray(self.center_features, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center_features must be finite with three coordinates")
        for value, (lower, upper) in zip(center, FEATURE_BOUNDS):
            if not lower <= value <= upper:
                raise ValueError("controlled profile center lies outside candidate domain")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("profile seed must be a non-negative integer")
        if not np.isfinite(self.field_strength) or self.field_strength < 0.0:
            raise ValueError("field_strength must be finite and non-negative")
        if self.field_strength > 0.20:
            raise ValueError("field_strength exceeds the frozen stress-test limit")
        if self.scenario == "COMMON_OPTIMUM" and self.field_strength != 0.0:
            raise ValueError("COMMON_OPTIMUM profiles must have zero field strength")
        if self.scenario == "DIVERGENT_OPTIMA" and self.field_strength <= 0.0:
            raise ValueError("DIVERGENT_OPTIMA profiles require a positive field strength")

    def gain(self, features: np.ndarray | list[float] | tuple[float, ...]) -> float:
        """Return the deterministic synthetic response gain for one candidate."""
        x = np.asarray(features, dtype=float)
        if x.shape != (3,) or not np.isfinite(x).all():
            raise ValueError("candidate features must be finite with three coordinates")
        if self.scenario == "COMMON_OPTIMUM":
            return 1.0
        center = np.asarray(self.center_features, dtype=float)
        # Subtract the reference distance so gain(reference) is exactly one.
        relative_distance = float(np.sum((x - center) ** 2) - np.sum(center**2))
        gain = float(np.exp(self.field_strength * relative_distance))
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError("controlled response field produced an invalid gain")
        return gain

    def as_manifest_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["center_features"] = list(self.center_features)
        return record


def _profile(
    profile_id: str,
    scenario: str,
    split: str,
    arm: str,
    seed: int,
    center: tuple[float, float, float],
    strength: float,
) -> ControlledProfile:
    return ControlledProfile(
        profile_id=profile_id,
        scenario=scenario,
        split=split,
        arm=arm,
        seed=seed,
        center_features=center,
        field_strength=strength,
    )


def build_profiles(root_seed: int = DEFAULT_ROOT_SEED) -> tuple[ControlledProfile, ...]:
    """Build the frozen development/confirmatory profile list.

    The seed is recorded for provenance and profile identity.  The current
    stress field is deterministic and does not draw random values; this keeps
    the positive-control landscape fixed before any algorithm result exists.
    """
    if not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    profiles: list[ControlledProfile] = []
    arm_centers = {
        "DEVELOPMENT": {
            "A": ((-4.0, 4.0, -2.0), (-3.0, 3.0, -1.5), (-2.0, 2.0, -1.0), (-1.0, 1.0, -0.5)),
            "B": ((4.0, -4.0, 2.0), (3.0, -3.0, 1.5), (2.0, -2.0, 1.0), (1.0, -1.0, 0.5)),
        },
        "CONFIRMATORY": {
            "A": ((-3.5, 3.5, -1.75), (-2.5, 2.5, -1.25), (-1.5, 1.5, -0.75), (-0.5, 0.5, -0.25)),
            "B": ((3.5, -3.5, 1.75), (2.5, -2.5, 1.25), (1.5, -1.5, 0.75), (0.5, -0.5, 0.25)),
        },
    }
    for split, split_tag, offset in (("DEVELOPMENT", "DEV", 0), ("CONFIRMATORY", "CONF", 100)):
        for index in range(4):
            profiles.append(_profile(
                f"CTRL_COMMON_{split_tag}_{index:02d}", "COMMON_OPTIMUM", split,
                "COMMON", root_seed + offset + index, (0.0, 0.0, 0.0), 0.0,
            ))
        for arm in ("A", "B"):
            for index, center in enumerate(arm_centers[split][arm]):
                profiles.append(_profile(
                    f"CTRL_DIVERGENT_{arm}_{split_tag}_{index:02d}",
                    "DIVERGENT_OPTIMA", split, arm, root_seed + offset + 20 + (4 if arm == "B" else 0) + index,
                    center, 0.01,
                ))
    validate_profiles(profiles)
    return tuple(profiles)


def validate_profile(profile: ControlledProfile) -> None:
    if not np.isclose(profile.gain(REFERENCE_FEATURES), 1.0, atol=0.0, rtol=0.0):
        raise ValueError(f"reference gain is not one for {profile.profile_id}")
    probe = np.asarray([
        [FEATURE_BOUNDS[0][0], FEATURE_BOUNDS[1][0], FEATURE_BOUNDS[2][0]],
        [FEATURE_BOUNDS[0][1], FEATURE_BOUNDS[1][1], FEATURE_BOUNDS[2][1]],
        [0.0, 0.0, 0.0],
    ])
    gains = np.asarray([profile.gain(x) for x in probe])
    if not np.isfinite(gains).all() or np.any(gains <= 0.0):
        raise ValueError(f"invalid controlled gains for {profile.profile_id}")


def validate_profiles(
    profiles: tuple[ControlledProfile, ...] | list[ControlledProfile],
    *,
    require_complete_splits: bool = True,
) -> None:
    """Check the complete cohort design before any simulator is run."""
    profiles = tuple(profiles)
    if not profiles:
        raise ValueError("controlled cohort cannot be empty")
    ids = [p.profile_id for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("controlled profile IDs must be unique")
    for profile in profiles:
        validate_profile(profile)
    for scenario in SCENARIOS:
        rows = [p for p in profiles if p.scenario == scenario]
        if not rows:
            raise ValueError(f"missing scenario {scenario}")
        if set(p.split for p in rows) != set(SPLITS):
            if require_complete_splits:
                raise ValueError(f"scenario {scenario} must contain both splits")
    positive = [p for p in profiles if p.scenario == "DIVERGENT_OPTIMA"]
    if {p.arm for p in positive} != {"A", "B"}:
        raise ValueError("positive cohort must contain both divergent arms")
    centers = {tuple(p.center_features) for p in positive}
    if len(centers) < 2:
        raise ValueError("positive cohort arms must have distinct centers")


def _manifest_payload(profiles: tuple[ControlledProfile, ...], root_seed: int) -> dict[str, Any]:
    return {
        "cohort_id": COHORT_ID,
        "schema_version": SCHEMA_VERSION,
        "root_seed": root_seed,
        "task_scope": "controlled synthetic stress test; no physiological interpretation",
        "base_truth": {
            "backend": "native MyoLeg V1 prescribed-state simulator",
            "base_subject_id": "MYOLEG_NATIVE_P0",
            "v1_cohort_unchanged": True,
            "learner_oracle_access": False,
            "candidate_landscape_access": False,
            "truth_isolation": "wrapper releases only requested(point) response",
        },
        "candidate_feature_definition": {
            "coordinates": ["beta_flex/0.03", "beta_extend/0.03", "share_shift/0.05"],
            "bounds": [list(bound) for bound in FEATURE_BOUNDS],
            "reference_features": [0.0, 0.0, 0.0],
        },
        "scenarios": {
            "COMMON_OPTIMUM": {
                "role": "null control for false-positive personalization",
                "field_strength": 0.0,
                "expected_behavior": "all profiles share the native response landscape",
            },
            "DIVERGENT_OPTIMA": {
                "role": "positive control for detectable subject-by-trajectory interaction",
                "field_strength": 0.01,
                "expected_behavior": "arm A and arm B receive opposite predeclared smooth fields",
                "interpretation": "software stress test only; not a patient or muscle truth claim",
            },
        },
        "response_formula_version": "multiplicative_log_gain_relative_to_reference_v2",
        "splits": {
            "DEVELOPMENT": "may be used to freeze algorithm and gate parameters",
            "CONFIRMATORY": "sealed until the algorithm and all thresholds are frozen",
        },
        "profiles": [p.as_manifest_record() for p in profiles],
    }


def manifest(root_seed: int = DEFAULT_ROOT_SEED) -> dict[str, Any]:
    profiles = build_profiles(root_seed)
    payload = _manifest_payload(profiles, root_seed)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["manifest_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_manifest(path: str | Path = DEFAULT_MANIFEST, root_seed: int = DEFAULT_ROOT_SEED) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest(root_seed)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def load_profiles(
    split: str | None = None,
    *,
    path: str | Path = DEFAULT_MANIFEST,
) -> tuple[ControlledProfile, ...]:
    """Load profiles while keeping confirmatory profiles opt-in."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"controlled cohort manifest not found: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("cohort_id") != COHORT_ID:
        raise ValueError("CONTROLLED_COHORT_ID_MISMATCH")
    expected = document.get("manifest_fingerprint_sha256")
    unsigned = dict(document)
    unsigned.pop("manifest_fingerprint_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if expected != hashlib.sha256(canonical).hexdigest():
        raise ValueError("CONTROLLED_COHORT_MANIFEST_FINGERPRINT_MISMATCH")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("CONTROLLED_COHORT_SCHEMA_VERSION_MISMATCH")
    records = document["profiles"]
    selected_records = records if split is None else [record for record in records if record.get("split") == split]
    profiles = tuple(
        ControlledProfile(**{**record, "center_features": tuple(record["center_features"])})
        for record in selected_records
    )
    validate_profiles(profiles, require_complete_splits=split is None)
    return profiles


class ControlledCohortBackend:
    """Private response wrapper for one controlled profile.

    The native backend remains responsible for all MuJoCo dynamics and cache
    validation.  This class only applies the frozen stress field after a
    requested trace is returned and releases no other candidate information.
    """

    def __init__(self, base_backend: RequestedBackend, profile: ControlledProfile):
        validate_profile(profile)
        self.base_backend = base_backend
        self.profile = profile

    def requested(self, point: Any) -> np.ndarray:
        base = np.asarray(self.base_backend.requested(point), dtype=float)
        if base.ndim != 2 or base.shape[1] != 2 or not np.isfinite(base).all():
            raise ValueError("INVALID_BASE_MYOLEG_RESPONSE")
        gain = self.profile.gain(point.features)
        shaped = base * gain
        if not np.isfinite(shaped).all() or shaped.shape != base.shape:
            raise ValueError("INVALID_CONTROLLED_RESPONSE")
        if np.allclose(np.asarray(point.features, dtype=float), REFERENCE_FEATURES, atol=0.0, rtol=0.0):
            if not np.array_equal(shaped, base):
                raise ValueError("REFERENCE_RESPONSE_WAS_CHANGED")
        return shaped

    def metadata(self) -> dict[str, Any]:
        return {
            "cohort_id": COHORT_ID,
            "profile_id": self.profile.profile_id,
            "scenario": self.profile.scenario,
            "split": self.profile.split,
            "truth_scope": self.profile.truth_scope,
            "oracle_exposed": False,
            "candidate_landscape_exposed": False,
            "response_model": self.profile.response_model,
            "field_strength": self.profile.field_strength,
        }


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    args = parser.parse_args()
    path = write_manifest(args.manifest, args.root_seed)
    print(json.dumps({"manifest": str(path), "profiles": len(build_profiles(args.root_seed))}, ensure_ascii=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "COHORT_ID", "COMMON_OPTIMUM", "ControlledCohortBackend", "ControlledProfile", "DEFAULT_MANIFEST",
    "DIVERGENT_OPTIMA", "SCENARIOS", "SPLITS", "build_profiles", "load_profiles",
    "manifest", "validate_profile", "validate_profiles", "write_manifest",
]
