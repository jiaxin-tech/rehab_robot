"""Declared resistance-interaction stress test for the MyoLeg policy gate.

This is a controlled analytical response field.  Its parameters have a
mechanical interpretation (where resistance starts, how speed changes it, and
how resistance is split between hip and knee), but it is not a patient model,
muscle simulation, or native MyoLeg evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "external_simulation_audits/myoleg_controlled_resistance_interaction_v1/MYOLEG_CONTROLLED_RESISTANCE_INTERACTION_V1_MANIFEST.json"
COHORT_ID = "MYOLEG_CONTROLLED_RESISTANCE_INTERACTION_V1"
SCENARIOS = ("COMMON_RESISTANCE", "DIVERGENT_RESISTANCE")
SPLITS = ("DEVELOPMENT", "CONFIRMATORY")
DURATION_SCALES = (0.9, 1.0, 1.1)
ASSISTANCE_TIMINGS = (0.25, 0.50, 0.75)
HIP_SHARES = (0.25, 0.50, 0.75)
REFERENCE_FEATURES = np.asarray((1.0, 0.50, 0.50), dtype=float)


@dataclass(frozen=True)
class ResistanceProfile:
    profile_id: str
    scenario: str
    split: str
    arm: str
    seed: int
    resistance_onset_phase: float
    velocity_sensitivity: float
    hip_resistance_share: float
    field_strength: float = 1.0
    response_model: str = "angle_onset_velocity_sensitive_load_share"
    truth_scope: str = "CONTROLLED_SYNTHETIC_MECHANISM_STRESS_TEST"

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS or self.split not in SPLITS:
            raise ValueError("UNKNOWN_RESISTANCE_PROFILE_SCOPE")
        if not self.profile_id.startswith("RESIST_"):
            raise ValueError("RESISTANCE_PROFILE_ID_PREFIX")
        values = (self.resistance_onset_phase, self.velocity_sensitivity,
                  self.hip_resistance_share, self.field_strength)
        if not np.isfinite(values).all():
            raise ValueError("NONFINITE_RESISTANCE_PROFILE")
        if not 0.20 <= self.resistance_onset_phase <= 0.80:
            raise ValueError("RESISTANCE_ONSET_OUT_OF_RANGE")
        if not 0.0 <= self.velocity_sensitivity <= 2.0:
            raise ValueError("VELOCITY_SENSITIVITY_OUT_OF_RANGE")
        if not 0.20 <= self.hip_resistance_share <= 0.80:
            raise ValueError("RESISTANCE_SHARE_OUT_OF_RANGE")
        if not 0.0 <= self.field_strength <= 1.0:
            raise ValueError("RESISTANCE_FIELD_STRENGTH_OUT_OF_RANGE")
        if self.scenario == "COMMON_RESISTANCE" and self.field_strength != 0.0:
            raise ValueError("COMMON_RESISTANCE_MUST_HAVE_ZERO_FIELD")
        if self.scenario == "DIVERGENT_RESISTANCE" and self.field_strength <= 0.0:
            raise ValueError("DIVERGENT_RESISTANCE_REQUIRES_FIELD")

    def mechanism_cost(self, features: tuple[float, float, float] | np.ndarray) -> float:
        """Return a dimensionless, reference-normalized resistance cost.

        ``features`` are (duration scale, assistance timing, hip share).  The
        cost is deliberately smooth and bounded; no candidate outcome is
        consulted when it is evaluated.
        """
        x = np.asarray(features, dtype=float)
        if x.shape != (3,) or not np.isfinite(x).all():
            raise ValueError("INVALID_RESISTANCE_CANDIDATE_FEATURES")
        duration, timing, hip_share = map(float, x)
        if duration not in DURATION_SCALES or timing not in ASSISTANCE_TIMINGS or hip_share not in HIP_SHARES:
            raise ValueError("UNDECLARED_RESISTANCE_CANDIDATE")
        speed = 1.0 / duration
        onset_mismatch = ((timing - self.resistance_onset_phase) / 0.25) ** 2
        share_mismatch = ((hip_share - self.hip_resistance_share) / 0.25) ** 2
        # The declared interaction is acceleration-sensitive: resistance rises
        # when the motion is faster than reference, while slowing does not
        # create an equal and opposite artificial benefit.
        velocity_cost = self.velocity_sensitivity * max((speed - 1.0) / 0.1, 0.0) ** 2
        return float(self.field_strength * (0.020 * onset_mismatch +
                                             0.020 * share_mismatch +
                                             0.012 * velocity_cost))

    def value(self, features: tuple[float, float, float] | np.ndarray) -> float:
        """Reference-normalized synthetic E3-like loss; lower is better."""
        x = np.asarray(features, dtype=float)
        duration, timing, hip_share = map(float, x)
        common = 0.002 * ((duration - 1.0) / 0.1) ** 2
        common += 0.001 * ((timing - 0.50) / 0.25) ** 2
        common += 0.001 * ((hip_share - 0.50) / 0.25) ** 2
        reference_cost = self.mechanism_cost(REFERENCE_FEATURES)
        return float(1.0 + common + self.mechanism_cost(x) - reference_cost)

    def as_manifest_record(self) -> dict[str, Any]:
        return asdict(self)


def _profile(profile_id: str, scenario: str, split: str, arm: str,
             seed: int, onset: float, velocity: float, share: float,
             strength: float) -> ResistanceProfile:
    return ResistanceProfile(profile_id, scenario, split, arm, seed, onset,
                              velocity, share, strength)


def build_profiles(root_seed: int = 20260925) -> tuple[ResistanceProfile, ...]:
    if not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("INVALID_RESISTANCE_ROOT_SEED")
    profiles: list[ResistanceProfile] = []
    for split, offset in (("DEVELOPMENT", 0), ("CONFIRMATORY", 100)):
        profiles.extend([
            _profile(f"RESIST_COMMON_{split[:4]}_00", "COMMON_RESISTANCE", split, "COMMON",
                     root_seed + offset, 0.50, 0.50, 0.50, 0.0),
            _profile(f"RESIST_COMMON_{split[:4]}_01", "COMMON_RESISTANCE", split, "COMMON",
                     root_seed + offset + 1, 0.50, 0.50, 0.50, 0.0),
            _profile(f"RESIST_EARLY_{split[:4]}", "DIVERGENT_RESISTANCE", split, "EARLY",
                     root_seed + offset + 10, 0.25, 1.20, 0.25, 1.0),
            _profile(f"RESIST_LATE_{split[:4]}", "DIVERGENT_RESISTANCE", split, "LATE",
                     root_seed + offset + 11, 0.75, 1.20, 0.75, 1.0),
            _profile(f"RESIST_FAST_{split[:4]}", "DIVERGENT_RESISTANCE", split, "FAST",
                     root_seed + offset + 12, 0.50, 1.80, 0.50, 1.0),
            _profile(f"RESIST_HIP_{split[:4]}", "DIVERGENT_RESISTANCE", split, "HIP",
                     root_seed + offset + 13, 0.50, 0.60, 0.75, 1.0),
        ])
    validate_profiles(profiles)
    return tuple(profiles)


def validate_profiles(profiles: tuple[ResistanceProfile, ...] | list[ResistanceProfile]) -> None:
    profiles = tuple(profiles)
    if not profiles or len({p.profile_id for p in profiles}) != len(profiles):
        raise ValueError("INVALID_RESISTANCE_PROFILE_SET")
    for profile in profiles:
        if not np.isclose(profile.value(REFERENCE_FEATURES), 1.0, atol=1e-15, rtol=0.0):
            raise ValueError("REFERENCE_RESISTANCE_VALUE_NOT_ONE")
    for scenario in SCENARIOS:
        rows = [p for p in profiles if p.scenario == scenario]
        if not rows or {p.split for p in rows} != set(SPLITS):
            raise ValueError("RESISTANCE_SCENARIO_SPLIT_INCOMPLETE")
    positive = [p for p in profiles if p.scenario == "DIVERGENT_RESISTANCE"]
    if len({(p.resistance_onset_phase, p.velocity_sensitivity, p.hip_resistance_share) for p in positive}) < 2:
        raise ValueError("RESISTANCE_PROFILES_NOT_DIVERGENT")


def manifest(root_seed: int = 20260925) -> dict[str, Any]:
    profiles = build_profiles(root_seed)
    payload: dict[str, Any] = {
        "cohort_id": COHORT_ID,
        "schema_version": 1,
        "root_seed": root_seed,
        "task_scope": "controlled synthetic mechanism stress test; no physiological interpretation",
        "candidate_features": {
            "coordinates": ["duration_scale", "assistance_timing_within_branch", "hip_share"],
            "values": [list(DURATION_SCALES), list(ASSISTANCE_TIMINGS), list(HIP_SHARES)],
            "reference": list(REFERENCE_FEATURES),
        },
        "mechanism": {
            "factors": ["resistance_onset_phase", "velocity_sensitivity", "hip_resistance_share"],
            "response": "reference-normalized E3-like loss; smooth angle-onset, speed, and load-share mismatch",
            "learner_oracle_access": False,
            "candidate_landscape_access": False,
        },
        "splits": {"DEVELOPMENT": "freeze policy and gate", "CONFIRMATORY": "sealed until freeze"},
        "profiles": [p.as_manifest_record() for p in profiles],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["manifest_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_manifest(path: str | Path = DEFAULT_MANIFEST, root_seed: int = 20260925) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest(root_seed), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def load_profiles(split: str | None = None, *, path: str | Path = DEFAULT_MANIFEST) -> tuple[ResistanceProfile, ...]:
    if split is not None and split not in SPLITS:
        raise ValueError("UNKNOWN_RESISTANCE_SPLIT")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = document.get("manifest_fingerprint_sha256")
    unsigned = dict(document)
    unsigned.pop("manifest_fingerprint_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if expected != hashlib.sha256(canonical).hexdigest():
        raise ValueError("RESISTANCE_MANIFEST_FINGERPRINT_MISMATCH")
    profiles = tuple(ResistanceProfile(**record) for record in document["profiles"])
    validate_profiles(profiles)
    return tuple(p for p in profiles if split is None or p.split == split)


__all__ = ["COHORT_ID", "SCENARIOS", "SPLITS", "DURATION_SCALES", "ASSISTANCE_TIMINGS", "HIP_SHARES",
           "REFERENCE_FEATURES", "ResistanceProfile", "build_profiles", "load_profiles", "manifest",
           "validate_profiles", "write_manifest", "DEFAULT_MANIFEST"]
