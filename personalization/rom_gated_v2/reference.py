"""Subject-specific reference and frozen V3-domain adapters."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np
from scipy.interpolate import make_interp_spline

from external_simulation.myoleg_v3_trajectory_parameterization_design_v1.parameterization import (
    PARAMETERIZATION_ID,
    V3Trajectory,
    generate_v3_trajectory,
)

from ..candidates import Candidate, V3CandidateDomain
from .rom import SubjectROMProfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REFERENCE = (
    ROOT
    / "external_simulation_audits"
    / "myoleg_knee_rom_compatibility_audit_v1"
    / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
)
FORMAL_TIMING_REFERENCE = (
    ROOT / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
)
SUBJECT_REFERENCE_VERSION = "SUBJECT_SPECIFIC_REFERENCE_ADAPTER_V1"
ROM_TOLERANCE_RAD = 2.0e-5


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source_reference() -> dict[str, Any]:
    with SOURCE_REFERENCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with FORMAL_TIMING_REFERENCE.open(newline="", encoding="utf-8") as stream:
        timing_rows = list(csv.DictReader(stream))
    if len(rows) != 401 or len(timing_rows) != 401:
        raise RuntimeError("frozen source reference sample count changed")
    time_s = np.asarray([float(row["time_s"]) for row in rows])
    global_phase = np.asarray([float(row["global_phase"]) for row in rows])
    segment_phase = np.asarray([float(row["segment_phase"]) for row in rows])
    phases = np.asarray([row["cycle_phase"] for row in rows])
    q = np.asarray(
        [[float(row["q_hip_rad"]), float(row["q_knee_rad"])] for row in rows]
    )
    dq = np.asarray(
        [[float(row["dq_hip_rad_s"]), float(row["dq_knee_rad_s"])] for row in rows]
    )
    ddq = np.asarray(
        [
            [float(row["ddq_hip_rad_s2"]), float(row["ddq_knee_rad_s2"])]
            for row in rows
        ]
    )
    phase_rate = np.asarray(
        [float(row["minimum_jerk_phase_rate_s_inv"]) for row in timing_rows]
    )
    phase_accel = np.asarray(
        [
            float(row["minimum_jerk_phase_acceleration_s_inv2"])
            for row in timing_rows
        ]
    )
    peak = float(global_phase[phases == "flexion"][-1])
    return {
        "time_s": time_s,
        "global_phase": global_phase,
        "segment_phase": segment_phase,
        "phases": phases,
        "q": q,
        "dq": dq,
        "ddq": ddq,
        "phase_rate": phase_rate,
        "phase_accel": phase_accel,
        "peak": peak,
        "source_reference_sha256": _sha256(SOURCE_REFERENCE),
        "formal_timing_reference_sha256": _sha256(FORMAL_TIMING_REFERENCE),
    }


@dataclass(frozen=True)
class SubjectSpecificReference:
    profile_id: str
    profile_fingerprint: str
    reference_version: str
    time_s: np.ndarray = field(repr=False, compare=False)
    global_phase: np.ndarray = field(repr=False, compare=False)
    segment_phase: np.ndarray = field(repr=False, compare=False)
    phases: np.ndarray = field(repr=False, compare=False)
    q: np.ndarray = field(repr=False, compare=False)
    dq: np.ndarray = field(repr=False, compare=False)
    ddq: np.ndarray = field(repr=False, compare=False)
    phase_rate: np.ndarray = field(repr=False, compare=False)
    phase_accel: np.ndarray = field(repr=False, compare=False)
    peak: float
    source_reference_sha256: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "time_s",
            "global_phase",
            "segment_phase",
            "phases",
            "q",
            "dq",
            "ddq",
            "phase_rate",
            "phase_accel",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def as_v3_mapping(self) -> Mapping[str, Any]:
        return {
            "time_s": self.time_s,
            "global_phase": self.global_phase,
            "segment_phase": self.segment_phase,
            "phases": self.phases,
            "q": self.q,
            "dq": self.dq,
            "ddq": self.ddq,
            "phase_rate": self.phase_rate,
            "phase_accel": self.phase_accel,
            "peak": self.peak,
            "hip_spline": make_interp_spline(
                self.global_phase, self.q[:, 0], k=3, bc_type="periodic"
            ),
            "knee_spline": make_interp_spline(
                self.global_phase, self.q[:, 1], k=3, bc_type="periodic"
            ),
        }


class SubjectSpecificReferenceAdapter:
    """Affine normalized-progression mapping with an explicit audit."""

    def __init__(self, source: Mapping[str, Any] | None = None) -> None:
        self.source = dict(source or _load_source_reference())

    def adapt(self, profile: SubjectROMProfile) -> SubjectSpecificReference:
        if not profile.frozen:
            raise RuntimeError("ROM_PROFILE_FROZEN must be true before reference adaptation")
        source_q = np.asarray(self.source["q"], dtype=float)
        source_min = np.min(source_q, axis=0)
        source_max = np.max(source_q, axis=0)
        source_span = source_max - source_min
        if np.any(source_span <= 0.0):
            raise RuntimeError("source reference must have positive hip and knee ranges")
        target_min = np.asarray([profile.hip_min_rad, profile.knee_min_rad])
        target_max = np.asarray([profile.hip_max_rad, profile.knee_max_rad])
        target_span = target_max - target_min
        scale = target_span / source_span
        normalized = (source_q - source_min) / source_span
        q = target_min + normalized * target_span
        dq = np.asarray(self.source["dq"], dtype=float) * scale
        ddq = np.asarray(self.source["ddq"], dtype=float) * scale
        recovered = (q - target_min) / target_span
        extrema_error = max(
            float(np.max(np.abs(np.min(q, axis=0) - target_min))),
            float(np.max(np.abs(np.max(q, axis=0) - target_max))),
        )
        closure_error = max(
            float(np.max(np.abs(q[0] - q[-1]))),
            float(np.max(np.abs(dq[0] - dq[-1]))),
            float(np.max(np.abs(ddq[0] - ddq[-1]))),
        )
        progression_error = float(np.max(np.abs(recovered - normalized)))
        if extrema_error > 1.0e-12 or closure_error > 1.0e-10:
            raise RuntimeError("subject reference failed extrema or closure audit")
        audit = {
            "branch_timing_preserved": True,
            "duration_preserved": True,
            "closure_preserved": True,
            "affine_progression_preserves_step_sign": bool(np.all(scale > 0.0)),
            "normalized_progression_max_abs_error": progression_error,
            "extrema_max_abs_error_rad": extrema_error,
            "closure_max_abs_error": closure_error,
            "smoothness_transform": "positive affine joint scaling of q/dq/ddq",
            "pointwise_clipping": False,
            "reference_semantics": "CURRENT_TASK_SPECIFIC_ALLOWED_ROM",
        }
        return SubjectSpecificReference(
            profile_id=profile.profile_id,
            profile_fingerprint=profile.fingerprint,
            reference_version=(
                f"{SUBJECT_REFERENCE_VERSION}:{self.source['source_reference_sha256']}"
            ),
            time_s=np.asarray(self.source["time_s"]),
            global_phase=np.asarray(self.source["global_phase"]),
            segment_phase=np.asarray(self.source["segment_phase"]),
            phases=np.asarray(self.source["phases"]),
            q=q,
            dq=dq,
            ddq=ddq,
            phase_rate=np.asarray(self.source["phase_rate"]),
            phase_accel=np.asarray(self.source["phase_accel"]),
            peak=float(self.source["peak"]),
            source_reference_sha256=str(self.source["source_reference_sha256"]),
            audit=audit,
        )


@dataclass(frozen=True, order=True)
class SubjectSpecificCandidate(Candidate):
    canonical_beta_id: str = field(compare=False)
    rom_profile_id: str = field(compare=False)
    rom_profile_fingerprint: str = field(compare=False)
    v3_operator_version: str = field(compare=False)
    subject_reference_version: str = field(compare=False)
    trajectory: V3Trajectory = field(repr=False, compare=False)
    validation: Mapping[str, Any] = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            **super().as_dict(),
            "subject_candidate_id": self.candidate_id,
            "canonical_beta_id": self.canonical_beta_id,
            "rom_profile_id": self.rom_profile_id,
            "rom_profile_fingerprint": self.rom_profile_fingerprint,
            "v3_operator_version": self.v3_operator_version,
            "subject_reference_version": self.subject_reference_version,
            "validation": dict(self.validation),
        }


def _subject_candidate_id(
    profile: SubjectROMProfile,
    canonical: Candidate,
    reference_version: str,
) -> str:
    payload = {
        "rom_profile_id": profile.profile_id,
        "rom_profile_version": profile.version,
        "rom_profile_fingerprint": profile.fingerprint,
        "beta_flex": canonical.beta_flex,
        "beta_extend": canonical.beta_extend,
        "canonical_beta_id": canonical.candidate_id,
        "v3_operator_version": PARAMETERIZATION_ID,
        "subject_reference_version": reference_version,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"SUBJECT_V3_{digest[:24]}"


def _make_trajectory_readonly(trajectory: V3Trajectory) -> V3Trajectory:
    for array in (
        trajectory.q,
        trajectory.dq,
        trajectory.ddq,
        trajectory.warped_segment_phase,
        trajectory.warp_first_derivative,
        trajectory.warp_second_derivative,
    ):
        array.setflags(write=False)
    return trajectory


def _validate_candidate(
    trajectory: V3Trajectory,
    reference: SubjectSpecificReference,
    profile: SubjectROMProfile,
) -> dict[str, Any]:
    arrays = (trajectory.q, trajectory.dq, trajectory.ddq)
    if any(array.shape != reference.q.shape for array in arrays):
        raise RuntimeError("subject candidate q/dq/ddq shape mismatch")
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("subject candidate contains non-finite trajectory values")
    target_min = np.asarray([profile.hip_min_rad, profile.knee_min_rad])
    target_max = np.asarray([profile.hip_max_rad, profile.knee_max_rad])
    extrema_error = max(
        float(np.max(np.abs(np.min(trajectory.q, axis=0) - target_min))),
        float(np.max(np.abs(np.max(trajectory.q, axis=0) - target_max))),
    )
    outside = max(
        float(np.max(target_min - np.min(trajectory.q, axis=0))),
        float(np.max(np.max(trajectory.q, axis=0) - target_max)),
        0.0,
    )
    anchors = np.isclose(reference.segment_phase, 0.0, atol=1.0e-15) | np.isclose(
        reference.segment_phase, 1.0, atol=1.0e-15
    )
    anchor_error = max(
        float(np.max(np.abs(trajectory.q[anchors] - reference.q[anchors]))),
        float(np.max(np.abs(trajectory.dq[anchors] - reference.dq[anchors]))),
        float(np.max(np.abs(trajectory.ddq[anchors] - reference.ddq[anchors]))),
    )
    closure_error = max(
        float(np.max(np.abs(array[0] - array[-1]))) for array in arrays
    )
    minimum_warp_derivative = float(np.min(trajectory.warp_first_derivative))
    no_branch_fold = all(
        bool(
            np.all(
                np.diff(trajectory.warped_segment_phase[reference.phases == phase])
                >= -1.0e-12
            )
        )
        for phase in ("flexion", "extension")
    )
    passed = (
        extrema_error <= ROM_TOLERANCE_RAD
        and outside <= ROM_TOLERANCE_RAD
        and anchor_error <= 1.0e-12
        and closure_error <= 1.0e-9
        and minimum_warp_derivative > 0.0
        and no_branch_fold
    )
    if not passed:
        raise RuntimeError(
            "subject-specific candidate failed fixed-ROM invariants; it was rejected without clipping"
        )
    return {
        "kinematic_gate_pass": True,
        "hip_min_rad": float(np.min(trajectory.q[:, 0])),
        "hip_max_rad": float(np.max(trajectory.q[:, 0])),
        "knee_min_rad": float(np.min(trajectory.q[:, 1])),
        "knee_max_rad": float(np.max(trajectory.q[:, 1])),
        "extrema_max_abs_error_rad": extrema_error,
        "outside_rom_max_rad": outside,
        "duration_s": reference.duration_s,
        "endpoint_C2_anchor_max_abs_error": anchor_error,
        "closure_max_abs_error": closure_error,
        "minimum_warp_derivative": minimum_warp_derivative,
        "no_branch_fold": no_branch_fold,
        "pointwise_clipping": False,
    }


class SubjectSpecificV3CandidateDomain(V3CandidateDomain):
    """The frozen canonical beta grid instantiated inside one frozen ROM."""

    def __init__(
        self,
        candidates: tuple[SubjectSpecificCandidate, ...],
        *,
        profile: SubjectROMProfile,
        reference: SubjectSpecificReference,
    ) -> None:
        if not profile.frozen:
            raise RuntimeError("ROM_PROFILE_FROZEN must be true before V3 construction")
        super().__init__(candidates)
        self.profile = profile
        self.subject_reference = reference

    @classmethod
    def from_frozen_beta_grid(
        cls,
        profile: SubjectROMProfile,
        *,
        reference_adapter: SubjectSpecificReferenceAdapter | None = None,
    ) -> "SubjectSpecificV3CandidateDomain":
        if not profile.frozen:
            raise RuntimeError("ROM_PROFILE_FROZEN must be true before V3 construction")
        reference = (reference_adapter or SubjectSpecificReferenceAdapter()).adapt(profile)
        canonical_domain = V3CandidateDomain.from_frozen_artifact()
        mapping = reference.as_v3_mapping()
        candidates = []
        for canonical in canonical_domain:
            trajectory = _make_trajectory_readonly(
                generate_v3_trajectory(
                    mapping, canonical.beta_flex, canonical.beta_extend
                )
            )
            validation = _validate_candidate(trajectory, reference, profile)
            candidates.append(
                SubjectSpecificCandidate(
                    candidate_id=_subject_candidate_id(
                        profile, canonical, reference.reference_version
                    ),
                    beta_flex=canonical.beta_flex,
                    beta_extend=canonical.beta_extend,
                    candidate_index=canonical.candidate_index,
                    canonical_beta_id=canonical.candidate_id,
                    rom_profile_id=profile.profile_id,
                    rom_profile_fingerprint=profile.fingerprint,
                    v3_operator_version=PARAMETERIZATION_ID,
                    subject_reference_version=reference.reference_version,
                    trajectory=trajectory,
                    validation=MappingProxyType(validation),
                )
            )
        if len(candidates) != 625:
            raise RuntimeError("frozen V3 beta grid must contain 625 candidates")
        return cls(tuple(candidates), profile=profile, reference=reference)

    @property
    def reference(self) -> SubjectSpecificCandidate:
        return super().reference  # type: ignore[return-value]

    def by_id(self, candidate_id: str) -> SubjectSpecificCandidate:
        return super().by_id(candidate_id)  # type: ignore[return-value]

    def trajectory_for(self, candidate_id: str) -> V3Trajectory:
        return self.by_id(candidate_id).trajectory

    def __iter__(self) -> Iterator[SubjectSpecificCandidate]:
        return super().__iter__()  # type: ignore[return-value]

    def invariant_summary(self) -> dict[str, Any]:
        validations = [candidate.validation for candidate in self]
        return {
            "candidate_count": len(self),
            "all_kinematic_gates_pass": all(
                bool(item["kinematic_gate_pass"]) for item in validations
            ),
            "maximum_extrema_error_rad": max(
                float(item["extrema_max_abs_error_rad"]) for item in validations
            ),
            "maximum_closure_error": max(
                float(item["closure_max_abs_error"]) for item in validations
            ),
            "minimum_warp_derivative": min(
                float(item["minimum_warp_derivative"]) for item in validations
            ),
            "pointwise_clipping_count": sum(
                bool(item["pointwise_clipping"]) for item in validations
            ),
            "duration_s": self.subject_reference.duration_s,
            "rom_profile_id": self.profile.profile_id,
            "rom_profile_fingerprint": self.profile.fingerprint,
            "v3_operator_version": PARAMETERIZATION_ID,
        }
