"""Fail-closed membership in the reference-centered offline design region.

``REFERENCE_CENTERED_ADMISSIBLE_REGION_V1`` is an offline personalization
design region.  It is neither the global physical-model ROM nor a real-robot
safety region.  The saved phase-wise corridors are a deterministic audit of
the generator family; final membership always rechecks the full generated
trajectory and every existing continuous-generator constraint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    jacobian_condition_limit,
    jacobian_det_threshold,
)
from .continuous_reference_neighborhood import (
    DOMAIN_BOUNDS_PATH,
    GENERATOR_VERSION,
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    GeneratedTrajectory,
)
from .dynamic_subject import get_dynamic_subject
from .force_mapping import endpoint_force_from_joint_torque
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    PROJECT_ROOT,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .full_dynamics import inverse_dynamics
from .geometry_error_metrics import StateDomainBounds, classify_state_domain
from .jacobian import jacobian_diagnostics
from .kinematics import forward_kinematics


REGION_VERSION = "REFERENCE_CENTERED_ADMISSIBLE_REGION_V1"
REGION_CLASSIFICATION = "OFFLINE_PERSONALIZATION_ADMISSIBLE_REGION"
GLOBAL_ROM_CLASSIFICATION = "GLOBAL_PHYSICAL_MODEL_ROM"
REAL_ROBOT_SAFETY_REGION_STATUS = "NOT_DEFINED_NOT_APPROVED"
MODEL_RELIABILITY_RULE_STATUS = "NOT_FROZEN"
DEFAULT_REGION_DIRECTORY = (
    PROJECT_ROOT
    / "lower_limb_sim"
    / "formal_artifacts"
    / "admissible_personalization_region_v1"
)

# This is only a floating-point/CSV round-trip tolerance.  It does not enlarge
# an offline corridor and is not a physical or robot-safety margin.
CORRIDOR_NUMERICAL_TOLERANCE = 1e-10

_JOINT_CORRIDOR_COLUMNS = (
    "sample_index",
    "time_s",
    "global_phase",
    "cycle_phase",
    "segment_phase",
    "q_hip_ref_rad",
    "q_hip_min_rad",
    "q_hip_max_rad",
    "q_knee_ref_rad",
    "q_knee_min_rad",
    "q_knee_max_rad",
)
_PULL_CORRIDOR_COLUMNS = (
    "sample_index",
    "time_s",
    "global_phase",
    "cycle_phase",
    "segment_phase",
    "x_pull_ref_m",
    "x_pull_min_m",
    "x_pull_max_m",
    "z_pull_ref_m",
    "z_pull_min_m",
    "z_pull_max_m",
    "pull_radial_max_mm",
)


@dataclass(frozen=True)
class AdmissibleRegionArtifacts:
    directory: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    joint_corridor: pd.DataFrame
    pull_corridor: pd.DataFrame


@dataclass(frozen=True)
class AdmissibleRegionEvaluation:
    alpha_bounds_valid: bool
    global_rom_valid: bool
    joint_corridor_valid: bool
    pull_corridor_valid: bool
    workspace_valid: bool
    jacobian_valid: bool
    force_mapping_valid: bool
    domain_valid: bool
    velocity_valid: bool
    acceleration_valid: bool
    closure_valid: bool
    continuity_valid: bool
    asymmetry_valid: bool
    finite_valid: bool
    generator_gate_valid: bool
    trajectory_admissible: bool
    invalid_reason: str
    first_invalid_sample: int | None
    invalid_phase: float | None
    first_invalid_sample_reason: str
    checked_sample_count: int
    domain_coverage_percent: float
    minimum_required_domain_coverage_percent: float
    maximum_joint_corridor_excess_rad: float
    maximum_pull_corridor_excess_m: float
    sample_audit: pd.DataFrame

    def as_dict(self, *, include_sample_audit: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        sample_audit = payload.pop("sample_audit")
        if include_sample_audit:
            payload["sample_audit"] = sample_audit.to_dict(orient="records")
        return payload


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _verify_artifact_hashes(directory: Path, manifest: Mapping[str, Any]) -> None:
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise RuntimeError("admissible-region manifest lacks artifact hashes")
    for name, expected in hashes.items():
        path = directory / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"admissible-region artifact missing: {path}")
        actual = _file_sha256(path)
        if actual != str(expected):
            raise RuntimeError(
                f"admissible-region artifact SHA mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )


def load_admissible_personalization_region(
    directory: str | Path = DEFAULT_REGION_DIRECTORY,
) -> AdmissibleRegionArtifacts:
    """Load and verify the frozen offline region artifacts."""

    root = Path(directory)
    manifest = _load_json(root / "admissible_region_manifest.json")
    summary = _load_json(root / "admissible_region_summary.json")
    required_manifest = {
        "region_version": REGION_VERSION,
        "region_classification": REGION_CLASSIFICATION,
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "generator_version": GENERATOR_VERSION,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "real_robot_safety_region_status": REAL_ROBOT_SAFETY_REGION_STATUS,
        "model_reliability_rule_status": MODEL_RELIABILITY_RULE_STATUS,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"admissible-region manifest {key!r} is not the frozen value"
            )
    expected_bounds = {
        key: list(value) for key, value in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
    }
    if manifest.get("offline_personalization_alpha_bounds") != expected_bounds:
        raise RuntimeError("admissible-region alpha bounds differ from generator bounds")
    if manifest.get("global_physical_model_rom") != {
        "classification": GLOBAL_ROM_CLASSIFICATION,
        "hip_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_deg": list(FORMAL_KNEE_ROM_DEG),
    }:
        raise RuntimeError("admissible-region global ROM provenance is invalid")
    _verify_artifact_hashes(root, manifest)
    joint = pd.read_csv(root / "joint_corridor_by_phase.csv")
    pull = pd.read_csv(root / "pull_corridor_by_phase.csv")
    if tuple(joint.columns) != _JOINT_CORRIDOR_COLUMNS:
        raise RuntimeError("joint corridor schema is not the frozen schema")
    if tuple(pull.columns) != _PULL_CORRIDOR_COLUMNS:
        raise RuntimeError("pull corridor schema is not the frozen schema")
    if len(joint) != len(pull) or len(joint) < 3:
        raise RuntimeError("admissible corridors have inconsistent sample counts")
    if not np.array_equal(
        joint["sample_index"].to_numpy(dtype=int), np.arange(len(joint), dtype=int)
    ):
        raise RuntimeError("joint corridor sample indices are not contiguous")
    np.testing.assert_allclose(
        joint["global_phase"], pull["global_phase"], atol=0.0, rtol=0.0
    )
    numerical = np.column_stack(
        (
            joint.select_dtypes(include=[np.number]).to_numpy(dtype=float),
            pull.select_dtypes(include=[np.number]).to_numpy(dtype=float),
        )
    )
    if not np.isfinite(numerical).all():
        raise RuntimeError("admissible corridors contain non-finite values")
    if not (
        np.all(joint["q_hip_min_rad"] <= joint["q_hip_ref_rad"])
        and np.all(joint["q_hip_ref_rad"] <= joint["q_hip_max_rad"])
        and np.all(joint["q_knee_min_rad"] <= joint["q_knee_ref_rad"])
        and np.all(joint["q_knee_ref_rad"] <= joint["q_knee_max_rad"])
        and np.all(pull["x_pull_min_m"] <= pull["x_pull_ref_m"])
        and np.all(pull["x_pull_ref_m"] <= pull["x_pull_max_m"])
        and np.all(pull["z_pull_min_m"] <= pull["z_pull_ref_m"])
        and np.all(pull["z_pull_ref_m"] <= pull["z_pull_max_m"])
        and np.all(pull["pull_radial_max_mm"] >= 0.0)
    ):
        raise RuntimeError("active reference is not contained in saved corridors")
    return AdmissibleRegionArtifacts(root, manifest, summary, joint, pull)


def _alpha_values(generated: GeneratedTrajectory) -> dict[str, float]:
    metadata = generated.metadata
    values = {
        "hip_amplitude_delta_deg": float(metadata["hip_amplitude_delta_deg"]),
        "knee_amplitude_delta_deg": float(metadata["knee_amplitude_delta_deg"]),
        "knee_phase_shift": float(metadata["knee_phase_shift"]),
    }
    if not np.isfinite(np.asarray(list(values.values()), dtype=float)).all():
        raise ValueError("generated trajectory alpha contains non-finite values")
    return values


def _alpha_bounds_valid(values: Mapping[str, float]) -> bool:
    return all(
        OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name][0]
        <= float(value)
        <= OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[name][1]
        for name, value in values.items()
    )


def _load_domain_bounds(path: str | Path) -> StateDomainBounds:
    payload = _load_json(Path(path))
    if payload.get("active_reference_identifier") != ACTIVE_REFERENCE_ID:
        raise RuntimeError("identification domain belongs to another reference")
    if payload.get("active_reference_sha256") != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("identification domain parent SHA mismatch")
    values = payload.get("bounds")
    if not isinstance(values, Mapping):
        raise RuntimeError("identification domain bounds are missing")
    return StateDomainBounds(
        columns=tuple(map(str, values["columns"])),
        lower=tuple(float(item) for item in values["lower"]),
        upper=tuple(float(item) for item in values["upper"]),
        valid_training_samples=int(values["valid_training_samples"]),
    )


def _domain_membership(
    trajectory: pd.DataFrame,
    domain_bounds_path: str | Path,
) -> np.ndarray:
    state = trajectory[
        [
            "q_hip_rad",
            "q_knee_rad",
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].copy(deep=True)
    state.columns = (
        "q_hip_est_rad",
        "q_knee_est_rad",
        "dq_hip_est_rad_s",
        "dq_knee_est_rad_s",
        "ddq_hip_est_rad_s2",
        "ddq_knee_est_rad_s2",
    )
    state["state_estimation_valid"] = np.isfinite(
        state.to_numpy(dtype=float)
    ).all(axis=1)
    return np.asarray(
        classify_state_domain(state, _load_domain_bounds(domain_bounds_path)),
        dtype=bool,
    )


def _comparison_excess(
    value: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    return np.maximum(np.maximum(lower - value, value - upper), 0.0)


def evaluate_admissible_personalization_region(
    generated: GeneratedTrajectory,
    *,
    region: AdmissibleRegionArtifacts | None = None,
    region_directory: str | Path = DEFAULT_REGION_DIRECTORY,
    domain_bounds_path: str | Path = DOMAIN_BOUNDS_PATH,
) -> AdmissibleRegionEvaluation:
    """Evaluate all samples and every frozen offline gate without clipping."""

    if not isinstance(generated, GeneratedTrajectory):
        raise TypeError("generated must be a continuous-generator result")
    metadata = generated.metadata
    if metadata.get("parent_reference_id") != ACTIVE_REFERENCE_ID:
        raise PermissionError("legacy or non-active reference parent is prohibited")
    if metadata.get("parent_reference_sha256") != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("REFERENCE_HASH_MISMATCH: admissible-region candidate")
    if metadata.get("generator_version") != GENERATOR_VERSION:
        raise RuntimeError("candidate was not produced by the frozen generator version")
    artifacts = (
        load_admissible_personalization_region(region_directory)
        if region is None
        else region
    )
    if not isinstance(artifacts, AdmissibleRegionArtifacts):
        raise TypeError("region must be verified AdmissibleRegionArtifacts")
    trajectory = generated.trajectory.copy(deep=False)
    if len(trajectory) != len(artifacts.joint_corridor):
        raise ValueError("candidate sample count differs from admissible corridor")
    required = {
        "time_s",
        "global_phase",
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
        "theta_shank_rad",
        "x_pull_m",
        "z_pull_m",
    }
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(f"candidate is missing columns: {sorted(missing)}")
    phase = trajectory["global_phase"].to_numpy(dtype=float)
    corridor_phase = artifacts.joint_corridor["global_phase"].to_numpy(dtype=float)
    if not np.allclose(phase, corridor_phase, atol=1e-14, rtol=0.0):
        raise ValueError("candidate phase grid differs from admissible corridor")

    q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
    q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
    dq_hip = trajectory["dq_hip_rad_s"].to_numpy(dtype=float)
    dq_knee = trajectory["dq_knee_rad_s"].to_numpy(dtype=float)
    ddq_hip = trajectory["ddq_hip_rad_s2"].to_numpy(dtype=float)
    ddq_knee = trajectory["ddq_knee_rad_s2"].to_numpy(dtype=float)
    stored_x = trajectory["x_pull_m"].to_numpy(dtype=float)
    stored_z = trajectory["z_pull_m"].to_numpy(dtype=float)
    numerical = np.column_stack(
        (
            phase,
            q_hip,
            q_knee,
            dq_hip,
            dq_knee,
            ddq_hip,
            ddq_knee,
            trajectory["theta_shank_rad"].to_numpy(dtype=float),
            stored_x,
            stored_z,
        )
    )
    finite_sample = np.isfinite(numerical).all(axis=1)
    theta_sample = np.isclose(
        numerical[:, 7], q_hip - q_knee, atol=1e-12, rtol=0.0
    )
    finite_sample &= theta_sample
    hip_deg = np.rad2deg(q_hip)
    knee_deg = np.rad2deg(q_knee)
    rom_sample = (
        (hip_deg >= FORMAL_HIP_ROM_DEG[0])
        & (hip_deg <= FORMAL_HIP_ROM_DEG[1])
        & (knee_deg >= FORMAL_KNEE_ROM_DEG[0])
        & (knee_deg <= FORMAL_KNEE_ROM_DEG[1])
    )

    joint = artifacts.joint_corridor
    hip_excess = _comparison_excess(
        q_hip,
        joint["q_hip_min_rad"].to_numpy(dtype=float),
        joint["q_hip_max_rad"].to_numpy(dtype=float),
    )
    knee_excess = _comparison_excess(
        q_knee,
        joint["q_knee_min_rad"].to_numpy(dtype=float),
        joint["q_knee_max_rad"].to_numpy(dtype=float),
    )
    joint_excess = np.maximum(hip_excess, knee_excess)
    joint_corridor_sample = joint_excess <= CORRIDOR_NUMERICAL_TOLERANCE

    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    fk_consistent = (
        np.isclose(stored_x, x_pull, atol=1e-12, rtol=0.0)
        & np.isclose(stored_z, z_pull, atol=1e-12, rtol=0.0)
    )
    workspace_sample = (
        np.isfinite(np.column_stack((x_knee, z_knee, x_pull, z_pull))).all(axis=1)
        & (x_pull >= 0.0)
        & (z_pull >= 0.0)
        & (z_knee >= 0.0)
        & fk_consistent
    )
    pull = artifacts.pull_corridor
    x_excess = _comparison_excess(
        x_pull,
        pull["x_pull_min_m"].to_numpy(dtype=float),
        pull["x_pull_max_m"].to_numpy(dtype=float),
    )
    z_excess = _comparison_excess(
        z_pull,
        pull["z_pull_min_m"].to_numpy(dtype=float),
        pull["z_pull_max_m"].to_numpy(dtype=float),
    )
    radial_m = np.hypot(
        x_pull - pull["x_pull_ref_m"].to_numpy(dtype=float),
        z_pull - pull["z_pull_ref_m"].to_numpy(dtype=float),
    )
    radial_excess = np.maximum(
        radial_m - pull["pull_radial_max_mm"].to_numpy(dtype=float) / 1000.0,
        0.0,
    )
    pull_excess = np.maximum.reduce((x_excess, z_excess, radial_excess))
    pull_corridor_sample = pull_excess <= CORRIDOR_NUMERICAL_TOLERANCE

    diagnostics = jacobian_diagnostics(q_hip, q_knee, L1, L2)
    determinant = np.asarray(diagnostics.determinant, dtype=float)
    condition = np.asarray(diagnostics.condition_number, dtype=float)
    jacobian_sample = (
        np.isfinite(determinant)
        & np.isfinite(condition)
        & ~np.asarray(diagnostics.near_singular, dtype=bool)
        & (np.abs(determinant) >= jacobian_det_threshold)
        & (condition <= jacobian_condition_limit)
    )

    baseline = get_dynamic_subject("baseline")
    dynamics = inverse_dynamics(
        q_hip, q_knee, dq_hip, dq_knee, ddq_hip, ddq_knee, baseline, L1
    )
    force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        dynamics.tau_total_hip_nm,
        dynamics.tau_total_knee_nm,
        L1,
        L2,
    )
    force_sample = np.asarray(force.force_mapping_valid, dtype=bool)
    domain_member = _domain_membership(trajectory, domain_bounds_path)
    coverage = 100.0 * float(np.mean(domain_member))
    required_coverage = float(
        generated.constraints.minimum_required_domain_coverage_percent
    )
    domain_valid = bool(
        coverage >= required_coverage
        and generated.constraints.domain_coverage_valid
    )
    velocity_sample = np.isfinite(np.column_stack((dq_hip, dq_knee))).all(axis=1)
    acceleration_sample = np.isfinite(
        np.column_stack((ddq_hip, ddq_knee))
    ).all(axis=1)

    values = _alpha_values(generated)
    alpha_valid = _alpha_bounds_valid(values)
    global_rom_valid = bool(rom_sample.all() and generated.constraints.rom_valid)
    joint_valid = bool(joint_corridor_sample.all())
    pull_valid = bool(pull_corridor_sample.all())
    workspace_valid = bool(workspace_sample.all() and generated.constraints.workspace_valid)
    jacobian_valid = bool(jacobian_sample.all() and generated.constraints.jacobian_valid)
    force_valid = bool(force_sample.all() and generated.constraints.force_mapping_valid)
    velocity_valid = bool(velocity_sample.all() and generated.constraints.velocity_valid)
    acceleration_valid = bool(
        acceleration_sample.all() and generated.constraints.acceleration_valid
    )
    closure_valid = bool(generated.constraints.closure_valid)
    continuity_valid = bool(
        generated.continuity_audit.get("passed")
        and generated.continuity_audit.get("position_continuity_warning_count") == 0
        and generated.continuity_audit.get("velocity_continuity_warning_count") == 0
        and generated.continuity_audit.get("acceleration_continuity_warning_count") == 0
    )
    asymmetry_valid = bool(generated.constraints.asymmetry_valid)
    finite_valid = bool(finite_sample.all() and generated.constraints.finite_valid)
    generator_gate_valid = bool(generated.constraints.trajectory_feasible)
    gates = {
        "alpha_bounds_invalid": alpha_valid,
        "global_rom_invalid": global_rom_valid,
        "joint_corridor_invalid": joint_valid,
        "pull_corridor_invalid": pull_valid,
        "workspace_invalid": workspace_valid,
        "jacobian_invalid": jacobian_valid,
        "force_mapping_invalid": force_valid,
        "identification_domain_insufficient": domain_valid,
        "velocity_invalid": velocity_valid,
        "acceleration_invalid": acceleration_valid,
        "closure_invalid": closure_valid,
        "continuity_invalid": continuity_valid,
        "asymmetry_invalid": asymmetry_valid,
        "finite_invalid": finite_valid,
        "generator_full_trajectory_gate_invalid": generator_gate_valid,
    }
    invalid_reasons = [name for name, passed in gates.items() if not passed]
    trajectory_admissible = not invalid_reasons

    sample_audit = pd.DataFrame(
        {
            "sample_index": np.arange(len(trajectory), dtype=int),
            "global_phase": phase,
            "finite_valid": finite_sample,
            "global_rom_valid": rom_sample,
            "joint_corridor_valid": joint_corridor_sample,
            "pull_corridor_valid": pull_corridor_sample,
            "workspace_valid": workspace_sample,
            "jacobian_valid": jacobian_sample,
            "force_mapping_valid": force_sample,
            "identification_domain_member": domain_member,
            "velocity_valid": velocity_sample,
            "acceleration_valid": acceleration_sample,
        }
    )
    sample_hard_valid = sample_audit[
        [
            "finite_valid",
            "global_rom_valid",
            "joint_corridor_valid",
            "pull_corridor_valid",
            "workspace_valid",
            "jacobian_valid",
            "force_mapping_valid",
            "velocity_valid",
            "acceleration_valid",
        ]
    ].all(axis=1).to_numpy(dtype=bool)
    # The existing domain gate is coverage-based.  Every sample is checked and
    # recorded, but isolated outside-domain rows do not invalidate the active
    # reference when the frozen >=90% coverage gate remains satisfied.
    if not domain_valid:
        sample_hard_valid &= domain_member
    sample_audit["sample_admissibility_components_valid"] = sample_hard_valid
    failing = np.flatnonzero(~sample_hard_valid)
    first_invalid = int(failing[0]) if len(failing) else None
    if first_invalid is None and invalid_reasons:
        first_invalid = 0
    first_reason = ""
    invalid_phase = None
    if first_invalid is not None:
        invalid_phase = float(phase[first_invalid])
        row = sample_audit.iloc[first_invalid]
        sample_reason_map = {
            "finite_invalid": "finite_valid",
            "global_rom_invalid": "global_rom_valid",
            "joint_corridor_invalid": "joint_corridor_valid",
            "pull_corridor_invalid": "pull_corridor_valid",
            "workspace_invalid": "workspace_valid",
            "jacobian_invalid": "jacobian_valid",
            "force_mapping_invalid": "force_mapping_valid",
            "identification_domain_insufficient": "identification_domain_member",
            "velocity_invalid": "velocity_valid",
            "acceleration_invalid": "acceleration_valid",
        }
        first_reason = ";".join(
            reason
            for reason in invalid_reasons
            if reason not in sample_reason_map or not bool(row[sample_reason_map[reason]])
        )
        if not first_reason:
            first_reason = ";".join(invalid_reasons)
    return AdmissibleRegionEvaluation(
        alpha_bounds_valid=alpha_valid,
        global_rom_valid=global_rom_valid,
        joint_corridor_valid=joint_valid,
        pull_corridor_valid=pull_valid,
        workspace_valid=workspace_valid,
        jacobian_valid=jacobian_valid,
        force_mapping_valid=force_valid,
        domain_valid=domain_valid,
        velocity_valid=velocity_valid,
        acceleration_valid=acceleration_valid,
        closure_valid=closure_valid,
        continuity_valid=continuity_valid,
        asymmetry_valid=asymmetry_valid,
        finite_valid=finite_valid,
        generator_gate_valid=generator_gate_valid,
        trajectory_admissible=trajectory_admissible,
        invalid_reason=";".join(invalid_reasons),
        first_invalid_sample=first_invalid,
        invalid_phase=invalid_phase,
        first_invalid_sample_reason=first_reason,
        checked_sample_count=len(trajectory),
        domain_coverage_percent=coverage,
        minimum_required_domain_coverage_percent=required_coverage,
        maximum_joint_corridor_excess_rad=float(np.max(joint_excess)),
        maximum_pull_corridor_excess_m=float(np.max(pull_excess)),
        sample_audit=sample_audit,
    )


__all__ = [
    "AdmissibleRegionArtifacts",
    "AdmissibleRegionEvaluation",
    "CORRIDOR_NUMERICAL_TOLERANCE",
    "DEFAULT_REGION_DIRECTORY",
    "GLOBAL_ROM_CLASSIFICATION",
    "MODEL_RELIABILITY_RULE_STATUS",
    "REAL_ROBOT_SAFETY_REGION_STATUS",
    "REGION_CLASSIFICATION",
    "REGION_VERSION",
    "evaluate_admissible_personalization_region",
    "load_admissible_personalization_region",
]
