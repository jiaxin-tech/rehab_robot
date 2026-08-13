"""Start-anchored relative rehabilitation trajectory generation.

The active source is the approved measured-flexion/measured-extension periodic
C2 reference.  Its equivalent strap pull point is recomputed with the current
``L1/L2`` forward kinematics; no observed ankle point or absolute hip centre
participates in this mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lower_limb_sim.config import L1, L2
from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from lower_limb_sim.reference_release import load_reference_release_manifest
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
)
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)


ABSOLUTE_CALIBRATED_MODE = "absolute_calibrated"
START_ANCHORED_RELATIVE_MODE = "start_anchored_relative"
FIRST_ROBOT_TRIAL_TRAJECTORY_ID = (
    ACTIVE_REFERENCE_ID
)
APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256 = (
    ACTIVE_REFERENCE_SHA256
)
APPROVED_FIRST_ROBOT_TRIAL_L1_M = 0.42
APPROVED_FIRST_ROBOT_TRIAL_L2_M = 0.30
# Candidate whitelist only.  Membership does not constitute physical-motion
# approval; the release manifest and output metadata remain NO-GO.
ALLOWED_FIRST_ROBOT_TRIAL_TRAJECTORIES = frozenset(
    {FIRST_ROBOT_TRIAL_TRAJECTORY_ID}
)
MODEL_ANGLE_DEFINITION = f"theta_shank = {THETA_SHANK_DEFINITION}"
APPROVED_HIP_ROM_DEG = FORMAL_HIP_ROM_DEG
APPROVED_KNEE_ROM_DEG = FORMAL_KNEE_ROM_DEG
DEFAULT_FRAME_INPUT_TOLERANCE = 1e-3
DEFAULT_NUMERICAL_TOLERANCE = 1e-10


def _finite_vector(
    values: Sequence[float], size: int, field_name: str
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must contain exactly {size} finite values")
    return vector


@dataclass(frozen=True)
class RehabFrameConfig:
    """Reviewed-or-draft rehabilitation bed axes expressed in robot Base."""

    rehab_x_axis_in_base: tuple[float, float, float]
    rehab_z_axis_in_base: tuple[float, float, float]
    reviewed: bool
    notes: str = ""
    input_tolerance: float = DEFAULT_FRAME_INPUT_TOLERANCE

    def __post_init__(self) -> None:
        if type(self.reviewed) is not bool:
            raise ValueError("reviewed must be a JSON/Python boolean")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        tolerance = float(self.input_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 0.05:
            raise ValueError("input_tolerance must be finite and in (0, 0.05]")

        x = _finite_vector(self.rehab_x_axis_in_base, 3, "rehab_x_axis_in_base")
        z = _finite_vector(self.rehab_z_axis_in_base, 3, "rehab_z_axis_in_base")
        x_norm = float(np.linalg.norm(x))
        z_norm = float(np.linalg.norm(z))
        dot = float(np.dot(x, z))
        if abs(x_norm - 1.0) > tolerance or abs(z_norm - 1.0) > tolerance:
            raise ValueError("rehab x/z input axes must have norm approximately one")
        if abs(dot) > tolerance:
            raise ValueError("rehab x/z input axes must be approximately orthogonal")

        object.__setattr__(self, "rehab_x_axis_in_base", tuple(map(float, x)))
        object.__setattr__(self, "rehab_z_axis_in_base", tuple(map(float, z)))
        object.__setattr__(self, "input_tolerance", tolerance)

        rotation = self.rotation_base_from_rehab
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0.0):
            raise ValueError("orthogonalized R_base_from_rehab is not orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-12):
            raise ValueError("R_base_from_rehab is not right handed")

    @property
    def rotation_base_from_rehab(self) -> np.ndarray:
        """Construct right-handed ``R_base_from_rehab`` by Gram-Schmidt."""

        x_raw = np.asarray(self.rehab_x_axis_in_base, dtype=float)
        z_raw = np.asarray(self.rehab_z_axis_in_base, dtype=float)
        x = x_raw / np.linalg.norm(x_raw)
        z_projected = z_raw - float(np.dot(z_raw, x)) * x
        z = z_projected / np.linalg.norm(z_projected)
        # In a right-handed frame x cross y = z, hence y = z cross x.
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
        # Recompute z after orthogonalizing y to remove round-off skew.
        z = np.cross(x, y)
        z /= np.linalg.norm(z)
        return np.column_stack((x, y, z))

    @property
    def rehab_y_axis_in_base(self) -> tuple[float, float, float]:
        return tuple(map(float, self.rotation_base_from_rehab[:, 1]))

    def as_metadata(self) -> dict[str, object]:
        rotation = self.rotation_base_from_rehab
        return {
            "rehab_x_axis_in_base": list(map(float, rotation[:, 0])),
            "rehab_y_axis_in_base": list(map(float, rotation[:, 1])),
            "rehab_z_axis_in_base": list(map(float, rotation[:, 2])),
            "R_base_from_rehab": rotation.tolist(),
            "reviewed": self.reviewed,
            "notes": self.notes,
            "construction": "x_normalize_then_z_orthogonalize_then_y_equals_z_cross_x",
        }


def load_rehab_frame_config(path: str | Path) -> RehabFrameConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"rehab frame config not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("rehab frame config must be a JSON object")
    required = {"rehab_x_axis_in_base", "rehab_z_axis_in_base", "reviewed", "notes"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"rehab frame config missing fields: {sorted(missing)}")
    return RehabFrameConfig(
        rehab_x_axis_in_base=tuple(payload["rehab_x_axis_in_base"]),
        rehab_z_axis_in_base=tuple(payload["rehab_z_axis_in_base"]),
        reviewed=payload["reviewed"],
        notes=payload["notes"],
    )


@dataclass(frozen=True)
class RelativeTrajectoryAudit:
    trajectory_id: str
    sample_count: int
    first_relative_displacement_zero: bool
    final_relative_displacement_zero: bool
    first_target_equals_anchor: bool
    final_target_equals_anchor: bool
    orientation_constant: bool
    source_samples_valid: bool
    source_formal_gate_valid: bool
    approved_rom_valid: bool
    theta_shank_definition_valid: bool
    pull_forward_kinematics_valid: bool
    all_samples_finite: bool
    time_strictly_increasing: bool
    no_obvious_position_jump: bool
    no_obvious_velocity_jump: bool
    no_obvious_acceleration_jump: bool
    maximum_position_step_m: float | None
    maximum_velocity_step_m_s: float | None
    maximum_acceleration_step_m_s2: float | None
    trajectory_valid: bool
    invalid_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _unique_string(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        raise ValueError(f"reference missing required column: {column}")
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"reference {column} must contain exactly one value")
    return values[0]


def _bool_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"reference missing required column: {column}")
    values = frame[column]
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise ValueError(f"reference {column} contains non-boolean values")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def _append_reason(target: list[str], reason: str, condition: bool) -> None:
    if condition and reason not in target:
        target.append(reason)


def _no_obvious_jump(values: np.ndarray, *, multiplier: float = 20.0) -> tuple[bool, float | None]:
    if len(values) < 2 or not np.isfinite(values).all():
        return False, None
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    maximum = float(np.max(steps)) if len(steps) else 0.0
    finite = steps[np.isfinite(steps)]
    if len(finite) < 5:
        return True, maximum
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, median * 0.05, np.finfo(float).eps)
    limit = max(multiplier * max(median, np.finfo(float).eps), median + multiplier * scale)
    return bool(maximum <= limit), maximum


def _reference_and_metadata(
    reference: str | Path | pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if isinstance(reference, pd.DataFrame):
        frame = reference.copy(deep=True)
        metadata: dict[str, object] = {"path": None, "source": "in_memory_dataframe"}
    else:
        frame, metadata = load_closed_reference_trajectory(reference)
    required = {
        "time_s",
        "trajectory_id",
        "reference_version",
        "q_hip_rad",
        "q_knee_rad",
        "theta_shank_rad",
        "trajectory_sample_valid",
        "formal_execution_allowed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"reference missing columns: {sorted(missing)}")
    if (
        _unique_string(frame, "reference_version")
        != MEASURED_ASYMMETRIC_CLOSED_REFERENCE
    ):
        raise ValueError(
            "start-anchored mode requires "
            "reference_measured_asymmetric_closed"
        )
    return frame.reset_index(drop=True), metadata


def build_start_anchored_relative_trajectory(
    reference: str | Path | pd.DataFrame = DEFAULT_REFERENCE_PATH,
    *,
    current_tcp_start_pose: Sequence[float],
    rehab_frame: RehabFrameConfig,
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> tuple[pd.DataFrame, RelativeTrajectoryAudit, dict[str, object]]:
    """Build ``p_tcp_B = p_tcp_start_B + R_B_R @ (p_R-p_R(0))``.

    The first and final relative displacements are made exactly zero only after
    the source FK endpoints pass the numerical closure tolerance. Joint angles,
    periodic path samples, and ROM values are never changed.
    """

    tolerance = float(numerical_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("numerical_tolerance must be finite and positive")
    anchor_pose = _finite_vector(current_tcp_start_pose, 6, "current_tcp_start_pose")
    source, source_metadata = _reference_and_metadata(reference)
    if len(source) < 3:
        raise ValueError("reference trajectory must contain at least three samples")

    trajectory_id = _unique_string(source, "trajectory_id")
    time_s = source["time_s"].to_numpy(dtype=float)
    q_hip = source["q_hip_rad"].to_numpy(dtype=float)
    q_knee = source["q_knee_rad"].to_numpy(dtype=float)
    theta_source = source["theta_shank_rad"].to_numpy(dtype=float)
    source_valid = _bool_column(source, "trajectory_sample_valid")
    formal_valid = _bool_column(source, "formal_execution_allowed")

    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    pull_rehab = np.column_stack(
        (np.asarray(x_pull, dtype=float), np.zeros(len(source)), np.asarray(z_pull, dtype=float))
    )
    relative_rehab = pull_rehab - pull_rehab[0]
    endpoint_error_m = float(np.linalg.norm(relative_rehab[-1]))
    source_closed = endpoint_error_m <= tolerance
    relative_rehab[0] = 0.0
    if source_closed:
        relative_rehab[-1] = 0.0

    rotation = rehab_frame.rotation_base_from_rehab
    relative_base = relative_rehab @ rotation.T
    target_position = anchor_pose[:3] + relative_base
    if source_closed:
        target_position[0] = anchor_pose[:3]
        target_position[-1] = anchor_pose[:3]
    orientation = np.repeat(anchor_pose[None, 3:], len(source), axis=0)

    finite_source = np.isfinite(
        np.column_stack((time_s, q_hip, q_knee, theta_source, pull_rehab))
    ).all(axis=1)
    hip_min, hip_max = np.deg2rad(APPROVED_HIP_ROM_DEG)
    knee_min, knee_max = np.deg2rad(APPROVED_KNEE_ROM_DEG)
    rom_valid = (
        (q_hip >= hip_min - tolerance)
        & (q_hip <= hip_max + tolerance)
        & (q_knee >= knee_min - tolerance)
        & (q_knee <= knee_max + tolerance)
    )
    theta_valid_rows = np.isclose(
        theta_source, q_hip - q_knee, atol=tolerance, rtol=0.0
    )

    if "x_pull_m" in source and "z_pull_m" in source:
        stored_pull = source[["x_pull_m", "z_pull_m"]].to_numpy(dtype=float)
        fk_pull = pull_rehab[:, (0, 2)]
        fk_valid_rows = np.linalg.norm(stored_pull - fk_pull, axis=1) <= tolerance
    else:
        fk_valid_rows = np.ones(len(source), dtype=bool)

    time_valid = bool(np.isfinite(time_s).all() and np.all(np.diff(time_s) > 0.0))
    sample_valid = (
        source_valid
        & formal_valid
        & finite_source
        & rom_valid
        & theta_valid_rows
        & fk_valid_rows
    )
    row_reasons: list[list[str]] = [[] for _ in range(len(source))]
    for index in range(len(source)):
        _append_reason(row_reasons[index], "source_sample_invalid", not source_valid[index])
        _append_reason(row_reasons[index], "source_formal_gate_not_allowed", not formal_valid[index])
        _append_reason(row_reasons[index], "non_finite_reference_sample", not finite_source[index])
        _append_reason(row_reasons[index], "outside_approved_experiment_rom", not rom_valid[index])
        _append_reason(row_reasons[index], "theta_shank_definition_invalid", not theta_valid_rows[index])
        _append_reason(row_reasons[index], "pull_point_fk_mismatch", not fk_valid_rows[index])

    positions_finite = bool(np.isfinite(target_position).all())
    orientation_constant = bool(
        np.array_equal(orientation, np.repeat(anchor_pose[None, 3:], len(source), axis=0))
    )
    first_relative_zero = bool(np.array_equal(relative_rehab[0], np.zeros(3)))
    final_relative_zero = bool(np.array_equal(relative_rehab[-1], np.zeros(3)))
    first_target_anchor = bool(np.array_equal(target_position[0], anchor_pose[:3]))
    final_target_anchor = bool(np.array_equal(target_position[-1], anchor_pose[:3]))

    velocity = np.full_like(target_position, np.nan)
    acceleration = np.full_like(target_position, np.nan)
    if time_valid and positions_finite:
        velocity = np.gradient(target_position, time_s, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, time_s, axis=0, edge_order=2)
    position_ok, max_position_step = _no_obvious_jump(target_position)
    velocity_ok, max_velocity_step = _no_obvious_jump(velocity)
    acceleration_ok, max_acceleration_step = _no_obvious_jump(acceleration)

    global_reasons: list[str] = []
    _append_reason(global_reasons, "reference_not_closed", not source_closed)
    _append_reason(global_reasons, "time_not_strictly_increasing", not time_valid)
    _append_reason(global_reasons, "non_finite_tcp_target", not positions_finite)
    _append_reason(global_reasons, "fixed_orientation_violation", not orientation_constant)
    _append_reason(global_reasons, "relative_start_not_zero", not first_relative_zero)
    _append_reason(global_reasons, "relative_end_not_zero", not final_relative_zero)
    _append_reason(global_reasons, "obvious_position_jump", not position_ok)
    _append_reason(global_reasons, "obvious_velocity_jump", not velocity_ok)
    _append_reason(global_reasons, "obvious_acceleration_jump", not acceleration_ok)
    geometry_matches_approved = bool(
        math.isclose(L1, APPROVED_FIRST_ROBOT_TRIAL_L1_M, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            L2,
            APPROVED_FIRST_ROBOT_TRIAL_L2_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    _append_reason(
        global_reasons,
        "equivalent_pull_point_geometry_not_approved",
        not geometry_matches_approved,
    )
    if global_reasons:
        sample_valid[:] = False
        for reasons in row_reasons:
            for reason in global_reasons:
                _append_reason(reasons, reason, True)

    trajectory = pd.DataFrame(
        {
            "time_s": time_s,
            "trajectory_id": trajectory_id,
            "trajectory_phase": source.get("cycle_phase", pd.Series([None] * len(source))),
            "delta_x_R": relative_rehab[:, 0],
            "delta_y_R": relative_rehab[:, 1],
            "delta_z_R": relative_rehab[:, 2],
            "tcp_x_base": target_position[:, 0],
            "tcp_y_base": target_position[:, 1],
            "tcp_z_base": target_position[:, 2],
            "tcp_rx": orientation[:, 0],
            "tcp_ry": orientation[:, 1],
            "tcp_rz": orientation[:, 2],
            "tcp_vx_base": velocity[:, 0],
            "tcp_vy_base": velocity[:, 1],
            "tcp_vz_base": velocity[:, 2],
            "tcp_ax_base": acceleration[:, 0],
            "tcp_ay_base": acceleration[:, 1],
            "tcp_az_base": acceleration[:, 2],
            "q_hip_ref": q_hip,
            "q_knee_ref": q_knee,
            "theta_shank_ref": q_hip - q_knee,
            "trajectory_valid": sample_valid,
            "invalid_reason": [";".join(reasons) for reasons in row_reasons],
            "experiment_mode": START_ANCHORED_RELATIVE_MODE,
            "tcp_orientation_strategy": "fixed_at_start_anchor",
        }
    )
    output_finite = bool(
        np.isfinite(
            trajectory[
                [
                    "time_s",
                    "delta_x_R",
                    "delta_y_R",
                    "delta_z_R",
                    "tcp_x_base",
                    "tcp_y_base",
                    "tcp_z_base",
                    "tcp_rx",
                    "tcp_ry",
                    "tcp_rz",
                    "q_hip_ref",
                    "q_knee_ref",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    audit_reasons = list(global_reasons)
    _append_reason(audit_reasons, "source_samples_invalid", not bool(source_valid.all()))
    _append_reason(audit_reasons, "source_formal_gate_invalid", not bool(formal_valid.all()))
    _append_reason(audit_reasons, "approved_rom_invalid", not bool(rom_valid.all()))
    _append_reason(audit_reasons, "theta_shank_definition_invalid", not bool(theta_valid_rows.all()))
    _append_reason(audit_reasons, "pull_point_fk_mismatch", not bool(fk_valid_rows.all()))
    _append_reason(audit_reasons, "non_finite_output", not output_finite)
    audit_valid = bool(trajectory["trajectory_valid"].all() and not audit_reasons)
    audit = RelativeTrajectoryAudit(
        trajectory_id=trajectory_id,
        sample_count=len(trajectory),
        first_relative_displacement_zero=first_relative_zero,
        final_relative_displacement_zero=final_relative_zero,
        first_target_equals_anchor=first_target_anchor,
        final_target_equals_anchor=final_target_anchor,
        orientation_constant=orientation_constant,
        source_samples_valid=bool(source_valid.all()),
        source_formal_gate_valid=bool(formal_valid.all()),
        approved_rom_valid=bool(rom_valid.all()),
        theta_shank_definition_valid=bool(theta_valid_rows.all()),
        pull_forward_kinematics_valid=bool(fk_valid_rows.all()),
        all_samples_finite=output_finite,
        time_strictly_increasing=time_valid,
        no_obvious_position_jump=position_ok,
        no_obvious_velocity_jump=velocity_ok,
        no_obvious_acceleration_jump=acceleration_ok,
        maximum_position_step_m=max_position_step,
        maximum_velocity_step_m_s=max_velocity_step,
        maximum_acceleration_step_m_s2=max_acceleration_step,
        trajectory_valid=audit_valid,
        invalid_reasons=tuple(audit_reasons),
    )
    metadata = {
        "experiment_mode": START_ANCHORED_RELATIVE_MODE,
        "trajectory_id": trajectory_id,
        "reference": source_metadata,
        "parent_reference_id": source_metadata.get("parent_reference_id"),
        "parent_reference_sha256": source_metadata.get(
            "parent_reference_sha256"
        ),
        "approved_first_robot_trial_reference_sha256": (
            APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256
        ),
        "reference_sha256_matches_approved_first_trial": bool(
            trajectory_id == FIRST_ROBOT_TRIAL_TRAJECTORY_ID
            and source_metadata.get("sha256")
            == APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256
            and load_reference_release_manifest()[
                "approved_for_first_robot_trial"
            ]
        ),
        "reference_sha256_matches_frozen_release": bool(
            trajectory_id == FIRST_ROBOT_TRIAL_TRAJECTORY_ID
            and source_metadata.get("sha256")
            == APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256
        ),
        "model_angle_definition": MODEL_ANGLE_DEFINITION,
        "approved_hip_rom_deg": list(APPROVED_HIP_ROM_DEG),
        "approved_knee_rom_deg": list(APPROVED_KNEE_ROM_DEG),
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "rehab_frame": rehab_frame.as_metadata(),
        "start_tcp_pose_base": anchor_pose.tolist(),
        "tcp_orientation_strategy": "fixed_at_start_anchor",
        "relative_formula": "p_tcp_B(t)=p_tcp_start_B+R_base_from_rehab@(p_R(t)-p_R(0))",
        "rehab_pull_point_formula": "p_R(t)=[x_pull_FK(t),0,z_pull_FK(t)]",
        "equivalent_pull_point_geometry": {
            "L1_hip_to_knee_m": float(L1),
            "L2_knee_to_equivalent_shank_strap_pull_point_m": float(L2),
            "physical_scope": (
                "supine passive hip-knee rehabilitation with an equivalent "
                "shank strap pull point"
            ),
            "matches_approved_first_trial_geometry": geometry_matches_approved,
        },
        "hip_center_required": False,
        "observed_ankle_used_as_pull_point": False,
        "tool_offset_applied_to_relative_increment": False,
        "tool_offset_retained_for_mode": ABSOLUTE_CALIBRATED_MODE,
        "allowed_for_first_robot_trial": bool(
            load_reference_release_manifest()["approved_for_first_robot_trial"]
        ),
        "first_robot_trial_candidate_whitelisted": bool(
            trajectory_id in ALLOWED_FIRST_ROBOT_TRIAL_TRAJECTORIES
        ),
        "trajectory_audit": audit.as_dict(),
        "robot_execution_approved": False,
    }
    return trajectory, audit, metadata


__all__ = [
    "ABSOLUTE_CALIBRATED_MODE",
    "ALLOWED_FIRST_ROBOT_TRIAL_TRAJECTORIES",
    "APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256",
    "APPROVED_FIRST_ROBOT_TRIAL_L1_M",
    "APPROVED_FIRST_ROBOT_TRIAL_L2_M",
    "FIRST_ROBOT_TRIAL_TRAJECTORY_ID",
    "RehabFrameConfig",
    "RelativeTrajectoryAudit",
    "START_ANCHORED_RELATIVE_MODE",
    "build_start_anchored_relative_trajectory",
    "load_rehab_frame_config",
]
