"""Stage 5C execution-ready reference versions and explicit ROM gate.

This module deliberately keeps the measured asymmetric cycle separate from the
closed, symmetric software execution reference.  The knee range used for a
Stage-5C run is a caller approval; it never mutates :mod:`lower_limb_sim.config`.

The model convention is always ``theta_shank = q_hip - q_knee``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import L1, L2, hip_range_deg, knee_range_deg
from .kinematics import forward_kinematics
from .reference_trajectory_retiming import (
    MODEL_ANGLE_DEFINITION,
    build_reference_phase_path,
    retime_reference_path,
)


MEASURED_REFERENCE = "reference_measured_asymmetric"
CLOSED_REFERENCE = "reference_closed_symmetric"
SOURCE_TIMING_STATUS = "unknown"


@dataclass(frozen=True)
class KneeRomApproval:
    """Run-local knee ROM approval; it does not alter the configured ROM."""

    minimum_deg: float
    maximum_deg: float

    def __post_init__(self) -> None:
        values = np.asarray((self.minimum_deg, self.maximum_deg), dtype=float)
        if not np.isfinite(values).all() or not values[0] < values[1]:
            raise ValueError("approved knee ROM must be two increasing finite values.")
        if values[0] < 0.0 or values[1] > 180.0:
            raise ValueError("approved knee ROM must remain within [0, 180] deg.")

    def as_list(self) -> list[float]:
        return [float(self.minimum_deg), float(self.maximum_deg)]


@dataclass(frozen=True)
class HipRomApproval:
    """Run-local hip ROM approval; it does not alter the configured ROM."""

    minimum_deg: float
    maximum_deg: float

    def __post_init__(self) -> None:
        values = np.asarray((self.minimum_deg, self.maximum_deg), dtype=float)
        if not np.isfinite(values).all() or not values[0] < values[1]:
            raise ValueError("approved hip ROM must be two increasing finite values.")
        if values[0] < 0.0 or values[1] > 180.0:
            raise ValueError("approved hip ROM must remain within [0, 180] deg.")
        configured = np.asarray(hip_range_deg, dtype=float)
        if values[0] < configured[0] or values[1] > configured[1]:
            raise ValueError(
                "run-local hip approval cannot expand the configured hip ROM."
            )

    def as_list(self) -> list[float]:
        return [float(self.minimum_deg), float(self.maximum_deg)]


@dataclass(frozen=True)
class ExecutionRomAudit:
    """The explicit decision that gates dynamics, identification and screening."""

    configured_hip_range_deg: tuple[float, float]
    configured_knee_range_deg: tuple[float, float]
    approved_hip_range_deg: tuple[float, float]
    approved_knee_range_deg: tuple[float, float] | None
    original_hip_range_deg: tuple[float, float]
    original_knee_range_deg: tuple[float, float]
    execution_hip_range_deg: tuple[float, float]
    execution_knee_range_deg: tuple[float, float]
    hip_approval_supplied: bool
    knee_approval_supplied: bool
    approval_supplied: bool
    rom_mapping_applied: bool
    mapping_formula: str | None
    formal_execution_allowed: bool
    trajectory_requires_rom_confirmation: bool
    block_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _range_deg(values: Sequence[float] | np.ndarray | pd.Series) -> tuple[float, float]:
    degrees = np.rad2deg(np.asarray(values, dtype=float))
    if degrees.size == 0 or not np.isfinite(degrees).all():
        raise ValueError("reference angles must be non-empty and finite.")
    return float(np.min(degrees)), float(np.max(degrees))


def _outside(values_rad: np.ndarray, limits_deg: Sequence[float]) -> np.ndarray:
    minimum, maximum = np.deg2rad(np.asarray(limits_deg, dtype=float))
    return (values_rad < minimum - 1e-12) | (values_rad > maximum + 1e-12)


def _attach_kinematics(dataframe: pd.DataFrame, hip_column: str, knee_column: str) -> None:
    q_hip = dataframe[hip_column].to_numpy(dtype=float)
    q_knee = dataframe[knee_column].to_numpy(dtype=float)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    dataframe["x_knee_m"] = x_knee
    dataframe["z_knee_m"] = z_knee
    dataframe["x_pull_m"] = x_pull
    dataframe["z_pull_m"] = z_pull


def build_execution_reference_versions(
    selected_cycle: pd.DataFrame,
    *,
    samples_per_segment: int = 201,
) -> pd.DataFrame:
    """Build the untouched measured version and a closed symmetric version.

    The phase-only table retains both copies of peak flexion.  Consequently the
    closed extension can be audited by the exact identity
    ``extension == flexion[::-1]`` before the duplicate join sample is removed
    by the existing retimer.
    """

    measured = build_reference_phase_path(
        selected_cycle, samples_per_segment=samples_per_segment
    ).copy(deep=True)
    measured["reference_version"] = MEASURED_REFERENCE
    measured["repeatable_loop"] = False
    measured["extension_source_is_measured"] = True
    measured["reference_provenance"] = "measured_flexion_and_extension"

    flexion = measured.loc[measured["cycle_phase"].eq("flexion")].copy(deep=True)
    if len(flexion) != samples_per_segment:
        raise RuntimeError("flexion phase sample count differs from the request.")
    extension = flexion.iloc[::-1].reset_index(drop=True)
    extension["cycle_phase"] = "extension"
    extension["segment_phase"] = np.linspace(0.0, 1.0, len(extension))
    extension["global_phase"] = 0.5 + 0.5 * extension["segment_phase"]
    extension["extension_source_is_measured"] = False
    extension["reference_provenance"] = "synthetic_time_reverse_of_measured_flexion"

    closed_flexion = flexion.copy(deep=True)
    closed_flexion["global_phase"] = 0.5 * closed_flexion["segment_phase"]
    closed_flexion["extension_source_is_measured"] = False
    closed_flexion["reference_provenance"] = "measured_flexion_branch_shape"
    closed = pd.concat((closed_flexion, extension), ignore_index=True)
    closed["reference_version"] = CLOSED_REFERENCE
    closed["repeatable_loop"] = True

    versions = pd.concat((measured, closed), ignore_index=True)
    versions["q_hip_original_rad"] = versions["q_hip_smoothed_rad"].to_numpy(float)
    versions["q_knee_original_rad"] = versions["q_knee_smoothed_rad"].to_numpy(float)
    versions["q_hip_reference_rad"] = versions["q_hip_original_rad"]
    versions["q_knee_reference_rad"] = versions["q_knee_original_rad"]
    versions["q_hip_rad"] = versions["q_hip_original_rad"]
    versions["q_knee_rad"] = versions["q_knee_original_rad"]
    versions["theta_shank_rad"] = (
        versions["q_hip_rad"] - versions["q_knee_rad"]
    )
    versions["theta_shank_reference_rad"] = (
        versions["q_hip_reference_rad"] - versions["q_knee_reference_rad"]
    )
    versions["model_angle_definition"] = MODEL_ANGLE_DEFINITION
    versions["source_timing_status"] = SOURCE_TIMING_STATUS
    versions["retimed_trajectory"] = False
    versions["retimed_timing_is_original"] = False
    versions["observed_ankle_is_pull_point"] = False
    versions["L1_m"] = L1
    versions["L2_m"] = L2
    versions["L2_definition"] = "knee_to_strap_equivalent_pull_point"
    _attach_kinematics(versions, "q_hip_reference_rad", "q_knee_reference_rad")

    closed_check = versions.loc[versions["reference_version"].eq(CLOSED_REFERENCE)]
    closed_flex = closed_check.loc[closed_check["cycle_phase"].eq("flexion")]
    closed_ext = closed_check.loc[closed_check["cycle_phase"].eq("extension")]
    for column in ("q_hip_original_rad", "q_knee_original_rad"):
        if not np.array_equal(
            closed_ext[column].to_numpy(float),
            closed_flex[column].to_numpy(float)[::-1],
        ):
            raise RuntimeError("closed extension is not the exact reversed flexion path.")
    if not np.allclose(
        versions["theta_shank_rad"],
        versions["q_hip_rad"] - versions["q_knee_rad"],
        atol=1e-14,
    ):
        raise RuntimeError("theta_shank convention was not preserved.")
    return versions


def apply_execution_rom_policy(
    reference_versions: pd.DataFrame,
    *,
    approved_knee_rom: KneeRomApproval | None,
    approved_hip_rom: HipRomApproval | None = None,
    apply_smooth_rom_mapping: bool = False,
) -> tuple[pd.DataFrame, ExecutionRomAudit]:
    """Apply an explicit, whole-path affine knee map or keep the run blocked.

    This function never performs pointwise clipping and never writes global
    configuration.  The measured version's original/reference fields are not
    modified; an optional mapping is applied only to the closed execution copy.
    """

    if apply_smooth_rom_mapping and approved_knee_rom is None:
        raise ValueError("ROM mapping requires an explicit approved knee range.")
    output = reference_versions.copy(deep=True)
    closed_mask = output["reference_version"].eq(CLOSED_REFERENCE).to_numpy()
    closed_original_hip = output.loc[closed_mask, "q_hip_original_rad"].to_numpy(float)
    closed_original_knee = output.loc[closed_mask, "q_knee_original_rad"].to_numpy(float)
    original_hip_range = _range_deg(closed_original_hip)
    original_knee_range = _range_deg(closed_original_knee)
    reasons: list[str] = []
    mapping_applied = False
    mapping_formula: str | None = None
    active_hip = (
        tuple(map(float, hip_range_deg))
        if approved_hip_rom is None
        else tuple(approved_hip_rom.as_list())
    )

    if bool(_outside(closed_original_hip, active_hip).any()):
        reasons.append("hip_outside_approved_rom")

    if approved_knee_rom is None:
        reasons.append("approved_knee_rom_missing")
        active_knee = tuple(map(float, knee_range_deg))
    else:
        active_knee = (
            float(approved_knee_rom.minimum_deg),
            float(approved_knee_rom.maximum_deg),
        )
        knee_outside = bool(_outside(closed_original_knee, active_knee).any())
        if knee_outside and not apply_smooth_rom_mapping:
            reasons.append("reference_outside_approved_knee_rom_mapping_not_authorized")
        elif knee_outside:
            raw_min = float(np.min(closed_original_knee))
            raw_max = float(np.max(closed_original_knee))
            if raw_max - raw_min <= 1e-12:
                raise ValueError("cannot amplitude-map a constant knee reference.")
            target_min, target_max = np.deg2rad(np.asarray(active_knee, dtype=float))
            normalized = (closed_original_knee - raw_min) / (raw_max - raw_min)
            mapped = target_min + normalized * (target_max - target_min)
            output.loc[closed_mask, "q_knee_reference_rad"] = mapped
            mapping_applied = True
            mapping_formula = (
                "q_new = approved_min + (q_original - original_min) / "
                "(original_max - original_min) * (approved_max - approved_min)"
            )

    execution_hip = output.loc[closed_mask, "q_hip_reference_rad"].to_numpy(float)
    execution_knee = output.loc[closed_mask, "q_knee_reference_rad"].to_numpy(float)
    if approved_knee_rom is not None and bool(_outside(execution_knee, active_knee).any()):
        reasons.append("knee_outside_approved_rom_after_policy")

    formal_allowed = len(reasons) == 0
    output["approved_hip_min_deg"] = active_hip[0]
    output["approved_hip_max_deg"] = active_hip[1]
    output["approved_hip_rom_supplied"] = approved_hip_rom is not None
    output["approved_knee_min_deg"] = active_knee[0]
    output["approved_knee_max_deg"] = active_knee[1]
    output["approved_knee_rom_supplied"] = approved_knee_rom is not None
    output["rom_mapping_applied"] = False
    output.loc[closed_mask, "rom_mapping_applied"] = mapping_applied
    output["trajectory_requires_rom_confirmation"] = not formal_allowed
    output["formal_execution_allowed"] = formal_allowed
    output["dynamics_allowed"] = formal_allowed
    output["rom_gate_reason"] = ";".join(reasons)
    output["q_hip_approved_min_deg"] = active_hip[0]
    output["q_hip_approved_max_deg"] = active_hip[1]
    output["q_knee_approved_min_deg"] = active_knee[0]
    output["q_knee_approved_max_deg"] = active_knee[1]
    output["q_hip_rad"] = output["q_hip_reference_rad"]
    output["q_knee_rad"] = output["q_knee_reference_rad"]
    output["theta_shank_rad"] = output["q_hip_rad"] - output["q_knee_rad"]
    output["theta_shank_reference_rad"] = output["theta_shank_rad"]
    output["joint_limit_valid"] = ~(
        _outside(output["q_hip_reference_rad"].to_numpy(float), active_hip)
        | _outside(output["q_knee_reference_rad"].to_numpy(float), active_knee)
    )
    _attach_kinematics(output, "q_hip_reference_rad", "q_knee_reference_rad")

    audit = ExecutionRomAudit(
        configured_hip_range_deg=tuple(map(float, hip_range_deg)),
        configured_knee_range_deg=tuple(map(float, knee_range_deg)),
        approved_hip_range_deg=tuple(active_hip),
        approved_knee_range_deg=(
            None if approved_knee_rom is None else tuple(active_knee)
        ),
        original_hip_range_deg=original_hip_range,
        original_knee_range_deg=original_knee_range,
        execution_hip_range_deg=_range_deg(execution_hip),
        execution_knee_range_deg=_range_deg(execution_knee),
        hip_approval_supplied=approved_hip_rom is not None,
        knee_approval_supplied=approved_knee_rom is not None,
        approval_supplied=approved_knee_rom is not None,
        rom_mapping_applied=mapping_applied,
        mapping_formula=mapping_formula,
        formal_execution_allowed=formal_allowed,
        trajectory_requires_rom_confirmation=not formal_allowed,
        block_reasons=tuple(reasons),
    )
    return output, audit


def closed_execution_phase_path(reference_versions: pd.DataFrame) -> pd.DataFrame:
    """Return the closed version in the schema expected by the Stage-5B retimer."""

    closed = reference_versions.loc[
        reference_versions["reference_version"].eq(CLOSED_REFERENCE)
    ].copy(deep=True)
    if closed.empty:
        raise ValueError("reference versions do not contain the closed reference.")
    closed["source_angle_valid"] = closed.get(
        "source_angle_valid", pd.Series(True, index=closed.index)
    ).fillna(False).astype(bool)
    return closed.reset_index(drop=True)


def retime_closed_reference(
    reference_versions: pd.DataFrame,
    *,
    profile: str,
    flexion_duration_s: float,
    extension_duration_s: float,
    samples_per_segment: int = 201,
) -> pd.DataFrame:
    """Reuse the existing minimum-jerk path retimer for the closed reference."""

    phase_path = closed_execution_phase_path(reference_versions)
    trajectory = retime_reference_path(
        phase_path,
        profile=profile,
        flexion_duration_s=flexion_duration_s,
        extension_duration_s=extension_duration_s,
        samples_per_segment=samples_per_segment,
    )
    trajectory.insert(0, "reference_version", CLOSED_REFERENCE)
    trajectory["repeatable_loop"] = True
    trajectory["extension_source_is_measured"] = False
    trajectory["formal_execution_allowed"] = bool(
        phase_path["formal_execution_allowed"].iloc[0]
    )
    return trajectory


def closure_metrics(dataframe: pd.DataFrame) -> dict[str, float]:
    """Return signed joint closure and Euclidean pull-point closure errors."""

    return {
        "q_hip_closure_error_deg": float(
            np.rad2deg(dataframe["q_hip_rad"].iloc[-1] - dataframe["q_hip_rad"].iloc[0])
        ),
        "q_knee_closure_error_deg": float(
            np.rad2deg(dataframe["q_knee_rad"].iloc[-1] - dataframe["q_knee_rad"].iloc[0])
        ),
        "pull_point_closure_error_m": float(
            np.hypot(
                dataframe["x_pull_m"].iloc[-1] - dataframe["x_pull_m"].iloc[0],
                dataframe["z_pull_m"].iloc[-1] - dataframe["z_pull_m"].iloc[0],
            )
        ),
    }


__all__ = [
    "CLOSED_REFERENCE",
    "ExecutionRomAudit",
    "HipRomApproval",
    "KneeRomApproval",
    "MEASURED_REFERENCE",
    "apply_execution_rom_policy",
    "build_execution_reference_versions",
    "closed_execution_phase_path",
    "closure_metrics",
    "retime_closed_reference",
]
