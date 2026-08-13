"""Active-asymmetric reference-local identification experiment primitives.

This module reuses the existing five-parameter estimator, numerical
identifiability analysis, minimum-jerk retiming, observation model, and
axis-aligned six-state domain definition.  It never regenerates or overwrites
the active reference and has no dependency on robot, hardware, control,
collection, or safety packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    identification_initial_guess,
    jacobian_condition_limit,
    jacobian_det_threshold,
    model_mismatch_nrmse_epsilon_nm,
    model_mismatch_nrmse_minimum_range_nm,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_ID as FORMAL_ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .dynamic_subject import get_dynamic_subject
from .geometry_error_metrics import StateDomainBounds
from .identifiability_analysis import IdentifiabilityResult, analyze_identifiability
from .jacobian import jacobian_diagnostics
from .kinematics import forward_kinematics
from .mismatch_metrics import (
    build_generic_vs_identified_comparison,
    compute_all_model_metrics,
)
from .parameter_estimator import (
    PARAMETER_NAMES,
    REQUIRED_OBSERVATION_COLUMNS,
    ParameterEstimationResult,
    baseline_template_from_dynamic_subject,
    measured_joint_torque,
    predict_joint_torque,
    valid_observations,
)
from .reference_local_excitation import (
    SUBJECT_IDS,
    build_local_domain_coverage,
    build_local_identification_dataset,
    fit_local_identification_domain,
    fit_local_subject_parameters,
    perturb_closed_phase_path,
)
from .reference_trajectory_retiming import retime_reference_path
from .reference_release import (
    RELEASE_ACTIVE_REFERENCE_PATH,
    RELEASE_METADATA_PATH,
    RELEASE_VERSION_MANIFEST_PATH,
    load_frozen_active_reference,
)


MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_DIRECTORY = MODULE_DIR / "data" / "reference_candidates"
ACTIVE_MANIFEST_PATH = RELEASE_VERSION_MANIFEST_PATH
ACTIVE_METADATA_PATH = RELEASE_METADATA_PATH
ACTIVE_REFERENCE_PATH = RELEASE_ACTIVE_REFERENCE_PATH
ACTIVE_NOMINAL_PATH = (
    REFERENCE_DIRECTORY / "reference_measured_asymmetric_closed_nominal.csv"
)
LEGACY_DATASET_PATH = (
    REFERENCE_DIRECTORY / "reference_local_identification_dataset.csv"
)
LEGACY_PARAMETER_PATH = (
    REFERENCE_DIRECTORY / "reference_local_parameter_estimates.csv"
)
LEGACY_COVERAGE_PATH = (
    REFERENCE_DIRECTORY / "reference_local_domain_coverage.csv"
)

ACTIVE_REFERENCE_ID = "reference_measured_asymmetric_closed_slow"
ACTIVE_NOMINAL_ID = "reference_measured_asymmetric_closed_nominal"
MODEL_VERSION = "lower_limb_sim_reference_local_active_asymmetric_v1"
RANDOM_SEED = 20260811
DOMAIN_MODEL = "axis_aligned_6d_active_asymmetric_reference_local_train_only"
SPLIT_DEFINITION_ID = "active_asymmetric_reference_local_split_v1"

if ACTIVE_REFERENCE_ID != FORMAL_ACTIVE_REFERENCE_ID:
    raise RuntimeError("active local-identification reference disagrees with manifest")


@dataclass(frozen=True)
class ActiveLocalTrajectorySpecification:
    trajectory_id: str
    dataset_split: str
    evaluation_role: str
    total_duration_s: float | None = None
    hip_amplitude_reduction_deg: float = 0.0
    knee_amplitude_reduction_deg: float = 0.0
    knee_phase_shift_fraction: float = 0.0
    exact_source_profile: str | None = None


TRAJECTORY_SPECIFICATIONS: tuple[ActiveLocalTrajectorySpecification, ...] = (
    ActiveLocalTrajectorySpecification(
        "train_hip_amplitude_minus_3deg_slow",
        "train",
        "reference_local_excitation",
        total_duration_s=24.0,
        hip_amplitude_reduction_deg=3.0,
    ),
    ActiveLocalTrajectorySpecification(
        "train_knee_amplitude_minus_3deg_slow",
        "train",
        "reference_local_excitation",
        total_duration_s=24.0,
        knee_amplitude_reduction_deg=3.0,
    ),
    ActiveLocalTrajectorySpecification(
        "train_knee_phase_advance_3pct_slow",
        "train",
        "reference_local_excitation",
        total_duration_s=24.0,
        knee_phase_shift_fraction=0.03,
    ),
    ActiveLocalTrajectorySpecification(
        "train_hip_amplitude_minus_3deg_nominal",
        "train",
        "reference_local_excitation",
        total_duration_s=12.0,
        hip_amplitude_reduction_deg=3.0,
    ),
    ActiveLocalTrajectorySpecification(
        "train_knee_amplitude_minus_3deg_nominal",
        "train",
        "reference_local_excitation",
        total_duration_s=12.0,
        knee_amplitude_reduction_deg=3.0,
    ),
    ActiveLocalTrajectorySpecification(
        "train_knee_phase_delay_3pct_nominal",
        "train",
        "reference_local_excitation",
        total_duration_s=12.0,
        knee_phase_shift_fraction=-0.03,
    ),
    ActiveLocalTrajectorySpecification(
        "validation_hip_amplitude_minus_2deg_intermediate",
        "validation",
        "reference_local_interpolation",
        total_duration_s=18.0,
        hip_amplitude_reduction_deg=2.0,
    ),
    ActiveLocalTrajectorySpecification(
        "validation_knee_phase_advance_2pct_intermediate",
        "validation",
        "reference_local_interpolation",
        total_duration_s=18.0,
        knee_phase_shift_fraction=0.02,
    ),
    ActiveLocalTrajectorySpecification(
        "heldout_knee_phase_delay_2pct_intermediate",
        "test",
        "held_out_within_reference_local_domain",
        total_duration_s=18.0,
        knee_phase_shift_fraction=-0.02,
    ),
    ActiveLocalTrajectorySpecification(
        "heldout_active_reference_slow",
        "test",
        "held_out_exact_active_reference",
        exact_source_profile="slow",
    ),
    ActiveLocalTrajectorySpecification(
        "heldout_active_reference_nominal",
        "test",
        "held_out_relevant_nominal_profile",
        exact_source_profile="nominal",
    ),
    ActiveLocalTrajectorySpecification(
        "heldout_boundary_speed_plus_10pct",
        "test",
        "held_out_boundary_near_speed",
        total_duration_s=10.8,
    ),
)

SPLIT_BY_TRAJECTORY = {
    item.trajectory_id: item.dataset_split for item in TRAJECTORY_SPECIFICATIONS
}
TRAINING_TRAJECTORY_IDS = tuple(
    item.trajectory_id
    for item in TRAJECTORY_SPECIFICATIONS
    if item.dataset_split == "train"
)
HELD_OUT_TRAJECTORY_IDS = tuple(
    item.trajectory_id
    for item in TRAJECTORY_SPECIFICATIONS
    if item.dataset_split == "test"
)


@dataclass(frozen=True)
class ActiveReferenceBundle:
    active: pd.DataFrame
    nominal: pd.DataFrame
    metadata: dict[str, object]
    summary: dict[str, object]


@dataclass(frozen=True)
class ActiveReferenceLocalResult:
    reference: ActiveReferenceBundle
    trajectories: dict[str, pd.DataFrame]
    excitation_metadata: pd.DataFrame
    dataset: pd.DataFrame
    identified_parameters: pd.DataFrame
    parameter_errors: pd.DataFrame
    optimizer_results: dict[str, ParameterEstimationResult]
    estimates_by_subject: dict[str, dict[str, float]]
    domain_bounds: StateDomainBounds
    domain_coverage: pd.DataFrame
    prediction_samples: pd.DataFrame
    prediction_metrics: pd.DataFrame
    generic_vs_identified: pd.DataFrame
    identifiability_summary: pd.DataFrame
    parameter_correlations: pd.DataFrame
    singular_values: pd.DataFrame
    legacy_identifiability_summary: pd.DataFrame
    legacy_prediction_metrics: pd.DataFrame
    legacy_comparison: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(dataframe: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _active_reference_summary(
    active: pd.DataFrame,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    time_s = active["time_s"].to_numpy(dtype=float)
    q_hip = active["q_hip_rad"].to_numpy(dtype=float)
    q_knee = active["q_knee_rad"].to_numpy(dtype=float)
    pull = active[["x_pull_m", "z_pull_m"]].to_numpy(dtype=float)
    dt = np.diff(time_s)
    branch_dt: dict[str, dict[str, float]] = {}
    for branch in ("flexion", "extension"):
        branch_time = active.loc[active["cycle_phase"].eq(branch), "time_s"].to_numpy(
            dtype=float
        )
        branch_steps = np.diff(branch_time)
        branch_dt[branch] = {
            "minimum_s": float(np.min(branch_steps)),
            "median_s": float(np.median(branch_steps)),
            "maximum_s": float(np.max(branch_steps)),
        }
    continuity = dict(metadata["continuity_audit"])
    asymmetry = dict(metadata["flexion_extension_asymmetry"])
    closure_q = np.array([q_hip[-1] - q_hip[0], q_knee[-1] - q_knee[0]])
    closure_pull = pull[-1] - pull[0]
    return {
        "source_file": str(ACTIVE_REFERENCE_PATH.resolve()),
        "metadata_file": str(ACTIVE_METADATA_PATH.resolve()),
        "manifest_file": str(ACTIVE_MANIFEST_PATH.resolve()),
        "active_reference_identifier": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": sha256_file(ACTIVE_REFERENCE_PATH),
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "formal_hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "formal_knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "trajectory_duration_s": float(time_s[-1] - time_s[0]),
        "sample_count": int(len(active)),
        "sampling_interval": {
            "nonuniform_across_asymmetric_branch_durations": bool(
                not np.allclose(dt, dt[0], atol=1e-14, rtol=0.0)
            ),
            "minimum_s": float(np.min(dt)),
            "median_s": float(np.median(dt)),
            "maximum_s": float(np.max(dt)),
            "by_branch": branch_dt,
        },
        "q_hip_range_deg": [
            float(np.rad2deg(np.min(q_hip))),
            float(np.rad2deg(np.max(q_hip))),
        ],
        "q_knee_range_deg": [
            float(np.rad2deg(np.min(q_knee))),
            float(np.rad2deg(np.max(q_knee))),
        ],
        "theta_shank_convention": f"theta_shank = {THETA_SHANK_DEFINITION}",
        "q_closure_error_rad": float(np.linalg.norm(closure_q)),
        "q_closure_error_deg": float(np.linalg.norm(np.rad2deg(closure_q))),
        "pull_point_closure_error_m": float(np.linalg.norm(closure_pull)),
        "continuity_level": f"C{int(continuity['continuity_order'])}",
        "continuity_audit": continuity,
        "asymmetry_audit": asymmetry,
        "is_asymmetric": bool(asymmetry["asymmetry_preserved"]),
        "q_start_equals_q_end": bool(
            np.allclose(closure_q, 0.0, atol=1e-12, rtol=0.0)
        ),
        "model_angle_identity_valid": bool(
            np.allclose(
                active["theta_shank_rad"].to_numpy(dtype=float),
                q_hip - q_knee,
                atol=1e-14,
                rtol=0.0,
            )
        ),
        "active_manifest_row_verified": True,
        "reference_file_modified_by_experiment": False,
    }


def load_active_reference() -> ActiveReferenceBundle:
    """Load and fail-closed verify the one manifest-declared active reference."""

    frozen = load_frozen_active_reference()
    for path in (
        ACTIVE_MANIFEST_PATH,
        ACTIVE_METADATA_PATH,
        ACTIVE_REFERENCE_PATH,
        ACTIVE_NOMINAL_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required active-reference artifact missing: {path}")
    manifest = pd.read_csv(ACTIVE_MANIFEST_PATH)
    _require_columns(
        manifest,
        {"trajectory_id", "active_reference", "path", "legacy_software_comparison"},
        "reference version manifest",
    )
    active_rows = manifest.loc[manifest["active_reference"].astype(bool)]
    if len(active_rows) != 1:
        raise RuntimeError("reference manifest must declare exactly one active row.")
    row = active_rows.iloc[0]
    if str(row["trajectory_id"]) != ACTIVE_REFERENCE_ID:
        raise RuntimeError("manifest active row is not the expected asymmetric reference.")
    if bool(row["legacy_software_comparison"]):
        raise RuntimeError("manifest active row is incorrectly marked legacy.")
    manifest_path = (ACTIVE_MANIFEST_PATH.parent / str(row["path"])).resolve()
    if manifest_path != ACTIVE_REFERENCE_PATH.resolve():
        raise RuntimeError(
            "manifest active path differs from the canonical asymmetric reference path."
        )

    with ACTIVE_METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("active_reference_trajectory") != ACTIVE_REFERENCE_ID:
        raise RuntimeError("metadata active trajectory identifier does not match manifest.")
    actual_sha = sha256_file(ACTIVE_REFERENCE_PATH)
    if actual_sha != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active-reference SHA-256 differs from formal manifest.")
    if metadata.get("active_reference_sha256") != actual_sha:
        raise RuntimeError("active-reference SHA-256 does not match release metadata.")
    if metadata.get("measured_extension_is_reversed_flexion") is not False:
        raise RuntimeError("active extension provenance is not measured asymmetric motion.")

    active = pd.read_csv(ACTIVE_REFERENCE_PATH)
    nominal = pd.read_csv(ACTIVE_NOMINAL_PATH)
    required = {
        "trajectory_id",
        "time_s",
        "cycle_phase",
        "segment_phase",
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
        "trajectory_sample_valid",
    }
    _require_columns(active, required, "active reference")
    _require_columns(nominal, required, "active nominal profile")
    if not active["trajectory_id"].astype(str).eq(ACTIVE_REFERENCE_ID).all():
        raise RuntimeError("active reference CSV contains another trajectory ID.")
    if not nominal["trajectory_id"].astype(str).eq(ACTIVE_NOMINAL_ID).all():
        raise RuntimeError("nominal profile CSV contains another trajectory ID.")
    if not active["active_reference"].astype(bool).all():
        raise RuntimeError("active reference CSV does not mark every row active.")
    if not active["trajectory_sample_valid"].astype(bool).all():
        raise RuntimeError("active reference contains invalid trajectory samples.")
    hip = np.rad2deg(active["q_hip_rad"].to_numpy(dtype=float))
    knee = np.rad2deg(active["q_knee_rad"].to_numpy(dtype=float))
    if not (
        ((hip >= FORMAL_HIP_ROM_DEG[0]) & (hip <= FORMAL_HIP_ROM_DEG[1])).all()
        and ((knee >= FORMAL_KNEE_ROM_DEG[0]) & (knee <= FORMAL_KNEE_ROM_DEG[1])).all()
    ):
        raise RuntimeError("active reference is outside ROM_PROTOCOL_V2")

    summary = _active_reference_summary(active, metadata)
    if not summary["q_start_equals_q_end"]:
        raise RuntimeError("active asymmetric reference is not closed.")
    if not summary["model_angle_identity_valid"]:
        raise RuntimeError("active reference violates theta_shank = q_hip - q_knee.")
    if summary["continuity_level"] != "C2":
        raise RuntimeError("active reference is not C2 according to release metadata.")
    if not summary["is_asymmetric"]:
        raise RuntimeError("active reference is not verified asymmetric.")
    summary["parent_reference_id"] = frozen.manifest["reference_id"]
    summary["parent_reference_sha256"] = frozen.manifest["sha256"]
    summary["approved_for_offline_personalization"] = frozen.manifest[
        "approved_for_offline_personalization"
    ]
    summary["approved_for_first_robot_trial"] = frozen.manifest[
        "approved_for_first_robot_trial"
    ]
    return ActiveReferenceBundle(active, nominal, metadata, summary)


def _phase_path_from_active(active: pd.DataFrame) -> pd.DataFrame:
    """Build a branch-complete phase table without changing source samples."""

    flexion = active.loc[active["cycle_phase"].eq("flexion")].copy(deep=True)
    extension = active.loc[active["cycle_phase"].eq("extension")].copy(deep=True)
    if flexion.empty or extension.empty:
        raise ValueError("active reference must contain flexion and extension.")
    peak_for_extension = flexion.iloc[[-1]].copy(deep=True)
    peak_for_extension["cycle_phase"] = "extension"
    peak_for_extension["segment_phase"] = 0.0
    extension = pd.concat((peak_for_extension, extension), ignore_index=True)
    output = pd.concat((flexion, extension), ignore_index=True)
    output["q_hip_reference_rad"] = output["q_hip_rad"].to_numpy(dtype=float)
    output["q_knee_reference_rad"] = output["q_knee_rad"].to_numpy(dtype=float)
    for joint in ("hip", "knee"):
        output[f"q_{joint}_raw_rad"] = output[f"q_{joint}_rad"].to_numpy(dtype=float)
        output[f"q_{joint}_smoothed_rad"] = output[f"q_{joint}_rad"].to_numpy(
            dtype=float
        )
    output["source_angle_valid"] = True
    output["rom_mapping_applied"] = False
    output["trajectory_requires_rom_confirmation"] = False
    output["dynamics_allowed"] = True
    output["q_hip_approved_min_deg"] = float(active["approved_hip_min_deg"].iloc[0])
    output["q_hip_approved_max_deg"] = float(active["approved_hip_max_deg"].iloc[0])
    output["q_knee_approved_min_deg"] = float(
        active["approved_knee_min_deg"].iloc[0]
    )
    output["q_knee_approved_max_deg"] = float(
        active["approved_knee_max_deg"].iloc[0]
    )
    for branch in ("flexion", "extension"):
        phase = output.loc[output["cycle_phase"].eq(branch), "segment_phase"].to_numpy(
            dtype=float
        )
        if (
            not np.all(np.diff(phase) > 0.0)
            or not np.isclose(phase[0], 0.0)
            or not np.isclose(phase[-1], 1.0)
        ):
            raise RuntimeError(f"active {branch} branch is not a complete phase path.")
    return output


def _duration_pair(total_duration_s: float, active: pd.DataFrame) -> tuple[float, float]:
    flexion_duration = float(
        active.loc[active["cycle_phase"].eq("flexion"), "time_s"].max()
        - active["time_s"].iloc[0]
    )
    total = float(active["time_s"].iloc[-1] - active["time_s"].iloc[0])
    ratio = flexion_duration / total
    return float(total_duration_s * ratio), float(total_duration_s * (1.0 - ratio))


def _annotate_trajectory(
    trajectory: pd.DataFrame,
    specification: ActiveLocalTrajectorySpecification,
    reference_sha256: str,
    *,
    exact_source_samples: bool,
) -> pd.DataFrame:
    output = trajectory.copy(deep=True)
    source_id = (
        output["trajectory_id"].astype(str).iloc[0]
        if "trajectory_id" in output
        else ""
    )
    output["source_trajectory_id"] = source_id
    output["trajectory_id"] = specification.trajectory_id
    output["dataset_split"] = specification.dataset_split
    output["evaluation_role"] = specification.evaluation_role
    output["active_reference_identifier"] = ACTIVE_REFERENCE_ID
    output["active_reference_sha256"] = reference_sha256
    output["parent_reference_id"] = ACTIVE_REFERENCE_ID
    output["parent_reference_sha256"] = reference_sha256
    output["active_reference_source_path"] = str(ACTIVE_REFERENCE_PATH.resolve())
    output["active_reference_exact_samples"] = exact_source_samples
    output["hip_amplitude_reduction_deg"] = (
        specification.hip_amplitude_reduction_deg
    )
    output["knee_amplitude_reduction_deg"] = (
        specification.knee_amplitude_reduction_deg
    )
    output["knee_phase_shift_fraction"] = (
        specification.knee_phase_shift_fraction
    )
    output["identification_trajectory_type"] = (
        "active_asymmetric_reference_neighbourhood_identification"
    )
    output["model_angle_definition"] = "theta_shank = q_hip - q_knee"
    output["simulation_status"] = "software_only"
    output["robot_execution_requested"] = False
    output["clinical_validation_status"] = "not_clinically_validated"
    if not np.allclose(
        output["theta_shank_rad"].to_numpy(dtype=float),
        output["q_hip_rad"].to_numpy(dtype=float)
        - output["q_knee_rad"].to_numpy(dtype=float),
        atol=1e-14,
        rtol=0.0,
    ):
        raise RuntimeError(f"{specification.trajectory_id} violates shank convention.")
    return output


def build_active_local_trajectories(
    reference: ActiveReferenceBundle,
    *,
    samples_per_segment: int = 201,
) -> dict[str, pd.DataFrame]:
    """Generate the frozen conservative active-reference-local trajectory set."""

    base_phase_path = _phase_path_from_active(reference.active)
    reference_sha = str(reference.summary["active_reference_sha256"])
    trajectories: dict[str, pd.DataFrame] = {}
    peak_global_phase = float(
        base_phase_path.loc[
            base_phase_path["cycle_phase"].eq("flexion"), "global_phase"
        ].max()
    )
    for specification in TRAJECTORY_SPECIFICATIONS:
        if specification.exact_source_profile is not None:
            source = (
                reference.active
                if specification.exact_source_profile == "slow"
                else reference.nominal
            )
            trajectory = _annotate_trajectory(
                source,
                specification,
                reference_sha,
                exact_source_samples=True,
            )
        else:
            phase_path = perturb_closed_phase_path(
                base_phase_path,
                trajectory_id=specification.trajectory_id,
                hip_amplitude_reduction_deg=(
                    specification.hip_amplitude_reduction_deg
                ),
                knee_amplitude_reduction_deg=(
                    specification.knee_amplitude_reduction_deg
                ),
                knee_phase_shift_fraction=(
                    specification.knee_phase_shift_fraction
                ),
            )
            flexion_s, extension_s = _duration_pair(
                float(specification.total_duration_s), reference.active
            )
            trajectory = retime_reference_path(
                phase_path,
                profile=specification.trajectory_id,
                flexion_duration_s=flexion_s,
                extension_duration_s=extension_s,
                samples_per_segment=samples_per_segment,
            )
            local_phase = trajectory["segment_phase"].to_numpy(dtype=float)
            flexion_mask = trajectory["cycle_phase"].eq("flexion").to_numpy()
            trajectory["global_phase"] = np.where(
                flexion_mask,
                peak_global_phase * local_phase,
                peak_global_phase + (1.0 - peak_global_phase) * local_phase,
            )
            trajectory["reference_version"] = (
                "reference_measured_asymmetric_closed_local_derivative"
            )
            trajectory["repeatable_loop"] = True
            trajectory["formal_execution_allowed"] = False
            trajectory["offline_identification_allowed"] = True
            trajectory = _annotate_trajectory(
                trajectory,
                specification,
                reference_sha,
                exact_source_samples=False,
            )
        trajectories[specification.trajectory_id] = trajectory
    if set(trajectories) != set(SPLIT_BY_TRAJECTORY):
        raise RuntimeError("active local trajectory set is incomplete.")
    if any(
        trajectory_id.startswith("reference_slow")
        or trajectory_id.startswith("reference_closed")
        for trajectory_id in trajectories
    ):
        raise RuntimeError("legacy symmetric trajectory silently entered active set.")
    return trajectories


def _trajectory_feasibility(trajectory: pd.DataFrame) -> dict[str, object]:
    q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
    q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    diagnostics = jacobian_diagnostics(q_hip, q_knee, L1, L2)
    determinant = np.asarray(diagnostics.determinant, dtype=float)
    condition = np.asarray(diagnostics.condition_number, dtype=float)
    near_singular = np.asarray(diagnostics.near_singular, dtype=bool)
    closure_q = np.array([q_hip[-1] - q_hip[0], q_knee[-1] - q_knee[0]])
    closure_pull = np.array([x_pull[-1] - x_pull[0], z_pull[-1] - z_pull[0]])
    endpoint_derivative = trajectory.iloc[[0, -1]][
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(dtype=float)
    rom_valid = bool(trajectory["joint_limit_valid"].astype(bool).all())
    finite_valid = bool(trajectory["trajectory_sample_valid"].astype(bool).all())
    workspace_valid = bool(
        np.all(x_pull >= -1e-12)
        and np.all(z_pull >= -1e-12)
        and np.all(z_knee >= -1e-12)
    )
    jacobian_valid = bool(
        not near_singular.any()
        and np.min(np.abs(determinant)) >= jacobian_det_threshold
        and np.max(condition) <= jacobian_condition_limit
    )
    closed = bool(
        np.allclose(closure_q, 0.0, atol=1e-12, rtol=0.0)
        and np.allclose(closure_pull, 0.0, atol=1e-12, rtol=0.0)
    )
    c2_stationary_seam = bool(
        np.allclose(endpoint_derivative, 0.0, atol=1e-12, rtol=0.0)
    )
    return {
        "trajectory_id": str(trajectory["trajectory_id"].iloc[0]),
        "parent_reference_id": str(
            trajectory["parent_reference_id"].iloc[0]
        ),
        "parent_reference_sha256": str(
            trajectory["parent_reference_sha256"].iloc[0]
        ),
        "dataset_split": str(trajectory["dataset_split"].iloc[0]),
        "evaluation_role": str(trajectory["evaluation_role"].iloc[0]),
        "sample_count": int(len(trajectory)),
        "total_duration_s": float(trajectory["time_s"].iloc[-1]),
        "q_hip_min_deg": float(np.rad2deg(np.min(q_hip))),
        "q_hip_max_deg": float(np.rad2deg(np.max(q_hip))),
        "q_knee_min_deg": float(np.rad2deg(np.min(q_knee))),
        "q_knee_max_deg": float(np.rad2deg(np.max(q_knee))),
        "max_abs_dq_hip_rad_s": float(np.max(np.abs(trajectory["dq_hip_rad_s"]))),
        "max_abs_dq_knee_rad_s": float(np.max(np.abs(trajectory["dq_knee_rad_s"]))),
        "max_abs_ddq_hip_rad_s2": float(
            np.max(np.abs(trajectory["ddq_hip_rad_s2"]))
        ),
        "max_abs_ddq_knee_rad_s2": float(
            np.max(np.abs(trajectory["ddq_knee_rad_s2"]))
        ),
        "velocity_limit_configured": False,
        "velocity_limit_status": "not_configured_offline_peak_report_only",
        "acceleration_limit_configured": False,
        "acceleration_limit_status": "not_configured_offline_peak_report_only",
        "rom_valid": rom_valid,
        "finite_state_valid": finite_valid,
        "workspace_valid": workspace_valid,
        "minimum_x_pull_m": float(np.min(x_pull)),
        "minimum_z_pull_m": float(np.min(z_pull)),
        "minimum_z_knee_m": float(np.min(z_knee)),
        "jacobian_valid": jacobian_valid,
        "minimum_abs_jacobian_determinant": float(np.min(np.abs(determinant))),
        "maximum_jacobian_condition": float(np.max(condition)),
        "closed": closed,
        "q_closure_error_rad": float(np.linalg.norm(closure_q)),
        "pull_closure_error_m": float(np.linalg.norm(closure_pull)),
        "c2_stationary_seam": c2_stationary_seam,
        "theta_shank_convention_valid": bool(
            np.allclose(
                trajectory["theta_shank_rad"].to_numpy(dtype=float),
                q_hip - q_knee,
                atol=1e-14,
                rtol=0.0,
            )
        ),
        "offline_feasible": bool(
            rom_valid
            and finite_valid
            and workspace_valid
            and jacobian_valid
            and closed
            and c2_stationary_seam
        ),
    }


def build_excitation_metadata(
    trajectories: Mapping[str, pd.DataFrame],
    dataset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [_trajectory_feasibility(trajectory) for trajectory in trajectories.values()]
    result = pd.DataFrame(rows)
    if dataset is not None and not dataset.empty:
        force = dataset.copy(deep=False)
        force["force_magnitude_observed_n"] = np.hypot(
            force["fx_observed_n"].to_numpy(dtype=float),
            force["fz_observed_n"].to_numpy(dtype=float),
        )
        force_summary = (
            force.groupby("trajectory_id", sort=False)
            .agg(
                force_mapping_valid_percent=(
                    "force_mapping_valid",
                    lambda values: 100.0 * float(np.mean(values.astype(bool))),
                ),
                maximum_observed_force_n=("force_magnitude_observed_n", "max"),
            )
            .reset_index()
        )
        result = result.merge(force_summary, on="trajectory_id", validate="one_to_one")
        result["offline_feasible"] &= result["force_mapping_valid_percent"].eq(100.0)
    return result


def _identified_parameter_table(
    optimizer_results: Mapping[str, ParameterEstimationResult],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, result in optimizer_results.items():
        rows.append(
            {
                "subject_id": subject_id,
                **result.estimated_parameters,
                "optimizer_success": result.optimizer_success,
                "optimizer_message": result.optimizer_message,
                "optimizer_cost": result.cost,
                "number_of_function_evaluations": result.number_of_function_evaluations,
                "valid_training_samples": result.valid_training_samples,
                "fit_split": "train_only",
                "training_trajectory_ids": ";".join(TRAINING_TRAJECTORY_IDS),
                "test_used_for_fit": False,
                "validation_used_for_fit": False,
            }
        )
    return pd.DataFrame(rows)


def _identifiability_tables(
    dataset: pd.DataFrame,
    estimates_by_subject: Mapping[str, Mapping[str, float]],
    *,
    analysis_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    summary_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    singular_rows: list[dict[str, object]] = []
    for subject_id, parameters in estimates_by_subject.items():
        training = dataset.loc[
            dataset["subject_id"].eq(subject_id)
            & dataset["dataset_split"].eq("train")
        ]
        result: IdentifiabilityResult = analyze_identifiability(
            training,
            template,
            parameters,
            L1,
            L2,
            analysis_set=f"{analysis_prefix}_{subject_id}_train",
        )
        correlation = np.asarray(result.parameter_correlation, dtype=float)
        upper_indices = np.triu_indices(len(PARAMETER_NAMES), 1)
        upper = correlation[upper_indices]
        strongest_index = int(np.argmax(np.abs(upper)))
        first_index = int(upper_indices[0][strongest_index])
        second_index = int(upper_indices[1][strongest_index])
        weakest_information_parameter = min(
            result.information_diagonal,
            key=result.information_diagonal.__getitem__,
        )
        row: dict[str, object] = {
            "subject_id": subject_id,
            "analysis_set": result.analysis_set,
            "valid_training_samples": result.valid_samples,
            "parameter_count": result.parameter_count,
            "numerical_rank": result.numerical_rank,
            "full_rank_five_parameter_model": result.numerical_rank == 5,
            "condition_number": result.condition_number,
            "maximum_absolute_parameter_correlation": float(
                np.max(np.abs(upper))
            ),
            "strongest_correlation_parameter_1": PARAMETER_NAMES[first_index],
            "strongest_correlation_parameter_2": PARAMETER_NAMES[second_index],
            "strongest_correlation": float(upper[strongest_index]),
            "high_correlation_threshold": 0.9,
            "highly_correlated_pair_count": len(result.highly_correlated_pairs),
            "near_collinearity_flag": bool(
                result.numerical_rank < 5 or result.highly_correlated_pairs
            ),
            "weakest_information_parameter": weakest_information_parameter,
            "weakest_information_diagonal": result.information_diagonal[
                weakest_information_parameter
            ],
        }
        for index, value in enumerate(result.singular_values, start=1):
            row[f"singular_value_{index}"] = value
            singular_rows.append(
                {
                    "subject_id": subject_id,
                    "analysis_set": result.analysis_set,
                    "singular_value_index": index,
                    "singular_value": value,
                }
            )
        for parameter, value in result.information_diagonal.items():
            row[f"information_diagonal_{parameter}"] = value
        summary_rows.append(row)
        for first in range(len(PARAMETER_NAMES)):
            for second in range(first + 1, len(PARAMETER_NAMES)):
                correlation_rows.append(
                    {
                        "subject_id": subject_id,
                        "analysis_set": result.analysis_set,
                        "parameter_1": PARAMETER_NAMES[first],
                        "parameter_2": PARAMETER_NAMES[second],
                        "correlation": float(correlation[first, second]),
                        "absolute_correlation": float(
                            abs(correlation[first, second])
                        ),
                        "highly_correlated": bool(
                            abs(correlation[first, second]) >= 0.9
                        ),
                    }
                )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(correlation_rows),
        pd.DataFrame(singular_rows),
    )


def _prediction_samples(
    dataset: pd.DataFrame,
    estimates_by_subject: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    frames: list[pd.DataFrame] = []
    for subject_id, identified_parameters in estimates_by_subject.items():
        subject_data = dataset.loc[dataset["subject_id"].eq(subject_id)]
        for trajectory_id, raw in subject_data.groupby("trajectory_id", sort=False):
            data = valid_observations(raw)
            true_hip, true_knee = measured_joint_torque(data, L1, L2)
            generic_hip, generic_knee = predict_joint_torque(
                data, template, identification_initial_guess, L1
            )
            identified_hip, identified_knee = predict_joint_torque(
                data, template, identified_parameters, L1
            )
            output = data[
                [
                    "subject_id",
                    "trajectory_id",
                    "dataset_split",
                    "time_s",
                    "q_hip_rad",
                    "q_knee_rad",
                    "dq_hip_rad_s",
                    "dq_knee_rad_s",
                    "ddq_hip_rad_s2",
                    "ddq_knee_rad_s2",
                    "fx_observed_n",
                    "fz_observed_n",
                    "sample_valid",
                ]
            ].copy(deep=True)
            if "evaluation_role" in data:
                output["evaluation_role"] = (
                    data["evaluation_role"].astype(str).to_numpy()
                )
            output["tau_true_hip_nm"] = true_hip
            output["tau_true_knee_nm"] = true_knee
            output["tau_generic_hip_nm"] = generic_hip
            output["tau_generic_knee_nm"] = generic_knee
            output["tau_identified_hip_nm"] = identified_hip
            output["tau_identified_knee_nm"] = identified_knee
            output["generic_prediction_valid"] = True
            output["identified_prediction_valid"] = True
            output["prediction_truth_source"] = "J_transpose_times_observed_force"
            output["identified_parameters_fit_split"] = "train_only"
            output["test_used_for_fit"] = False
            frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _prediction_metric_tables(
    prediction_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = compute_all_model_metrics(
        prediction_samples,
        nrmse_epsilon_nm=model_mismatch_nrmse_epsilon_nm,
        minimum_nrmse_range_nm=model_mismatch_nrmse_minimum_range_nm,
    )
    comparison = build_generic_vs_identified_comparison(metrics)
    return metrics, comparison


def _legacy_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    for path in (LEGACY_DATASET_PATH, LEGACY_PARAMETER_PATH, LEGACY_COVERAGE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"legacy comparison artifact missing: {path}")
    dataset = pd.read_csv(LEGACY_DATASET_PATH)
    parameter_table = pd.read_csv(LEGACY_PARAMETER_PATH)
    coverage = pd.read_csv(LEGACY_COVERAGE_PATH)
    estimates: dict[str, dict[str, float]] = {}
    for subject_id, rows in parameter_table.groupby("subject_id", sort=False):
        estimates[str(subject_id)] = {
            str(row["parameter"]): float(row["estimated_value"])
            for _, row in rows.iterrows()
        }
    identifiability, correlations, singular_values = _identifiability_tables(
        dataset,
        estimates,
        analysis_prefix="legacy_symmetric_retrospective",
    )
    samples = _prediction_samples(dataset, estimates)
    metrics, _ = _prediction_metric_tables(samples)
    return identifiability, correlations, singular_values, metrics, coverage


def _comparison_value(
    rows: list[dict[str, object]],
    metric: str,
    active_value: object,
    legacy_value: object,
    unit: str,
    interpretation: str,
) -> None:
    rows.append(
        {
            "metric": metric,
            "active_asymmetric_value": active_value,
            "legacy_symmetric_value": legacy_value,
            "unit": unit,
            "comparison_scope": "retrospective_same_definition_not_causal_superiority",
            "interpretation": interpretation,
        }
    )


def _legacy_comparison(
    active_coverage: pd.DataFrame,
    active_identifiability: pd.DataFrame,
    active_metrics: pd.DataFrame,
    active_parameter_errors: pd.DataFrame,
    legacy_coverage: pd.DataFrame,
    legacy_identifiability: pd.DataFrame,
    legacy_metrics: pd.DataFrame,
    legacy_parameter_errors: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_heldout_coverage = active_coverage.loc[
        active_coverage["dataset_split"].isin(["validation", "test"])
    ]
    legacy_heldout_coverage = legacy_coverage.loc[
        legacy_coverage["dataset_split"].isin(["validation", "test"])
    ]
    active_heldout_metrics = active_metrics.loc[
        active_metrics["dataset_split"].isin(["validation", "test"])
        & active_metrics["prediction_model"].eq("identified")
    ]
    legacy_heldout_metrics = legacy_metrics.loc[
        legacy_metrics["dataset_split"].isin(["validation", "test"])
        & legacy_metrics["prediction_model"].eq("identified")
    ]
    _comparison_value(
        rows,
        "minimum_heldout_axis_aligned_domain_coverage",
        float(active_heldout_coverage["in_domain_percent"].min()),
        float(legacy_heldout_coverage["in_domain_percent"].min()),
        "percent",
        "Different neighboring trajectories; compares transfer coverage, not reference quality.",
    )
    _comparison_value(
        rows,
        "mean_heldout_axis_aligned_domain_coverage",
        float(active_heldout_coverage["in_domain_percent"].mean()),
        float(legacy_heldout_coverage["in_domain_percent"].mean()),
        "percent",
        "Axis-aligned q/dq/ddq box uses the same six-state definition.",
    )
    _comparison_value(
        rows,
        "maximum_identifiability_condition_number",
        float(active_identifiability["condition_number"].max()),
        float(legacy_identifiability["condition_number"].max()),
        "dimensionless",
        "Same scaled sensitivity implementation; lower is not interpreted as clinical superiority.",
    )
    _comparison_value(
        rows,
        "maximum_absolute_parameter_correlation",
        float(active_identifiability["maximum_absolute_parameter_correlation"].max()),
        float(legacy_identifiability["maximum_absolute_parameter_correlation"].max()),
        "dimensionless",
        "Local numerical correlation for the adopted five-parameter model.",
    )
    _comparison_value(
        rows,
        "maximum_parameter_relative_error",
        float(active_parameter_errors["relative_error_percent"].max()),
        float(legacy_parameter_errors["relative_error_percent"].max()),
        "percent",
        "Matched clean virtual-subject recovery only.",
    )
    _comparison_value(
        rows,
        "maximum_heldout_identified_torque_rmse",
        float(active_heldout_metrics["combined_torque_rmse_nm"].max()),
        float(legacy_heldout_metrics["combined_torque_rmse_nm"].max()),
        "N m",
        "Held-out prediction under matched clean synthetic observations.",
    )
    _comparison_value(
        rows,
        "mean_heldout_identified_nrmse",
        float(active_heldout_metrics["combined_nrmse_percent"].mean()),
        float(legacy_heldout_metrics["combined_nrmse_percent"].mean()),
        "percent",
        "NRMSE uses the same combined true-torque range convention.",
    )
    _comparison_value(
        rows,
        "reference_geometry",
        "measured_flexion_and_measured_extension_asymmetric",
        "synthetic_time_reversed_symmetric",
        "categorical",
        "Geometry/provenance changed; no claim that asymmetry is intrinsically better.",
    )
    return pd.DataFrame(rows)


def run_active_reference_local_identification(
    *,
    subject_ids: Sequence[str] = SUBJECT_IDS,
    samples_per_segment: int = 201,
) -> ActiveReferenceLocalResult:
    """Run the deterministic in-memory P0.1 experiment without saving files."""

    reference = load_active_reference()
    trajectories = build_active_local_trajectories(
        reference, samples_per_segment=samples_per_segment
    )
    initial_metadata = build_excitation_metadata(trajectories)
    if not initial_metadata["offline_feasible"].astype(bool).all():
        failures = initial_metadata.loc[
            ~initial_metadata["offline_feasible"].astype(bool), "trajectory_id"
        ].tolist()
        raise RuntimeError(f"active-reference local trajectories failed gates: {failures}")

    dataset = build_local_identification_dataset(
        trajectories,
        subject_ids=subject_ids,
        expected_trajectory_ids=tuple(trajectories),
    )
    # Preserve evaluation roles after the shared estimator-facing table builder.
    role_by_id = {
        item.trajectory_id: item.evaluation_role for item in TRAJECTORY_SPECIFICATIONS
    }
    dataset["evaluation_role"] = dataset["trajectory_id"].map(role_by_id)
    dataset["active_reference_identifier"] = ACTIVE_REFERENCE_ID
    dataset["active_reference_sha256"] = reference.summary[
        "active_reference_sha256"
    ]
    dataset["parent_reference_id"] = ACTIVE_REFERENCE_ID
    dataset["parent_reference_sha256"] = reference.summary[
        "active_reference_sha256"
    ]
    excitation_metadata = build_excitation_metadata(trajectories, dataset)
    if not excitation_metadata["offline_feasible"].astype(bool).all():
        raise RuntimeError("force mapping invalidated an active local trajectory.")

    parameter_errors, optimizer_results, estimates = fit_local_subject_parameters(
        dataset,
        subject_ids=subject_ids,
        training_trajectory_ids=TRAINING_TRAJECTORY_IDS,
    )
    if not all(result.optimizer_success for result in optimizer_results.values()):
        raise RuntimeError("at least one active-local parameter fit failed.")
    identified_parameters = _identified_parameter_table(optimizer_results)
    domain_bounds = fit_local_identification_domain(
        dataset,
        training_trajectory_ids=TRAINING_TRAJECTORY_IDS,
    )
    coverage = build_local_domain_coverage(
        trajectories,
        domain_bounds,
        split_by_trajectory=SPLIT_BY_TRAJECTORY,
        domain_model=DOMAIN_MODEL,
    )
    coverage["evaluation_role"] = coverage["trajectory_id"].map(role_by_id)
    coverage["outside_domain_percent"] = 100.0 - coverage["in_domain_percent"]

    identifiability, correlations, singular_values = _identifiability_tables(
        dataset,
        estimates,
        analysis_prefix="active_asymmetric_reference_local",
    )
    prediction_samples = _prediction_samples(dataset, estimates)
    prediction_metrics, generic_comparison = _prediction_metric_tables(
        prediction_samples
    )

    (
        legacy_identifiability,
        _legacy_correlations,
        _legacy_singular_values,
        legacy_metrics,
        legacy_coverage,
    ) = _legacy_analysis()
    legacy_parameter_errors = pd.read_csv(LEGACY_PARAMETER_PATH)
    legacy_comparison = _legacy_comparison(
        coverage,
        identifiability,
        prediction_metrics,
        parameter_errors,
        legacy_coverage,
        legacy_identifiability,
        legacy_metrics,
        legacy_parameter_errors,
    )
    return ActiveReferenceLocalResult(
        reference=reference,
        trajectories=trajectories,
        excitation_metadata=excitation_metadata,
        dataset=dataset,
        identified_parameters=identified_parameters,
        parameter_errors=parameter_errors,
        optimizer_results=optimizer_results,
        estimates_by_subject=estimates,
        domain_bounds=domain_bounds,
        domain_coverage=coverage,
        prediction_samples=prediction_samples,
        prediction_metrics=prediction_metrics,
        generic_vs_identified=generic_comparison,
        identifiability_summary=identifiability,
        parameter_correlations=correlations,
        singular_values=singular_values,
        legacy_identifiability_summary=legacy_identifiability,
        legacy_prediction_metrics=legacy_metrics,
        legacy_comparison=legacy_comparison,
    )


__all__ = [
    "ACTIVE_MANIFEST_PATH",
    "ACTIVE_METADATA_PATH",
    "ACTIVE_NOMINAL_ID",
    "ACTIVE_NOMINAL_PATH",
    "ACTIVE_REFERENCE_ID",
    "ACTIVE_REFERENCE_PATH",
    "DOMAIN_MODEL",
    "HELD_OUT_TRAJECTORY_IDS",
    "MODEL_VERSION",
    "RANDOM_SEED",
    "SPLIT_DEFINITION_ID",
    "SPLIT_BY_TRAJECTORY",
    "TRAINING_TRAJECTORY_IDS",
    "TRAJECTORY_SPECIFICATIONS",
    "ActiveReferenceBundle",
    "ActiveReferenceLocalResult",
    "build_active_local_trajectories",
    "build_excitation_metadata",
    "load_active_reference",
    "run_active_reference_local_identification",
    "sha256_file",
]
