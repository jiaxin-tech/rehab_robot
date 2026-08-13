"""Generate the active measured-asymmetric periodic reference offline.

This runner reads Stage-5A products, audits all natural cycle candidates,
selects the minimum-closure eligible measured cycle, fits a small periodic C2
correction, retimes slow/nominal profiles, checks them against the frozen
Stage-5C local-identification domain, and persists provenance plus figures.

It deliberately has no robot SDK, connection, power, servo, or motion path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import L1, L2
from .geometry_error_metrics import (
    ESTIMATED_DOMAIN_STATE_COLUMNS,
    StateDomainBounds,
    classify_state_domain,
)
from .kinematics import forward_kinematics
from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .reference_cycle_closure import (
    ReferenceCycleClosureAuditResult,
    audit_reference_cycle_closure,
)
from .reference_local_excitation import fit_local_identification_domain
from .reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
    MEASURED_ASYMMETRIC_MODEL_VERSION,
    MEASURED_ASYMMETRIC_NOMINAL_ID,
    MEASURED_ASYMMETRIC_SLOW_ID,
    MEASURED_RAW_REFERENCE,
    PERIODIC_PATH_SMOOTHING_POLYNOMIAL_ORDER,
    PERIODIC_PATH_SMOOTHING_WINDOW,
    MeasuredAsymmetricPeriodicModel,
    build_reference_measured_raw,
    fit_measured_asymmetric_periodic_reference,
    retime_measured_asymmetric_periodic_reference,
)
from .run_reference_candidate_evaluation import LOCAL_DOMAIN_MINIMUM_PERCENT


MODULE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PROCESSED_DIRECTORY = (
    MODULE_DIRECTORY / "data" / "reference_trajectories" / "processed"
)
DEFAULT_FULL_ANGLES_PATH = DEFAULT_PROCESSED_DIRECTORY / "reference_full_angles.csv"
DEFAULT_DETECTED_CYCLES_PATH = DEFAULT_PROCESSED_DIRECTORY / "detected_cycles.csv"
DEFAULT_STAGE5A_METADATA_PATH = DEFAULT_PROCESSED_DIRECTORY / "metadata.json"
DEFAULT_OUTPUT_DIRECTORY = MODULE_DIRECTORY / "data" / "reference_candidates"
DEFAULT_FROZEN_LOCAL_DATASET_PATH = (
    DEFAULT_OUTPUT_DIRECTORY / "reference_local_identification_dataset.csv"
)
PROFILE_TOTAL_DURATIONS_S = {"slow": 24.0, "nominal": 12.0}
OUTPUT_FILENAMES = {
    "cycle_closure_audit": "reference_cycle_closure_audit.csv",
    "measured_raw": "reference_measured_raw.csv",
    "periodic_phase": "reference_measured_asymmetric_closed.csv",
    "slow": "reference_measured_asymmetric_closed_slow.csv",
    "nominal": "reference_measured_asymmetric_closed_nominal.csv",
    "domain_coverage": "reference_measured_asymmetric_domain_coverage.csv",
    "manifest": "reference_version_manifest.csv",
    "metadata": "reference_measured_asymmetric_metadata.json",
}
_PROJECT_ROOT = MODULE_DIRECTORY.parent
_PROTECTED_OUTPUT_ROOT_NAMES = (
    "hardware",
    "control",
    "collection",
    "safety",
    "config",
    "scripts",
)


@dataclass(frozen=True)
class ReferenceMeasuredAsymmetricRunResult:
    cycle_audit: ReferenceCycleClosureAuditResult
    model: MeasuredAsymmetricPeriodicModel
    trajectories: dict[str, pd.DataFrame]
    domain_coverage: pd.DataFrame
    manifest: pd.DataFrame
    metadata: dict[str, object]
    output_paths: dict[str, Path]
    visualization_paths: dict[str, Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata root must be an object: {path}")
    return value


def _legacy_selected_cycle_closure(
    full_angles: pd.DataFrame,
    detected_cycles: pd.DataFrame,
) -> dict[str, object] | None:
    """Reproduce the old selected-cycle endpoint error for provenance only."""

    if "selected" not in detected_cycles or "cycle_index" not in detected_cycles:
        return None
    selected_rows = detected_cycles.loc[detected_cycles["selected"].eq(True)]
    if len(selected_rows) != 1:
        return None
    row = selected_rows.iloc[0]
    start_frame = int(row["start_frame"])
    peak_frame = int(row["peak_flexion_frame"])
    end_frame = int(row["end_frame"])
    endpoints = full_angles.loc[
        full_angles["Frame"].isin((start_frame, end_frame)),
        ["Frame", "q_hip_rad", "q_knee_rad"],
    ].sort_values("Frame")
    if len(endpoints) != 2:
        return None
    q_hip = endpoints["q_hip_rad"].to_numpy(dtype=float)
    q_knee = endpoints["q_knee_rad"].to_numpy(dtype=float)
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    delta_hip_deg = float(np.rad2deg(q_hip[-1] - q_hip[0]))
    delta_knee_deg = float(np.rad2deg(q_knee[-1] - q_knee[0]))
    delta_x_mm = float(1000.0 * (x_pull[-1] - x_pull[0]))
    delta_z_mm = float(1000.0 * (z_pull[-1] - z_pull[0]))
    return {
        "cycle_index": int(row["cycle_index"]),
        "start_frame": start_frame,
        "peak_frame": peak_frame,
        "end_frame": end_frame,
        "cycle_quality_score": float(row["cycle_quality_score"]),
        "legacy_selection_used_closure": False,
        "delta_q_hip_deg": delta_hip_deg,
        "delta_q_knee_deg": delta_knee_deg,
        "delta_x_pull_mm": delta_x_mm,
        "delta_z_pull_mm": delta_z_mm,
        "pull_closure_error_mm": float(np.hypot(delta_x_mm, delta_z_mm)),
        "closure_score_under_current_definition": float(
            np.sqrt(
                delta_hip_deg**2
                + delta_knee_deg**2
                + (delta_x_mm / 5.0) ** 2
                + (delta_z_mm / 5.0) ** 2
            )
        ),
    }


def _validate_output_directory(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    for name in _PROTECTED_OUTPUT_ROOT_NAMES:
        protected = (_PROJECT_ROOT / name).resolve()
        if destination == protected or destination.is_relative_to(protected):
            raise ValueError(
                f"reference output is forbidden inside protected directory: {protected}"
            )
    return destination


def _frozen_domain_coverage(
    trajectories: Mapping[str, pd.DataFrame],
    frozen_dataset_path: Path,
) -> tuple[pd.DataFrame, StateDomainBounds]:
    if not frozen_dataset_path.is_file():
        raise FileNotFoundError(
            f"frozen local-identification dataset is missing: {frozen_dataset_path}"
        )
    frozen_dataset = pd.read_csv(frozen_dataset_path)
    bounds = fit_local_identification_domain(frozen_dataset)
    source_columns = (
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    rows: list[dict[str, object]] = []
    for profile, trajectory in trajectories.items():
        states = trajectory.loc[:, source_columns].copy(deep=True)
        states.columns = ESTIMATED_DOMAIN_STATE_COLUMNS
        states["state_estimation_valid"] = trajectory[
            "trajectory_sample_valid"
        ].astype(bool).to_numpy()
        membership = classify_state_domain(states, bounds)
        values = states.loc[:, bounds.columns].to_numpy(dtype=float)
        lower = np.asarray(bounds.lower, dtype=float)
        upper = np.asarray(bounds.upper, dtype=float)
        outside = (values < lower) | (values > upper) | ~np.isfinite(values)
        outside_counts = {
            column: int(outside[:, index].sum())
            for index, column in enumerate(source_columns)
        }
        in_count = int(membership.sum())
        percent = 100.0 * in_count / len(trajectory)
        phase_rows: dict[str, object] = {}
        for phase_name in ("flexion", "extension"):
            mask = trajectory["cycle_phase"].eq(phase_name).to_numpy()
            phase_in = int(membership[mask].sum())
            phase_count = int(mask.sum())
            phase_rows[f"{phase_name}_in_domain_sample_count"] = phase_in
            phase_rows[f"{phase_name}_sample_count"] = phase_count
            phase_rows[f"{phase_name}_in_domain_percent"] = (
                100.0 * phase_in / phase_count
            )
        missing_groups: list[str] = []
        if outside[:, :2].any():
            missing_groups.append("q")
        if outside[:, 2:4].any():
            missing_groups.append("dq")
        if outside[:, 4:6].any():
            missing_groups.append("ddq")
        rows.append(
            {
                "trajectory_id": str(trajectory["trajectory_id"].iloc[0]),
                "profile": profile,
                "sample_count": int(len(trajectory)),
                "in_domain_sample_count": in_count,
                "out_of_domain_sample_count": int(len(trajectory) - in_count),
                "in_domain_percent": percent,
                "minimum_required_percent": LOCAL_DOMAIN_MINIMUM_PERCENT,
                "coverage_gate_passed": bool(
                    percent >= LOCAL_DOMAIN_MINIMUM_PERCENT
                ),
                "missing_state_variable_groups": ";".join(missing_groups),
                **outside_counts,
                **phase_rows,
                "domain_model": (
                    "frozen_existing_Stage5C_axis_aligned_6d_train_domain"
                ),
                "domain_fitted_from_new_reference": False,
                "domain_threshold_changed": False,
                "domain_training_sample_count": bounds.valid_training_samples,
            }
        )
    return pd.DataFrame(rows), bounds


def _apply_domain_gate(
    trajectories: dict[str, pd.DataFrame], coverage: pd.DataFrame
) -> None:
    for row in coverage.itertuples(index=False):
        profile = str(row.profile)
        passed = bool(row.coverage_gate_passed)
        trajectory = trajectories[profile]
        trajectory["frozen_local_domain_in_domain_percent"] = float(
            row.in_domain_percent
        )
        trajectory["frozen_local_domain_minimum_required_percent"] = float(
            row.minimum_required_percent
        )
        trajectory["frozen_local_domain_coverage_valid"] = passed
        if not passed:
            trajectory["formal_execution_allowed"] = False
            previous = trajectory["invalid_reason"].fillna("").astype(str)
            trajectory["invalid_reason"] = np.where(
                previous.eq(""),
                "outside_frozen_local_identification_domain",
                previous + ";outside_frozen_local_identification_domain",
            )
        trajectory["active_reference"] = bool(profile == "slow" and passed)
        trajectory["allowed_for_first_robot_trial"] = bool(
            profile == "slow" and passed
        )


def _manifest(
    trajectories: Mapping[str, pd.DataFrame],
    output_paths: Mapping[str, Path] | None = None,
) -> pd.DataFrame:
    paths = output_paths or {}
    rows = [
        {
            "reference_version": MEASURED_RAW_REFERENCE,
            "trajectory_id": MEASURED_RAW_REFERENCE,
            "role": "immutable_selected_measured_source",
            "active_reference": False,
            "allowed_for_first_robot_trial": False,
            "legacy_software_comparison": False,
            "active": False,
            "legacy": False,
            "not_used_for_final_personalization": True,
            "not_used_for_robot_execution": True,
            "path": str(paths.get("measured_raw", "")),
        },
        {
            "reference_version": MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
            "trajectory_id": MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
            "role": "periodic_C2_phase_model",
            "active_reference": False,
            "allowed_for_first_robot_trial": False,
            "legacy_software_comparison": False,
            "active": False,
            "legacy": False,
            "not_used_for_final_personalization": True,
            "not_used_for_robot_execution": True,
            "path": str(paths.get("periodic_phase", "")),
        },
    ]
    for profile in ("slow", "nominal"):
        trajectory = trajectories[profile]
        rows.append(
            {
                "reference_version": MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
                "trajectory_id": str(trajectory["trajectory_id"].iloc[0]),
                "role": f"retimed_{profile}",
                "active_reference": bool(
                    trajectory["active_reference"].astype(bool).all()
                ),
                "allowed_for_first_robot_trial": False,
                "legacy_software_comparison": False,
                "active": profile == "slow",
                "legacy": False,
                "not_used_for_final_personalization": profile != "slow",
                "not_used_for_robot_execution": True,
                "path": str(paths.get(profile, "")),
            }
        )
    rows.extend(
        (
            {
                "reference_version": "reference_closed_symmetric",
                "trajectory_id": "reference_closed_symmetric",
                "role": "legacy_synthetic_reversal_software_comparison",
                "active_reference": False,
                "allowed_for_first_robot_trial": False,
                "legacy_software_comparison": True,
                "active": False,
                "legacy": True,
                "not_used_for_final_personalization": True,
                "not_used_for_robot_execution": True,
                "path": "reference_execution_versions.csv",
            },
            {
                "reference_version": "reference_closed_c2",
                "trajectory_id": "reference_closed_c2_slow",
                "role": "legacy_symmetric_C2_software_comparison",
                "active_reference": False,
                "allowed_for_first_robot_trial": False,
                "legacy_software_comparison": True,
                "active": False,
                "legacy": True,
                "not_used_for_final_personalization": True,
                "not_used_for_robot_execution": True,
                "path": "reference_closed_c2_slow.csv",
            },
            {
                "reference_version": "reference_closed_c2",
                "trajectory_id": "reference_closed_c2_nominal",
                "role": "legacy_symmetric_C2_software_comparison",
                "active_reference": False,
                "allowed_for_first_robot_trial": False,
                "legacy_software_comparison": True,
                "active": False,
                "legacy": True,
                "not_used_for_final_personalization": True,
                "not_used_for_robot_execution": True,
                "path": "reference_closed_c2_nominal.csv",
            },
        )
    )
    return pd.DataFrame(rows)


def run_reference_measured_asymmetric(
    *,
    full_angles_path: str | Path = DEFAULT_FULL_ANGLES_PATH,
    detected_cycles_path: str | Path = DEFAULT_DETECTED_CYCLES_PATH,
    stage5a_metadata_path: str | Path = DEFAULT_STAGE5A_METADATA_PATH,
    frozen_local_dataset_path: str | Path = DEFAULT_FROZEN_LOCAL_DATASET_PATH,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    save_outputs: bool = True,
    generate_plots: bool = True,
) -> ReferenceMeasuredAsymmetricRunResult:
    """Run the complete offline measured-asymmetric reference release."""

    full_path = Path(full_angles_path).expanduser().resolve()
    cycles_path = Path(detected_cycles_path).expanduser().resolve()
    stage5a_path = Path(stage5a_metadata_path).expanduser().resolve()
    frozen_dataset_path = Path(frozen_local_dataset_path).expanduser().resolve()
    for path in (full_path, cycles_path, stage5a_path, frozen_dataset_path):
        if not path.is_file():
            raise FileNotFoundError(f"required reference input is missing: {path}")
    destination = _validate_output_directory(output_directory)
    full_angles = pd.read_csv(full_path)
    detected_cycles = pd.read_csv(cycles_path)
    stage5a_metadata = _load_json(stage5a_path)
    legacy_selected_cycle = _legacy_selected_cycle_closure(
        full_angles, detected_cycles
    )
    source_fps = stage5a_metadata.get("fps")
    cycle_audit = audit_reference_cycle_closure(
        full_angles,
        detected_cycles,
        source_fps=source_fps,
    )
    selected = cycle_audit.selected_candidate
    if selected is None:
        raise PermissionError(
            "no complete projection-valid approved-ROM natural cycle passed closure selection"
        )
    start_frame = int(selected["start_frame"])
    peak_frame = int(selected["peak_frame"])
    end_frame = int(selected["end_frame"])
    measured_raw = build_reference_measured_raw(
        full_angles,
        start_frame=start_frame,
        peak_frame=peak_frame,
        end_frame=end_frame,
    )
    model = fit_measured_asymmetric_periodic_reference(measured_raw)
    if not model.fit_accepted:
        raise PermissionError(
            "periodic measured reference rejected: "
            + ";".join(model.rejection_reasons)
        )
    trajectories = {
        profile: retime_measured_asymmetric_periodic_reference(
            model,
            profile=profile,
            total_duration_s=duration,
        )
        for profile, duration in PROFILE_TOTAL_DURATIONS_S.items()
    }
    domain_coverage, frozen_bounds = _frozen_domain_coverage(
        trajectories, frozen_dataset_path
    )
    _apply_domain_gate(trajectories, domain_coverage)
    slow_coverage = domain_coverage.loc[domain_coverage["profile"].eq("slow")]
    if len(slow_coverage) != 1 or not bool(
        slow_coverage["coverage_gate_passed"].iloc[0]
    ):
        raise PermissionError(
            "new slow reference is outside the frozen local-identification domain"
        )

    output_paths: dict[str, Path] = {}
    visualization_paths: dict[str, Path] = {}
    manifest = _manifest(trajectories)
    if save_outputs:
        destination.mkdir(parents=True, exist_ok=True)
        tables = {
            "cycle_closure_audit": cycle_audit.closure_audit,
            "measured_raw": measured_raw,
            "periodic_phase": model.phase_path,
            "slow": trajectories["slow"],
            "nominal": trajectories["nominal"],
            "domain_coverage": domain_coverage,
        }
        for key, table in tables.items():
            path = destination / OUTPUT_FILENAMES[key]
            table.to_csv(path, index=False)
            output_paths[key] = path
        manifest = _manifest(trajectories, output_paths)
        manifest_path = destination / OUTPUT_FILENAMES["manifest"]
        manifest.to_csv(manifest_path, index=False)
        output_paths["manifest"] = manifest_path
        if generate_plots:
            from .visualize_reference_measured_asymmetric import (
                generate_measured_asymmetric_reference_visualizations,
            )

            visualization_paths = (
                generate_measured_asymmetric_reference_visualizations(
                    full_angles,
                    cycle_audit.closure_audit,
                    model,
                    trajectories,
                    destination,
                )
            )
            output_paths.update(visualization_paths)

    source_csv_path = Path(
        str(stage5a_metadata.get("input_path", ""))
    ).expanduser()
    active_path = output_paths.get("slow")
    metadata: dict[str, object] = {
        "stage": "measured_asymmetric_periodic_reference_release",
        "model_version": MEASURED_ASYMMETRIC_MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bone_csv_path": str(source_csv_path),
        "source_bone_csv_sha256": (
            _sha256(source_csv_path.resolve())
            if source_csv_path.is_file()
            else None
        ),
        "stage5a_full_angles_path": str(full_path),
        "stage5a_full_angles_sha256": _sha256(full_path),
        "stage5a_detected_cycles_path": str(cycles_path),
        "stage5a_detected_cycles_sha256": _sha256(cycles_path),
        "stage5a_metadata_path": str(stage5a_path),
        "stage5a_metadata_sha256": _sha256(stage5a_path),
        "source_fps": source_fps,
        "physical_derivative_closure_available": source_fps is not None,
        "physical_derivative_closure_unavailable_reason": (
            None if source_fps is not None else "source_fps_not_provided"
        ),
        "cycle_detection": cycle_audit.metadata,
        "detected_natural_cycle_count": int(len(cycle_audit.closure_audit)),
        "selected_cycle_candidate": _json_ready(selected.to_dict()),
        "legacy_stage5a_selected_cycle": legacy_selected_cycle,
        "selected_measured_source_row_count": int(len(measured_raw)),
        "reference_measured_raw_values_modified": False,
        "periodic_closure_deviation": model.deviation_audit.as_dict(),
        "flexion_extension_asymmetry": model.asymmetry_audit.as_dict(),
        "continuity_audit": model.continuity_audit.as_dict(),
        "periodic_spline_anchor_count": model.spline_anchor_count,
        "periodic_spline_subdivision_factor": model.spline_subdivision_factor,
        "periodic_path_smoothing": {
            "family": "Savitzky-Golay",
            "window_samples": PERIODIC_PATH_SMOOTHING_WINDOW,
            "polynomial_order": PERIODIC_PATH_SMOOTHING_POLYNOMIAL_ORDER,
            "raw_reference_modified": False,
        },
        "periodic_fit_accepted": model.fit_accepted,
        "periodic_fit_rejection_reasons": list(model.rejection_reasons),
        "measured_extension_is_reversed_flexion": False,
        "extension_definition": "measured_extension_from_source_CSV",
        "profiles": {
            profile: {
                "trajectory_id": str(trajectory["trajectory_id"].iloc[0]),
                "total_duration_s": float(trajectory["time_s"].iloc[-1]),
                "flexion_duration_s": float(
                    trajectory.loc[
                        trajectory["cycle_phase"].eq("flexion"), "time_s"
                    ].iloc[-1]
                ),
                "extension_duration_s": float(
                    trajectory["time_s"].iloc[-1]
                    - trajectory.loc[
                        trajectory["cycle_phase"].eq("flexion"), "time_s"
                    ].iloc[-1]
                ),
                "sample_count": int(len(trajectory)),
                "active_reference": bool(
                    trajectory["active_reference"].astype(bool).all()
                ),
                "formal_execution_allowed": bool(
                    trajectory["formal_execution_allowed"].astype(bool).all()
                ),
            }
            for profile, trajectory in trajectories.items()
        },
        "frozen_local_domain_dataset_path": str(frozen_dataset_path),
        "frozen_local_domain_dataset_sha256": _sha256(frozen_dataset_path),
        "frozen_local_domain_bounds": frozen_bounds.as_serializable_dict(),
        "frozen_local_domain_minimum_percent": LOCAL_DOMAIN_MINIMUM_PERCENT,
        "local_domain_refit_with_new_reference": False,
        "local_domain_threshold_changed": False,
        "domain_coverage": domain_coverage.to_dict(orient="records"),
        "active_reference_trajectory": MEASURED_ASYMMETRIC_SLOW_ID,
        "active_reference_path": str(active_path) if active_path else None,
        "active_reference_sha256": _sha256(active_path) if active_path else None,
        "parent_reference_id": MEASURED_ASYMMETRIC_SLOW_ID,
        "parent_reference_sha256": _sha256(active_path) if active_path else None,
        "legacy_reference_closed_symmetric_active_reference": False,
        "legacy_reference_closed_c2_active_reference": False,
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "approved_hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "approved_knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "L1_m": float(model.phase_path["L1_m"].iloc[0]),
        "L2_m": float(model.phase_path["L2_m"].iloc[0]),
        "L2_definition": "knee_to_strap_equivalent_pull_point",
        "source_reference_overwritten": False,
        "legacy_reference_overwritten": False,
        "real_robot_sdk_imported": False,
        "real_robot_connected": False,
        "robot_command_sent": False,
        "hardware_code_modified": False,
        "control_execution_code_modified": False,
        "safety_thresholds_modified": False,
    }
    if save_outputs:
        if metadata["active_reference_sha256"] != ACTIVE_REFERENCE_SHA256:
            raise RuntimeError(
                "generated active reference differs from the formally pinned SHA-256"
            )
        metadata["generated_file_sha256"] = {
            key: _sha256(path) for key, path in output_paths.items()
        }
        metadata_path = destination / OUTPUT_FILENAMES["metadata"]
        metadata_path.write_text(
            json.dumps(
                _json_ready(metadata),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        output_paths["metadata"] = metadata_path
    return ReferenceMeasuredAsymmetricRunResult(
        cycle_audit=cycle_audit,
        model=model,
        trajectories=trajectories,
        domain_coverage=domain_coverage,
        manifest=manifest,
        metadata=metadata,
        output_paths=output_paths,
        visualization_paths=visualization_paths,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and audit the measured-asymmetric periodic reference "
            "offline. This command never imports or connects to robot hardware."
        )
    )
    parser.add_argument("--full-angles", type=Path, default=DEFAULT_FULL_ANGLES_PATH)
    parser.add_argument(
        "--detected-cycles", type=Path, default=DEFAULT_DETECTED_CYCLES_PATH
    )
    parser.add_argument(
        "--stage5a-metadata", type=Path, default=DEFAULT_STAGE5A_METADATA_PATH
    )
    parser.add_argument(
        "--frozen-local-dataset",
        type=Path,
        default=DEFAULT_FROZEN_LOCAL_DATASET_PATH,
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_reference_measured_asymmetric(
        full_angles_path=arguments.full_angles,
        detected_cycles_path=arguments.detected_cycles,
        stage5a_metadata_path=arguments.stage5a_metadata,
        frozen_local_dataset_path=arguments.frozen_local_dataset,
        output_directory=arguments.output_directory,
        save_outputs=True,
        generate_plots=not arguments.no_plots,
    )
    selected = result.cycle_audit.selected_candidate
    print("measured-asymmetric periodic reference generated offline")
    print(f"detected natural cycles: {len(result.cycle_audit.closure_audit)}")
    print(
        "selected: "
        f"{int(selected['start_frame'])} -> {int(selected['peak_frame'])} "
        f"-> {int(selected['end_frame'])}"
    )
    print(result.domain_coverage.to_string(index=False))
    print(f"active reference: {MEASURED_ASYMMETRIC_SLOW_ID}")
    print("real robot connected: false")
    print("robot command sent: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DETECTED_CYCLES_PATH",
    "DEFAULT_FROZEN_LOCAL_DATASET_PATH",
    "DEFAULT_FULL_ANGLES_PATH",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_STAGE5A_METADATA_PATH",
    "OUTPUT_FILENAMES",
    "PROFILE_TOTAL_DURATIONS_S",
    "ReferenceMeasuredAsymmetricRunResult",
    "run_reference_measured_asymmetric",
]
