"""Build the offline ``reference_closed_c2`` execution candidate.

This runner reads the formally approved Stage-5C reference, keeps that source
file unchanged, and writes independent C2 phase/slow/nominal artifacts.  It has
no robot SDK, connection, servo, power, or motion path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .reference_closed_c2 import (
    APPROVED_HIP_ROM_DEG,
    APPROVED_KNEE_ROM_DEG,
    C2_MODEL_VERSION,
    C2ReferenceModel,
    compare_c2_with_pchip,
    fit_reference_closed_c2,
    retime_reference_closed_c2,
)


DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "reference_candidates"
    / "reference_execution_versions.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "data" / "reference_candidates"
PROFILE_DURATIONS_S = {"slow": 12.0, "nominal": 6.0}
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_PROTECTED_OUTPUT_ROOT_NAMES = (
    "hardware",
    "control",
    "collection",
    "safety",
    "config",
    "scripts",
    "sdk",
)


@dataclass(frozen=True)
class ReferenceC2RunResult:
    model: C2ReferenceModel
    trajectories: dict[str, pd.DataFrame]
    original_trajectories: dict[str, pd.DataFrame]
    comparison: pd.DataFrame
    metadata: dict[str, object]
    output_paths: dict[str, Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _validate_output_directory(path: str | Path) -> Path:
    """Refuse C2 artifacts inside real-robot or shared configuration roots."""

    destination = Path(path).expanduser().resolve()
    for name in _PROTECTED_OUTPUT_ROOT_NAMES:
        protected = (_REPOSITORY_ROOT / name).resolve()
        if destination == protected or destination.is_relative_to(protected):
            raise ValueError(
                f"reference C2 output is forbidden inside protected directory: {protected}"
            )
    return destination


def _metadata(
    source_path: Path,
    model: C2ReferenceModel,
    trajectories: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
) -> dict[str, object]:
    maxima = {
        profile: {
            "max_abs_dq_hip_rad_s": float(frame["dq_hip_rad_s"].abs().max()),
            "max_abs_dq_knee_rad_s": float(frame["dq_knee_rad_s"].abs().max()),
            "max_abs_ddq_hip_rad_s2": float(frame["ddq_hip_rad_s2"].abs().max()),
            "max_abs_ddq_knee_rad_s2": float(frame["ddq_knee_rad_s2"].abs().max()),
        }
        for profile, frame in trajectories.items()
    }
    warning_counts = {
        str(row.profile): {
            "original_acceleration_warning_count": int(
                row.original_acceleration_warning_count
            ),
            "c2_acceleration_warning_count": int(row.c2_acceleration_warning_count),
        }
        for row in comparison.itertuples(index=False)
    }
    return {
        "stage": "reference_closed_c2_offline_preparation",
        "model_version": C2_MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_reference_path": str(source_path),
        "source_reference_sha256": _sha256(source_path),
        "source_reference_version": "reference_closed_symmetric",
        "source_reference_overwritten": False,
        "reference_version": "reference_closed_c2",
        "approved_hip_rom_deg": list(APPROVED_HIP_ROM_DEG),
        "approved_knee_rom_deg": list(APPROVED_KNEE_ROM_DEG),
        "rom_approval_status": "approved",
        "rom_approval_source": "user_explicit_reference_experiment_approval",
        "rom_mapping_applied": False,
        "reference_path_preserved": True,
        "shape_preserved_within_audit_gates": model.shape_audit.fit_accepted,
        "reference_path_preserved_meaning": (
            "immutable_source_retained_and_c2_shape_within_explicit_deviation_gates"
        ),
        "pointwise_angle_clipping_applied": False,
        "spline_family": "quintic_B_spline",
        "spline_degree": model.shape_audit.spline_degree,
        "continuity_order": model.shape_audit.continuity_order,
        "internal_spline_continuity_order": (
            model.shape_audit.internal_spline_continuity_order
        ),
        "full_cycle_global_phase_continuity_order": (
            model.shape_audit.full_cycle_global_phase_continuity_order
        ),
        "reflection_boundary_conditions": {
            "flexion_start": {"derivative_order_1": 0.0, "derivative_order_3": 0.0},
            "peak_flexion": {"derivative_order_1": 0.0, "derivative_order_3": 0.0},
            "reason": (
                "odd path derivatives change sign under exact reversal; zero first "
                "and third derivatives make the full global-phase cycle C4"
            ),
        },
        "extension_definition": "exact_time_reverse_of_c2_flexion_path",
        "minimum_jerk_controls": "path_phase_not_joint_endpoint_line",
        "profiles": {
            profile: {
                "flexion_duration_s": duration,
                "extension_duration_s": duration,
                "retimed_timing_is_original": False,
            }
            for profile, duration in PROFILE_DURATIONS_S.items()
        },
        "shape_audit": model.shape_audit.as_dict(),
        "acceleration_warning_definition": (
            "robust ratio-10 outlier diagnostic on finite-difference equivalent-"
            "pull-point acceleration increments; data-quality audit, not safety limit"
        ),
        "acceleration_warning_is_sampling_resolution_dependent": True,
        "acceleration_warning_samples_per_segment": {
            profile: int((len(frame) + 1) // 2)
            for profile, frame in trajectories.items()
        },
        "acceleration_warning_counts": warning_counts,
        "retimed_joint_state_maxima": maxima,
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "L2_definition": "knee_to_strap_equivalent_pull_point",
        "observed_ankle_is_pull_point": False,
        "software_only": True,
        "formal_execution_scope": "offline_reference_rom_and_shape_gate_only",
        "robot_execution_approved": False,
        "real_robot_sdk_imported": False,
        "real_robot_connected": False,
        "robot_command_sent": False,
    }


def run_reference_c2(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    samples_per_segment: int = 201,
    save_outputs: bool = True,
    generate_plots: bool = True,
) -> ReferenceC2RunResult:
    """Generate, audit, save, and optionally visualize the formal C2 candidate."""

    source_path = Path(input_path).expanduser().resolve()
    destination = _validate_output_directory(output_directory)
    if not source_path.is_file():
        raise FileNotFoundError(f"approved Stage-5C reference does not exist: {source_path}")
    reference_versions = pd.read_csv(source_path)
    model = fit_reference_closed_c2(reference_versions)
    trajectories = {
        profile: retime_reference_closed_c2(
            model,
            profile=profile,
            flexion_duration_s=duration,
            extension_duration_s=duration,
            samples_per_segment=samples_per_segment,
        )
        for profile, duration in PROFILE_DURATIONS_S.items()
    }
    comparison, originals = compare_c2_with_pchip(
        reference_versions,
        model,
        trajectories,
        durations_s=PROFILE_DURATIONS_S,
        samples_per_segment=samples_per_segment,
    )
    metadata = _metadata(source_path, model, trajectories, comparison)
    output_paths: dict[str, Path] = {}
    if save_outputs:
        destination.mkdir(parents=True, exist_ok=True)
        tables = {
            "reference_closed_c2_phase.csv": model.phase_path,
            "reference_closed_c2_slow.csv": trajectories["slow"],
            "reference_closed_c2_nominal.csv": trajectories["nominal"],
            "reference_c2_comparison.csv": comparison,
        }
        for filename, table in tables.items():
            path = destination / filename
            table.to_csv(path, index=False)
            output_paths[filename] = path
        metadata_path = destination / "reference_c2_metadata.json"
        metadata_path.write_text(
            json.dumps(_json_value(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_paths[metadata_path.name] = metadata_path
        if generate_plots:
            from .visualize_reference_c2 import generate_reference_c2_visualizations

            output_paths.update(
                generate_reference_c2_visualizations(
                    model.phase_path,
                    originals,
                    trajectories,
                    destination,
                )
            )
    return ReferenceC2RunResult(
        model=model,
        trajectories=trajectories,
        original_trajectories=originals,
        comparison=comparison,
        metadata=metadata,
        output_paths=output_paths,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the offline reference_closed_c2 candidate from the formally "
            "approved Stage-5C reference. No robot SDK or motion is used."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--samples-per-segment", type=int, default=201)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_reference_c2(
        input_path=arguments.input,
        output_directory=arguments.output_directory,
        samples_per_segment=arguments.samples_per_segment,
        save_outputs=True,
        generate_plots=not arguments.no_plots,
    )
    print("reference_closed_c2 generated offline only")
    print(f"approved hip ROM: {list(APPROVED_HIP_ROM_DEG)} deg")
    print(f"approved knee ROM: {list(APPROVED_KNEE_ROM_DEG)} deg")
    print(result.comparison.to_string(index=False))
    print("robot execution approved: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_DIRECTORY",
    "PROFILE_DURATIONS_S",
    "ReferenceC2RunResult",
    "run_reference_c2",
]
