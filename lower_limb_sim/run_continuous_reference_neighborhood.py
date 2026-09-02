"""Generate the offline audit grid for the continuous reference neighbourhood."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import (
    FIXED_TIME_SCALE,
    GENERATOR_VERSION,
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    GeneratedTrajectory,
    generate_personalized_trajectory,
)
from .formal_protocol import ACTIVE_REFERENCE_ID, ACTIVE_REFERENCE_SHA256, PROJECT_ROOT
from .reference_release import load_frozen_active_reference


DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "lower_limb_sim"
    / "formal_artifacts"
    / "continuous_reference_neighborhood"
)
GRID_HIP_DELTAS_DEG = (-5.0, -2.5, 0.0)
GRID_KNEE_DELTAS_DEG = (-5.0, -2.5, 0.0)
GRID_PHASE_SHIFTS = (-0.03, 0.0, 0.03)
REPRESENTATIVE_POINTS = {
    "neutral": (0.0, 0.0, 0.0),
    "hip_minus_3deg": (-3.0, 0.0, 0.0),
    "hip_minus_5deg": (-5.0, 0.0, 0.0),
    "knee_minus_3deg": (0.0, -3.0, 0.0),
    "knee_minus_5deg": (0.0, -5.0, 0.0),
    "phase_plus_3pct": (0.0, 0.0, 0.03),
    "phase_minus_3pct": (0.0, 0.0, -0.03),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(result: GeneratedTrajectory, *, label: str) -> dict[str, object]:
    metadata = result.metadata
    return {
        "label": label,
        "trajectory_id": metadata["trajectory_id"],
        "hip_amplitude_delta_deg": metadata["hip_amplitude_delta_deg"],
        "knee_amplitude_delta_deg": metadata["knee_amplitude_delta_deg"],
        "knee_phase_shift": metadata["knee_phase_shift"],
        "time_scale": metadata["time_scale"],
        "trajectory_sha256": metadata["trajectory_sha256"],
        "parent_reference_id": metadata["parent_reference_id"],
        "parent_reference_sha256": metadata["parent_reference_sha256"],
        "hip_max_deviation_deg": metadata["hip_max_deviation_deg"],
        "hip_rms_deviation_deg": metadata["hip_rms_deviation_deg"],
        "knee_max_deviation_deg": metadata["knee_max_deviation_deg"],
        "knee_rms_deviation_deg": metadata["knee_rms_deviation_deg"],
        "pull_max_deviation_mm": metadata["pull_max_deviation_mm"],
        "pull_rms_deviation_mm": metadata["pull_rms_deviation_mm"],
        "generator_version": metadata["generator_version"],
        "generator_git_commit": metadata["generator_git_commit"],
        "generator_source_sha256": metadata["generator_source_sha256"],
        "domain_bounds_sha256": metadata["domain_bounds_sha256"],
        **result.constraints.as_dict(),
    }


def _plot_joint_family(
    results: Iterable[tuple[str, GeneratedTrajectory]],
    *,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    for label, result in results:
        axis.plot(
            result.trajectory["time_s"],
            np.rad2deg(result.trajectory[column]),
            label=label,
            linewidth=2.0 if label == "neutral" else 1.3,
        )
    axis.set(xlabel="Time (s)", ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_pull_paths(
    results: Iterable[tuple[str, GeneratedTrajectory]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 6.2))
    for label, result in results:
        trajectory = result.trajectory
        axis.plot(
            1000.0 * trajectory["x_pull_m"],
            1000.0 * trajectory["z_pull_m"],
            label=label,
            linewidth=2.0 if label == "neutral" else 1.2,
        )
    axis.set(
        xlabel="Pull point x (mm)",
        ylabel="Pull point z (mm)",
        title="Continuous asymmetric candidate pull paths",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_continuous_reference_neighborhood(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    generate_plots: bool = True,
) -> dict[str, Path]:
    """Generate deterministic verification artifacts; never overwrite a reference."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    parent = load_frozen_active_reference()
    representative = {
        label: generate_personalized_trajectory(parent, *parameters)
        for label, parameters in REPRESENTATIVE_POINTS.items()
    }
    grid_rows: list[dict[str, object]] = []
    for hip in GRID_HIP_DELTAS_DEG:
        for knee in GRID_KNEE_DELTAS_DEG:
            for phase in GRID_PHASE_SHIFTS:
                result = generate_personalized_trajectory(parent, hip, knee, phase)
                grid_rows.append(_summary(result, label="grid_verification_point"))
    grid = pd.DataFrame(grid_rows)
    representative_frame = pd.DataFrame(
        [_summary(result, label=label) for label, result in representative.items()]
    )
    grid_path = output / "continuous_candidate_parameter_space.csv"
    representative_path = output / "representative_regression_points.csv"
    grid.to_csv(grid_path, index=False)
    representative_frame.to_csv(representative_path, index=False)
    paths: dict[str, Path] = {
        grid_path.name: grid_path,
        representative_path.name: representative_path,
    }

    if generate_plots:
        hip_family = [
            ("neutral", representative["neutral"]),
            ("hip -3 deg", representative["hip_minus_3deg"]),
            ("hip -5 deg", representative["hip_minus_5deg"]),
        ]
        knee_family = [
            ("neutral", representative["neutral"]),
            ("knee -3 deg", representative["knee_minus_3deg"]),
            ("knee -5 deg", representative["knee_minus_5deg"]),
        ]
        phase_family = [
            ("phase -3%", representative["phase_minus_3pct"]),
            ("neutral", representative["neutral"]),
            ("phase +3%", representative["phase_plus_3pct"]),
        ]
        plot_specs = {
            "candidate_family_hip.png": lambda p: _plot_joint_family(
                hip_family,
                column="q_hip_rad",
                ylabel="Hip flexion (deg)",
                title="Hip-amplitude family",
                path=p,
            ),
            "candidate_family_knee.png": lambda p: _plot_joint_family(
                knee_family,
                column="q_knee_rad",
                ylabel="Knee flexion (deg)",
                title="Knee-amplitude family",
                path=p,
            ),
            "candidate_family_phase.png": lambda p: _plot_joint_family(
                phase_family,
                column="q_knee_rad",
                ylabel="Knee flexion (deg)",
                title="Knee phase-warp family",
                path=p,
            ),
            "candidate_pull_paths.png": lambda p: _plot_pull_paths(
                [
                    ("neutral", representative["neutral"]),
                    ("hip -3 deg", representative["hip_minus_3deg"]),
                    ("knee -3 deg", representative["knee_minus_3deg"]),
                    ("phase +3%", representative["phase_plus_3pct"]),
                    ("phase -3%", representative["phase_minus_3pct"]),
                ],
                p,
            ),
        }
        for filename, plotter in plot_specs.items():
            path = output / filename
            plotter(path)
            paths[filename] = path

    neutral_metadata = representative["neutral"].metadata
    artifact_sha256 = {
        name: _sha256_file(path) for name, path in sorted(paths.items())
    }
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_status": "offline_generator_verification_only",
        "generator_version": GENERATOR_VERSION,
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "parent_sha_verified_before_generation": True,
        "generator_git_commit": neutral_metadata["generator_git_commit"],
        "generator_source_sha256": neutral_metadata["generator_source_sha256"],
        "domain_bounds_source": neutral_metadata["domain_bounds_source"],
        "domain_bounds_sha256": neutral_metadata["domain_bounds_sha256"],
        "offline_personalization_search_bounds": {
            key: list(value)
            for key, value in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
        },
        "time_scale": FIXED_TIME_SCALE,
        "duration_optimization_enabled": False,
        "grid_axes": {
            "hip_amplitude_delta_deg": list(GRID_HIP_DELTAS_DEG),
            "knee_amplitude_delta_deg": list(GRID_KNEE_DELTAS_DEG),
            "knee_phase_shift": list(GRID_PHASE_SHIFTS),
        },
        "grid_sample_count": int(len(grid)),
        "grid_feasible_count": int(grid["trajectory_feasible"].astype(bool).sum()),
        "representative_points": REPRESENTATIVE_POINTS,
        "neutral_generator_sha": neutral_metadata["neutral_generator_sha"],
        "neutral_reference_max_abs_state_error": neutral_metadata[
            "neutral_reference_max_abs_state_error"
        ],
        "neutral_exact_numeric_state_copy": neutral_metadata[
            "neutral_exact_numeric_state_copy"
        ],
        "trajectory_sha256_definition": neutral_metadata[
            "trajectory_sha256_definition"
        ],
        "trajectory_sha256_columns": neutral_metadata[
            "trajectory_sha256_columns"
        ],
        "rom_protocol_version": neutral_metadata["rom_protocol_version"],
        "hip_rom_deg": neutral_metadata["hip_rom_deg"],
        "knee_rom_deg": neutral_metadata["knee_rom_deg"],
        "theta_shank_definition": neutral_metadata["theta_shank_definition"],
        "continuity_audit_neutral": representative["neutral"].continuity_audit,
        "asymmetry_audit_neutral": representative["neutral"].asymmetry_audit,
        "measured_extension_is_reversed_flexion": False,
        "optimizer_implemented": False,
        "robot_connection_performed": False,
        "robot_motion_authorized": False,
        "hardware_safety_thresholds_modified": False,
        "files": sorted(paths),
        "artifact_sha256": artifact_sha256,
    }
    metadata_path = output / "continuous_generator_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths[metadata_path.name] = metadata_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    paths = run_continuous_reference_neighborhood(
        args.output_dir,
        generate_plots=not args.no_plots,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
