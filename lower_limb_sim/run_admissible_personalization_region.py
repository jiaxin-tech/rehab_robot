"""Build the formal reference-centered offline admissible-region audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .admissible_personalization_region import (
    DEFAULT_REGION_DIRECTORY,
    GLOBAL_ROM_CLASSIFICATION,
    MODEL_RELIABILITY_RULE_STATUS,
    REAL_ROBOT_SAFETY_REGION_STATUS,
    REGION_CLASSIFICATION,
    REGION_VERSION,
)
from .config import L1, L2, workspace_csv_path
from .continuous_reference_neighborhood import (
    DOMAIN_BOUNDS_PATH,
    FIXED_TIME_SCALE,
    GENERATOR_VERSION,
    OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
    GeneratedTrajectory,
    generate_personalized_trajectory,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    PROJECT_ROOT,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .reference_release import load_frozen_active_reference
from .sequential_personalization import (
    MINIMUM_STEP_HIP_DEG,
    MINIMUM_STEP_KNEE_DEG,
    MINIMUM_STEP_PHASE,
)


GRID_HIP_STEP_DEG = MINIMUM_STEP_HIP_DEG
GRID_KNEE_STEP_DEG = MINIMUM_STEP_KNEE_DEG
GRID_PHASE_STEP = MINIMUM_STEP_PHASE
PLOT_PARAMETER_STRIDE = 160
PLOT_PHASE_STRIDE = 4

REPRESENTATIVE_POINTS: dict[str, tuple[float, float, float]] = {
    "neutral": (0.0, 0.0, 0.0),
    "hip_negative": (-3.0, 0.0, 0.0),
    "knee_negative": (0.0, -3.0, 0.0),
    "positive_phase": (0.0, 0.0, 0.03),
    "negative_phase": (0.0, 0.0, -0.03),
    "combined_perturbation": (-3.0, -3.0, 0.03),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def _axis(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(round((upper - lower) / step))
    values = lower + step * np.arange(count + 1, dtype=float)
    if not np.isclose(values[-1], upper, atol=1e-14, rtol=0.0):
        raise RuntimeError("grid resolution does not exactly span a frozen bound")
    values[-1] = upper
    return np.round(values, 12)


def parameter_grid_axes() -> dict[str, np.ndarray]:
    hip_bounds = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[
        "hip_amplitude_delta_deg"
    ]
    knee_bounds = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS[
        "knee_amplitude_delta_deg"
    ]
    phase_bounds = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS["knee_phase_shift"]
    return {
        "hip_amplitude_delta_deg": _axis(*hip_bounds, GRID_HIP_STEP_DEG),
        "knee_amplitude_delta_deg": _axis(*knee_bounds, GRID_KNEE_STEP_DEG),
        "knee_phase_shift": _axis(*phase_bounds, GRID_PHASE_STEP),
    }


def _normalized_invalid_reason(value: str) -> str:
    reasons = []
    for reason in str(value).split(";"):
        if not reason:
            continue
        reasons.append(
            "identification_domain_insufficient"
            if reason == "domain_coverage_insufficient"
            else reason
        )
    return ";".join(reasons)


def _parameter_row(
    result: GeneratedTrajectory,
    hip: float,
    knee: float,
    phase: float,
) -> dict[str, Any]:
    constraints = result.constraints
    metadata = result.metadata
    return {
        "hip_delta": float(hip),
        "knee_delta": float(knee),
        "phase_delta": float(phase),
        "trajectory_id": str(metadata["trajectory_id"]),
        "trajectory_sha256": str(metadata["trajectory_sha256"]),
        "trajectory_admissible": bool(constraints.trajectory_feasible),
        "trajectory_feasible": bool(constraints.trajectory_feasible),
        "domain_coverage": float(constraints.domain_coverage),
        "max_joint_deviation_deg": float(
            max(
                metadata["hip_max_deviation_deg"],
                metadata["knee_max_deviation_deg"],
            )
        ),
        "hip_max_deviation_deg": float(metadata["hip_max_deviation_deg"]),
        "knee_max_deviation_deg": float(metadata["knee_max_deviation_deg"]),
        "max_pull_deviation_mm": float(metadata["pull_max_deviation_mm"]),
        "invalid_reason": _normalized_invalid_reason(constraints.invalid_reason),
        "alpha_bounds_valid": True,
        "global_rom_valid": bool(constraints.rom_valid),
        "workspace_valid": bool(constraints.workspace_valid),
        "jacobian_valid": bool(constraints.jacobian_valid),
        "force_mapping_valid": bool(constraints.force_mapping_valid),
        "domain_valid": bool(constraints.domain_coverage_valid),
        "velocity_valid": bool(constraints.velocity_valid),
        "acceleration_valid": bool(constraints.acceleration_valid),
        "closure_valid": bool(constraints.closure_valid),
        "continuity_valid": bool(result.continuity_audit["passed"]),
        "asymmetry_valid": bool(constraints.asymmetry_valid),
        "finite_valid": bool(constraints.finite_valid),
        "pointwise_clipping_applied": bool(metadata["pointwise_clipping_applied"]),
        "parent_reference_sha256": str(metadata["parent_reference_sha256"]),
    }


def _new_extreme(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": math.inf if kind == "minimum" else -math.inf,
        "sample_index": None,
        "global_phase": None,
        "hip_delta": None,
        "knee_delta": None,
        "phase_delta": None,
    }


def _update_extreme(
    record: dict[str, Any],
    values: np.ndarray,
    global_phase: np.ndarray,
    alpha: tuple[float, float, float],
) -> None:
    if record["kind"] == "minimum":
        index = int(np.argmin(values))
        better = float(values[index]) < float(record["value"])
    else:
        index = int(np.argmax(values))
        better = float(values[index]) > float(record["value"])
    if better:
        record.update(
            {
                "value": float(values[index]),
                "sample_index": index,
                "global_phase": float(global_phase[index]),
                "hip_delta": float(alpha[0]),
                "knee_delta": float(alpha[1]),
                "phase_delta": float(alpha[2]),
            }
        )


def _plot_joint_corridor(
    corridor: pd.DataFrame,
    *,
    joint: str,
    output_path: Path,
) -> None:
    phase = corridor["global_phase"].to_numpy(dtype=float)
    lower = np.rad2deg(corridor[f"q_{joint}_min_rad"].to_numpy(dtype=float))
    reference = np.rad2deg(corridor[f"q_{joint}_ref_rad"].to_numpy(dtype=float))
    upper = np.rad2deg(corridor[f"q_{joint}_max_rad"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.fill_between(phase, lower, upper, alpha=0.25, color="C0", label="admissible envelope audit")
    ax.plot(phase, lower, color="C0", linewidth=0.9, label="lower envelope")
    ax.plot(phase, reference, color="black", linewidth=1.8, label="active reference")
    ax.plot(phase, upper, color="C1", linewidth=0.9, label="upper envelope")
    ax.set(
        xlabel="Global cycle phase",
        ylabel=f"{joint.capitalize()} flexion (deg)",
        title=f"Reference-centered {joint} corridor",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_joint_space(
    reference: pd.DataFrame,
    admissible_points: list[np.ndarray],
    rejected_points: list[np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    if admissible_points:
        values = np.concatenate(admissible_points, axis=0)
        ax.scatter(values[:, 0], values[:, 1], s=2, alpha=0.18, color="C2", label="admissible generated samples")
    if rejected_points:
        values = np.concatenate(rejected_points, axis=0)
        ax.scatter(values[:, 0], values[:, 1], s=3, alpha=0.18, color="C3", label="rejected generated samples")
    ax.plot(
        np.rad2deg(reference["q_hip_rad"]),
        np.rad2deg(reference["q_knee_rad"]),
        color="black",
        linewidth=2.0,
        label="active reference",
    )
    hip = FORMAL_HIP_ROM_DEG
    knee = FORMAL_KNEE_ROM_DEG
    ax.plot(
        [hip[0], hip[1], hip[1], hip[0], hip[0]],
        [knee[0], knee[0], knee[1], knee[1], knee[0]],
        "k--",
        linewidth=1.0,
        label="global ROM boundary",
    )
    ax.set(
        xlabel="Hip flexion (deg)",
        ylabel="Knee flexion (deg)",
        title="Joint-space admissible-region audit",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pull_corridor(
    corridor: pd.DataFrame,
    admissible_points: list[np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    if admissible_points:
        values = np.concatenate(admissible_points, axis=0)
        ax.scatter(values[:, 0], values[:, 1], s=2, alpha=0.16, color="C2", label="admissible generated pull samples")
    ax.plot(
        1000.0 * corridor["x_pull_ref_m"],
        1000.0 * corridor["z_pull_ref_m"],
        color="black",
        linewidth=2.0,
        label="active reference pull path",
    )
    for x_name, z_name, label, color in (
        ("x_pull_min_m", "z_pull_min_m", "phase-wise lower pair", "C0"),
        ("x_pull_max_m", "z_pull_max_m", "phase-wise upper pair", "C1"),
    ):
        ax.plot(
            1000.0 * corridor[x_name],
            1000.0 * corridor[z_name],
            color=color,
            linewidth=0.9,
            label=label,
        )
    ax.set(
        xlabel="Pull point x (mm)",
        ylabel="Pull point z (mm)",
        title="Reference-centered pull-point corridor audit",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pull_radial(corridor: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(
        corridor["global_phase"],
        corridor["pull_radial_max_mm"],
        color="C4",
        linewidth=1.8,
    )
    ax.set(
        xlabel="Global cycle phase",
        ylabel="Maximum radial deviation (mm)",
        title="Maximum admissible same-phase pull deviation from reference",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_parameter_slices(table: pd.DataFrame, output_path: Path) -> None:
    phases = (-0.03, 0.0, 0.03)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), sharex=True, sharey=True)
    for ax, phase in zip(axes, phases):
        subset = table.loc[np.isclose(table["phase_delta"], phase, atol=1e-12)]
        admissible = subset["trajectory_admissible"].astype(bool).to_numpy()
        ax.scatter(
            subset.loc[~admissible, "hip_delta"],
            subset.loc[~admissible, "knee_delta"],
            marker="s",
            s=24,
            color="C3",
            label="rejected",
        )
        ax.scatter(
            subset.loc[admissible, "hip_delta"],
            subset.loc[admissible, "knee_delta"],
            marker="s",
            s=24,
            color="C2",
            label="admissible",
        )
        ax.set_title(f"phase = {phase:+.3f}")
        ax.set_xlabel("Hip amplitude delta (deg)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Knee amplitude delta (deg)")
    axes[0].legend()
    fig.suptitle("Offline parameter-space admissibility slices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_admissible_personalization_region(
    output_directory: str | Path = DEFAULT_REGION_DIRECTORY,
    *,
    progress_every: int = 1000,
) -> dict[str, Path]:
    """Scan the frozen trust-lattice resolution and build formal artifacts."""

    started = time.perf_counter()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    parent = load_frozen_active_reference()
    neutral = generate_personalized_trajectory(parent, 0.0, 0.0, 0.0)
    if not neutral.constraints.trajectory_feasible:
        raise RuntimeError("active reference is not feasible before region construction")
    reference = neutral.trajectory
    sample_count = len(reference)
    phase_values = reference["global_phase"].to_numpy(dtype=float)
    q_hip_ref = reference["q_hip_rad"].to_numpy(dtype=float)
    q_knee_ref = reference["q_knee_rad"].to_numpy(dtype=float)
    x_ref = reference["x_pull_m"].to_numpy(dtype=float)
    z_ref = reference["z_pull_m"].to_numpy(dtype=float)

    q_hip_min = np.full(sample_count, np.inf)
    q_hip_max = np.full(sample_count, -np.inf)
    q_knee_min = np.full(sample_count, np.inf)
    q_knee_max = np.full(sample_count, -np.inf)
    x_min = np.full(sample_count, np.inf)
    x_max = np.full(sample_count, -np.inf)
    z_min = np.full(sample_count, np.inf)
    z_max = np.full(sample_count, -np.inf)
    radial_max_mm = np.zeros(sample_count, dtype=float)
    extrema = {
        "maximum_negative_hip_deviation_deg": _new_extreme("minimum"),
        "maximum_positive_hip_deviation_deg": _new_extreme("maximum"),
        "maximum_negative_knee_deviation_deg": _new_extreme("minimum"),
        "maximum_positive_knee_deviation_deg": _new_extreme("maximum"),
        "minimum_delta_x_pull_mm": _new_extreme("minimum"),
        "maximum_delta_x_pull_mm": _new_extreme("maximum"),
        "minimum_delta_z_pull_mm": _new_extreme("minimum"),
        "maximum_delta_z_pull_mm": _new_extreme("maximum"),
        "maximum_radial_pull_deviation_mm": _new_extreme("maximum"),
    }

    axes = parameter_grid_axes()
    total = int(np.prod([len(axis) for axis in axes.values()]))
    rows: list[dict[str, Any]] = []
    admissible_joint_plot: list[np.ndarray] = []
    rejected_joint_plot: list[np.ndarray] = []
    admissible_pull_plot: list[np.ndarray] = []
    index = 0
    for hip in axes["hip_amplitude_delta_deg"]:
        for knee in axes["knee_amplitude_delta_deg"]:
            for phase in axes["knee_phase_shift"]:
                result = generate_personalized_trajectory(
                    parent, float(hip), float(knee), float(phase)
                )
                row = _parameter_row(result, float(hip), float(knee), float(phase))
                rows.append(row)
                trajectory = result.trajectory
                plot_joint = np.column_stack(
                    (
                        np.rad2deg(trajectory["q_hip_rad"].to_numpy(dtype=float)),
                        np.rad2deg(trajectory["q_knee_rad"].to_numpy(dtype=float)),
                    )
                )[::PLOT_PHASE_STRIDE]
                if row["trajectory_admissible"]:
                    q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
                    q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
                    x_pull = trajectory["x_pull_m"].to_numpy(dtype=float)
                    z_pull = trajectory["z_pull_m"].to_numpy(dtype=float)
                    q_hip_min = np.minimum(q_hip_min, q_hip)
                    q_hip_max = np.maximum(q_hip_max, q_hip)
                    q_knee_min = np.minimum(q_knee_min, q_knee)
                    q_knee_max = np.maximum(q_knee_max, q_knee)
                    x_min = np.minimum(x_min, x_pull)
                    x_max = np.maximum(x_max, x_pull)
                    z_min = np.minimum(z_min, z_pull)
                    z_max = np.maximum(z_max, z_pull)
                    delta_x_mm = 1000.0 * (x_pull - x_ref)
                    delta_z_mm = 1000.0 * (z_pull - z_ref)
                    radial_mm = np.hypot(delta_x_mm, delta_z_mm)
                    radial_max_mm = np.maximum(radial_max_mm, radial_mm)
                    alpha = (float(hip), float(knee), float(phase))
                    _update_extreme(
                        extrema["maximum_negative_hip_deviation_deg"],
                        np.rad2deg(q_hip - q_hip_ref),
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["maximum_positive_hip_deviation_deg"],
                        np.rad2deg(q_hip - q_hip_ref),
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["maximum_negative_knee_deviation_deg"],
                        np.rad2deg(q_knee - q_knee_ref),
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["maximum_positive_knee_deviation_deg"],
                        np.rad2deg(q_knee - q_knee_ref),
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["minimum_delta_x_pull_mm"],
                        delta_x_mm,
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["maximum_delta_x_pull_mm"],
                        delta_x_mm,
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["minimum_delta_z_pull_mm"],
                        delta_z_mm,
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["maximum_delta_z_pull_mm"],
                        delta_z_mm,
                        phase_values,
                        alpha,
                    )
                    _update_extreme(
                        extrema["maximum_radial_pull_deviation_mm"],
                        radial_mm,
                        phase_values,
                        alpha,
                    )
                    if index % PLOT_PARAMETER_STRIDE == 0:
                        admissible_joint_plot.append(plot_joint)
                        admissible_pull_plot.append(
                            1000.0
                            * trajectory[["x_pull_m", "z_pull_m"]]
                            .to_numpy(dtype=float)[::PLOT_PHASE_STRIDE]
                        )
                elif index % PLOT_PARAMETER_STRIDE == 0:
                    rejected_joint_plot.append(plot_joint)
                index += 1
                if progress_every > 0 and index % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"admissible-region scan {index}/{total} "
                        f"({100.0 * index / total:.1f}%), elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    table = pd.DataFrame(rows)
    if len(table) != total:
        raise RuntimeError("parameter-space scan is incomplete")
    admissible_mask = table["trajectory_admissible"].astype(bool)
    if not admissible_mask.any():
        raise RuntimeError("formal parameter scan found no admissible trajectory")
    if not np.isfinite(
        np.column_stack(
            (
                q_hip_min,
                q_hip_max,
                q_knee_min,
                q_knee_max,
                x_min,
                x_max,
                z_min,
                z_max,
                radial_max_mm,
            )
        )
    ).all():
        raise RuntimeError("phase-wise envelope construction is incomplete")

    joint_corridor = pd.DataFrame(
        {
            "sample_index": np.arange(sample_count, dtype=int),
            "time_s": reference["time_s"].to_numpy(dtype=float),
            "global_phase": phase_values,
            "cycle_phase": reference["cycle_phase"].astype(str).to_numpy(),
            "segment_phase": reference["segment_phase"].to_numpy(dtype=float),
            "q_hip_ref_rad": q_hip_ref,
            "q_hip_min_rad": q_hip_min,
            "q_hip_max_rad": q_hip_max,
            "q_knee_ref_rad": q_knee_ref,
            "q_knee_min_rad": q_knee_min,
            "q_knee_max_rad": q_knee_max,
        }
    )
    pull_corridor = pd.DataFrame(
        {
            "sample_index": np.arange(sample_count, dtype=int),
            "time_s": reference["time_s"].to_numpy(dtype=float),
            "global_phase": phase_values,
            "cycle_phase": reference["cycle_phase"].astype(str).to_numpy(),
            "segment_phase": reference["segment_phase"].to_numpy(dtype=float),
            "x_pull_ref_m": x_ref,
            "x_pull_min_m": x_min,
            "x_pull_max_m": x_max,
            "z_pull_ref_m": z_ref,
            "z_pull_min_m": z_min,
            "z_pull_max_m": z_max,
            "pull_radial_max_mm": radial_max_mm,
        }
    )
    if not (
        np.all(q_hip_min <= q_hip_ref)
        and np.all(q_hip_ref <= q_hip_max)
        and np.all(q_knee_min <= q_knee_ref)
        and np.all(q_knee_ref <= q_knee_max)
        and np.all(x_min <= x_ref)
        and np.all(x_ref <= x_max)
        and np.all(z_min <= z_ref)
        and np.all(z_ref <= z_max)
    ):
        raise RuntimeError("active reference is outside its constructed corridor")

    paths: dict[str, Path] = {}
    for filename, frame in (
        ("joint_corridor_by_phase.csv", joint_corridor),
        ("pull_corridor_by_phase.csv", pull_corridor),
        ("parameter_space_admissibility.csv", table),
        ("admissible_parameter_samples.csv", table.loc[admissible_mask]),
        ("rejected_parameter_samples.csv", table.loc[~admissible_mask]),
    ):
        path = output / filename
        frame.to_csv(path, index=False, float_format="%.17g")
        paths[filename] = path

    representative_rows = []
    for label, (hip, knee, phase) in REPRESENTATIVE_POINTS.items():
        match = table.loc[
            np.isclose(table["hip_delta"], hip, atol=1e-12)
            & np.isclose(table["knee_delta"], knee, atol=1e-12)
            & np.isclose(table["phase_delta"], phase, atol=1e-12)
        ]
        if len(match) != 1:
            raise RuntimeError(f"representative point missing from formal grid: {label}")
        representative_rows.append({"label": label, **match.iloc[0].to_dict()})
    representative = pd.DataFrame(representative_rows)
    representative_path = output / "representative_trajectory_admissibility.csv"
    representative.to_csv(representative_path, index=False, float_format="%.17g")
    paths[representative_path.name] = representative_path

    bounds = OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
    boundary_mask = (
        table["hip_delta"].isin(bounds["hip_amplitude_delta_deg"])
        | table["knee_delta"].isin(bounds["knee_amplitude_delta_deg"])
        | table["phase_delta"].isin(bounds["knee_phase_shift"])
    )
    interior_rejected = (~admissible_mask) & (~boundary_mask)
    rejected_reason_counts: dict[str, int] = {}
    for reasons in table.loc[~admissible_mask, "invalid_reason"].astype(str):
        for reason in filter(None, reasons.split(";")):
            rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1
    width_mm = 1000.0 * np.hypot(x_max - x_min, z_max - z_min)
    summary = {
        "region_version": REGION_VERSION,
        "artifact_status": "FORMAL_OFFLINE_SOFTWARE_REGION_AUDIT",
        "parameter_sample_count": int(len(table)),
        "admissible_parameter_sample_count": int(admissible_mask.sum()),
        "rejected_parameter_sample_count": int((~admissible_mask).sum()),
        "admissible_fraction": float(admissible_mask.mean()),
        "boundary_rejected_parameter_sample_count": int(
            ((~admissible_mask) & boundary_mask).sum()
        ),
        "interior_rejected_parameter_sample_count": int(interior_rejected.sum()),
        "parameter_box_contains_interior_infeasible_samples": bool(
            interior_rejected.any()
        ),
        "parameter_box_claimed_fully_feasible": False,
        "rejected_reason_counts": rejected_reason_counts,
        "trajectory_sample_count": sample_count,
        "joint_extrema": {
            key: value
            for key, value in extrema.items()
            if "hip_deviation" in key or "knee_deviation" in key
        },
        "pull_extrema": {
            key: value
            for key, value in extrema.items()
            if "pull" in key
        },
        "rms_cartesian_envelope_width_mm": float(
            np.sqrt(np.mean(width_mm**2))
        ),
        "rms_cartesian_envelope_width_definition": (
            "sqrt(mean((1000*sqrt((x_max-x_min)^2+(z_max-z_min)^2))^2))"
        ),
        "global_pull_x_min_m": float(x_min.min()),
        "global_pull_x_max_m": float(x_max.max()),
        "global_pull_z_min_m": float(z_min.min()),
        "global_pull_z_max_m": float(z_max.max()),
        "maximum_phasewise_pull_radial_deviation_mm": float(radial_max_mm.max()),
        "representative_trajectories": representative[
            ["label", "hip_delta", "knee_delta", "phase_delta", "trajectory_admissible", "invalid_reason"]
        ].to_dict(orient="records"),
        "visual_envelope_is_robot_safety_boundary": False,
        "real_robot_safety_region_status": REAL_ROBOT_SAFETY_REGION_STATUS,
        "model_reliability_rule_status": MODEL_RELIABILITY_RULE_STATUS,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    summary_path = output / "admissible_region_summary.json"
    _write_json(summary_path, summary)
    paths[summary_path.name] = summary_path

    figure_specs = {
        "hip_corridor_vs_phase.png": lambda path: _plot_joint_corridor(
            joint_corridor, joint="hip", output_path=path
        ),
        "knee_corridor_vs_phase.png": lambda path: _plot_joint_corridor(
            joint_corridor, joint="knee", output_path=path
        ),
        "joint_space_admissible_region.png": lambda path: _plot_joint_space(
            reference, admissible_joint_plot, rejected_joint_plot, path
        ),
        "pull_point_admissible_corridor.png": lambda path: _plot_pull_corridor(
            pull_corridor, admissible_pull_plot, path
        ),
        "pull_deviation_vs_phase.png": lambda path: _plot_pull_radial(
            pull_corridor, path
        ),
        "parameter_space_admissibility_slices.png": lambda path: _plot_parameter_slices(
            table, path
        ),
    }
    for filename, plotter in figure_specs.items():
        path = output / filename
        plotter(path)
        paths[filename] = path

    workspace_path = Path(workspace_csv_path)
    if not workspace_path.is_file():
        raise FileNotFoundError(f"formal ROM V2 workspace artifact missing: {workspace_path}")
    manifest = {
        "region_version": REGION_VERSION,
        "region_classification": REGION_CLASSIFICATION,
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "generator_version": GENERATOR_VERSION,
        "generator_source_sha256": neutral.metadata["generator_source_sha256"],
        "generator_source_path": str(
            Path(__file__).with_name("continuous_reference_neighborhood.py").resolve()
        ),
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "global_physical_model_rom": {
            "classification": GLOBAL_ROM_CLASSIFICATION,
            "hip_deg": list(FORMAL_HIP_ROM_DEG),
            "knee_deg": list(FORMAL_KNEE_ROM_DEG),
        },
        "offline_personalization_alpha_bounds": {
            key: list(value)
            for key, value in OFFLINE_PERSONALIZATION_SEARCH_BOUNDS.items()
        },
        "time_scale": FIXED_TIME_SCALE,
        "duration_s": 24.0,
        "workspace_artifact": str(workspace_path.resolve()),
        "workspace_artifact_sha256": _file_sha256(workspace_path),
        "identification_domain_artifact": str(Path(DOMAIN_BOUNDS_PATH).resolve()),
        "identification_domain_sha256": _file_sha256(DOMAIN_BOUNDS_PATH),
        "envelope_construction_method": (
            "deterministic_full_cartesian_lattice_at_existing_frozen_"
            "minimum_trust_region_steps;phasewise_envelope_of_all_"
            "generator_samples_passing_existing_full_trajectory_gates"
        ),
        "grid_resolution": {
            "hip_amplitude_delta_deg": GRID_HIP_STEP_DEG,
            "knee_amplitude_delta_deg": GRID_KNEE_STEP_DEG,
            "knee_phase_shift": GRID_PHASE_STEP,
        },
        "grid_resolution_source": (
            "existing_frozen_minimum_trust_region_steps_read_only;"
            "trust_region_algorithm_not_modified"
        ),
        "grid_axis_counts": {key: int(len(value)) for key, value in axes.items()},
        "grid_sampling_method": "inclusive_cartesian_product_lexical_nested_order",
        "grid_sample_count": int(len(table)),
        "corners_included": True,
        "bounds_edges_included": True,
        "neutral_included": True,
        "representative_points_included": list(REPRESENTATIVE_POINTS),
        "visual_envelope_is_final_membership_test": False,
        "formal_membership_test": (
            "alpha_bounds_plus_phasewise_joint_and_pull_corridors_plus_"
            "direct_full_trajectory_existing_generator_gates"
        ),
        "pointwise_clipping_allowed": False,
        "real_robot_safety_region_status": REAL_ROBOT_SAFETY_REGION_STATUS,
        "offline_region_is_real_robot_safety_region": False,
        "model_reliability_rule_status": MODEL_RELIABILITY_RULE_STATUS,
        "mechanical_objective_modified": False,
        "optimizer_ranking_modified": False,
        "trust_region_steps_modified": False,
        "formal_sequential_experiment_rerun": False,
        "created_git_commit": _git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "L1_m": L1,
        "L2_m": L2,
        "artifact_sha256": {
            name: _file_sha256(path) for name, path in sorted(paths.items())
        },
        "files": sorted(paths),
    }
    manifest_path = output / "admissible_region_manifest.json"
    _write_json(manifest_path, manifest)
    paths[manifest_path.name] = manifest_path
    print(
        f"admissible-region complete: {len(table)} samples, "
        f"{int(admissible_mask.sum())} admissible, "
        f"{int((~admissible_mask).sum())} rejected, "
        f"runtime={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build REFERENCE_CENTERED_ADMISSIBLE_REGION_V1 offline artifacts."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REGION_DIRECTORY)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    paths = run_admissible_personalization_region(
        args.output_dir, progress_every=args.progress_every
    )
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, indent=2))


if __name__ == "__main__":
    main()
