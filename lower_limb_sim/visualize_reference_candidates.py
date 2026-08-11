"""Stage 5C plots for reference execution and software-only candidate screening.

The plotting boundary intentionally mirrors the Stage 5C execution boundary.
Measured and closed reference geometry may be audited without ROM approval, but
local-identification, candidate-dynamics, and Pareto figures are generated only
after the caller records an explicit approved knee ROM.  Missing data are never
replaced with synthetic zero curves: every requested figure is either returned
in ``paths`` or accompanied by an explicit entry in ``skipped``.

All force quantities shown here are virtual, software-only comparison metrics.
They are not real-robot safety limits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any


_MPL_CONFIG_DIRECTORY = Path(tempfile.gettempdir()) / "lower_limb_sim_matplotlib"
_MPL_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIRECTORY))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .config import L1, L2
from .kinematics import forward_kinematics


FIGURE_FILENAMES = (
    "measured_vs_closed_reference.png",
    "reference_closure_comparison.png",
    "local_excitation_trajectories.png",
    "local_identification_domain.png",
    "candidate_joint_paths.png",
    "candidate_torque_comparison.png",
    "candidate_subject_comparison.png",
    "candidate_pareto.png",
)
GEOMETRY_AUDIT_FIGURES = FIGURE_FILENAMES[:2]
FORMAL_EVALUATION_FIGURES = FIGURE_FILENAMES[2:]

_HIP_COLOR = "#4C78A8"
_KNEE_COLOR = "#F58518"
_CLOSED_COLOR = "#54A24B"
_MEASURED_COLOR = "#79706E"
_IN_DOMAIN_COLOR = "#54A24B"
_OUT_DOMAIN_COLOR = "#E45756"
_PARETO_COLOR = "#4C78A8"
_NON_PARETO_COLOR = "#BAB0AC"

_EXECUTION_VERSION_COLUMNS = (
    "reference_version",
    "execution_version",
    "trajectory_version",
    "version",
    "source_trajectory_type",
)
_TRAJECTORY_COLUMNS = (
    "trajectory_id",
    "trajectory_name",
    "excitation_id",
    "excitation_name",
    "profile",
    "candidate_id",
    "candidate_name",
)
_CANDIDATE_COLUMNS = (
    "candidate_id",
    "candidate_name",
    "trajectory_id",
    "trajectory_name",
)
_SUBJECT_COLUMNS = ("subject_id", "virtual_subject_id")


@dataclass(frozen=True)
class ReferenceCandidateVisualizationResult:
    """Generated paths and explicit reasons for omitted requested figures."""

    paths: dict[str, Path]
    skipped: dict[str, str]

    @property
    def generated_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.paths[name] for name in FIGURE_FILENAMES if name in self.paths
        )

    @property
    def all_requested_outputs_accounted_for(self) -> bool:
        return set(self.paths) | set(self.skipped) == set(FIGURE_FILENAMES)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _as_dataframe(value: pd.DataFrame | None, name: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame or None.")
    return value.copy(deep=False)


def _column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in dataframe), None)


def _numeric(dataframe: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray | None:
    column = _column(dataframe, candidates)
    if column is None:
        return None
    return pd.to_numeric(dataframe[column], errors="coerce").to_numpy(float)


def _finite_numeric_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
) -> tuple[str, np.ndarray] | None:
    for name in candidates:
        if name not in dataframe:
            continue
        values = pd.to_numeric(dataframe[name], errors="coerce").to_numpy(float)
        if np.isfinite(values).any():
            return name, values
    return None


def _boolean(dataframe: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray | None:
    column = _column(dataframe, candidates)
    if column is None:
        return None
    values = dataframe[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).to_numpy(float) != 0.0
    return values.astype(str).str.strip().str.lower().isin(
        ("true", "yes", "1", "inside", "in_domain", "feasible", "pareto")
    ).to_numpy(bool)


def _finite(*arrays: np.ndarray | None) -> np.ndarray:
    available = [np.asarray(value) for value in arrays if value is not None]
    if not available:
        return np.empty(0, dtype=bool)
    mask = np.ones(len(available[0]), dtype=bool)
    for value in available:
        if len(value) != len(mask):
            raise ValueError("Plotting arrays must have equal lengths.")
        mask &= np.isfinite(value)
    return mask


def _phase(dataframe: pd.DataFrame) -> np.ndarray:
    phase_value = _finite_numeric_column(
        dataframe,
        ("global_phase", "cycle_phase", "normalized_phase", "phase"),
    )
    if phase_value is None:
        time_value = _finite_numeric_column(dataframe, ("time_s", "retimed_time_s"))
        if time_value is not None:
            _, time_values = time_value
            finite_time = time_values[np.isfinite(time_values)]
            if finite_time.size and np.ptp(finite_time) > 0.0:
                return (time_values - np.nanmin(finite_time)) / np.ptp(finite_time)
        if len(dataframe) <= 1:
            return np.zeros(len(dataframe), dtype=float)
        return np.linspace(0.0, 1.0, len(dataframe))
    _, values = phase_value
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmax(finite) > 1.5:
        values = values / 100.0
    return values


def _angle_degrees(dataframe: pd.DataFrame, joint: str) -> np.ndarray | None:
    degree = _numeric(
        dataframe,
        (
            f"q_{joint}_deg",
            f"q_{joint}_reference_deg",
            f"q_{joint}_smoothed_deg",
            f"q_{joint}_raw_deg",
        ),
    )
    if degree is not None:
        return degree
    radians = _numeric(
        dataframe,
        (
            f"q_{joint}_rad",
            f"q_{joint}_reference_rad",
            f"q_{joint}_smoothed_rad",
            f"q_{joint}_raw_rad",
        ),
    )
    return None if radians is None else np.rad2deg(radians)


def _angle_radians(dataframe: pd.DataFrame, joint: str) -> np.ndarray | None:
    radians = _numeric(
        dataframe,
        (
            f"q_{joint}_rad",
            f"q_{joint}_reference_rad",
            f"q_{joint}_smoothed_rad",
            f"q_{joint}_raw_rad",
        ),
    )
    if radians is not None:
        return radians
    degrees = _angle_degrees(dataframe, joint)
    return None if degrees is None else np.deg2rad(degrees)


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    for nested_name in (
        "rom",
        "rom_approval",
        "rom_audit",
        "execution",
        "candidate_evaluation",
        "reference",
        "stage5c",
    ):
        nested = metadata.get(nested_name)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {
        "true",
        "yes",
        "1",
        "approved",
        "confirmed",
        "allowed",
        "pass",
        "passed",
    }


def _explicit_rom_gate(metadata: Mapping[str, Any]) -> tuple[bool, str]:
    """Return true only when the caller records explicit Stage 5C approval."""

    formal_gate = _metadata_value(
        metadata,
        "formal_evaluation_allowed",
        "formal_execution_allowed",
    )
    approval_gate = _metadata_value(
        metadata,
        "rom_approved",
        "approved_knee_rom_provided",
        "knee_rom_approval_provided",
        "approved_knee_rom_supplied",
        "approval_supplied",
    )
    gate = formal_gate if formal_gate is not None else approval_gate
    minimum = _metadata_value(
        metadata,
        "approved_knee_min_deg",
        "knee_min_deg",
    )
    maximum = _metadata_value(
        metadata,
        "approved_knee_max_deg",
        "knee_max_deg",
    )
    if minimum is None or maximum is None:
        approved_range = _metadata_value(
            metadata,
            "approved_knee_range_deg",
            "approved_knee_rom_deg",
        )
        if isinstance(approved_range, Sequence) and not isinstance(
            approved_range, (str, bytes)
        ):
            try:
                minimum, maximum = approved_range
            except ValueError:
                pass
    try:
        lower = float(minimum)
        upper = float(maximum)
        finite_range = np.isfinite(lower) and np.isfinite(upper) and lower < upper
    except (TypeError, ValueError):
        finite_range = False
    if not (_truth(gate) and finite_range):
        return (
            False,
            "explicit approved knee ROM is unavailable; local identification, "
            "formal dynamics, candidate screening, and Pareto figures were "
            "not generated",
        )
    return True, ""


def _save(figure: plt.Figure, destination: Path, filename: str) -> Path:
    path = destination / filename
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _named_groups(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    fallback: str,
) -> list[tuple[str, pd.DataFrame]]:
    if dataframe.empty:
        return []
    column = _column(dataframe, candidates)
    if column is None:
        return [(fallback, dataframe.copy(deep=False))]
    groups = [
        (str(name), group.copy(deep=False))
        for name, group in dataframe.groupby(column, sort=False, dropna=False)
    ]
    return sorted(groups, key=lambda item: _natural_name_key(item[0]))


def _natural_name_key(name: str) -> tuple[int, str]:
    normalized = name.strip().lower()
    if normalized.startswith("c") and normalized[1:].isdigit():
        return int(normalized[1:]), normalized
    preferred = {
        "reference_measured_asymmetric": 0,
        "measured_asymmetric": 0,
        "reference_closed_symmetric": 1,
        "closed_symmetric": 1,
        "reference_slow": 2,
        "reference_nominal": 3,
        "hip_amplitude_minus_3deg": 4,
        "knee_amplitude_minus_3deg": 5,
        "knee_phase_advance_3pct": 6,
        "knee_phase_delay_3pct": 7,
    }
    return preferred.get(normalized, 1000), normalized


def _representative_path(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated subject rows while preserving one geometric phase path."""

    if dataframe.empty:
        return dataframe
    phase_value = _finite_numeric_column(
        dataframe,
        (
            "global_phase",
            "cycle_phase",
            "normalized_phase",
            "phase",
            "time_s",
            "retimed_time_s",
        ),
    )
    if phase_value is None:
        return dataframe
    _, phase_values = phase_value
    ordered = dataframe.assign(_plot_phase=phase_values).sort_values(
        "_plot_phase", kind="mergesort"
    )
    return ordered.drop_duplicates("_plot_phase", keep="first").drop(
        columns="_plot_phase"
    )


def _version_groups(execution_versions: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = _named_groups(
        execution_versions,
        _EXECUTION_VERSION_COLUMNS,
        "reference",
    )
    return [(name, _representative_path(frame)) for name, frame in groups]


def _version_role(name: str) -> str:
    normalized = name.strip().lower()
    if "measured" in normalized or "asymmetric" in normalized:
        return "measured"
    if "closed" in normalized or "symmetric" in normalized:
        return "closed"
    return "other"


def _pull_path(
    dataframe: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    x_pull = _numeric(dataframe, ("x_pull_m", "x_pull"))
    z_pull = _numeric(dataframe, ("z_pull_m", "z_pull"))
    if x_pull is not None and z_pull is not None:
        return x_pull, z_pull
    q_hip = _angle_radians(dataframe, "hip")
    q_knee = _angle_radians(dataframe, "knee")
    if q_hip is None or q_knee is None:
        return None
    try:
        l1 = float(_metadata_value(metadata, "L1", "L1_m") or L1)
        l2 = float(_metadata_value(metadata, "L2", "L2_m") or L2)
    except (TypeError, ValueError):
        l1, l2 = float(L1), float(L2)
    _, _, x_pull, z_pull = forward_kinematics(q_hip, q_knee, l1, l2)
    return np.asarray(x_pull, dtype=float), np.asarray(z_pull, dtype=float)


def _plot_measured_vs_closed(
    execution_versions: pd.DataFrame,
    destination: Path,
) -> Path | None:
    groups = _version_groups(execution_versions)
    groups = [(name, frame) for name, frame in groups if _version_role(name) != "other"]
    if not groups:
        return None
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    plotted = False
    for name, dataframe in groups:
        role = _version_role(name)
        phase = 100.0 * _phase(dataframe)
        color = _MEASURED_COLOR if role == "measured" else _CLOSED_COLOR
        linestyle = "--" if role == "measured" else "-"
        label = (
            "Measured asymmetric path (source geometry)"
            if role == "measured"
            else "Closed symmetric path (synthetic reverse extension)"
        )
        for axis, joint in zip(axes, ("hip", "knee")):
            angle = _angle_degrees(dataframe, joint)
            if angle is None:
                continue
            valid = _finite(phase, angle)
            if np.any(valid):
                axis.plot(
                    phase[valid],
                    angle[valid],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=label,
                )
                plotted = True
    if not plotted:
        plt.close(figure)
        return None
    for axis, joint in zip(axes, ("hip", "knee")):
        axis.axvline(50.0, color="#999999", linestyle=":", linewidth=1.0)
        axis.set_ylabel(f"{joint.title()} angle (deg)")
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=8)
    axes[-1].set_xlabel("Reference cycle phase (%)")
    axes[0].set_title(
        "Measured asymmetric reference versus closed symmetric execution path"
    )
    axes[0].text(
        0.01,
        0.05,
        "The symmetric extension is generated by reversing flexion; it is not measured skeleton motion.",
        transform=axes[0].transAxes,
        fontsize=8,
    )
    return _save(figure, destination, "measured_vs_closed_reference.png")


def _plot_closure_comparison(
    execution_versions: pd.DataFrame,
    metadata: Mapping[str, Any],
    destination: Path,
) -> Path | None:
    groups = _version_groups(execution_versions)
    groups = [(name, frame) for name, frame in groups if _version_role(name) != "other"]
    if not groups:
        return None
    figure = plt.figure(figsize=(12, 6.5))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.55, 1.0))
    path_axis = figure.add_subplot(grid[:, 0])
    angle_axis = figure.add_subplot(grid[0, 1])
    pull_axis = figure.add_subplot(grid[1, 1])
    labels: list[str] = []
    hip_errors: list[float] = []
    knee_errors: list[float] = []
    pull_errors: list[float] = []
    path_plotted = False
    for index, (name, dataframe) in enumerate(groups):
        role = _version_role(name)
        display = "measured asymmetric" if role == "measured" else "closed symmetric"
        color = _MEASURED_COLOR if role == "measured" else _CLOSED_COLOR
        path = _pull_path(dataframe, metadata)
        hip = _angle_degrees(dataframe, "hip")
        knee = _angle_degrees(dataframe, "knee")
        if path is None or hip is None or knee is None:
            continue
        x_pull, z_pull = path
        valid = _finite(x_pull, z_pull, hip, knee)
        indices = np.flatnonzero(valid)
        if len(indices) < 2:
            continue
        first, last = int(indices[0]), int(indices[-1])
        path_axis.plot(
            x_pull[valid],
            z_pull[valid],
            color=color,
            linestyle="--" if role == "measured" else "-",
            linewidth=1.7,
            label=display,
        )
        path_axis.scatter(
            [x_pull[first]],
            [z_pull[first]],
            color=color,
            marker="o",
            s=45,
            zorder=4,
        )
        path_axis.scatter(
            [x_pull[last]],
            [z_pull[last]],
            color=color,
            marker="x",
            s=55,
            zorder=4,
        )
        labels.append(display)
        hip_errors.append(abs(float(hip[last] - hip[first])))
        knee_errors.append(abs(float(knee[last] - knee[first])))
        pull_errors.append(float(np.hypot(x_pull[last] - x_pull[first], z_pull[last] - z_pull[first])))
        path_plotted = True
    if not path_plotted:
        plt.close(figure)
        return None
    y = np.arange(len(labels))
    width = 0.36
    angle_axis.barh(y - width / 2.0, hip_errors, height=width, color=_HIP_COLOR, label="Hip")
    angle_axis.barh(y + width / 2.0, knee_errors, height=width, color=_KNEE_COLOR, label="Knee")
    angle_axis.set_yticks(y, labels)
    angle_axis.set_xlabel("Absolute end-to-start angle error (deg)")
    angle_axis.set_title("Joint closure error")
    angle_axis.legend(fontsize=8)
    pull_axis.barh(y, pull_errors, color="#B279A2")
    pull_axis.set_yticks(y, labels)
    pull_axis.set_xlabel("End-to-start pull-point error (m)")
    pull_axis.set_title("Equivalent pull-point closure error")
    for row, value in enumerate(pull_errors):
        pull_axis.text(value, row, f" {value:.4g} m", va="center", fontsize=8)
    path_axis.axhline(0.0, color="#222222", linewidth=1.0, label="Bed: z = 0")
    path_axis.set_xlabel("Equivalent pull point x (m)")
    path_axis.set_ylabel("Equivalent pull point z (m)")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_title("L2 equivalent pull paths; ○ start, × end")
    path_axis.legend(fontsize=8)
    figure.suptitle("Reference closure audit — path geometry only")
    return _save(figure, destination, "reference_closure_comparison.png")


def _trajectory_groups(dataframe: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = _named_groups(dataframe, _TRAJECTORY_COLUMNS, "trajectory")
    return [(name, _representative_path(frame)) for name, frame in groups]


def _split_label(dataframe: pd.DataFrame) -> str | None:
    column = _column(dataframe, ("dataset_split", "split"))
    if column is None:
        return None
    values = dataframe[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) == 1 else None


def _plot_local_excitations(
    local_dataset: pd.DataFrame,
    destination: Path,
) -> Path | None:
    groups = _trajectory_groups(local_dataset)
    if not groups:
        return None
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    palette = plt.get_cmap("tab10")
    plotted = False
    for index, (name, dataframe) in enumerate(groups):
        phase = 100.0 * _phase(dataframe)
        split = _split_label(dataframe)
        label = name if split is None else f"{name} [{split}]"
        color = palette(index % 10)
        for axis, joint in zip(axes, ("hip", "knee")):
            angle = _angle_degrees(dataframe, joint)
            if angle is None:
                continue
            valid = _finite(phase, angle)
            if np.any(valid):
                axis.plot(phase[valid], angle[valid], color=color, label=label)
                plotted = True
    if not plotted:
        plt.close(figure)
        return None
    for axis, joint in zip(axes, ("hip", "knee")):
        axis.set_ylabel(f"{joint.title()} angle (deg)")
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            fontsize=7.5,
        )
    axes[-1].set_xlabel("Closed-reference cycle phase (%)")
    axes[0].set_title(
        "Reference-neighborhood identification excitations — software-retimed"
    )
    return _save(figure, destination, "local_excitation_trajectories.png")


def _coverage_table(domain_coverage: pd.DataFrame) -> pd.DataFrame:
    if domain_coverage.empty:
        return pd.DataFrame()
    name_column = _column(domain_coverage, _TRAJECTORY_COLUMNS)
    if name_column is None:
        names = pd.Series("reference", index=domain_coverage.index)
    else:
        names = domain_coverage[name_column].astype(str)
    percent = _numeric(
        domain_coverage,
        ("in_domain_percent", "domain_coverage_percent", "in_domain_percentage"),
    )
    if percent is not None:
        working = pd.DataFrame({"trajectory": names.to_numpy(), "percent": percent})
        return (
            working.groupby("trajectory", sort=False, as_index=False)["percent"]
            .mean()
            .sort_values("trajectory", key=lambda series: series.map(_natural_name_key))
        )
    membership = _boolean(
        domain_coverage,
        ("domain_membership_estimated", "inside_estimated_domain", "in_domain"),
    )
    if membership is None:
        inside = _numeric(domain_coverage, ("in_domain_sample_count",))
        outside = _numeric(domain_coverage, ("out_of_domain_sample_count",))
        if inside is None or outside is None:
            return pd.DataFrame()
        denominator = inside + outside
        percent = np.divide(
            100.0 * inside,
            denominator,
            out=np.full(len(inside), np.nan),
            where=denominator > 0.0,
        )
    else:
        percent = 100.0 * membership.astype(float)
    working = pd.DataFrame({"trajectory": names.to_numpy(), "percent": percent})
    return (
        working.groupby("trajectory", sort=False, as_index=False)["percent"]
        .mean()
        .sort_values("trajectory", key=lambda series: series.map(_natural_name_key))
    )


def _plot_local_domain(
    local_dataset: pd.DataFrame,
    domain_coverage: pd.DataFrame,
    destination: Path,
) -> Path | None:
    coverage = _coverage_table(domain_coverage)
    if coverage.empty:
        return None
    figure, (coverage_axis, state_axis) = plt.subplots(
        1,
        2,
        figsize=(13, max(5.5, 0.42 * len(coverage) + 2.5)),
        gridspec_kw={"width_ratios": (1.05, 1.4)},
    )
    names = coverage["trajectory"].astype(str).tolist()
    values = coverage["percent"].to_numpy(float)
    y = np.arange(len(names))
    colors = [
        _IN_DOMAIN_COLOR if np.isfinite(value) and value >= 90.0 else _OUT_DOMAIN_COLOR
        for value in values
    ]
    coverage_axis.barh(y, values, color=colors)
    coverage_axis.set_yticks(y, names)
    coverage_axis.set_xlim(0.0, 100.0)
    coverage_axis.set_xlabel("Estimated-state samples in local domain (%)")
    coverage_axis.set_title("Reference-local domain coverage")
    coverage_axis.axvline(90.0, color="#777777", linestyle="--", linewidth=1.0, label="Nominal target: 90%")
    coverage_axis.axvline(95.0, color="#444444", linestyle=":", linewidth=1.0, label="Slow target: 95%")
    coverage_axis.legend(fontsize=8)
    for row, value in enumerate(values):
        if np.isfinite(value):
            coverage_axis.text(min(value + 1.0, 98.0), row, f"{value:.1f}%", va="center", fontsize=8)

    hip = _angle_degrees(local_dataset, "hip")
    knee = _angle_degrees(local_dataset, "knee")
    split_column = _column(local_dataset, ("dataset_split", "split"))
    if hip is not None and knee is not None:
        valid = _finite(hip, knee)
        if split_column is None:
            state_axis.scatter(hip[valid], knee[valid], s=10, alpha=0.35, color=_HIP_COLOR, label="Estimated-state samples")
        else:
            splits = local_dataset[split_column].astype(str).to_numpy()
            markers = {"train": "o", "validation": "^", "test": "x"}
            split_colors = {"train": _HIP_COLOR, "validation": _KNEE_COLOR, "test": _OUT_DOMAIN_COLOR}
            for split in ("train", "validation", "test"):
                selected = valid & (np.char.lower(splits.astype(str)) == split)
                if np.any(selected):
                    state_axis.scatter(
                        hip[selected],
                        knee[selected],
                        s=12,
                        alpha=0.45,
                        marker=markers[split],
                        color=split_colors[split],
                        label=f"{split} estimated states",
                    )
        state_axis.set_xlabel("Estimated hip angle (deg)")
        state_axis.set_ylabel("Estimated knee angle (deg)")
        state_axis.set_title("State support used for local-domain audit")
        state_axis.legend(fontsize=8)
    else:
        state_axis.axis("off")
        state_axis.text(
            0.5,
            0.5,
            "Per-sample estimated joint states unavailable;\ncoverage bars use supplied domain audit only.",
            transform=state_axis.transAxes,
            ha="center",
            va="center",
        )
    figure.suptitle("Reference-neighborhood identification domain — estimated states only")
    return _save(figure, destination, "local_identification_domain.png")


def _candidate_path_source(
    candidate_trajectories: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    local_dataset: pd.DataFrame,
) -> pd.DataFrame:
    for dataframe in (candidate_trajectories, candidate_metrics, local_dataset):
        if dataframe.empty:
            continue
        if (
            _column(dataframe, _CANDIDATE_COLUMNS) is not None
            and _angle_degrees(dataframe, "hip") is not None
            and _angle_degrees(dataframe, "knee") is not None
        ):
            return dataframe
    return pd.DataFrame()


def _plot_candidate_paths(
    candidate_paths: pd.DataFrame,
    destination: Path,
) -> Path | None:
    groups = _named_groups(candidate_paths, _CANDIDATE_COLUMNS, "candidate")
    groups = [(name, _representative_path(frame)) for name, frame in groups if "fast" not in name.lower()]
    if not groups:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    palette = plt.get_cmap("tab10")
    plotted = False
    for index, (name, dataframe) in enumerate(groups):
        phase = 100.0 * _phase(dataframe)
        hip = _angle_degrees(dataframe, "hip")
        knee = _angle_degrees(dataframe, "knee")
        if hip is None or knee is None:
            continue
        color = _HIP_COLOR if name.strip().upper() == "C0" else palette(index % 10)
        linewidth = 2.2 if name.strip().upper() == "C0" else 1.1
        valid = _finite(phase, hip, knee)
        if not np.any(valid):
            continue
        axes[0].plot(phase[valid], hip[valid], color=color, linewidth=linewidth, label=name)
        axes[1].plot(phase[valid], knee[valid], color=color, linewidth=linewidth, label=name)
        axes[2].plot(hip[valid], knee[valid], color=color, linewidth=linewidth, label=name)
        plotted = True
    if not plotted:
        plt.close(figure)
        return None
    axes[0].set_xlabel("Cycle phase (%)")
    axes[0].set_ylabel("Hip angle (deg)")
    axes[0].set_title("Candidate hip paths")
    axes[1].set_xlabel("Cycle phase (%)")
    axes[1].set_ylabel("Knee angle (deg)")
    axes[1].set_title("Candidate knee paths")
    axes[2].set_xlabel("Hip angle (deg)")
    axes[2].set_ylabel("Knee angle (deg)")
    axes[2].set_title("Candidate joint-space paths")
    handles, labels = axes[2].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[2].legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    figure.suptitle("Stage 5C candidate geometry — fast stress test excluded from ranking")
    return _save(figure, destination, "candidate_joint_paths.png")


def _summary_by_candidate(
    dataframe: pd.DataFrame,
    metrics: Sequence[str],
    *,
    baseline_only: bool,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()
    candidate_column = _column(dataframe, _CANDIDATE_COLUMNS)
    if candidate_column is None:
        return pd.DataFrame()
    working = dataframe.copy()
    subject_column = _column(working, _SUBJECT_COLUMNS)
    if baseline_only and subject_column is not None:
        baseline = working.loc[working[subject_column].astype(str).str.lower().eq("baseline")]
        if not baseline.empty:
            working = baseline
    selected = [column for column in metrics if column in working]
    if not selected:
        return pd.DataFrame()
    for column in selected:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    aggregation = {column: "max" for column in selected}
    result = working.groupby(candidate_column, sort=False, as_index=False).agg(aggregation)
    return result.sort_values(candidate_column, key=lambda series: series.map(_natural_name_key))


def _feasible_lookup(dataframe: pd.DataFrame) -> dict[str, bool]:
    candidate_column = _column(dataframe, _CANDIDATE_COLUMNS)
    flags = _boolean(dataframe, ("candidate_feasible", "feasible"))
    if candidate_column is None or flags is None:
        return {}
    return {
        str(name): bool(np.all(flags[np.asarray(dataframe[candidate_column].astype(str) == str(name))]))
        for name in dataframe[candidate_column].astype(str).unique()
    }


def _plot_candidate_torque(
    candidate_metrics: pd.DataFrame,
    destination: Path,
) -> Path | None:
    metrics = (
        "peak_abs_tau_hip_nm",
        "peak_abs_tau_knee_nm",
        "rms_combined_torque_nm",
    )
    summary = _summary_by_candidate(candidate_metrics, metrics, baseline_only=True)
    if summary.empty or not all(column in summary for column in metrics):
        return None
    candidate_column = _column(summary, _CANDIDATE_COLUMNS)
    assert candidate_column is not None
    names = summary[candidate_column].astype(str).tolist()
    feasibility = _feasible_lookup(candidate_metrics)
    colors = [_PARETO_COLOR if feasibility.get(name, True) else _OUT_DOMAIN_COLOR for name in names]
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = (
        "Peak |hip torque| (N·m)",
        "Peak |knee torque| (N·m)",
        "RMS combined torque (N·m)",
    )
    x = np.arange(len(names))
    for axis, metric, label in zip(axes, metrics, labels):
        values = summary[metric].to_numpy(float)
        axis.bar(x, values, color=colors)
        axis.set_ylabel(label)
    axes[-1].set_xticks(x, names, rotation=35, ha="right")
    axes[0].set_title(
        "Worst-case candidate torque across four virtual subjects"
    )
    axes[0].legend(
        handles=(
            Line2D([], [], marker="s", linestyle="None", color=_PARETO_COLOR, label="Feasible or not flagged"),
            Line2D([], [], marker="s", linestyle="None", color=_OUT_DOMAIN_COLOR, label="Rejected by hard constraint"),
        ),
        fontsize=8,
    )
    figure.text(
        0.5,
        0.005,
        "Software-only virtual dynamics; endpoint-force values are not real-robot safety thresholds.",
        ha="center",
        fontsize=8,
    )
    return _save(figure, destination, "candidate_torque_comparison.png")


def _metric_matrix(
    dataframe: pd.DataFrame,
    metric: str,
) -> tuple[list[str], list[str], np.ndarray] | None:
    candidate_column = _column(dataframe, _CANDIDATE_COLUMNS)
    subject_column = _column(dataframe, _SUBJECT_COLUMNS)
    if candidate_column is None or subject_column is None or metric not in dataframe:
        return None
    working = dataframe[[candidate_column, subject_column, metric]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    pivot = working.pivot_table(
        index=candidate_column,
        columns=subject_column,
        values=metric,
        aggfunc="max",
        sort=False,
    )
    if pivot.empty:
        return None
    pivot = pivot.loc[sorted(pivot.index, key=_natural_name_key)]
    preferred_subjects = ["baseline", "hip_stiff", "knee_stiff", "heavy_leg"]
    columns = [name for name in preferred_subjects if name in pivot.columns]
    columns += [name for name in pivot.columns if name not in columns]
    pivot = pivot[columns]
    return (
        [str(value) for value in pivot.index],
        [str(value) for value in pivot.columns],
        pivot.to_numpy(float),
    )


def _plot_candidate_subjects(
    comparison: pd.DataFrame,
    destination: Path,
) -> Path | None:
    specifications = (
        ("peak_abs_tau_hip_nm", "Peak |hip torque| (N·m)"),
        ("peak_abs_tau_knee_nm", "Peak |knee torque| (N·m)"),
        ("rms_combined_torque_nm", "RMS combined torque (N·m)"),
    )
    matrices = [(metric, title, _metric_matrix(comparison, metric)) for metric, title in specifications]
    matrices = [(metric, title, matrix) for metric, title, matrix in matrices if matrix is not None]
    if not matrices:
        return None
    figure, axes = plt.subplots(1, len(matrices), figsize=(5.3 * len(matrices), 7.2), squeeze=False)
    for axis, (_, title, matrix_data) in zip(axes[0], matrices):
        assert matrix_data is not None
        candidates, subjects, values = matrix_data
        image = axis.imshow(values, aspect="auto", cmap="viridis")
        axis.set_xticks(np.arange(len(subjects)), subjects, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(candidates)), candidates)
        axis.set_title(title)
        finite = values[np.isfinite(values)]
        threshold = float(np.nanmedian(finite)) if finite.size else np.nan
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value >= threshold else "black",
                        fontsize=7,
                    )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Candidate comparison across virtual subjects — software only")
    return _save(figure, destination, "candidate_subject_comparison.png")


def _pareto_flag(dataframe: pd.DataFrame) -> np.ndarray:
    values = _boolean(
        dataframe,
        ("pareto_optimal", "pareto_front", "on_pareto_front", "is_pareto"),
    )
    return np.zeros(len(dataframe), dtype=bool) if values is None else values


def _plot_candidate_pareto(
    pareto: pd.DataFrame,
    destination: Path,
) -> Path | None:
    candidate_column = _column(pareto, _CANDIDATE_COLUMNS)
    required = (
        "peak_abs_tau_hip_nm",
        "peak_abs_tau_knee_nm",
        "rms_combined_torque_nm",
        "joint_jerk_cost",
    )
    if candidate_column is None or not all(column in pareto for column in required):
        return None
    duration_column = _column(
        pareto,
        ("total_duration_s", "duration_s", "trajectory_duration_s"),
    )
    rate_column = _column(pareto, ("peak_torque_rate_nm_s",))
    working_columns = [candidate_column, *required]
    if duration_column is not None:
        working_columns.append(duration_column)
    if rate_column is not None:
        working_columns.append(rate_column)
    flag_column = _column(pareto, ("pareto_optimal", "pareto_front", "on_pareto_front", "is_pareto"))
    feasible_column = _column(pareto, ("candidate_feasible", "feasible"))
    if flag_column is not None:
        working_columns.append(flag_column)
    if feasible_column is not None and feasible_column not in working_columns:
        working_columns.append(feasible_column)
    working = pareto[working_columns].drop_duplicates(candidate_column, keep="first").copy()
    for column in required:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    pareto_flag = _pareto_flag(working)
    feasible = _boolean(working, ("candidate_feasible", "feasible"))
    if feasible is None:
        feasible = np.ones(len(working), dtype=bool)
    names = working[candidate_column].astype(str).to_numpy()
    colors = np.where(
        ~feasible,
        _OUT_DOMAIN_COLOR,
        np.where(pareto_flag, _PARETO_COLOR, _NON_PARETO_COLOR),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    panel_specs: list[tuple[plt.Axes, str, str, str, str]] = [
        (axes[0, 0], "peak_abs_tau_hip_nm", "peak_abs_tau_knee_nm", "Peak |hip torque| (N·m)", "Peak |knee torque| (N·m)"),
        (axes[0, 1], "rms_combined_torque_nm", "joint_jerk_cost", "RMS combined torque (N·m)", "Joint jerk cost (rad²/s⁵, discrete integral)"),
    ]
    if duration_column is not None:
        working[duration_column] = pd.to_numeric(working[duration_column], errors="coerce")
        panel_specs.append((axes[1, 0], duration_column, "rms_combined_torque_nm", "Total duration (s)", "RMS combined torque (N·m)"))
    if rate_column is not None:
        working[rate_column] = pd.to_numeric(working[rate_column], errors="coerce")
        panel_specs.append((axes[1, 1], rate_column, "joint_jerk_cost", "Peak torque rate (N·m/s)", "Joint jerk cost (rad²/s⁵, discrete integral)"))

    used_axes: set[plt.Axes] = set()
    for axis, x_column, y_column, xlabel, ylabel in panel_specs:
        used_axes.add(axis)
        x = working[x_column].to_numpy(float)
        y = working[y_column].to_numpy(float)
        valid = _finite(x, y)
        for index in np.flatnonzero(valid):
            marker = "o" if feasible[index] else "x"
            axis.scatter(x[index], y[index], color=colors[index], marker=marker, s=52)
            axis.annotate(names[index], (x[index], y[index]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    for axis in axes.flat:
        if axis not in used_axes:
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "Requested Pareto dimension unavailable;\nno value was synthesized.",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    axes[0, 0].legend(
        handles=(
            Line2D([], [], marker="o", linestyle="None", color=_PARETO_COLOR, label="Pareto front / feasible"),
            Line2D([], [], marker="o", linestyle="None", color=_NON_PARETO_COLOR, label="Feasible, not Pareto"),
            Line2D([], [], marker="x", linestyle="None", color=_OUT_DOMAIN_COLOR, label="Rejected by hard constraint"),
        ),
        fontsize=8,
    )
    figure.suptitle(
        "Pareto-style candidate comparison — no comfort score or arbitrary weighting"
    )
    return _save(figure, destination, "candidate_pareto.png")


def generate_reference_candidate_visualizations(
    execution_versions: pd.DataFrame,
    local_dataset: pd.DataFrame | None,
    domain_coverage: pd.DataFrame | None,
    candidate_metrics: pd.DataFrame | None,
    candidate_subject_comparison: pd.DataFrame | None,
    pareto: pd.DataFrame | None,
    metadata: Mapping[str, Any] | None,
    output_dir: str | Path,
    *,
    candidate_trajectories: pd.DataFrame | None = None,
) -> ReferenceCandidateVisualizationResult:
    """Generate the eight Stage 5C figures without bypassing the ROM gate.

    ``execution_versions`` contains the measured-asymmetric and
    closed-symmetric geometry and is the only input plotted without explicit
    ROM approval.  All remaining figures require metadata that records both a
    positive approval gate and a finite approved knee minimum/maximum.

    ``candidate_trajectories`` is optional for API compatibility.  When it is
    omitted, a long-form candidate path may instead be supplied through
    ``candidate_metrics``; summary-only metrics cannot be turned into a path
    figure and therefore cause an explicit skip.
    """

    _configure_style()
    execution = _as_dataframe(execution_versions, "execution_versions")
    local = _as_dataframe(local_dataset, "local_dataset")
    domain = _as_dataframe(domain_coverage, "domain_coverage")
    metrics = _as_dataframe(candidate_metrics, "candidate_metrics")
    comparison = _as_dataframe(
        candidate_subject_comparison,
        "candidate_subject_comparison",
    )
    pareto_frame = _as_dataframe(pareto, "pareto")
    paths_frame = _as_dataframe(candidate_trajectories, "candidate_trajectories")
    metadata_map: Mapping[str, Any] = metadata if metadata is not None else {}
    if not isinstance(metadata_map, Mapping):
        raise TypeError("metadata must be a mapping or None.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    skipped: dict[str, str] = {}

    audit_producers = (
        (
            "measured_vs_closed_reference.png",
            lambda: _plot_measured_vs_closed(execution, destination),
            "execution versions lack measured/closed finite joint paths",
        ),
        (
            "reference_closure_comparison.png",
            lambda: _plot_closure_comparison(execution, metadata_map, destination),
            "execution versions lack reconstructable measured/closed pull paths",
        ),
    )
    for filename, producer, reason in audit_producers:
        path = producer()
        if path is None:
            skipped[filename] = reason
        else:
            paths[path.name] = path

    approved, gate_reason = _explicit_rom_gate(metadata_map)
    if not approved:
        for filename in FORMAL_EVALUATION_FIGURES:
            skipped[filename] = gate_reason
    else:
        candidate_paths = _candidate_path_source(paths_frame, metrics, local)
        formal_producers = (
            (
                "local_excitation_trajectories.png",
                lambda: _plot_local_excitations(local, destination),
                "reference-local dataset lacks finite phase joint paths",
            ),
            (
                "local_identification_domain.png",
                lambda: _plot_local_domain(local, domain, destination),
                "domain audit lacks estimated-state coverage data",
            ),
            (
                "candidate_joint_paths.png",
                lambda: _plot_candidate_paths(candidate_paths, destination),
                "candidate summary lacks per-sample phase and joint paths; no path was reconstructed from summary metrics",
            ),
            (
                "candidate_torque_comparison.png",
                lambda: _plot_candidate_torque(metrics, destination),
                "candidate metrics lack baseline peak hip/knee and RMS torque fields",
            ),
            (
                "candidate_subject_comparison.png",
                lambda: _plot_candidate_subjects(comparison, destination),
                "candidate subject comparison lacks subject-resolved torque metrics",
            ),
            (
                "candidate_pareto.png",
                lambda: _plot_candidate_pareto(pareto_frame, destination),
                "Pareto table lacks candidate torque, RMS, and smoothness dimensions",
            ),
        )
        for filename, producer, reason in formal_producers:
            path = producer()
            if path is None:
                skipped[filename] = reason
            else:
                paths[path.name] = path

    for filename in FIGURE_FILENAMES:
        if filename not in paths and filename not in skipped:
            skipped[filename] = "no visualization producer accounted for this output"
    return ReferenceCandidateVisualizationResult(paths=paths, skipped=skipped)


__all__ = [
    "FIGURE_FILENAMES",
    "FORMAL_EVALUATION_FIGURES",
    "GEOMETRY_AUDIT_FIGURES",
    "ReferenceCandidateVisualizationResult",
    "generate_reference_candidate_visualizations",
]
