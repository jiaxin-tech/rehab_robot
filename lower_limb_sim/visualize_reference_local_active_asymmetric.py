"""Scientific figures for active-asymmetric reference-local identification."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Mapping


_MPL_CONFIG_DIRECTORY = Path(tempfile.gettempdir()) / "lower_limb_sim_matplotlib"
_MPL_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIRECTORY))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reference_local_active_asymmetric import ActiveReferenceLocalResult


FIGURE_DEFINITIONS: dict[str, dict[str, object]] = {
    "active_reference_local_excitation_family.png": {
        "scientific_question": (
            "Do the conservative amplitude, phase, and duration perturbations "
            "remain local to the active asymmetric joint/task path?"
        ),
        "source_data": ["excitation_metadata.csv", "trajectory_*.csv"],
        "manuscript_interpretation": (
            "Software trajectory construction around the unchanged active "
            "asymmetric reference; no physical execution claim."
        ),
    },
    "active_reference_local_domain_coverage.png": {
        "scientific_question": (
            "How much of each validation/test trajectory lies inside the "
            "axis-aligned six-state q/dq/ddq training box?"
        ),
        "source_data": ["domain_coverage.csv", "state_domain_bounds.json"],
        "manuscript_interpretation": (
            "Coverage is a train-fitted support diagnostic, not a confidence "
            "region or physical safety guarantee."
        ),
    },
    "active_reference_local_identifiability.png": {
        "scientific_question": (
            "Does the active-reference-local training set provide full-rank, "
            "well-scaled local numerical sensitivity for five parameters?"
        ),
        "source_data": [
            "identifiability_summary.csv",
            "sensitivity_singular_values.csv",
        ],
        "manuscript_interpretation": (
            "Local numerical identifiability of the adopted equivalent model, "
            "not unique physiological identification."
        ),
    },
    "active_reference_heldout_torque_prediction.png": {
        "scientific_question": (
            "How do generic and train-only identified models predict torque "
            "along the exact held-out active slow reference?"
        ),
        "source_data": ["held_out_predictions.csv"],
        "manuscript_interpretation": (
            "Matched clean offline virtual-subject prediction; parameter "
            "recovery is reported separately."
        ),
    },
    "active_reference_generic_vs_identified.png": {
        "scientific_question": (
            "Does subject-specific equivalent fitting reduce held-out torque "
            "error relative to the frozen generic parameter vector?"
        ),
        "source_data": ["generic_vs_identified.csv"],
        "manuscript_interpretation": (
            "Relative software prediction result under matched clean dynamics, "
            "not robot, human, comfort, or clinical validation."
        ),
    },
}


def _style() -> None:
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


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _excitation_figure(result: ActiveReferenceLocalResult, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    colors = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
    labels_seen: set[str] = set()
    for trajectory_id, trajectory in result.trajectories.items():
        split = str(trajectory["dataset_split"].iloc[0])
        exact_active = trajectory_id == "heldout_active_reference_slow"
        color = "#222222" if exact_active else colors[split]
        width = 2.4 if exact_active else 0.9
        alpha = 1.0 if exact_active else 0.58
        label = "exact active slow (held out)" if exact_active else split
        plot_label = label if label not in labels_seen else None
        labels_seen.add(label)
        phase = trajectory["global_phase"].to_numpy(dtype=float)
        axes[0].plot(
            phase,
            np.rad2deg(trajectory["q_hip_rad"]),
            color=color,
            lw=width,
            alpha=alpha,
            label=plot_label,
        )
        axes[1].plot(
            phase,
            np.rad2deg(trajectory["q_knee_rad"]),
            color=color,
            lw=width,
            alpha=alpha,
        )
        axes[2].plot(
            trajectory["x_pull_m"],
            trajectory["z_pull_m"],
            color=color,
            lw=width,
            alpha=alpha,
        )
    axes[0].set(xlabel="Cycle phase", ylabel="Hip flexion (deg)")
    axes[1].set(xlabel="Cycle phase", ylabel="Knee flexion (deg)")
    axes[2].set(
        xlabel="Traction-point x (m)",
        ylabel="Traction-point z (m)",
        aspect="equal",
    )
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Active asymmetric reference and reference-local excitation set")
    return _save(fig, path)


def _coverage_figure(result: ActiveReferenceLocalResult, path: Path) -> Path:
    coverage = result.domain_coverage.copy(deep=False)
    colors = coverage["dataset_split"].map(
        {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    positions = np.arange(len(coverage))
    axes[0].barh(positions, coverage["in_domain_percent"], color=colors)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(coverage["trajectory_id"], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Samples inside train-fitted 6-D box (%)", xlim=(0, 101))
    for y, value in zip(positions, coverage["in_domain_percent"]):
        axes[0].text(min(float(value) + 0.5, 99.2), y, f"{value:.1f}", va="center", fontsize=7)

    train = result.dataset.loc[result.dataset["dataset_split"].eq("train")]
    axes[1].scatter(
        np.rad2deg(train["q_hip_rad"]),
        np.rad2deg(train["q_knee_rad"]),
        s=4,
        alpha=0.08,
        color="#4C78A8",
        label="training states",
    )
    for trajectory_id in (
        "heldout_active_reference_slow",
        "heldout_boundary_speed_plus_10pct",
    ):
        trajectory = result.trajectories[trajectory_id]
        axes[1].plot(
            np.rad2deg(trajectory["q_hip_rad"]),
            np.rad2deg(trajectory["q_knee_rad"]),
            lw=1.5,
            label=trajectory_id.replace("heldout_", ""),
        )
    axes[1].set(
        xlabel="Hip flexion (deg)",
        ylabel="Knee flexion (deg)",
        title="Joint-space projection (coverage uses q, dq, ddq)",
    )
    axes[1].legend(fontsize=7)
    fig.suptitle("Active-reference-local state-domain coverage")
    return _save(fig, path)


def _identifiability_figure(
    result: ActiveReferenceLocalResult,
    path: Path,
) -> Path:
    singular = result.singular_values
    summary = result.identifiability_summary
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for subject_id, group in singular.groupby("subject_id", sort=False):
        axes[0].plot(
            group["singular_value_index"],
            group["singular_value"],
            marker="o",
            label=str(subject_id),
        )
    axes[0].set(
        xlabel="Singular-value index",
        ylabel="Scaled sensitivity singular value",
        yscale="log",
        xticks=range(1, 6),
    )
    axes[0].legend(fontsize=7)
    positions = np.arange(len(summary))
    axes[1].bar(positions, summary["condition_number"], color="#4C78A8")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(summary["subject_id"], rotation=25, ha="right")
    axes[1].set(ylabel="Condition number", title="Rank = 5 for every subject")
    fig.suptitle("Five-parameter local numerical identifiability")
    return _save(fig, path)


def _prediction_figure(result: ActiveReferenceLocalResult, path: Path) -> Path:
    data = result.prediction_samples.loc[
        result.prediction_samples["subject_id"].eq("baseline")
        & result.prediction_samples["trajectory_id"].eq(
            "heldout_active_reference_slow"
        )
    ]
    if data.empty:
        raise ValueError("baseline exact-active held-out predictions are missing.")
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.7), sharex=True)
    for axis, joint in zip(axes, ("hip", "knee")):
        axis.plot(
            data["time_s"],
            data[f"tau_true_{joint}_nm"],
            color="#222222",
            lw=2.0,
            label="synthetic observed",
        )
        axis.plot(
            data["time_s"],
            data[f"tau_generic_{joint}_nm"],
            color="#E45756",
            lw=1.0,
            label="generic",
        )
        axis.plot(
            data["time_s"],
            data[f"tau_identified_{joint}_nm"],
            color="#4C78A8",
            lw=1.1,
            linestyle="--",
            label="identified (train only)",
        )
        axis.set_ylabel(f"{joint.capitalize()} torque (N m)")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_xlabel("Time (s)")
    fig.suptitle("Exact active slow reference held out from parameter fitting")
    return _save(fig, path)


def _comparison_figure(result: ActiveReferenceLocalResult, path: Path) -> Path:
    comparison = result.generic_vs_identified.loc[
        result.generic_vs_identified["split"].eq("test")
    ].copy()
    aggregate = (
        comparison.groupby("trajectory_id", sort=False)
        .agg(
            generic_rmse=("generic_rmse", "mean"),
            identified_rmse=("identified_rmse", "mean"),
        )
        .reset_index()
    )
    positions = np.arange(len(aggregate))
    width = 0.38
    fig, axis = plt.subplots(figsize=(9.5, 4.2))
    axis.bar(
        positions - width / 2,
        aggregate["generic_rmse"],
        width,
        label="generic",
        color="#E45756",
    )
    axis.bar(
        positions + width / 2,
        aggregate["identified_rmse"],
        width,
        label="identified",
        color="#4C78A8",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        aggregate["trajectory_id"].str.replace("heldout_", "", regex=False),
        rotation=25,
        ha="right",
        fontsize=8,
    )
    axis.set_ylabel("Mean combined torque RMSE (N m, log scale)")
    axis.legend()
    axis.set_title("Held-out generic versus identified prediction")
    return _save(fig, path)


def generate_active_reference_local_figures(
    result: ActiveReferenceLocalResult,
    output_directory: str | Path,
) -> Mapping[str, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    _style()
    generators = {
        "active_reference_local_excitation_family.png": _excitation_figure,
        "active_reference_local_domain_coverage.png": _coverage_figure,
        "active_reference_local_identifiability.png": _identifiability_figure,
        "active_reference_heldout_torque_prediction.png": _prediction_figure,
        "active_reference_generic_vs_identified.png": _comparison_figure,
    }
    return {
        filename: generator(result, destination / filename)
        for filename, generator in generators.items()
    }


__all__ = [
    "FIGURE_DEFINITIONS",
    "generate_active_reference_local_figures",
]
