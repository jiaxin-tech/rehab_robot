"""Deterministically generate figures and GIFs for the finite method audit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Sequence

_MPL_CACHE = Path(tempfile.gettempdir()) / "rehab_robot_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image

from .formal_protocol import ACTIVE_REFERENCE_PATH


FIGURE_NAMES = (
    "01_method_overview.png",
    "02_subject_specific_landscape_example.png",
    "03_frozen_shortlist_on_landscape.png",
    "04_sequential_validation_progress.png",
    "05_reference_vs_validated_J.png",
    "06_predicted_vs_truth_optimum.png",
    "07_subject_specific_alpha.png",
    "08_budget_sensitivity.png",
    "09_p2_vs_finite_method_trial_cost.png",
)
GIF_NAMES = (
    "FINAL_METHOD_WORKFLOW_ANIMATION.gif",
    "SUBJECT_SPECIFIC_LANDSCAPE_COMPARISON.gif",
)

_BLUE = "#276FBF"
_ORANGE = "#E07A2D"
_GREEN = "#2A9D6F"
_RED = "#C44536"
_GRAY = "#7C8797"
_LIGHT = "#F4F7FA"
_DARK = "#1D2A39"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#AAB4C0",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E4E9EF",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _flow_axis(figure: plt.Figure, title: str) -> plt.Axes:
    axis = figure.add_axes((0.04, 0.08, 0.92, 0.84))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    axis.set_title(title, fontsize=19, weight="bold", color=_DARK, pad=12)
    return axis


def _node(axis: plt.Axes, x: float, y: float, text: str, color: str = _BLUE) -> None:
    box = patches.FancyBboxPatch(
        (x - 0.075, y - 0.055),
        0.15,
        0.11,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="white",
        edgecolor=color,
        linewidth=2.0,
    )
    axis.add_patch(box)
    axis.text(x, y, text, ha="center", va="center", fontsize=10, color=_DARK)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": _GRAY, "lw": 1.8},
    )


def _pivot_slice(frame: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot_table(
        index="phase_delta", columns="hip_delta", values=value, aggfunc="first"
    ).sort_index().sort_index(axis=1)
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def _heatmap(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    shortlist: pd.DataFrame | None = None,
) -> None:
    hip, phase, values = _pivot_slice(frame, value)
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=(hip.min(), hip.max(), phase.min(), phase.max()),
        cmap="viridis_r",
    )
    if shortlist is not None and not shortlist.empty:
        axis.scatter(
            shortlist["hip_delta"],
            shortlist["phase_delta"],
            s=90,
            marker="o",
            facecolors="white",
            edgecolors=_RED,
            linewidths=2,
            zorder=4,
        )
        for row in shortlist.itertuples(index=False):
            axis.text(
                float(row.hip_delta),
                float(row.phase_delta),
                str(row.candidate_id),
                ha="center",
                va="center",
                fontsize=8,
                weight="bold",
                color=_RED,
                zorder=5,
            )
        axis.text(
            0.02,
            0.02,
            "C1–C3 markers are hip–phase projections",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=_DARK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    axis.set_title(title)
    axis.set_xlabel("Hip amplitude delta (deg)")
    axis.set_ylabel("Knee phase shift")
    plt.colorbar(image, ax=axis, label=value.replace("_", " "))


def generate_static_figures(artifact_directory: str | Path) -> tuple[Path, ...]:
    directory = Path(artifact_directory)
    _style()
    shortlist = pd.read_csv(directory / "candidate_shortlist_manifest.csv")
    execution = pd.read_csv(directory / "candidate_execution_history.csv")
    subjects = pd.read_csv(directory / "final_subject_summary.csv")
    optimum = pd.read_csv(directory / "predicted_vs_truth_optimum.csv")
    budget = pd.read_csv(directory / "budget_sensitivity.csv")
    comparison = pd.read_csv(directory / "p2_vs_finite_method_comparison.csv")
    slices = pd.read_csv(directory / "visualization_landscape_slice.csv")

    figure = plt.figure(figsize=(14, 7.875))
    axis = _flow_axis(figure, "Model-screened finite sequential validation")
    labels = (
        "Measured\nreference",
        "Initial ID\n$\\hat\\theta_0$",
        "Full $J_{pred}$\nlandscape",
        "Frozen\nC1–C3",
        "Validate → refit\n→ rerank",
        "Best validated\ntrajectory",
    )
    xs = np.linspace(0.09, 0.91, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        _node(axis, float(x), 0.56, label, _GREEN if index == len(labels) - 1 else _BLUE)
        if index:
            _arrow(axis, (float(xs[index - 1]) + 0.08, 0.56), (float(x) - 0.08, 0.56))
    axis.text(
        0.5,
        0.27,
        "Large virtual search, small physical search  •  shortlist frozen before validation  •  budget ≤ 3",
        ha="center",
        fontsize=13,
        color=_DARK,
    )
    _save(figure, directory / FIGURE_NAMES[0])

    example_id = str(slices["case_id"].iloc[0])
    example = slices.loc[slices["case_id"].eq(example_id)]
    example_shortlist = shortlist.loc[shortlist["case_id"].eq(example_id)]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    _heatmap(
        axes[0],
        example,
        value="J_pred",
        title="Initial model prediction (minimum over knee)",
    )
    _heatmap(
        axes[1],
        example,
        value="J_truth",
        title="Post-freeze virtual truth (minimum over knee)",
    )
    figure.suptitle(f"Subject-specific whole-trajectory landscape: {example_id}", fontsize=16)
    _save(figure, directory / FIGURE_NAMES[1])

    figure, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    _heatmap(
        axis,
        example,
        value="J_pred",
        title=f"Frozen shortlist on hip–phase projection ({example_id})",
        shortlist=example_shortlist,
    )
    _save(figure, directory / FIGURE_NAMES[2])

    example_history = execution.loc[execution["case_id"].eq(example_id)].sort_values("round")
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    rounds = np.r_[0, example_history["round"].to_numpy(dtype=int)]
    best = np.r_[1.0, example_history["best_validated_J_after"].to_numpy(dtype=float)]
    axis.step(rounds, best, where="post", color=_GREEN, lw=2.5, label="Best validated J")
    axis.scatter(
        example_history["round"], example_history["actual_J"], color=_ORANGE, s=55, label="Candidate truth J"
    )
    axis.plot(
        example_history["round"], example_history["J_pred_before_execution"],
        "o--", color=_BLUE, label="Prediction before execution"
    )
    axis.axhline(1.0, color=_GRAY, ls=":", label="Reference J = 1")
    axis.set(xlabel="Complete candidate validation round", ylabel="Whole-trajectory mechanical J", title=f"Finite validation progress: {example_id}")
    axis.set_xticks(range(0, 4))
    axis.legend(frameon=False)
    _save(figure, directory / FIGURE_NAMES[3])

    ordered = subjects.sort_values("case_id").reset_index(drop=True)
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    axis.bar(x - 0.18, np.ones(len(x)), width=0.36, color="#CBD4DE", label="Reference")
    axis.bar(x + 0.18, ordered["B2_final_best_validated_J"], width=0.36, color=_GREEN, label="Finite B2")
    axis.set_xticks(x, [str(value).replace("__", "\n") for value in ordered["case_id"]], rotation=60, ha="right")
    axis.set_ylabel("Whole-trajectory mechanical J")
    axis.set_title("Reference versus best validated trajectory")
    axis.legend(frameon=False)
    _save(figure, directory / FIGURE_NAMES[4])

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    axis.scatter(optimum["initial_predicted_optimum_J_pred"], optimum["truth_J_at_initial_predicted_optimum"], c=np.where(optimum["case_class"].eq("MATCHED"), _BLUE, _ORANGE), s=60)
    lo = min(optimum["initial_predicted_optimum_J_pred"].min(), optimum["truth_J_at_initial_predicted_optimum"].min())
    hi = max(optimum["initial_predicted_optimum_J_pred"].max(), optimum["truth_J_at_initial_predicted_optimum"].max())
    axis.plot([lo, hi], [lo, hi], color=_GRAY, ls="--")
    axis.set(xlabel="Initial predicted J", ylabel="Post-freeze truth J", title="Predicted versus truth objective at screened optimum")
    _save(figure, directory / FIGURE_NAMES[5])

    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for axis, column, label in zip(
        axes,
        ("best_validated_alpha_hip", "best_validated_alpha_knee", "best_validated_alpha_phase"),
        ("Hip delta (deg)", "Knee delta (deg)", "Phase shift"),
    ):
        axis.scatter(x, ordered[column], color=_GREEN, s=45)
        axis.set_ylabel(label)
    axes[-1].set_xticks(x, [str(value).replace("__", "\n") for value in ordered["case_id"]], rotation=60, ha="right")
    axes[0].set_title("Subject-specific best validated alpha")
    _save(figure, directory / FIGURE_NAMES[6])

    aggregate_budget = budget.groupby("budget", as_index=False).agg(mean_final_J=("final_best_validated_J", "mean"), mean_regret=("global_regret", "mean"))
    figure, axis = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    axis.plot(aggregate_budget["budget"], aggregate_budget["mean_final_J"], "o-", lw=2.4, color=_GREEN, label="Mean final J")
    axis.plot(aggregate_budget["budget"], aggregate_budget["mean_regret"], "s--", lw=2.0, color=_ORANGE, label="Mean regret")
    axis.set(xlabel="Maximum complete candidate validations", ylabel="Mean metric", title="Budget sensitivity on one frozen shortlist")
    axis.set_xticks([0, 1, 2, 3])
    axis.legend(frameon=False)
    _save(figure, directory / FIGURE_NAMES[7])

    figure, axis = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    axis.bar(comparison["method_id"], comparison["trial_count"], color=[_GRAY, _ORANGE, _RED, _BLUE, _GREEN, "#7B61A8"][: len(comparison)])
    axis.set_ylabel("Candidate/personalization trials across 15 cases")
    axis.set_title("Old P2 versus finite candidate-validation cost")
    axis.tick_params(axis="x", rotation=35)
    _save(figure, directory / FIGURE_NAMES[8])
    return tuple(directory / name for name in FIGURE_NAMES)


def _canvas_image(figure: plt.Figure) -> Image.Image:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    image = Image.fromarray(rgb)
    plt.close(figure)
    return image


def _new_frame(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    axis = _flow_axis(figure, title)
    if subtitle:
        axis.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=12, color=_GRAY)
    return figure, axis


def generate_workflow_animation(
    artifact_directory: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    directory = Path(artifact_directory)
    output = Path(output_path) if output_path is not None else directory / GIF_NAMES[0]
    _style()
    reference = pd.read_csv(ACTIVE_REFERENCE_PATH)
    shortlist = pd.read_csv(directory / "candidate_shortlist_manifest.csv")
    execution = pd.read_csv(directory / "candidate_execution_history.csv")
    updates = pd.read_csv(directory / "model_update_history.csv")
    slices = pd.read_csv(directory / "visualization_landscape_slice.csv")
    example_id = str(slices["case_id"].iloc[0])
    example_slice = slices.loc[slices["case_id"].eq(example_id)]
    example_shortlist = shortlist.loc[shortlist["case_id"].eq(example_id)]
    example_execution = execution.loc[execution["case_id"].eq(example_id)].sort_values("round")
    example_updates = updates.loc[updates["case_id"].eq(example_id)].sort_values("model_iteration")
    frames: list[Image.Image] = []

    figure, axis = _new_frame("REFERENCE", "Measured Reference Trajectory")
    plot_axis = figure.add_axes((0.12, 0.2, 0.76, 0.55))
    plot_axis.plot(reference["time_s"], np.rad2deg(reference["q_hip_rad"]), color=_BLUE, lw=2.4, label="$q_{hip}(t)$")
    plot_axis.plot(reference["time_s"], np.rad2deg(reference["q_knee_rad"]), color=_ORANGE, lw=2.4, label="$q_{knee}(t)$")
    plot_axis.set(xlabel="Time (s)", ylabel="Joint angle (deg)")
    plot_axis.legend(frameon=False)
    frames.append(_canvas_image(figure))

    figure, axis = _new_frame("INITIAL IDENTIFICATION", "1–5 complete excitation trajectories")
    for index, label in enumerate(("q(t)", "dq(t)", "ddq(t)", "tau(t)")):
        _node(axis, 0.16 + index * 0.18, 0.63, label, _BLUE)
        if index:
            _arrow(axis, (0.16 + (index - 1) * 0.18 + 0.08, 0.63), (0.16 + index * 0.18 - 0.08, 0.63))
    _arrow(axis, (0.78, 0.63), (0.84, 0.63))
    _node(axis, 0.91, 0.63, "$\\hat\\theta_0$\n[m, Kh, Kk, Bh, Bk]", _GREEN)
    axis.text(0.5, 0.3, "Subject-specific local equivalent dynamics", ha="center", fontsize=15, color=_DARK)
    frames.append(_canvas_image(figure))

    figure, axis = _new_frame("FULL MODEL-SPACE EXPLORATION", "Each point is one complete rehabilitation trajectory")
    rng = np.random.default_rng(20260826)
    points = rng.uniform((0.12, 0.2), (0.88, 0.74), size=(1100, 2))
    axis.scatter(points[:, 0], points[:, 1], s=7, color=_BLUE, alpha=0.3)
    axis.scatter(points[:340, 0], points[:340, 1], s=9, color="#C7CDD5", alpha=0.9)
    axis.text(0.5, 0.81, "21,025 trajectory candidates", ha="center", fontsize=18, weight="bold", color=_DARK)
    axis.text(0.5, 0.12, "ROM • workspace • Jacobian • patient envelope → invalid regions greyed", ha="center", fontsize=12, color=_GRAY)
    frames.append(_canvas_image(figure))

    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    _heatmap(axis, example_slice, value="J_pred", title="Predicted mechanical interaction landscape\n$J_{pred}(\\alpha;\\hat\\theta_0)$ — minimum over knee")
    frames.append(_canvas_image(figure))

    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    _heatmap(axis, example_slice, value="J_pred", title="MODEL SCREENING — FROZEN SHORTLIST\nHip–phase projection; shortlist frozen before validation", shortlist=example_shortlist)
    frames.append(_canvas_image(figure))

    first = example_execution.iloc[0]
    figure, axis = _new_frame("VALIDATION ROUND 1", f"Execute one complete candidate: {first['candidate_id']}")
    for x, label in zip((0.16, 0.39, 0.62, 0.85), ("Complete\ntrajectory", "Observation", f"J measured\n{first['actual_J']:.4f}", "$D_1=D_{ID}\\cup D_{C1}$")):
        _node(axis, x, 0.57, label, _ORANGE if x == 0.16 else _BLUE)
    for start, end in zip((0.24, 0.47, 0.70), (0.31, 0.54, 0.77)):
        _arrow(axis, (start, 0.57), (end, 0.57))
    frames.append(_canvas_image(figure))

    updated = example_updates.loc[example_updates["model_iteration"].eq(1)].iloc[0]
    figure, axis = _new_frame("MODEL UPDATE", "$\\hat\\theta_0 \\rightarrow \\hat\\theta_1$ and recompute the full landscape")
    _node(axis, 0.23, 0.58, "$\\hat\\theta_0$", _BLUE)
    _arrow(axis, (0.31, 0.58), (0.43, 0.58))
    _node(axis, 0.51, 0.58, "$\\hat\\theta_1$", _GREEN)
    _arrow(axis, (0.59, 0.58), (0.70, 0.58))
    _node(axis, 0.80, 0.58, "New full\n$J_{pred}$ map", _BLUE)
    axis.text(0.5, 0.3, f"Diagnostic optimum: {updated['diagnostic_full_predicted_optimum_id']}", ha="center", fontsize=12, color=_GRAY)
    axis.text(0.5, 0.21, "New candidates are NOT added", ha="center", fontsize=15, weight="bold", color=_RED)
    frames.append(_canvas_image(figure))

    figure, axis = _new_frame("RERANK REMAINING FROZEN CANDIDATES", "Only unexecuted C1/C2/C3 can move in rank")
    remaining = example_execution.iloc[1:]
    for index, row in enumerate(remaining.itertuples(index=False), start=1):
        _node(axis, 0.35, 0.7 - index * 0.18, f"{row.candidate_id}\ncurrent rank {index}", _BLUE)
    if not remaining.empty:
        _arrow(axis, (0.45, 0.43), (0.62, 0.43))
        _node(axis, 0.73, 0.43, f"Round 2\n{remaining.iloc[0]['candidate_id']}", _ORANGE)
    frames.append(_canvas_image(figure))

    figure, axis = _new_frame("FINITE VALIDATION", "validate → observe → refit → rerank")
    for index in range(1, 4):
        _node(axis, 0.2 + (index - 1) * 0.3, 0.56, f"Validation budget\n{index} / 3", _GREEN if index <= len(example_execution) else _GRAY)
        if index > 1:
            _arrow(axis, (0.2 + (index - 2) * 0.3 + 0.08, 0.56), (0.2 + (index - 1) * 0.3 - 0.08, 0.56))
    frames.append(_canvas_image(figure))

    figure, axis = _new_frame("STOP", "MAX VALIDATION BUDGET REACHED")
    axis.text(0.5, 0.52, "3 complete candidate trajectories", ha="center", fontsize=27, weight="bold", color=_DARK)
    axis.text(0.5, 0.35, "No claim that the global optimum was found", ha="center", fontsize=15, color=_RED)
    frames.append(_canvas_image(figure))

    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    labels = ["Reference"] + example_execution["candidate_id"].astype(str).tolist()
    values = [1.0] + example_execution["actual_J"].astype(float).tolist()
    colors = [_GRAY] + [_BLUE] * len(example_execution)
    best_index = int(np.argmin(values))
    colors[best_index] = _GREEN
    axis.bar(labels, values, color=colors)
    axis.set_ylim(min(values) - 0.01, max(values) + 0.01)
    axis.set_ylabel("Measured / truth whole-trajectory J")
    axis.set_title("FINAL SELECTION — BEST VALIDATED TRAJECTORY")
    frames.append(_canvas_image(figure))

    figure, axis = _new_frame("FINAL SUMMARY")
    labels = ("Reference", "Initial ID", "Full virtual\nlandscape", "Frozen\nshortlist", "Finite sequential\nvalidation", "Best validated\ntrajectory")
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        _node(axis, float(x), 0.58, label, _GREEN if index == 5 else _BLUE)
        if index:
            _arrow(axis, (float(xs[index - 1]) + 0.08, 0.58), (float(x) - 0.08, 0.58))
    axis.text(0.5, 0.28, "Large virtual search, small physical search", ha="center", fontsize=19, weight="bold", color=_DARK)
    axis.text(0.5, 0.18, "Model screens candidates; measurements decide the final trajectory.", ha="center", fontsize=14, color=_GREEN)
    frames.append(_canvas_image(figure))

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[900] * len(frames),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output


def generate_subject_landscape_animation(
    artifact_directory: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    directory = Path(artifact_directory)
    output = Path(output_path) if output_path is not None else directory / GIF_NAMES[1]
    _style()
    slices = pd.read_csv(directory / "visualization_landscape_slice.csv")
    shortlist = pd.read_csv(directory / "candidate_shortlist_manifest.csv")
    selected_cases = list(dict.fromkeys(slices["case_id"].astype(str)))[:3]
    if len(selected_cases) < 3:
        raise RuntimeError("subject-specific GIF requires three real simulation cases")
    frames: list[Image.Image] = []
    for index, case_id in enumerate(selected_cases, start=1):
        frame = slices.loc[slices["case_id"].eq(case_id)]
        candidates = shortlist.loc[shortlist["case_id"].eq(case_id)]
        figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
        _heatmap(
            axis,
            frame,
            value="J_pred",
            title=(
                f"Subject {index}: {case_id}\n"
                f"Subject-specific $\\hat\\theta_0$; hip–phase projection (minimum over knee)"
            ),
            shortlist=candidates,
        )
        frames.append(_canvas_image(figure))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[2200] * len(frames),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output


def generate_all_visuals(artifact_directory: str | Path) -> tuple[Path, ...]:
    static = generate_static_figures(artifact_directory)
    workflow = generate_workflow_animation(artifact_directory)
    subjects = generate_subject_landscape_animation(artifact_directory)
    return (*static, workflow, subjects)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", type=Path)
    arguments = parser.parse_args(argv)
    generated = generate_all_visuals(arguments.artifact_directory)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
