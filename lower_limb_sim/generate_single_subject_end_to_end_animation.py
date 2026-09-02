"""Generate one real-simulation subject's complete finite-validation journey.

This is a visualization-only supplement.  It reads the already frozen formal
artifacts and regenerates the three complete candidate trajectories from their
recorded alpha values.  It does not rerun personalization, read held-out data,
or connect to hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

_MPL_CACHE = Path(tempfile.gettempdir()) / "rehab_robot_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .decision_relevant_global_model_reliability import (
    geometrically_valid_parameter_lattice,
)
from .formal_protocol import ACTIVE_REFERENCE_PATH, ACTIVE_REFERENCE_SHA256
from .generate_final_method_animation import (
    _BLUE,
    _DARK,
    _GRAY,
    _GREEN,
    _LIGHT,
    _ORANGE,
    _RED,
    _canvas_image,
    _heatmap,
    _style,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


MODULE_DIR = Path(__file__).resolve().parent
SOURCE_ARTIFACT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "final_model_screened_finite_sequential_validation_v1"
)
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "final_model_screened_finite_sequential_validation_v1_single_subject_animation"
)
DEFAULT_CASE_ID = "baseline__combined_mild"
GIF_NAME = "SINGLE_SIMULATED_SUBJECT_END_TO_END.gif"
METADATA_NAME = "metadata.json"
GUIDE_NAME = "README.md"

_PARAMETER_COLUMNS = (
    "mass_scale",
    "k_hip_nm_per_rad",
    "k_knee_nm_per_rad",
    "b_hip_nm_s_per_rad",
    "b_knee_nm_s_per_rad",
)
_PARAMETER_LABELS = ("mass scale", "$K_h$", "$K_k$", "$B_h$", "$B_k$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_table(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _select_exact_case(table: pd.DataFrame, case_id: str, name: str) -> pd.DataFrame:
    selected = table.loc[table["case_id"].astype(str).eq(case_id)].copy()
    if selected.empty:
        raise ValueError(f"{case_id!r} is absent from {name}")
    return selected


def _load_case(
    artifact_directory: str | Path,
    case_id: str,
) -> dict[str, Any]:
    directory = Path(artifact_directory)
    subjects = _select_exact_case(
        _read_table(directory, "final_subject_summary.csv"), case_id, "subject summary"
    )
    if len(subjects) != 1:
        raise ValueError(f"{case_id!r} must have exactly one subject-summary row")
    shortlist = _select_exact_case(
        _read_table(directory, "candidate_shortlist_manifest.csv"),
        case_id,
        "shortlist",
    ).sort_values("shortlist_ordinal", kind="mergesort")
    execution = _select_exact_case(
        _read_table(directory, "candidate_execution_history.csv"),
        case_id,
        "execution history",
    ).sort_values("round", kind="mergesort")
    updates = _select_exact_case(
        _read_table(directory, "model_update_history.csv"),
        case_id,
        "model-update history",
    ).sort_values("model_iteration", kind="mergesort")
    landscape = _select_exact_case(
        _read_table(directory, "visualization_landscape_slice.csv"),
        case_id,
        "landscape slice",
    )
    if len(shortlist) != 3 or len(execution) != 3:
        raise ValueError("single-subject animation requires exactly three frozen/executed candidates")
    if execution["round"].astype(int).tolist() != [1, 2, 3]:
        raise ValueError("candidate rounds must be exactly 1, 2, 3")
    if shortlist["truth_read_before_freeze"].astype(bool).any():
        raise RuntimeError("shortlist is not truth-independent")
    if execution["truth_used_for_shortlist_or_ranking"].astype(bool).any():
        raise RuntimeError("execution history reports truth-based ranking")
    if not execution["whole_trajectory_execution"].astype(bool).all():
        raise RuntimeError("execution history contains a non-trajectory trial")
    if not execution["trajectory_sample_count"].astype(int).eq(401).all():
        raise RuntimeError("execution history does not contain complete 401-sample cycles")

    reference = pd.read_csv(ACTIVE_REFERENCE_PATH)
    trajectories: dict[str, pd.DataFrame] = {"Reference": reference}
    lattice = geometrically_valid_parameter_lattice(
        pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    ).set_index("trajectory_id")
    for row in execution.itertuples(index=False):
        if str(row.trajectory_id) not in lattice.index:
            raise RuntimeError("formal candidate is absent from the frozen 21,025-point lattice")
        lattice_row = lattice.loc[str(row.trajectory_id)]
        if isinstance(lattice_row, pd.DataFrame):
            raise RuntimeError("formal candidate identity is not unique in the lattice")
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=float(lattice_row.hip_delta),
            knee_amplitude_delta_deg=float(lattice_row.knee_delta),
            knee_phase_shift=float(lattice_row.phase_delta),
        )
        if str(generated.metadata["trajectory_id"]) != str(row.trajectory_id):
            raise RuntimeError("regenerated candidate identity does not match formal history")
        if str(generated.metadata["trajectory_sha256"]) != str(row.trajectory_sha256):
            raise RuntimeError("regenerated candidate SHA does not match formal history")
        trajectory = generated.trajectory
        theta_error = np.max(
            np.abs(
                trajectory["theta_shank_rad"].to_numpy(dtype=float)
                - (
                    trajectory["q_hip_rad"].to_numpy(dtype=float)
                    - trajectory["q_knee_rad"].to_numpy(dtype=float)
                )
            )
        )
        if theta_error > 1e-12:
            raise RuntimeError("theta_shank != q_hip - q_knee")
        trajectories[str(row.candidate_id)] = trajectory

    return {
        "directory": directory,
        "subject": subjects.iloc[0],
        "shortlist": shortlist,
        "execution": execution,
        "updates": updates,
        "landscape": landscape,
        "trajectories": trajectories,
    }


def _base_frame(
    stage: int,
    title: str,
    subtitle: str,
    *,
    case_id: str,
) -> tuple[plt.Figure, plt.Axes]:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    axis = figure.add_axes((0.04, 0.08, 0.92, 0.84))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.0, 1.01, f"STAGE {stage:02d}", color=_BLUE, weight="bold", fontsize=11)
    axis.text(0.5, 0.95, title, ha="center", va="top", fontsize=22, weight="bold", color=_DARK)
    axis.text(0.5, 0.88, subtitle, ha="center", va="top", fontsize=12, color=_GRAY)
    axis.text(
        0.0,
        -0.03,
        f"Offline virtual subject: {case_id}",
        ha="left",
        va="bottom",
        fontsize=9,
        color=_GRAY,
    )
    axis.text(
        1.0,
        -0.03,
        "Simulation only • not human-ready • not robot-approved",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_RED,
    )
    return figure, axis


def _box(axis: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    rectangle = patches.FancyBboxPatch(
        (x - 0.09, y - 0.06),
        0.18,
        0.12,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    axis.add_patch(rectangle)
    axis.text(x, y, text, ha="center", va="center", fontsize=10, color=_DARK)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": _GRAY},
    )


def _trajectory_axes(
    figure: plt.Figure,
    trajectories: Mapping[str, pd.DataFrame],
    labels: Sequence[str],
    *,
    bounds: tuple[float, float, float, float] = (0.10, 0.18, 0.58, 0.58),
) -> tuple[plt.Axes, plt.Axes]:
    left, bottom, width, height = bounds
    hip_axis = figure.add_axes((left, bottom + height / 2 + 0.035, width, height / 2 - 0.035))
    knee_axis = figure.add_axes((left, bottom, width, height / 2 - 0.035), sharex=hip_axis)
    palette = {"Reference": _GRAY, "C1": _GREEN, "C2": _BLUE, "C3": _ORANGE}
    for label in labels:
        frame = trajectories[label]
        line_width = 2.7 if label != "Reference" else 1.8
        hip_axis.plot(
            frame["time_s"],
            np.rad2deg(frame["q_hip_rad"]),
            color=palette[label],
            lw=line_width,
            label=label,
        )
        knee_axis.plot(
            frame["time_s"],
            np.rad2deg(frame["q_knee_rad"]),
            color=palette[label],
            lw=line_width,
            label=label,
        )
    hip_axis.set_ylabel("Hip (deg)")
    knee_axis.set_ylabel("Knee (deg)")
    knee_axis.set_xlabel("Time (s)")
    hip_axis.legend(frameon=False, ncol=len(labels), loc="upper right")
    plt.setp(hip_axis.get_xticklabels(), visible=False)
    return hip_axis, knee_axis


def _append_title_frame(data: Mapping[str, Any], frames: list[Image.Image]) -> None:
    subject = data["subject"]
    figure, axis = _base_frame(
        1,
        "ONE SIMULATED SUBJECT — COMPLETE JOURNEY",
        "A real recorded case from the frozen offline evaluation",
        case_id=str(subject.case_id),
    )
    _box(axis, 0.22, 0.58, "Subject\nbaseline", _BLUE)
    _arrow(axis, (0.32, 0.58), (0.43, 0.58))
    _box(axis, 0.53, 0.58, "Scenario\ncombined mild", _ORANGE)
    _arrow(axis, (0.63, 0.58), (0.74, 0.58))
    _box(axis, 0.84, 0.58, "Model class\nmismatch", _RED)
    axis.text(
        0.5,
        0.31,
        "Goal: choose the lowest validated whole-trajectory mechanical J",
        ha="center",
        fontsize=15,
        color=_DARK,
    )
    frames.append(_canvas_image(figure))


def _append_reference_frame(data: Mapping[str, Any], frames: list[Image.Image]) -> None:
    case_id = str(data["subject"].case_id)
    figure, axis = _base_frame(
        2,
        "MEASURED REFERENCE",
        "One complete 24 s asymmetric rehabilitation cycle • normalized J = 1",
        case_id=case_id,
    )
    _trajectory_axes(figure, data["trajectories"], ["Reference"], bounds=(0.13, 0.18, 0.74, 0.58))
    frames.append(_canvas_image(figure))


def _append_identification_frame(data: Mapping[str, Any], frames: list[Image.Image]) -> None:
    subject = data["subject"]
    theta = json.loads(str(subject.initial_theta_hat_json))
    figure, axis = _base_frame(
        3,
        "INITIAL IDENTIFICATION",
        f"{int(subject.initial_identification_trial_count)} existing complete identification trajectories",
        case_id=str(subject.case_id),
    )
    for index, label in enumerate(("q(t)", "dq(t)", "ddq(t)", "tau(t)")):
        x = 0.12 + index * 0.18
        _box(axis, x, 0.65, label, _BLUE)
        if index:
            _arrow(axis, (x - 0.27 + 0.09, 0.65), (x - 0.09, 0.65))
    _arrow(axis, (0.75, 0.65), (0.80, 0.65))
    _box(axis, 0.90, 0.65, "$\\hat\\theta_0$", _GREEN)
    values = [theta[name] for name in _PARAMETER_COLUMNS]
    table = "   ".join(f"{label}={value:.3f}" for label, value in zip(_PARAMETER_LABELS, values))
    axis.text(0.5, 0.37, table, ha="center", fontsize=12, color=_DARK)
    axis.text(
        0.5,
        0.24,
        "Equivalent local dynamics; not physiological tissue parameters",
        ha="center",
        fontsize=11,
        color=_GRAY,
    )
    frames.append(_canvas_image(figure))


def _append_landscape_frames(data: Mapping[str, Any], frames: list[Image.Image]) -> None:
    case_id = str(data["subject"].case_id)
    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    _heatmap(
        axis,
        data["landscape"],
        value="J_pred",
        title=(
            "STAGE 04 — FULL MODEL SCREENING\n"
            "21,025 complete trajectories; hip–phase projection, minimum over formal knee axis"
        ),
    )
    axis.text(
        0.01,
        -0.17,
        f"Offline virtual subject: {case_id} • geometry + frozen 90% support gate",
        transform=axis.transAxes,
        fontsize=9,
        color=_GRAY,
    )
    frames.append(_canvas_image(figure))

    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    heat_axis = figure.add_axes((0.07, 0.15, 0.58, 0.72))
    _heatmap(
        heat_axis,
        data["landscape"],
        value="J_pred",
        title="STAGE 05 — FROZEN SHORTLIST BEFORE TRUTH",
        shortlist=data["shortlist"],
    )
    text_axis = figure.add_axes((0.70, 0.17, 0.27, 0.68))
    text_axis.axis("off")
    text_axis.text(0.5, 0.96, "Three complete candidates", ha="center", weight="bold", fontsize=13)
    for index, row in enumerate(data["shortlist"].itertuples(index=False)):
        text_axis.text(
            0.02,
            0.76 - index * 0.25,
            (
                f"{row.candidate_id}\n"
                f"α = ({row.hip_delta:+.2f}°, {row.knee_delta:+.2f}°, {row.phase_delta:+.4f})\n"
                f"initial Jpred = {row.initial_J_pred:.6f}\n"
                f"support = {row.initial_domain_coverage:.2f}%"
            ),
            va="top",
            fontsize=10.5,
            color=(_GREEN, _BLUE, _ORANGE)[index],
        )
    text_axis.text(
        0.5,
        0.02,
        "LOCKED: no C4 can be added",
        ha="center",
        color=_RED,
        fontsize=12,
        weight="bold",
    )
    frames.append(_canvas_image(figure))


def _append_candidate_overview(data: Mapping[str, Any], frames: list[Image.Image]) -> None:
    figure, axis = _base_frame(
        6,
        "CANDIDATES ARE COMPLETE TRAJECTORIES",
        "Reference and C1–C3 each contain 401 samples over 24 s",
        case_id=str(data["subject"].case_id),
    )
    _trajectory_axes(
        figure,
        data["trajectories"],
        ["Reference", "C1", "C2", "C3"],
        bounds=(0.10, 0.18, 0.80, 0.58),
    )
    frames.append(_canvas_image(figure))


def _append_round_frame(
    data: Mapping[str, Any],
    frames: list[Image.Image],
    round_number: int,
    stage: int,
) -> None:
    row = data["execution"].loc[data["execution"]["round"].astype(int).eq(round_number)].iloc[0]
    candidate = str(row.candidate_id)
    figure, axis = _base_frame(
        stage,
        f"VALIDATION ROUND {round_number} — {candidate}",
        "Execute one whole trajectory → observe → compute whole-trajectory J",
        case_id=str(data["subject"].case_id),
    )
    _trajectory_axes(
        figure,
        data["trajectories"],
        ["Reference", candidate],
        bounds=(0.07, 0.19, 0.56, 0.55),
    )
    metric_axis = figure.add_axes((0.70, 0.23, 0.24, 0.47))
    labels = ["Predicted", "Virtual truth", "Best so far"]
    values = [
        float(row.J_pred_before_execution),
        float(row.actual_J),
        float(row.best_validated_J_after),
    ]
    metric_axis.bar(labels, values, color=[_BLUE, _ORANGE, _GREEN])
    metric_axis.axhline(1.0, color=_GRAY, ls=":", lw=1.5)
    metric_axis.set_ylim(min(values) - 0.008, 1.003)
    metric_axis.set_ylabel("Whole-trajectory J")
    metric_axis.tick_params(axis="x", rotation=18)
    for index, value in enumerate(values):
        metric_axis.text(index, value + 0.001, f"{value:.6f}", ha="center", fontsize=9)
    frames.append(_canvas_image(figure))


def _append_update_frame(
    data: Mapping[str, Any],
    frames: list[Image.Image],
    after_round: int,
    stage: int,
) -> None:
    update = data["updates"].loc[
        data["updates"]["model_iteration"].astype(int).eq(after_round)
    ].iloc[0]
    remaining = json.loads(str(update.remaining_frozen_candidates_json))
    figure, axis = _base_frame(
        stage,
        f"REFIT AFTER ROUND {after_round}",
        "Append the observed trajectory, refit five parameters, recompute all 21,025 predictions",
        case_id=str(data["subject"].case_id),
    )
    _box(axis, 0.16, 0.61, f"D{after_round - 1}\n+ candidate", _ORANGE)
    _arrow(axis, (0.26, 0.61), (0.36, 0.61))
    _box(axis, 0.46, 0.61, f"$\\hat\\theta_{after_round}$\nrefitted", _GREEN)
    _arrow(axis, (0.56, 0.61), (0.66, 0.61))
    _box(axis, 0.76, 0.61, "Full Jpred\nrecomputed", _BLUE)
    axis.text(
        0.5,
        0.36,
        "Remaining frozen candidates: " + (", ".join(remaining) if remaining else "none"),
        ha="center",
        fontsize=13,
        color=_DARK,
    )
    axis.text(
        0.5,
        0.24,
        f"Diagnostic optimum: {update.diagnostic_full_predicted_optimum_id}",
        ha="center",
        fontsize=10,
        color=_GRAY,
    )
    axis.text(
        0.5,
        0.15,
        "New predicted candidates remain diagnostic only",
        ha="center",
        fontsize=11,
        weight="bold",
        color=_RED,
    )
    frames.append(_canvas_image(figure))


def _append_final_frames(data: Mapping[str, Any], frames: list[Image.Image]) -> None:
    execution = data["execution"]
    subject = data["subject"]
    figure, axis = _base_frame(
        13,
        "FINAL MEASUREMENT-BASED SELECTION",
        "The model screened; virtual measurements decide among Reference + C1 + C2 + C3",
        case_id=str(subject.case_id),
    )
    chart = figure.add_axes((0.18, 0.20, 0.64, 0.56))
    labels = ["Reference"] + execution["candidate_id"].astype(str).tolist()
    values = [1.0] + execution["actual_J"].astype(float).tolist()
    best = int(np.argmin(values))
    colors = [_GRAY, _BLUE, _BLUE, _BLUE]
    colors[best] = _GREEN
    chart.bar(labels, values, color=colors)
    chart.set_ylim(min(values) - 0.008, 1.004)
    chart.set_ylabel("Measured / virtual-truth whole-trajectory J")
    for index, value in enumerate(values):
        chart.text(index, value + 0.001, f"{value:.6f}", ha="center", fontsize=10)
    chart.text(best, min(values) - 0.004, "BEST VALIDATED", ha="center", color=_GREEN, weight="bold")
    frames.append(_canvas_image(figure))

    figure, axis = _base_frame(
        14,
        "BEST VALIDATED TRAJECTORY",
        f"{subject.best_validated_trajectory_id}",
        case_id=str(subject.case_id),
    )
    _trajectory_axes(
        figure,
        data["trajectories"],
        ["Reference", "C1"],
        bounds=(0.08, 0.21, 0.58, 0.53),
    )
    summary_axis = figure.add_axes((0.71, 0.23, 0.25, 0.48))
    summary_axis.axis("off")
    summary_axis.text(0.0, 0.90, "Selected α", fontsize=13, weight="bold", color=_DARK)
    summary_axis.text(
        0.0,
        0.76,
        (
            f"hip  {subject.best_validated_alpha_hip:+.2f}°\n"
            f"knee {subject.best_validated_alpha_knee:+.2f}°\n"
            f"phase {subject.best_validated_alpha_phase:+.4f}"
        ),
        fontsize=12,
        linespacing=1.5,
        color=_GREEN,
    )
    summary_axis.text(0.0, 0.43, f"Reference J     1.000000\nValidated J     {subject.B2_final_best_validated_J:.6f}", fontsize=11, linespacing=1.6)
    summary_axis.text(0.0, 0.19, f"Reduction        {100 * subject.reference_improvement:.2f}%\nGlobal regret   {subject.global_regret:.6f}", fontsize=11, linespacing=1.6)
    frames.append(_canvas_image(figure))

    figure, axis = _base_frame(
        15,
        "SUBJECT JOURNEY COMPLETE",
        "Finite search stopped after three frozen complete-trajectory validations",
        case_id=str(subject.case_id),
    )
    labels = (
        "Reference",
        "5 initial ID\ntrajectories",
        "$\\hat\\theta_0$",
        "21,025 virtual\ncandidates",
        "Frozen C1–C3",
        "3 validations",
        "Best validated\nC1",
    )
    xs = np.linspace(0.07, 0.93, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        _box(axis, float(x), 0.58, label, _GREEN if index == len(labels) - 1 else _BLUE)
        if index:
            _arrow(axis, (float(xs[index - 1]) + 0.095, 0.58), (float(x) - 0.095, 0.58))
    axis.text(
        0.5,
        0.27,
        "Prediction screens candidates; complete-trajectory observations make the final choice.",
        ha="center",
        fontsize=14,
        weight="bold",
        color=_DARK,
    )
    frames.append(_canvas_image(figure))


def generate_single_subject_animation(
    artifact_directory: str | Path = SOURCE_ARTIFACT_DIRECTORY,
    *,
    case_id: str = DEFAULT_CASE_ID,
    output_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Generate the deterministic end-to-end GIF and return its provenance."""

    _style()
    data = _load_case(artifact_directory, case_id)
    output = (
        Path(output_path)
        if output_path is not None
        else DEFAULT_OUTPUT_DIRECTORY / GIF_NAME
    )
    frames: list[Image.Image] = []
    _append_title_frame(data, frames)
    _append_reference_frame(data, frames)
    _append_identification_frame(data, frames)
    _append_landscape_frames(data, frames)
    _append_candidate_overview(data, frames)
    _append_round_frame(data, frames, 1, 7)
    _append_update_frame(data, frames, 1, 8)
    _append_round_frame(data, frames, 2, 9)
    _append_update_frame(data, frames, 2, 10)
    _append_round_frame(data, frames, 3, 11)
    _append_update_frame(data, frames, 3, 12)
    _append_final_frames(data, frames)
    durations_ms = [1800, 1700, 1900, 1900, 2400, 1800, 1900, 1600, 1900, 1600, 1900, 1500, 2200, 2200, 2500]
    if len(frames) != len(durations_ms):
        raise RuntimeError("frame-duration contract mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    subject = data["subject"]
    metadata = {
        "visualization_id": "SINGLE_SIMULATED_SUBJECT_END_TO_END_V1",
        "case_id": case_id,
        "subject_id": str(subject.subject_id),
        "scenario_name": str(subject.scenario_name),
        "case_class": str(subject.case_class),
        "source_formal_manifest_sha256": _sha256(Path(artifact_directory) / "FINAL_METHOD_MANIFEST_V1.json"),
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "candidate_ids": data["execution"]["candidate_id"].astype(str).tolist(),
        "candidate_trajectory_sha256": data["execution"]["trajectory_sha256"].astype(str).tolist(),
        "actual_J": data["execution"]["actual_J"].astype(float).tolist(),
        "best_validated_trajectory_id": str(subject.best_validated_trajectory_id),
        "best_validated_J": float(subject.B2_final_best_validated_J),
        "reference_improvement": float(subject.reference_improvement),
        "global_regret": float(subject.global_regret),
        "shortlist_frozen_before_truth": True,
        "truth_used_to_create_or_rank_shortlist": False,
        "whole_trajectory_trials": True,
        "held_out_final_test_read": False,
        "robot_connected": False,
        "frame_count": len(frames),
        "resolution_px": [1280, 720],
        "duration_ms": int(sum(durations_ms)),
        "loop": True,
        "gif_sha256": _sha256(output),
        "generator_source_sha256": _sha256(Path(__file__)),
    }
    return output, metadata


def _guide(metadata: Mapping[str, Any]) -> str:
    return f"""# Single simulated subject end-to-end animation

This supplementary GIF follows the frozen offline case
`{metadata['case_id']}` from its measured reference through initial
identification, full model screening, truth-independent C1–C3 shortlist freeze,
three complete candidate validations, model updates, and the final
measurement-based selection.

- Output: `{GIF_NAME}`
- Frames: `{metadata['frame_count']}` at 1280×720
- Duration: `{metadata['duration_ms'] / 1000:.1f} s`, looping
- Selected trajectory: `{metadata['best_validated_trajectory_id']}`
- Best validated whole-trajectory J: `{metadata['best_validated_J']:.6f}`

The subject and every plotted trajectory come from the existing offline
simulation artifacts.  The GIF is not human, clinical, comfort, safety, or
robot-motion evidence.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=SOURCE_ARTIFACT_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    arguments = parser.parse_args(argv)
    output, metadata = generate_single_subject_animation(
        arguments.artifact_directory,
        case_id=str(arguments.case_id),
        output_path=arguments.output_directory / GIF_NAME,
    )
    _atomic_bytes(
        arguments.output_directory / METADATA_NAME,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_bytes(
        arguments.output_directory / GUIDE_NAME,
        _guide(metadata).encode("utf-8"),
    )
    print(json.dumps({"gif": str(output), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
