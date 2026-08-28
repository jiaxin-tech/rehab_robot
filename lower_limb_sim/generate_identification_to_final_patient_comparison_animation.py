"""Animate identification excitations through final selection for three subjects.

The compared frozen offline subjects are baseline (normal equivalent dynamics),
hip stiffness, and knee stiffness under the matched-linear scenario.  Every
motion, force response, identified parameter, shortlist, and validation J is
reconstructed from or checked against the existing formal offline evidence.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image

from .config import L1, L2
from .continuous_reference_neighborhood import generate_personalized_trajectory
from .decision_relevant_global_model_reliability import (
    geometrically_valid_parameter_lattice,
)
from .generate_final_method_animation import (
    _BLUE,
    _DARK,
    _GRAY,
    _GREEN,
    _LIGHT,
    _ORANGE,
    _RED,
    _style,
)
from .generate_single_subject_end_to_end_animation import (
    DEFAULT_OUTPUT_DIRECTORY,
    SOURCE_ARTIFACT_DIRECTORY,
    _atomic_bytes,
    _sha256,
)
from .kinematics import forward_kinematics
from .research_decision_guarded_sequential_personalization import (
    build_initial_research_state,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


GIF_NAME = "NORMAL_VS_STIFFNESS_IDENTIFICATION_TO_FINAL.gif"
METADATA_NAME = "normal_vs_stiffness_animation_metadata.json"
GUIDE_NAME = "NORMAL_VS_STIFFNESS_ANIMATION_README.md"
DEFAULT_SAMPLES_PER_MOTION = 31

PATIENTS = (
    {
        "display_name": "Normal",
        "subject_id": "baseline",
        "scenario_name": "matched_linear",
        "case_id": "baseline__matched_linear",
        "color": _GREEN,
    },
    {
        "display_name": "Hip stiffness",
        "subject_id": "hip_stiff",
        "scenario_name": "matched_linear",
        "case_id": "hip_stiff__matched_linear",
        "color": _BLUE,
    },
    {
        "display_name": "Knee stiffness",
        "subject_id": "knee_stiff",
        "scenario_name": "matched_linear",
        "case_id": "knee_stiff__matched_linear",
        "color": _ORANGE,
    },
)

_TIMELINE_LABELS = (
    "Excitation 1",
    "Excitation 2",
    "Fit θ",
    "Screen 21,025",
    "Validate C1",
    "Validate C2",
    "Validate C3",
    "Final",
)


def _read_formal(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _complete_excitation(state: Any, trial_id: int) -> pd.DataFrame:
    table = state.sequential_result.executed_identification_data
    selected = table.loc[table["trial_id"].astype(int).eq(int(trial_id))].copy()
    if selected.empty:
        raise RuntimeError(f"identification trial {trial_id} is absent")
    selected = selected.sort_values("trajectory_sample_index", kind="mergesort")
    if selected["trajectory_sample_index"].duplicated().any() or len(selected) != 401:
        raise RuntimeError("identification excitation is not one complete 401-sample trajectory")
    q_hip = selected["q_hip_rad"].to_numpy(dtype=float)
    q_knee = selected["q_knee_rad"].to_numpy(dtype=float)
    x_knee, z_knee, x_pull, z_pull = forward_kinematics(q_hip, q_knee, L1, L2)
    selected["x_knee_m"] = x_knee
    selected["z_knee_m"] = z_knee
    selected["x_pull_m"] = x_pull
    selected["z_pull_m"] = z_pull
    selected["theta_shank_rad"] = q_hip - q_knee
    duration = float(selected["excitation_duration_s"].iloc[0])
    selected["time_s"] = (
        selected["trajectory_sample_index"].to_numpy(dtype=float) / 400.0 * duration
    )
    selected["force_magnitude_n"] = np.hypot(
        selected["fx_observed_n"].to_numpy(dtype=float),
        selected["fz_observed_n"].to_numpy(dtype=float),
    )
    return selected.reset_index(drop=True)


def _load_animation_data(artifact_directory: str | Path) -> dict[str, Any]:
    directory = Path(artifact_directory)
    shortlist_all = _read_formal(directory, "candidate_shortlist_manifest.csv")
    execution_all = _read_formal(directory, "candidate_execution_history.csv")
    summary_all = _read_formal(directory, "final_subject_summary.csv")
    lattice = geometrically_valid_parameter_lattice(
        pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    ).set_index("trajectory_id")
    patient_data: list[dict[str, Any]] = []

    for specification in PATIENTS:
        state = build_initial_research_state(
            str(specification["subject_id"]), str(specification["scenario_name"])
        )
        if int(state.selected_trial_id) != 2:
            raise RuntimeError("comparison expects the frozen two-excitation diagnostic model")
        summary_rows = summary_all.loc[
            summary_all["case_id"].astype(str).eq(str(specification["case_id"]))
        ]
        if len(summary_rows) != 1:
            raise RuntimeError("formal subject summary must contain exactly one comparison row")
        summary = summary_rows.iloc[0]
        formal_parameters = json.loads(str(summary.initial_theta_hat_json))
        for name, value in state.parameters.items():
            if not np.isclose(float(value), float(formal_parameters[name]), atol=1e-10, rtol=0.0):
                raise RuntimeError("reconstructed identification parameters differ from formal output")
        shortlist = shortlist_all.loc[
            shortlist_all["case_id"].astype(str).eq(str(specification["case_id"]))
        ].sort_values("shortlist_ordinal", kind="mergesort")
        execution = execution_all.loc[
            execution_all["case_id"].astype(str).eq(str(specification["case_id"]))
        ].sort_values("round", kind="mergesort")
        if len(shortlist) != 3 or len(execution) != 3:
            raise RuntimeError("comparison requires three frozen and three executed candidates")
        if shortlist["truth_read_before_freeze"].astype(bool).any():
            raise RuntimeError("comparison shortlist is not truth-independent")
        if execution["truth_used_for_shortlist_or_ranking"].astype(bool).any():
            raise RuntimeError("comparison execution used truth to rank candidates")

        candidates: dict[str, pd.DataFrame] = {}
        for row in execution.itertuples(index=False):
            trajectory_id = str(row.trajectory_id)
            if trajectory_id not in lattice.index:
                raise RuntimeError("formal candidate is absent from the frozen lattice")
            point = lattice.loc[trajectory_id]
            if isinstance(point, pd.DataFrame):
                raise RuntimeError("formal candidate identity is not unique")
            generated = generate_personalized_trajectory(
                hip_amplitude_delta_deg=float(point.hip_delta),
                knee_amplitude_delta_deg=float(point.knee_delta),
                knee_phase_shift=float(point.phase_delta),
            )
            if str(generated.metadata["trajectory_sha256"]) != str(row.trajectory_sha256):
                raise RuntimeError("candidate trajectory SHA does not match formal execution")
            trajectory = generated.trajectory.copy(deep=True)
            if np.max(
                np.abs(
                    trajectory["theta_shank_rad"].to_numpy(dtype=float)
                    - (
                        trajectory["q_hip_rad"].to_numpy(dtype=float)
                        - trajectory["q_knee_rad"].to_numpy(dtype=float)
                    )
                )
            ) > 1e-12:
                raise RuntimeError("theta_shank subtraction convention changed")
            candidates[str(row.candidate_id)] = trajectory

        patient_data.append(
            {
                **specification,
                "state": state,
                "parameters": dict(state.parameters),
                "excitation_1": _complete_excitation(state, 1),
                "excitation_2": _complete_excitation(state, 2),
                "shortlist": shortlist,
                "execution": execution,
                "summary": summary,
                "candidates": candidates,
            }
        )
    return {"directory": directory, "patients": patient_data}


def _draw_timeline(figure: plt.Figure, active_step: int) -> None:
    axis = figure.add_axes((0.08, 0.035, 0.84, 0.065))
    axis.set_xlim(0, len(_TIMELINE_LABELS))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, label in enumerate(_TIMELINE_LABELS):
        completed = index <= active_step
        face = _BLUE if completed else "white"
        rectangle = patches.FancyBboxPatch(
            (index + 0.04, 0.17),
            0.84,
            0.62,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor=face,
            edgecolor=_BLUE,
            linewidth=1.1,
            alpha=0.92 if completed else 0.45,
        )
        axis.add_patch(rectangle)
        axis.text(
            index + 0.46,
            0.48,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if completed else _GRAY,
            weight="bold" if index == active_step else "normal",
        )
        if index < len(_TIMELINE_LABELS) - 1:
            axis.annotate(
                "",
                xy=(index + 1.02, 0.48),
                xytext=(index + 0.90, 0.48),
                arrowprops={"arrowstyle": "-|>", "color": _GRAY, "lw": 0.9},
            )


def _draw_supine_panel(
    axis: plt.Axes,
    patient: Mapping[str, Any],
    trajectory: pd.DataFrame,
    sample_index: int,
    *,
    stage_text: str,
    response_text: str,
) -> None:
    color = str(patient["color"])
    axis.set_xlim(-1.0, 0.82)
    axis.set_ylim(-0.20, 0.66)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    mattress = patches.FancyBboxPatch(
        (-0.96, -0.115),
        1.72,
        0.085,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        facecolor="#DDEAF4",
        edgecolor="#7D95A8",
        linewidth=1.2,
        zorder=0,
    )
    axis.add_patch(mattress)
    axis.plot([-0.88, -0.88], [-0.12, -0.18], color="#7D95A8", lw=3)
    axis.plot([0.68, 0.68], [-0.12, -0.18], color="#7D95A8", lw=3)
    axis.add_patch(
        patches.FancyBboxPatch(
            (-0.91, -0.002),
            0.31,
            0.10,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="white",
            edgecolor="#AAB9C5",
            linewidth=1.0,
        )
    )
    skin = "#D9A07E"
    axis.add_patch(patches.Circle((-0.72, 0.105), 0.050, facecolor=skin, edgecolor=_DARK, lw=1.2, zorder=3))
    axis.plot([-0.66, -0.08], [0.045, 0.015], color=_DARK, lw=15, solid_capstyle="round", zorder=2)
    axis.plot([-0.49, -0.15], [0.035, 0.10], color=skin, lw=6, solid_capstyle="round", zorder=3)
    axis.plot([-0.08, 0.0], [0.015, 0.0], color=skin, lw=13, solid_capstyle="round", zorder=3)

    row = trajectory.iloc[int(sample_index)]
    knee = (float(row.x_knee_m), float(row.z_knee_m))
    distal = (float(row.x_pull_m), float(row.z_pull_m))
    axis.plot(trajectory["x_pull_m"], trajectory["z_pull_m"], color=color, lw=1.0, alpha=0.18)
    axis.plot(
        trajectory["x_pull_m"].iloc[: int(sample_index) + 1],
        trajectory["z_pull_m"].iloc[: int(sample_index) + 1],
        color=color,
        lw=2.2,
        alpha=0.75,
    )
    axis.plot([0, knee[0]], [0, knee[1]], color=skin, lw=16, solid_capstyle="round", zorder=3)
    axis.plot([0, knee[0]], [0, knee[1]], color=_BLUE, lw=7, solid_capstyle="round", zorder=4)
    axis.plot([knee[0], distal[0]], [knee[1], distal[1]], color=skin, lw=14, solid_capstyle="round", zorder=3)
    axis.plot([knee[0], distal[0]], [knee[1], distal[1]], color=_ORANGE, lw=6, solid_capstyle="round", zorder=4)
    axis.scatter([0], [0], s=75, color=_DARK, edgecolor="white", lw=1, zorder=5)
    axis.scatter([knee[0]], [knee[1]], s=70, color="white", edgecolor=_DARK, lw=1.5, zorder=5)
    axis.scatter([distal[0]], [distal[1]], s=70, color=color, edgecolor="white", lw=1.2, zorder=5)
    q_hip = np.rad2deg(float(row.q_hip_rad))
    q_knee = np.rad2deg(float(row.q_knee_rad))
    axis.text(0.5, 1.05, str(patient["display_name"]), transform=axis.transAxes, ha="center", fontsize=14, weight="bold", color=color)
    axis.text(0.5, 0.98, stage_text, transform=axis.transAxes, ha="center", fontsize=8.8, color=_GRAY)
    axis.text(0.02, 0.04, f"hip {q_hip:5.1f}°   knee {q_knee:5.1f}°", transform=axis.transAxes, fontsize=8.5, color=_DARK)
    axis.text(0.02, 0.91, response_text, transform=axis.transAxes, fontsize=8.5, color=color, weight="bold")


def _motion_frame(
    data: Mapping[str, Any],
    *,
    source_key: str,
    sample_index: int,
    title: str,
    subtitle: str,
    active_step: int,
    validation_round: int | None = None,
) -> Image.Image:
    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    figure.text(0.5, 0.965, title, ha="center", va="top", fontsize=22, weight="bold", color=_DARK)
    figure.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=11, color=_GRAY)
    positions = (0.025, 0.345, 0.665)
    for patient, left in zip(data["patients"], positions):
        axis = figure.add_axes((left, 0.16, 0.30, 0.69))
        if validation_round is None:
            trajectory = patient[source_key]
            row = trajectory.iloc[int(sample_index)]
            response = (
                f"|F|={float(row.force_magnitude_n):.2f} N   "
                f"Fx={float(row.fx_observed_n):+.2f}   Fz={float(row.fz_observed_n):+.2f}"
            )
            stage = str(trajectory["candidate_id"].iloc[0])
        else:
            label = f"C{validation_round}"
            trajectory = patient["candidates"][label]
            execution = patient["execution"].loc[
                patient["execution"]["round"].astype(int).eq(validation_round)
            ].iloc[0]
            response = (
                f"Jpred={float(execution.J_pred_before_execution):.6f}   "
                f"Jtruth={float(execution.actual_J):.6f}"
            )
            stage = f"{label}: α=({float(execution.hip_delta):+.2f}°, {float(execution.knee_delta):+.2f}°, {float(execution.phase_delta):+.4f})"
        _draw_supine_panel(
            axis,
            patient,
            trajectory,
            int(sample_index),
            stage_text=stage,
            response_text=response,
        )
    _draw_timeline(figure, active_step)
    figure.text(
        0.99,
        0.012,
        "Frozen offline simulation • theta_shank = q_hip - q_knee • no robot motion",
        ha="right",
        fontsize=8.5,
        color=_RED,
    )
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
    image = Image.fromarray(np.ascontiguousarray(rgba[:, :, :3]))
    plt.close(figure)
    return image


def _title_frame(data: Mapping[str, Any]) -> Image.Image:
    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    axis = figure.add_axes((0.05, 0.08, 0.90, 0.84))
    axis.axis("off")
    axis.text(0.5, 0.92, "IDENTIFICATION → SCREENING → VALIDATION → FINAL TRAJECTORY", ha="center", fontsize=24, weight="bold", color=_DARK)
    axis.text(0.5, 0.84, "Three frozen offline virtual patients", ha="center", fontsize=14, color=_GRAY)
    xs = (0.18, 0.50, 0.82)
    for patient, x in zip(data["patients"], xs):
        color = str(patient["color"])
        rectangle = patches.FancyBboxPatch(
            (x - 0.12, 0.42),
            0.24,
            0.25,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="white",
            edgecolor=color,
            linewidth=2.5,
        )
        axis.add_patch(rectangle)
        axis.text(x, 0.61, str(patient["display_name"]), ha="center", fontsize=16, weight="bold", color=color)
        parameters = patient["parameters"]
        axis.text(
            x,
            0.50,
            f"Kh={parameters['k_hip_nm_per_rad']:.1f}\nKk={parameters['k_knee_nm_per_rad']:.1f}",
            ha="center",
            fontsize=12,
            color=_DARK,
        )
    axis.text(0.5, 0.23, "Same excitation motion; different force response identifies different equivalent dynamics.", ha="center", fontsize=15, color=_DARK)
    axis.text(0.5, 0.10, "Every candidate shown later is one complete 24 s supine hip–knee trajectory.", ha="center", fontsize=12, color=_GRAY)
    figure.canvas.draw()
    image = Image.fromarray(np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]))
    plt.close(figure)
    return image


def _parameter_frame(data: Mapping[str, Any]) -> Image.Image:
    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    figure.suptitle("IDENTIFIED SUBJECT-SPECIFIC FIVE-PARAMETER MODELS", fontsize=22, weight="bold", color=_DARK, y=0.96)
    baseline = data["patients"][0]["parameters"]
    names = ("mass_scale", "k_hip_nm_per_rad", "k_knee_nm_per_rad", "b_hip_nm_s_per_rad", "b_knee_nm_s_per_rad")
    labels = ("mass", "Kh", "Kk", "Bh", "Bk")
    positions = (0.04, 0.36, 0.68)
    for patient, left in zip(data["patients"], positions):
        axis = figure.add_axes((left, 0.20, 0.28, 0.62))
        values = np.asarray([patient["parameters"][name] / baseline[name] for name in names])
        bars = axis.bar(labels, values, color=str(patient["color"]))
        axis.axhline(1.0, color=_GRAY, ls="--", lw=1.2)
        axis.set_ylim(0, max(2.8, float(values.max()) + 0.3))
        axis.set_ylabel("Identified value / normal value")
        axis.set_title(str(patient["display_name"]), color=str(patient["color"]), weight="bold")
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}×", ha="center", fontsize=10)
        axis.grid(True, axis="y", color="#E6EBF0")
    figure.text(0.5, 0.13, "Normal: Kh=15, Kk=12   •   Hip stiffness: Kh=30   •   Knee stiffness: Kk=30", ha="center", fontsize=13, color=_DARK)
    figure.text(0.5, 0.08, "Equivalent model parameters—not physiological tissue constants", ha="center", fontsize=10, color=_RED)
    _draw_timeline(figure, 2)
    figure.canvas.draw()
    image = Image.fromarray(np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]))
    plt.close(figure)
    return image


def _screening_frame(data: Mapping[str, Any]) -> Image.Image:
    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    figure.suptitle("MODEL SCREENING: FREEZE C1–C3 BEFORE CANDIDATE TRUTH", fontsize=22, weight="bold", color=_DARK, y=0.96)
    positions = (0.04, 0.36, 0.68)
    for patient, left in zip(data["patients"], positions):
        axis = figure.add_axes((left, 0.22, 0.28, 0.58))
        shortlist = patient["shortlist"]
        labels = shortlist["candidate_id"].astype(str).tolist()
        values = shortlist["initial_J_pred"].astype(float).to_numpy()
        bars = axis.barh(labels, values, color=str(patient["color"]))
        for bar, alpha in zip(bars, (0.95, 0.72, 0.50)):
            bar.set_alpha(alpha)
        axis.invert_yaxis()
        axis.set_xlim(min(0.968, float(values.min()) - 0.002), max(0.993, float(values.max()) + 0.002))
        axis.set_xlabel("Initial predicted whole-trajectory J")
        axis.set_title(str(patient["display_name"]), color=str(patient["color"]), weight="bold")
        for bar, row in zip(bars, shortlist.itertuples(index=False)):
            axis.text(
                float(row.initial_J_pred) + 0.0003,
                bar.get_y() + bar.get_height() / 2,
                f"({row.hip_delta:+.2f}, {row.knee_delta:+.2f}, {row.phase_delta:+.4f})",
                va="center",
                fontsize=8,
            )
        axis.grid(True, axis="x", color="#E6EBF0")
    figure.text(0.5, 0.14, "21,025 complete candidates screened per patient • only the frozen three may be validated", ha="center", fontsize=13, color=_DARK)
    _draw_timeline(figure, 3)
    figure.canvas.draw()
    image = Image.fromarray(np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]))
    plt.close(figure)
    return image


def _final_frame(data: Mapping[str, Any]) -> Image.Image:
    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    figure.suptitle("FINAL BEST VALIDATED TRAJECTORY — NORMAL VS STIFFNESS", fontsize=22, weight="bold", color=_DARK, y=0.96)
    left = figure.add_axes((0.08, 0.23, 0.38, 0.56))
    patients = data["patients"]
    labels = [str(patient["display_name"]) for patient in patients]
    values = [float(patient["summary"].B2_final_best_validated_J) for patient in patients]
    colors = [str(patient["color"]) for patient in patients]
    bars = left.bar(labels, values, color=colors)
    left.set_ylim(min(values) - 0.006, 1.002)
    left.axhline(1.0, color=_GRAY, ls="--", lw=1.2)
    left.set_ylabel("Best validated whole-trajectory J")
    left.set_title("Mechanical objective after validation")
    for bar, value in zip(bars, values):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.6f}", ha="center", fontsize=10)

    right = figure.add_axes((0.54, 0.20, 0.40, 0.60))
    right.axis("off")
    right.text(0.5, 0.96, "Selected trajectories", ha="center", fontsize=15, weight="bold", color=_DARK)
    for index, patient in enumerate(patients):
        summary = patient["summary"]
        y = 0.77 - index * 0.25
        right.text(0.02, y + 0.08, str(patient["display_name"]), color=str(patient["color"]), fontsize=13, weight="bold")
        right.text(
            0.02,
            y,
            (
                f"α=({summary.best_validated_alpha_hip:+.2f}°, "
                f"{summary.best_validated_alpha_knee:+.2f}°, "
                f"{summary.best_validated_alpha_phase:+.4f})\n"
                f"J={summary.B2_final_best_validated_J:.6f}   regret={summary.global_regret:.6f}"
            ),
            fontsize=10.5,
            color=_DARK,
        )
    right.text(
        0.5,
        0.04,
        "All three select C1 in this frozen cohort.\nTheir force responses, identified parameters, C2/C3 shortlists, and J differ.",
        ha="center",
        fontsize=11,
        color=_RED,
    )
    _draw_timeline(figure, 7)
    figure.canvas.draw()
    image = Image.fromarray(np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]))
    plt.close(figure)
    return image


def generate_identification_to_final_comparison(
    artifact_directory: str | Path = SOURCE_ARTIFACT_DIRECTORY,
    *,
    output_path: str | Path | None = None,
    samples_per_motion: int = DEFAULT_SAMPLES_PER_MOTION,
) -> tuple[Path, dict[str, Any]]:
    if isinstance(samples_per_motion, bool) or int(samples_per_motion) < 3:
        raise ValueError("samples_per_motion must be an integer >= 3")
    samples_per_motion = int(samples_per_motion)
    _style()
    data = _load_animation_data(artifact_directory)
    indices = np.rint(np.linspace(0, 400, samples_per_motion)).astype(int)
    frames: list[Image.Image] = [_title_frame(data)]
    durations: list[int] = [2200]

    for trial_id, source_key, active_step in ((1, "excitation_1", 0), (2, "excitation_2", 1)):
        candidate_id = str(data["patients"][0][source_key]["candidate_id"].iloc[0])
        for position, sample_index in enumerate(indices):
            frames.append(
                _motion_frame(
                    data,
                    source_key=source_key,
                    sample_index=int(sample_index),
                    title=f"INITIAL IDENTIFICATION EXCITATION {trial_id}",
                    subtitle=f"{candidate_id} • same joint excitation, patient-specific force response",
                    active_step=active_step,
                )
            )
            durations.append(600 if position == 0 else 400 if position == len(indices) - 1 else 70)

    frames.append(_parameter_frame(data))
    durations.append(2500)
    frames.append(_screening_frame(data))
    durations.append(2800)

    for round_number in (1, 2, 3):
        for position, sample_index in enumerate(indices):
            frames.append(
                _motion_frame(
                    data,
                    source_key="",
                    sample_index=int(sample_index),
                    title=f"FINITE VALIDATION ROUND {round_number}: COMPLETE C{round_number} TRAJECTORY",
                    subtitle="Observe whole-trajectory J, refit each subject model, rerank only remaining frozen candidates",
                    active_step=3 + round_number,
                    validation_round=round_number,
                )
            )
            durations.append(600 if position == 0 else 400 if position == len(indices) - 1 else 70)

    frames.append(_final_frame(data))
    durations.append(3200)
    output = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_DIRECTORY / GIF_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    metadata = {
        "visualization_id": "NORMAL_VS_STIFFNESS_IDENTIFICATION_TO_FINAL_V1",
        "source_formal_manifest_sha256": _sha256(Path(artifact_directory) / "FINAL_METHOD_MANIFEST_V1.json"),
        "patients": [
            {
                "display_name": patient["display_name"],
                "subject_id": patient["subject_id"],
                "case_id": patient["case_id"],
                "identified_parameters": patient["parameters"],
                "executed_identification_candidate_ids": [
                    str(patient["excitation_1"]["candidate_id"].iloc[0]),
                    str(patient["excitation_2"]["candidate_id"].iloc[0]),
                ],
                "frozen_shortlist_ids": patient["shortlist"]["trajectory_id"].astype(str).tolist(),
                "best_validated_trajectory_id": str(patient["summary"].best_validated_trajectory_id),
                "best_validated_J": float(patient["summary"].B2_final_best_validated_J),
            }
            for patient in data["patients"]
        ],
        "stage_order": list(_TIMELINE_LABELS),
        "samples_per_motion": samples_per_motion,
        "frame_count": len(frames),
        "duration_ms": int(sum(durations)),
        "resolution_px": [1600, 900],
        "loop": True,
        "theta_shank_definition": "q_hip - q_knee",
        "initial_identification_reconstruction_matches_formal_parameters": True,
        "candidate_trajectory_sha_verified": True,
        "truth_used_to_create_or_rank_shortlist": False,
        "held_out_final_test_read": False,
        "robot_connected": False,
        "gif_sha256": _sha256(output),
        "generator_source_sha256": _sha256(Path(__file__)),
    }
    return output, metadata


def _guide(metadata: Mapping[str, Any]) -> str:
    return f"""# Normal versus stiffness: identification to final trajectory

`{GIF_NAME}` compares Normal, Hip stiffness, and Knee stiffness virtual subjects
through two frozen diagnostic identification excitations, five-parameter model
reconstruction, 21,025-point model screening, C1–C3 complete-trajectory
validation, and final measured/virtual-truth selection.

The same excitation motion produces different force responses.  The identified
equivalent parameters then produce patient-specific C2/C3 shortlists and J
values.  In this frozen matched-linear cohort, all three ultimately select the
same C1 alpha; the animation reports that result rather than inventing visual
personalization.

- Frames: {metadata['frame_count']}
- Resolution: 1600×900
- Duration: {metadata['duration_ms'] / 1000:.2f} s, looping

Offline simulation only.  No held-out final test, human observation, clinical
claim, or robot connection is involved.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=SOURCE_ARTIFACT_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--samples-per-motion", type=int, default=DEFAULT_SAMPLES_PER_MOTION)
    arguments = parser.parse_args(argv)
    output, metadata = generate_identification_to_final_comparison(
        arguments.artifact_directory,
        output_path=arguments.output_directory / GIF_NAME,
        samples_per_motion=int(arguments.samples_per_motion),
    )
    _atomic_bytes(
        arguments.output_directory / METADATA_NAME,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_bytes(arguments.output_directory / GUIDE_NAME, _guide(metadata).encode("utf-8"))
    print(json.dumps({"gif": str(output), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
