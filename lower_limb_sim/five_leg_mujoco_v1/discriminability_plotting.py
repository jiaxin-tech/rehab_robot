"""The five required figures for the V3 discriminability audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .discriminability_audit import COMPONENT_NAMES, SELECTED_BETAS


COLORS = ("#3b4cc0", "#1fa187", "#d73027", "#fdae61", "#7b3294")


def _group_rows(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in payload["response_rows"]:
        grouped.setdefault(row["leg_id"], []).append(row)
    return grouped


def plot_selected_joint_paths(payload: dict[str, Any], path: Path) -> None:
    domains = payload["_plot_data"]["domains"]
    figure, axes = plt.subplots(1, 5, figsize=(15.5, 3.5), constrained_layout=True)
    labels = {
        (0.0, 0.0): "reference",
        (-0.03, 0.03): "oracle boundary",
        (0.03, -0.03): "opposite boundary",
        (-0.03, -0.03): "corner --",
        (0.03, 0.03): "corner ++",
    }
    for axis, (leg_id, domain) in zip(axes, domains.items()):
        for beta, color in zip(SELECTED_BETAS, COLORS):
            candidate = next(item for item in domain if item.beta == beta)
            q_deg = np.degrees(candidate.trajectory.q)
            axis.plot(
                q_deg[:, 0],
                q_deg[:, 1],
                color=color,
                linewidth=1.4,
                label=labels[beta],
            )
        axis.set_title(leg_id, fontsize=8)
        axis.set_xlabel("hip angle (deg)")
        axis.set_ylabel("knee angle (deg)")
    handles, labels_text = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_text, loc="outside lower center", ncol=5, fontsize=8)
    figure.suptitle("Selected V3 joint-space coordination paths", fontsize=13)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_kinematic_spread(payload: dict[str, Any], path: Path) -> None:
    curves = payload["_plot_data"]["trajectory_curves"]
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 6.7), constrained_layout=True)
    scale_and_unit = {
        "q_spread": (180.0 / np.pi, "angle spread (deg)"),
        "dq_spread": (180.0 / np.pi, "velocity spread (deg/s)"),
        "ddq_spread": (180.0 / np.pi, "acceleration spread (deg/s2)"),
    }
    for column, name in enumerate(("q_spread", "dq_spread", "ddq_spread")):
        scale, ylabel = scale_and_unit[name]
        for row, joint_name in enumerate(("hip", "knee")):
            axis = axes[row, column]
            for leg_index, (leg_id, item) in enumerate(curves.items()):
                axis.plot(
                    item["time_s"],
                    scale * item[name][:, row],
                    linewidth=1.1,
                    color=COLORS[leg_index],
                    label=leg_id,
                )
            axis.set_title(f"{joint_name}: {name.replace('_spread', '')}")
            axis.set_xlabel("time (s)")
            axis.set_ylabel(ylabel)
    handles, legend_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=5, fontsize=8)
    figure.suptitle("Across-candidate q/dq/ddq spread versus time", fontsize=13)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_normalized_landscapes(payload: dict[str, Any], path: Path) -> None:
    fields = payload["_plot_data"]["gradient_fields"]
    figure, axes = plt.subplots(1, 5, figsize=(15.5, 3.35), constrained_layout=True)
    image = None
    for axis, (leg_id, item) in zip(axes, fields.items()):
        flex = item["beta_flex"]
        extend = item["beta_extend"]
        image = axis.imshow(
            item["normalized"].T,
            origin="lower",
            extent=(flex[0], flex[-1], extend[0], extend[-1]),
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axis.scatter(-0.03, 0.03, s=65, marker="*", c="white", edgecolors="black")
        axis.set_title(leg_id, fontsize=8)
        axis.set_xlabel(r"$\beta_{flex}$")
        axis.set_ylabel(r"$\beta_{extend}$")
    figure.colorbar(image, ax=axes, shrink=0.78, label="within-leg normalized J")
    figure.suptitle("Five normalized primary objective landscapes", fontsize=13)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_component_sensitivity(payload: dict[str, Any], path: Path) -> None:
    grouped = _group_rows(payload)
    figure, axes = plt.subplots(2, 4, figsize=(14.5, 7.0), constrained_layout=True)
    image = None
    for axis, component in zip(axes.flat, COMPONENT_NAMES):
        normalized_grids = []
        flex = extend = None
        for rows in grouped.values():
            flex = np.asarray(sorted({row["beta_flex"] for row in rows}), dtype=float)
            extend = np.asarray(sorted({row["beta_extend"] for row in rows}), dtype=float)
            grid = np.empty((len(flex), len(extend)), dtype=float)
            fi = {value: index for index, value in enumerate(flex)}
            ei = {value: index for index, value in enumerate(extend)}
            for row in rows:
                grid[fi[row["beta_flex"]], ei[row["beta_extend"]]] = row[
                    f"{component}_rms_nm"
                ]
            reference = grid[np.flatnonzero(np.isclose(flex, 0.0))[0], np.flatnonzero(np.isclose(extend, 0.0))[0]]
            if reference > 1.0e-12:
                normalized_grids.append(100.0 * (grid / reference - 1.0))
            elif float(np.max(np.abs(grid))) > 1.0e-12:
                normalized_grids.append(100.0 * grid / float(np.max(np.abs(grid))))
        mean_change = np.mean(normalized_grids, axis=0)
        maximum = max(float(np.max(np.abs(mean_change))), 1.0e-12)
        image = axis.imshow(
            mean_change.T,
            origin="lower",
            extent=(flex[0], flex[-1], extend[0], extend[-1]),
            aspect="equal",
            cmap="coolwarm",
            vmin=-maximum,
            vmax=maximum,
        )
        axis.set_title(component.replace("_", " ").title())
        axis.set_xlabel(r"$\beta_{flex}$")
        axis.set_ylabel(r"$\beta_{extend}$")
        figure.colorbar(
            image,
            ax=axis,
            shrink=0.72,
            label="mean normalized change (%)",
        )
    for axis in axes.flat[len(COMPONENT_NAMES) :]:
        axis.axis("off")
    figure.suptitle("Mechanical component RMS sensitivity across beta", fontsize=13)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_branch_and_joint_response(payload: dict[str, Any], path: Path) -> None:
    grouped = _group_rows(payload)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), constrained_layout=True)
    for leg_index, (leg_id, rows) in enumerate(grouped.items()):
        reference = next(
            row for row in rows if row["beta_flex"] == 0.0 and row["beta_extend"] == 0.0
        )
        flexion = np.asarray([row["flexion_rms_nm"] for row in rows]) / reference[
            "flexion_rms_nm"
        ]
        extension = np.asarray([row["extension_rms_nm"] for row in rows]) / reference[
            "extension_rms_nm"
        ]
        hip = np.asarray([row["hip_rms_nm"] for row in rows]) / reference["hip_rms_nm"]
        knee = np.asarray([row["knee_rms_nm"] for row in rows]) / reference[
            "knee_rms_nm"
        ]
        axes[0].scatter(flexion, extension, s=7, alpha=0.35, color=COLORS[leg_index], label=leg_id)
        axes[1].scatter(hip, knee, s=7, alpha=0.35, color=COLORS[leg_index], label=leg_id)
    axes[0].set_xlabel("normalized flexion RMS")
    axes[0].set_ylabel("normalized extension RMS")
    axes[0].set_title("Branch response")
    axes[1].set_xlabel("normalized hip RMS")
    axes[1].set_ylabel("normalized knee RMS")
    axes[1].set_title("Joint response")
    for axis in axes:
        axis.axvline(1.0, color="#777777", linewidth=0.7)
        axis.axhline(1.0, color="#777777", linewidth=0.7)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    figure.suptitle("Flexion/extension and hip/knee candidate response", fontsize=13)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_required_figures(payload: dict[str, Any], output_dir: Path) -> None:
    plot_selected_joint_paths(payload, output_dir / "figure_1_selected_joint_space_paths.png")
    plot_kinematic_spread(payload, output_dir / "figure_2_q_dq_ddq_spread_vs_time.png")
    plot_normalized_landscapes(payload, output_dir / "figure_3_normalized_landscapes.png")
    plot_component_sensitivity(payload, output_dir / "figure_4_component_sensitivity.png")
    plot_branch_and_joint_response(payload, output_dir / "figure_5_branch_joint_response.png")


__all__ = ["write_required_figures"]
