"""Only the three figures required by the five-leg benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _by_leg(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in payload["landscape_rows"]:
        grouped.setdefault(row["leg_id"], []).append(row)
    return grouped


def _grid(rows: list[dict[str, Any]], field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta_flex = np.asarray(sorted({row["beta_flex"] for row in rows}), dtype=float)
    beta_extend = np.asarray(sorted({row["beta_extend"] for row in rows}), dtype=float)
    values = np.full((len(beta_flex), len(beta_extend)), np.nan)
    flex_index = {value: index for index, value in enumerate(beta_flex)}
    extend_index = {value: index for index, value in enumerate(beta_extend)}
    for row in rows:
        values[flex_index[row["beta_flex"]], extend_index[row["beta_extend"]]] = row[
            field
        ]
    return beta_flex, beta_extend, values


def plot_truth_landscapes(payload: dict[str, Any], output_path: Path) -> None:
    grouped = _by_leg(payload)
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), constrained_layout=True)
    image = None
    for axis, (leg_id, rows) in zip(axes.flat, grouped.items()):
        beta_flex, beta_extend, values = _grid(rows, "endpoint_value_nm")
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=(beta_flex[0], beta_flex[-1], beta_extend[0], beta_extend[-1]),
            aspect="equal",
            cmap="viridis",
        )
        best = min(rows, key=lambda row: (row["endpoint_value_nm"], row["candidate_index"]))
        axis.scatter(best["beta_flex"], best["beta_extend"], marker="*", s=90, c="white", edgecolors="black")
        axis.set_title(leg_id.replace("LEG_", "Leg ").replace("_", " ").title(), fontsize=9)
        axis.set_xlabel(r"$\beta_{flex}$")
        axis.set_ylabel(r"$\beta_{extend}$")
        figure.colorbar(image, ax=axis, shrink=0.76, label="J (N m)")
    axes.flat[-1].axis("off")
    figure.suptitle("Five frozen MuJoCo truth landscapes", fontsize=13)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_truth_vs_gray_box(payload: dict[str, Any], output_path: Path) -> None:
    grouped = _by_leg(payload)
    quality = {
        row["leg_id"]: row for row in payload["gray_box_prediction_quality"]
    }
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), constrained_layout=True)
    for axis, (leg_id, rows) in zip(axes.flat, grouped.items()):
        truth = np.asarray([row["endpoint_value_nm"] for row in rows], dtype=float)
        predicted = np.asarray(
            [row["gray_box_predicted_J_nm"] for row in rows], dtype=float
        )
        lower = min(float(np.min(truth)), float(np.min(predicted)))
        upper = max(float(np.max(truth)), float(np.max(predicted)))
        axis.scatter(truth, predicted, s=8, alpha=0.45, color="#31688e")
        axis.plot([lower, upper], [lower, upper], "--", color="#444444", lw=1)
        axis.set_title(
            f"{leg_id}\nSpearman={quality[leg_id]['spearman_rank_correlation']:.3f}",
            fontsize=9,
        )
        axis.set_xlabel("MuJoCo truth J (N m)")
        axis.set_ylabel("Gray-box predicted J (N m)")
    axes.flat[-1].axis("off")
    figure.suptitle("Reference-fitted gray-box versus MuJoCo truth", fontsize=13)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_oracle_near_oracle(payload: dict[str, Any], output_path: Path) -> None:
    grouped = _by_leg(payload)
    minimax = payload["personalization_necessity_analysis"]["minimax_common_candidate"]
    figure, axes = plt.subplots(1, 5, figsize=(15.0, 3.4), constrained_layout=True)
    for axis, (leg_id, rows) in zip(axes, grouped.items()):
        oracle = min(rows, key=lambda row: (row["endpoint_value_nm"], row["candidate_index"]))
        oracle_value = float(oracle["endpoint_value_nm"])
        beta_flex = np.asarray([row["beta_flex"] for row in rows], dtype=float)
        beta_extend = np.asarray([row["beta_extend"] for row in rows], dtype=float)
        relative = np.asarray(
            [float(row["endpoint_value_nm"]) / oracle_value - 1.0 for row in rows]
        )
        axis.scatter(beta_flex, beta_extend, s=6, c="#dddddd", label="other")
        near_5 = relative <= 0.05
        near_1 = relative <= 0.01
        axis.scatter(beta_flex[near_5], beta_extend[near_5], s=12, c="#2a788e", label="within 5%")
        axis.scatter(beta_flex[near_1], beta_extend[near_1], s=17, c="#fca636", label="within 1%")
        axis.scatter(oracle["beta_flex"], oracle["beta_extend"], s=90, marker="*", c="#d73027", edgecolors="black", label="oracle")
        axis.scatter(minimax["beta"][0], minimax["beta"][1], s=50, marker="D", facecolors="none", edgecolors="black", label="minimax common")
        axis.set_title(leg_id, fontsize=8)
        axis.set_xlabel(r"$\beta_{flex}$")
        axis.set_ylabel(r"$\beta_{extend}$")
        axis.set_aspect("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    figure.suptitle("Oracle and near-oracle V3 candidates", fontsize=13)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_required_figures(payload: dict[str, Any], output_dir: Path) -> None:
    plot_truth_landscapes(payload, output_dir / "figure_1_truth_landscapes.png")
    plot_truth_vs_gray_box(payload, output_dir / "figure_2_truth_vs_gray_box.png")
    plot_oracle_near_oracle(payload, output_dir / "figure_3_oracle_near_oracle.png")


__all__ = [
    "plot_oracle_near_oracle",
    "plot_truth_landscapes",
    "plot_truth_vs_gray_box",
    "write_required_figures",
]
