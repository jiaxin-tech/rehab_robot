"""二维下肢可达空间和样例姿态可视化。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import workspace_csv_path, workspace_data_dir


def _plot_colored_workspace(
    reachable_atlas: pd.DataFrame,
    color_column: str,
    colorbar_label: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        reachable_atlas["x_pull"],
        reachable_atlas["z_pull"],
        c=reachable_atlas[color_column],
        cmap="viridis",
        s=8,
        alpha=0.8,
        linewidths=0,
    )
    figure.colorbar(scatter, ax=axis, label=colorbar_label)
    axis.axhline(0.0, color="black", linewidth=1.2, label="bed: z = 0")
    axis.set_xlabel("x_pull (m)")
    axis.set_ylabel("z_pull (m)")
    axis.set_title(title)
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_sample_postures(
    reachable_atlas: pd.DataFrame,
    output_path: Path,
    sample_count: int,
    random_seed: int,
) -> None:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    count = min(sample_count, len(reachable_atlas))
    samples = reachable_atlas.sample(n=count, random_state=random_seed)

    figure, axis = plt.subplots(figsize=(9, 6))
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, count))
    for color, (_, row) in zip(colors, samples.iterrows()):
        x_points = [0.0, row["x_knee"], row["x_pull"]]
        z_points = [0.0, row["z_knee"], row["z_pull"]]
        axis.plot(
            x_points,
            z_points,
            "-",
            color=color,
            linewidth=1.6,
            alpha=0.85,
        )
        axis.scatter(
            [row["x_knee"]],
            [row["z_knee"]],
            marker="o",
            s=30,
            color=color,
            alpha=0.9,
        )
        axis.scatter(
            [row["x_pull"]],
            [row["z_pull"]],
            marker="x",
            s=38,
            color=color,
            alpha=0.9,
        )

    all_x = np.concatenate(
        (
            np.array([0.0]),
            samples["x_knee"].to_numpy(),
            samples["x_pull"].to_numpy(),
        )
    )
    axis.axhline(0.0, color="black", linewidth=1.5, label="bed: z = 0")
    axis.scatter([0.0], [0.0], marker="s", s=55, color="black", label="hip")
    axis.scatter([], [], marker="o", s=30, color="gray", label="knee")
    axis.scatter([], [], marker="x", s=38, color="gray", label="pull point")
    axis.set_xlim(min(-0.05, float(all_x.min()) - 0.05), float(all_x.max()) + 0.05)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_title(f"{count} randomly selected reachable postures")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_workspace_plots(
    atlas: pd.DataFrame,
    output_dir: str | Path = workspace_data_dir,
    sample_count: int = 10,
    random_seed: int = 42,
) -> tuple[Path, Path, Path]:
    """生成两张工作空间角度图和一张随机姿态图。"""

    required_columns = {
        "x_knee",
        "z_knee",
        "x_pull",
        "z_pull",
        "q_hip_deg",
        "q_knee_deg",
        "reachable",
    }
    missing = required_columns.difference(atlas.columns)
    if missing:
        raise ValueError(f"atlas is missing columns: {sorted(missing)}")

    reachable_atlas = atlas.loc[atlas["reachable"].astype(bool)].copy()
    if reachable_atlas.empty:
        raise ValueError("atlas contains no reachable posture.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hip_path = output_path / "workspace_hip_angle.png"
    knee_path = output_path / "workspace_knee_angle.png"
    sample_path = output_path / "sample_postures.png"

    _plot_colored_workspace(
        reachable_atlas,
        "q_hip_deg",
        "q_hip (deg)",
        "Reachable workspace colored by hip flexion",
        hip_path,
    )
    _plot_colored_workspace(
        reachable_atlas,
        "q_knee_deg",
        "q_knee (deg)",
        "Reachable workspace colored by knee flexion",
        knee_path,
    )
    _plot_sample_postures(
        reachable_atlas,
        sample_path,
        sample_count,
        random_seed,
    )
    return hip_path, knee_path, sample_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        type=Path,
        default=workspace_csv_path,
        help="workspace_atlas.csv 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_data_dir,
        help="图片输出目录。",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    atlas = pd.read_csv(args.atlas)
    for path in generate_workspace_plots(
        atlas,
        args.output_dir,
        random_seed=args.seed,
    ):
        print(path)


if __name__ == "__main__":
    main()
