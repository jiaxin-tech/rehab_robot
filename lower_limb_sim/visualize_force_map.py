"""准静态力地图与虚拟受试者比较可视化。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import force_map_data_dir
from .virtual_subject import VIRTUAL_SUBJECTS


def _valid_mask(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
        .to_numpy()
    )


def _color_limits(values: np.ndarray, sequential: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite values are available for plotting.")
    if sequential:
        lower = max(0.0, float(np.percentile(finite, 1.0)))
        upper = float(np.percentile(finite, 99.0))
        if upper <= lower:
            upper = lower + 1.0
        return lower, upper

    bound = float(np.percentile(np.abs(finite), 99.0))
    if bound <= 0.0:
        bound = 1.0
    return -bound, bound


def _scatter_map(
    valid_map: pd.DataFrame,
    value_column: str,
    colorbar_label: str,
    title: str,
    output_path: Path,
    sequential: bool,
) -> None:
    values = valid_map[value_column].to_numpy(dtype=float)
    vmin, vmax = _color_limits(values, sequential)
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        valid_map["x_pull"],
        valid_map["z_pull"],
        c=values,
        cmap="viridis" if sequential else "coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=8,
        alpha=0.82,
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


def _force_vector_field(
    valid_map: pd.DataFrame,
    subject_id: str,
    output_path: Path,
    max_arrows: int = 500,
) -> None:
    step = max(1, int(np.ceil(len(valid_map) / max_arrows)))
    sampled = valid_map.iloc[::step].copy()
    fx = sampled["fx_robot_on_leg_n"].to_numpy(dtype=float)
    fz = sampled["fz_robot_on_leg_n"].to_numpy(dtype=float)
    magnitude = sampled["force_magnitude_n"].to_numpy(dtype=float)
    display_magnitude = np.maximum(magnitude, np.finfo(float).eps)
    arrow_length_m = 0.025
    u = arrow_length_m * fx / display_magnitude
    v = arrow_length_m * fz / display_magnitude
    vmin, vmax = _color_limits(magnitude, sequential=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    quiver = axis.quiver(
        sampled["x_pull"],
        sampled["z_pull"],
        u,
        v,
        np.clip(magnitude, vmin, vmax),
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
    )
    quiver.set_clim(vmin, vmax)
    figure.colorbar(quiver, ax=axis, label="force magnitude (N)")
    axis.axhline(0.0, color="black", linewidth=1.2, label="bed: z = 0")
    axis.set_xlabel("x_pull (m)")
    axis.set_ylabel("z_pull (m)")
    axis.set_title(
        f"Robot-on-leg force direction field — subject: {subject_id}"
    )
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_force_map_plots(
    force_map: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """为一名虚拟受试者生成五张标量图和一张力方向图。"""

    required = {
        "subject_id",
        "x_pull",
        "z_pull",
        "force_mapping_valid",
        "force_magnitude_n",
        "fx_robot_on_leg_n",
        "fz_robot_on_leg_n",
        "tau_total_hip_nm",
        "tau_total_knee_nm",
    }
    missing = required.difference(force_map.columns)
    if missing:
        raise ValueError(f"force map is missing columns: {sorted(missing)}")
    subject_ids = force_map["subject_id"].astype(str).unique()
    if len(subject_ids) != 1:
        raise ValueError("force map must contain exactly one subject_id.")
    subject_id = subject_ids[0]
    valid_map = force_map.loc[
        _valid_mask(force_map["force_mapping_valid"])
    ].copy()
    if valid_map.empty:
        raise ValueError("force map contains no valid force mapping.")

    destination = (
        Path(output_dir)
        if output_dir is not None
        else force_map_data_dir / subject_id
    )
    destination.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        (
            "force_magnitude_n",
            "force magnitude (N)",
            "Force magnitude",
            "force_magnitude_map.png",
            True,
        ),
        (
            "fx_robot_on_leg_n",
            "Fx robot on leg (N)",
            "Robot-on-leg force Fx",
            "fx_map.png",
            False,
        ),
        (
            "fz_robot_on_leg_n",
            "Fz robot on leg (N)",
            "Robot-on-leg force Fz",
            "fz_map.png",
            False,
        ),
        (
            "tau_total_hip_nm",
            "hip torque (N·m)",
            "Total hip quasi-static torque",
            "hip_torque_map.png",
            False,
        ),
        (
            "tau_total_knee_nm",
            "knee torque (N·m)",
            "Total knee quasi-static torque",
            "knee_torque_map.png",
            False,
        ),
    )

    paths: list[Path] = []
    for column, label, title, filename, sequential in plot_specs:
        path = destination / filename
        _scatter_map(
            valid_map,
            column,
            label,
            f"{title} — subject: {subject_id}",
            path,
            sequential,
        )
        paths.append(path)

    vector_path = destination / "force_vector_field.png"
    _force_vector_field(valid_map, subject_id, vector_path)
    paths.append(vector_path)
    return tuple(paths)


def plot_virtual_subject_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """绘制共同姿态下四名虚拟受试者的力矩和力幅值比较。"""

    required = {
        "subject_id",
        "q_hip_deg",
        "q_knee_deg",
        "tau_total_hip_nm",
        "tau_total_knee_nm",
        "force_magnitude_n",
    }
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"comparison is missing columns: {sorted(missing)}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    metrics = (
        ("tau_total_hip_nm", "total hip torque (N·m)"),
        ("tau_total_knee_nm", "total knee torque (N·m)"),
        ("force_magnitude_n", "force magnitude (N)"),
    )
    posture_labels = (
        comparison[["q_hip_deg", "q_knee_deg"]]
        .drop_duplicates()
        .sort_values(["q_hip_deg", "q_knee_deg"])
    )
    labels = [
        f"H{row.q_hip_deg:.0f}/K{row.q_knee_deg:.0f}"
        for row in posture_labels.itertuples()
    ]
    x = np.arange(len(labels))

    for subject_id in VIRTUAL_SUBJECTS:
        subject_rows = comparison.loc[
            comparison["subject_id"] == subject_id
        ].sort_values(["q_hip_deg", "q_knee_deg"])
        for axis, (column, ylabel) in zip(axes, metrics):
            axis.plot(
                x,
                subject_rows[column],
                "-o",
                linewidth=1.7,
                markersize=5,
                label=subject_id,
            )
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)

    axes[0].set_title("Virtual-subject quasi-static comparison")
    axes[0].legend(ncol=2)
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("common posture: hip/knee (deg)")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_id", choices=tuple(VIRTUAL_SUBJECTS))
    parser.add_argument(
        "--force-map",
        type=Path,
        default=None,
        help="输入 CSV；默认读取 data/force_maps/force_map_<subject>.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="图片目录；默认使用 data/force_maps/<subject>/。",
    )
    args = parser.parse_args()
    input_path = args.force_map or (
        force_map_data_dir / f"force_map_{args.subject_id}.csv"
    )
    force_map = pd.read_csv(input_path)
    for path in generate_force_map_plots(force_map, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
