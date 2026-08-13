"""遍历髋膝关节空间并生成二维可达空间图谱。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    angle_step_deg,
    hip_range_deg,
    knee_range_deg,
    singularity_threshold,
    workspace_data_dir,
)
from .kinematics import forward_kinematics
from .jacobian import jacobian_diagnostics
from .formal_protocol import (
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)


def _inclusive_angle_values(
    angle_range_deg: tuple[float, float],
    step_deg: float,
) -> np.ndarray:
    if not np.isfinite(step_deg) or step_deg <= 0.0:
        raise ValueError("step_deg must be finite and positive.")
    start, stop = angle_range_deg
    interval_count = int(np.floor((stop - start) / step_deg))
    values = start + np.arange(interval_count + 1, dtype=float) * step_deg
    if values[-1] < stop - 1e-12:
        values = np.append(values, stop)
    else:
        values[-1] = stop
    return values


def build_workspace_atlas(step_deg: float = angle_step_deg) -> pd.DataFrame:
    """生成包含全部关节网格点及有效性标记的 DataFrame。"""

    hip_values_deg = _inclusive_angle_values(hip_range_deg, step_deg)
    knee_values_deg = _inclusive_angle_values(knee_range_deg, step_deg)
    q_hip_deg, q_knee_deg = np.meshgrid(
        hip_values_deg,
        knee_values_deg,
        indexing="ij",
    )
    q_hip_rad = np.deg2rad(q_hip_deg)
    q_knee_rad = np.deg2rad(q_knee_deg)

    x_knee, z_knee, x_pull, z_pull = forward_kinematics(
        q_hip_rad,
        q_knee_rad,
        L1,
        L2,
    )
    finite = (
        np.isfinite(x_knee)
        & np.isfinite(z_knee)
        & np.isfinite(x_pull)
        & np.isfinite(z_pull)
    )
    reachable = finite & (x_pull >= 0.0) & (z_pull >= 0.0)
    near_singular = np.abs(np.sin(q_knee_rad)) < singularity_threshold
    jacobian = jacobian_diagnostics(q_hip_rad, q_knee_rad, L1, L2)
    jacobian_near_singular = np.asarray(jacobian.near_singular, dtype=bool)

    return pd.DataFrame(
        {
            "rom_protocol_version": ROM_PROTOCOL_VERSION,
            "theta_shank_definition": THETA_SHANK_DEFINITION,
            "q_hip_rad": q_hip_rad.ravel(),
            "q_knee_rad": q_knee_rad.ravel(),
            "theta_shank_rad": (q_hip_rad - q_knee_rad).ravel(),
            "q_hip_deg": q_hip_deg.ravel(),
            "q_knee_deg": q_knee_deg.ravel(),
            "x_knee": x_knee.ravel(),
            "z_knee": z_knee.ravel(),
            "x_pull": x_pull.ravel(),
            "z_pull": z_pull.ravel(),
            "reachable": reachable.ravel(),
            "near_singular": near_singular.ravel(),
            "jacobian_determinant": np.asarray(jacobian.determinant).ravel(),
            "jacobian_condition_number": np.asarray(
                jacobian.condition_number
            ).ravel(),
            "jacobian_near_singular": jacobian_near_singular.ravel(),
            "jacobian_mapping_valid": (~jacobian_near_singular).ravel(),
        }
    )


def save_workspace_atlas(
    atlas: pd.DataFrame,
    output_dir: str | Path = workspace_data_dir,
) -> tuple[Path, Path]:
    """将图谱保存为 CSV 和不依赖 pickle 的结构化 NPY。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "workspace_atlas.csv"
    npy_path = output_path / "workspace_atlas.npy"
    atlas.to_csv(csv_path, index=False)
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for column in atlas.columns:
        values = atlas[column].to_numpy()
        if values.dtype == object:
            strings = atlas[column].astype(str).to_numpy()
            maximum_length = max(1, max(map(len, strings)))
            values = strings.astype(f"<U{maximum_length}")
        arrays.append(values)
        names.append(str(column))
    np.save(
        npy_path,
        np.rec.fromarrays(arrays, names=names),
        allow_pickle=False,
    )
    metadata = {
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "angle_step_deg": float(
            np.min(np.diff(np.sort(atlas["q_hip_deg"].unique())))
        ),
        "sample_count": int(len(atlas)),
        "reachable_sample_count": int(atlas["reachable"].astype(bool).sum()),
        "jacobian_mapping_valid_sample_count": int(
            atlas["jacobian_mapping_valid"].astype(bool).sum()
        ),
        "knee_above_legacy_upper_bound_sample_count": int(
            atlas["q_knee_deg"].gt(FORMAL_KNEE_ROM_DEG[1] - 15.0).sum()
        ),
        "legacy_workspace_overwritten": False,
        "real_robot_safety_thresholds_modified": False,
    }
    (output_path / "workspace_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, npy_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-deg",
        type=float,
        default=angle_step_deg,
        help="关节网格角度步长，默认 1 deg。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_data_dir,
        help="CSV、NPY 和图片输出目录。",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="仅保存图谱数据，不生成图片。",
    )
    args = parser.parse_args()

    atlas = build_workspace_atlas(args.step_deg)
    csv_path, npy_path = save_workspace_atlas(atlas, args.output_dir)
    reachable_count = int(atlas["reachable"].sum())
    print(f"joint-space samples: {len(atlas)}")
    print(f"reachable samples: {reachable_count}")
    print(f"filtered samples: {len(atlas) - reachable_count}")
    print(f"CSV: {csv_path}")
    print(f"NPY: {npy_path}")

    if not args.no_plots:
        from .visualize import generate_workspace_plots

        figure_paths = generate_workspace_plots(atlas, args.output_dir)
        for figure_path in figure_paths:
            print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
