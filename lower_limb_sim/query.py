"""查询牵引目标在离散二维工作空间图谱中的最近点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import query_max_distance_m, workspace_csv_path


def _load_atlas(atlas_path: str | Path) -> pd.DataFrame:
    path = Path(atlas_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Workspace atlas not found: {path}. "
            "Run `python -m lower_limb_sim.workspace_atlas` first."
        )
    return pd.read_csv(path)


def _reachable_mask(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"}).to_numpy()


def query_position(
    x: float,
    z: float,
    atlas: pd.DataFrame | None = None,
    atlas_path: str | Path = workspace_csv_path,
    max_distance: float = query_max_distance_m,
) -> dict[str, float | bool]:
    """返回离目标最近的有效图谱点及其髋膝角。

    ``q_hip`` 和 ``q_knee`` 为 rad。即使误差超过 ``max_distance``，结果仍
    给出最近图谱点，但 ``reachable`` 为 False。
    """

    if not np.isfinite(x) or not np.isfinite(z):
        raise ValueError("x and z must be finite.")
    if not np.isfinite(max_distance) or max_distance < 0.0:
        raise ValueError("max_distance must be finite and non-negative.")

    workspace = _load_atlas(atlas_path) if atlas is None else atlas
    required_columns = {
        "x_pull",
        "z_pull",
        "q_hip_rad",
        "q_knee_rad",
        "q_hip_deg",
        "q_knee_deg",
        "reachable",
    }
    missing = required_columns.difference(workspace.columns)
    if missing:
        raise ValueError(f"atlas is missing columns: {sorted(missing)}")

    candidates = workspace.loc[_reachable_mask(workspace["reachable"])]
    if candidates.empty:
        raise ValueError("atlas contains no reachable posture.")

    dx = candidates["x_pull"].to_numpy(dtype=float) - float(x)
    dz = candidates["z_pull"].to_numpy(dtype=float) - float(z)
    distances = np.hypot(dx, dz)
    nearest_position = int(np.argmin(distances))
    nearest = candidates.iloc[nearest_position]
    error = float(distances[nearest_position])

    return {
        "reachable": error <= max_distance,
        "x_pull": float(nearest["x_pull"]),
        "z_pull": float(nearest["z_pull"]),
        "q_hip": float(nearest["q_hip_rad"]),
        "q_knee": float(nearest["q_knee_rad"]),
        "q_hip_deg": float(nearest["q_hip_deg"]),
        "q_knee_deg": float(nearest["q_knee_deg"]),
        "distance_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, help="目标 x 坐标，单位 m。")
    parser.add_argument("z", type=float, help="目标 z 坐标，单位 m。")
    parser.add_argument(
        "--atlas",
        type=Path,
        default=workspace_csv_path,
        help="workspace_atlas.csv 路径。",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=query_max_distance_m,
        help="最大允许最近点误差，单位 m。",
    )
    args = parser.parse_args()
    result = query_position(
        args.x,
        args.z,
        atlas_path=args.atlas,
        max_distance=args.max_distance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
