"""Offline PINN identification from valid, base-frame collection snapshots."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import settings
from models.pinn import run_pinn, run_tangential_pinn
from utils.logger import get_logger


logger = get_logger("TrainPINN")
VERIFIED_BASE_WRENCH_TRANSFORMS = {
    "sdk_base",
    "rotation_only_verified_by_project_procedure",
}


def _finite_row_values(row: dict[str, str], names: list[str]) -> list[float] | None:
    values: list[float] = []
    for name in names:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def _valid_rows(fpath: str) -> list[dict[str, str]]:
    with open(fpath, encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"空 episode: {fpath}")
    valid: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=2):
        if int(row.get("schema_version", -1)) != settings.DATA_SCHEMA_VERSION:
            raise ValueError(f"schema 不兼容: {fpath}:{index}")
        if row.get("frame") != settings.CONTROL_FRAME:
            raise ValueError(f"坐标系不兼容: {fpath}:{index}")
        if row.get("units") != settings.SI_UNITS:
            raise ValueError(f"单位元数据不兼容: {fpath}:{index}")
        if row.get("valid") != "1" or row.get("force_estimate_valid") != "1":
            continue
        if row.get("raw_force_frame") in {"", "unknown", None}:
            continue
        if row.get("base_wrench_transform_kind") not in VERIFIED_BASE_WRENCH_TRANSFORMS:
            continue
        valid.append(row)
    if len(valid) < 8:
        raise ValueError(f"有效且坐标一致的样本不足 8 行: {fpath}")
    return valid


def _relative_strict_time(rows: list[dict[str, str]], fpath: str) -> np.ndarray:
    values = _finite_row_values(rows[0], ["sample_time_s"])
    assert values is not None
    t0 = values[0]
    time_s = []
    for row in rows:
        values = _finite_row_values(row, ["sample_time_s"])
        if values is None:
            raise ValueError(f"无效时间字段: {fpath}")
        time_s.append(values[0] - t0)
    result = np.asarray(time_s, dtype=float)
    if np.any(np.diff(result) <= 0):
        raise ValueError(f"有效样本时间必须严格递增: {fpath}")
    return result


def load_episode(fpath: str):
    """Load valid 3D base-frame data for an explicitly Cartesian PINN study."""
    rows = _valid_rows(fpath)
    time_s = _relative_strict_time(rows, fpath)
    xyz = []
    force = []
    for row in rows:
        position = _finite_row_values(row, ["x_m", "y_m", "z_m"])
        wrench = _finite_row_values(row, ["fx_base_n", "fy_base_n", "fz_base_n"])
        if position is None or wrench is None:
            continue
        xyz.append(position)
        force.append(wrench)
    if len(xyz) < 8:
        raise ValueError(f"有效 base 三维力/位姿不足 8 行: {fpath}")
    # Rebuild time after dropping any incomplete row, preserving actual timestamps.
    selected_rows = [
        row for row in rows
        if _finite_row_values(row, ["x_m", "y_m", "z_m", "fx_base_n", "fy_base_n", "fz_base_n"])
        is not None
    ]
    return _relative_strict_time(selected_rows, fpath), np.asarray(xyz), np.asarray(force)


def load_tangential_episode(fpath: str):
    """Load arc length (m) and corrected base-frame tangent force (N)."""
    rows = _valid_rows(fpath)
    selected_rows = [
        row for row in rows
        if _finite_row_values(row, ["trajectory_arc_length_m", "force_tangent_n"]) is not None
    ]
    if len(selected_rows) < 8:
        raise ValueError(f"有效切向力/弧长不足 8 行: {fpath}")
    time_s = _relative_strict_time(selected_rows, fpath)
    arc = np.asarray(
        [float(row["trajectory_arc_length_m"]) for row in selected_rows], dtype=float
    )
    force = np.asarray(
        [float(row["force_tangent_n"]) for row in selected_rows], dtype=float
    )
    return time_s, arc, force


def run_validation(data_dir: str, n_episodes: int = 5, cartesian: bool = False) -> None:
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))[:n_episodes]
    if not files:
        raise FileNotFoundError(f"未找到 episode CSV: {data_dir}")
    results = []
    for index, fpath in enumerate(files, start=1):
        logger.info("── Episode %d: %s", index, os.path.basename(fpath))
        if cartesian:
            time_s, xyz, force = load_episode(fpath)
            _, params = run_pinn(time_s, xyz, force, epochs=settings.PINN_EPOCHS, verbose=True)
        else:
            time_s, arc, force_tangent = load_tangential_episode(fpath)
            _, params = run_tangential_pinn(
                time_s,
                arc,
                force_tangent,
                epochs=settings.PINN_EPOCHS,
                verbose=True,
            )
        results.append(params)
    if len(results) > 1:
        keys = ("Mx", "My", "Mz", "Bx", "By", "Bz", "Kx", "Ky", "Kz") if cartesian else ("M", "B", "K")
        for key in keys:
            values = [result[key] for result in results]
            logger.info("%s: mean=%.4f std=%.4f", key, float(np.mean(values)), float(np.std(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description="从有效 ROKAE 状态快照离线辨识 PINN")
    parser.add_argument("--data-dir", default=settings.DATA_DIR)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="使用显式验证过的三维 Cartesian 模型；默认使用弧长/切向力模型",
    )
    args = parser.parse_args()
    run_validation(args.data_dir, args.n_episodes, cartesian=args.cartesian)


if __name__ == "__main__":
    main()
