# collection/synthetic_risk.py
# 离线合成危险/明显不适负样本：只写CSV，不向机器人发送任何运动指令。

import csv
import glob
import os

import numpy as np

from collection.collector import DataCollector
from config import settings
from utils.logger import get_logger
from utils.signal_processing import compute_acceleration, smooth_differentiate

logger = get_logger("SyntheticRisk")


def _count_existing(out_dir: str) -> int:
    max_idx = 0
    for fpath in glob.glob(os.path.join(out_dir, "episode_*.csv")):
        stem = os.path.splitext(os.path.basename(fpath))[0]
        try:
            max_idx = max(max_idx, int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_idx


def _orientation_tuple(orientation=None) -> tuple[float, float, float]:
    if orientation is None:
        orientation = getattr(settings, "TOOL_DOWN_ORIENTATION", None)
    if orientation is None:
        orientation = [0.0, 90.0, 0.0]
    return tuple(float(v) for v in orientation[:3])


def _synthetic_rows(spec: dict, orientation=None) -> list[dict]:
    dt = float(spec.get("dt", settings.COLLECT_DT))
    duration = float(spec.get("duration", settings.REHAB_DURATION))
    cycles = float(spec.get("cycles", settings.REHAB_CYCLES))
    range_scale = float(spec.get("range_scale", 1.0))
    label = int(spec.get("comfort", 2))

    if dt <= 0.0 or duration <= 0.0:
        raise ValueError("synthetic risk dt/duration必须大于0")

    t = np.arange(0.0, duration, dt)
    center = np.asarray(settings.JOINT_CENTER, dtype=float)
    radius = float(settings.JOINT_RADIUS)
    angle_min = float(settings.JOINT_ANGLE_MIN)
    angle_max = float(settings.JOINT_ANGLE_MAX)
    angle_center = 0.5 * (angle_min + angle_max)
    angle_amp = 0.5 * (angle_max - angle_min) * range_scale

    phase = 2.0 * np.pi * cycles * t / duration
    angle = angle_center + angle_amp * np.sin(phase)

    xyz = np.zeros((len(t), 3), dtype=float)
    xyz[:, 0] = center[0] + radius * np.cos(angle)
    xyz[:, 1] = center[1]
    xyz[:, 2] = center[2] + radius * np.sin(angle)

    vel = smooth_differentiate(xyz, t)
    accel = compute_acceleration(xyz, t)
    joint_speed = smooth_differentiate(np.degrees(angle), t)

    rx, ry, rz = _orientation_tuple(orientation)
    force_n = float(spec.get("force_n", settings.MAX_FORCE_N * 1.1))
    force = np.zeros((len(t), 3), dtype=float)
    force[:, 0] = 0.35 * force_n * np.sin(phase)
    force[:, 1] = 0.15 * force_n * np.sin(2.0 * phase)
    force[:, 2] = force_n * (0.75 + 0.25 * np.cos(phase))

    mass_scale = float(spec.get("mass_scale", 2.0))
    damping_scale = float(spec.get("damping_scale", 3.0))
    stiffness_scale = float(spec.get("stiffness_scale", 5.0))
    params = {
        "Mx": settings.PINN_M_INIT * mass_scale,
        "My": settings.PINN_M_INIT * mass_scale,
        "Mz": settings.PINN_M_INIT * mass_scale,
        "Bx": settings.PINN_B_INIT * damping_scale,
        "By": settings.PINN_B_INIT * damping_scale,
        "Bz": settings.PINN_B_INIT * damping_scale,
        "Kx": settings.PINN_K_INIT * stiffness_scale,
        "Ky": settings.PINN_K_INIT * stiffness_scale,
        "Kz": settings.PINN_K_INIT * stiffness_scale,
    }

    rows = []
    variant = str(spec.get("name", "unsafe_sim"))
    for i, ts in enumerate(t):
        row = {
            "t": round(float(ts), 4),
            "trajectory_type": "rehab",
            "trajectory_variant": variant,
            "x": round(float(xyz[i, 0]), 4),
            "y": round(float(xyz[i, 1]), 4),
            "z": round(float(xyz[i, 2]), 4),
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "vx": round(float(vel[i, 0]), 4),
            "vy": round(float(vel[i, 1]), 4),
            "vz": round(float(vel[i, 2]), 4),
            "ax": round(float(accel[i, 0]), 4),
            "ay": round(float(accel[i, 1]), 4),
            "az": round(float(accel[i, 2]), 4),
            "j1": round(float(np.degrees(angle[i])), 4),
            "j2": 0.0,
            "j3": 0.0,
            "j4": 0.0,
            "j5": 0.0,
            "j6": 0.0,
            "dj1": round(float(joint_speed[i]), 4),
            "dj2": 0.0,
            "dj3": 0.0,
            "dj4": 0.0,
            "dj5": 0.0,
            "dj6": 0.0,
            "fx": round(float(force[i, 0]), 4),
            "fy": round(float(force[i, 1]), 4),
            "fz": round(float(force[i, 2]), 4),
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
            "Mx": params["Mx"],
            "My": params["My"],
            "Mz": params["Mz"],
            "Bx": params["Bx"],
            "By": params["By"],
            "Bz": params["Bz"],
            "Kx": params["Kx"],
            "Ky": params["Ky"],
            "Kz": params["Kz"],
            "mode": "synthetic_risk",
            "comfort": label,
        }
        rows.append(row)
    return rows


def write_synthetic_risk_episodes(
    subject_id: str,
    session_id: str,
    n_episodes: int,
    orientation=None,
) -> list[str]:
    """
    写入离线危险/明显不适episode。

    这些样本只用于训练负类：不会连接机器人，不会读取力传感器，也不会执行轨迹。
    """
    if n_episodes <= 0:
        return []

    specs = list(getattr(settings, "SYNTHETIC_RISK_VARIANTS", []))
    if not specs:
        raise ValueError("settings.SYNTHETIC_RISK_VARIANTS为空，无法生成合成风险样本")

    out_dir = os.path.join(settings.DATA_DIR, subject_id, session_id)
    os.makedirs(out_dir, exist_ok=True)
    ep_idx = _count_existing(out_dir)

    paths = []
    for i in range(n_episodes):
        spec = specs[i % len(specs)]
        rows = _synthetic_rows(spec, orientation=orientation)
        ep_idx += 1
        fname = os.path.join(out_dir, f"episode_{ep_idx:04d}.csv")
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DataCollector.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        paths.append(fname)
        logger.info(
            f"已写入离线风险episode: {fname} "
            f"({spec.get('name', 'unsafe_sim')}, {len(rows)}行, comfort=2)"
        )

    return paths
