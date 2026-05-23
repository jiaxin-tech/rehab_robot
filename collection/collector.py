# collection/collector.py
# 数据采集主逻辑：同步读传感器、存CSV、计算加速度

import csv
import glob
import os
import time
import numpy as np
from config import settings
from utils.signal_processing import smooth_differentiate, compute_acceleration
from utils.logger import get_logger

logger = get_logger("Collector")


class DataCollector:
    FIELDNAMES = [
        "t",
        "trajectory_type",
        "x", "y", "z", "rx", "ry", "rz",
        "vx", "vy", "vz",
        "ax", "ay", "az",
        "j1", "j2", "j3", "j4", "j5", "j6",
        "dj1", "dj2", "dj3", "dj4", "dj5", "dj6",
        "fx", "fy", "fz", "tx", "ty", "tz",
        "Mx", "My", "Mz", "Bx", "By", "Bz", "Kx", "Ky", "Kz",
        "mode",
        "comfort",
    ]

    def __init__(self, robot, force_sensor,
                 subject_id: str, session_id: str,
                 mode: str = "passive"):
        self.robot = robot
        self.force = force_sensor
        self.mode = mode
        self._buf = []
        self._t0 = 0.0
        self._active = False
        self._sample_errors = 0
        self.trajectory_type = "unknown"

        self.out_dir = os.path.join(settings.DATA_DIR, subject_id, session_id)
        os.makedirs(self.out_dir, exist_ok=True)

        self._ep_count = self._count_existing()
        logger.info(f"存储路径: {self.out_dir}，已有 {self._ep_count} 个episode")

    def _count_existing(self) -> int:
        max_idx = 0
        for fpath in glob.glob(os.path.join(self.out_dir, "episode_*.csv")):
            stem = os.path.splitext(os.path.basename(fpath))[0]
            try:
                max_idx = max(max_idx, int(stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max_idx

    # ── Episode控制 ──────────────────────────────────
    def start_episode(self):
        self._buf = []
        self._t0 = time.perf_counter()
        self._active = True
        self._sample_errors = 0
        logger.info(f"Episode {self._ep_count + 1} 开始")

    def set_trajectory_type(self, trajectory_type: str):
        self.trajectory_type = trajectory_type

    @staticmethod
    def _as_six(values, name: str):
        if values is None or len(values) < 6:
            raise ValueError(f"{name} 长度不足6")
        return [float(v) for v in values[:6]]

    @staticmethod
    def _pinn_params_or_default(pinn_params=None) -> dict:
        params = {
            "Mx": settings.PINN_M_INIT, "My": settings.PINN_M_INIT, "Mz": settings.PINN_M_INIT,
            "Bx": settings.PINN_B_INIT, "By": settings.PINN_B_INIT, "Bz": settings.PINN_B_INIT,
            "Kx": settings.PINN_K_INIT, "Ky": settings.PINN_K_INIT, "Kz": settings.PINN_K_INIT,
        }
        if pinn_params:
            for key in params:
                if key in pinn_params:
                    params[key] = float(pinn_params[key])
        return params

    def record_sample(self, pinn_params=None) -> dict | None:
        if not self._active:
            raise RuntimeError("请先调用 start_episode() 再采样")

        try:
            t_rel = time.perf_counter() - self._t0
            pose  = self._as_six(self.robot.get_cartesian_pose(), "cartesian_pose")
            jnt   = self._as_six(self.robot.get_joint_angles(), "joint_angles")
            djnt  = self._as_six(self.robot.get_actual_joint_speeds(), "joint_speeds")
            force = self.force.get()
            params = self._pinn_params_or_default(pinn_params)

            row = {
                "t": round(t_rel, 4),
                "trajectory_type": self.trajectory_type,
                "x": pose[0], "y": pose[1], "z": pose[2],
                "rx": pose[3], "ry": pose[4], "rz": pose[5],
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                "j1": jnt[0], "j2": jnt[1], "j3": jnt[2],
                "j4": jnt[3], "j5": jnt[4], "j6": jnt[5],
                "dj1": djnt[0], "dj2": djnt[1], "dj3": djnt[2],
                "dj4": djnt[3], "dj5": djnt[4], "dj6": djnt[5],
                "fx": force["fx"], "fy": force["fy"], "fz": force["fz"],
                "tx": force["tx"], "ty": force["ty"], "tz": force["tz"],
                "Mx": params["Mx"], "My": params["My"], "Mz": params["Mz"],
                "Bx": params["Bx"], "By": params["By"], "Bz": params["Bz"],
                "Kx": params["Kx"], "Ky": params["Ky"], "Kz": params["Kz"],
                "mode": self.mode,
                "comfort": -1,
            }
        except Exception as e:
            self._sample_errors += 1
            logger.warning(f"采样失败，已跳过本帧: {e}")
            return None

        self._buf.append(row)
        return row

    # ── M/B/K推理支持 ────────────────────────────────
    def get_current_episode_buffer(self) -> dict | None:
        """
        返回当前episode缓冲区中的t/xyz/F数据，供OnlinePINN.infer_mbk()调用。
        必须在end_episode()之前调用，否则缓冲区已清空。
        返回 None 表示数据不足。
        """
        if len(self._buf) < 30:
            return None
        return {
            "t":   [r["t"] for r in self._buf],
            "xyz": [[r["x"], r["y"], r["z"]] for r in self._buf],
            "F":   [[r["fx"], r["fy"], r["fz"]] for r in self._buf],
        }

    def write_mbk_to_episode(self, params: dict):
        """
        把PINN推理出的M/B/K写入当前episode缓冲区的每一帧。
        同一episode内M/B/K是常数（整段轨迹共享一组参数）。
        必须在end_episode()之前调用。
        """
        if not self._buf:
            logger.warning("write_mbk_to_episode: 缓冲区为空，跳过")
            return
        keys = ["Mx", "My", "Mz", "Bx", "By", "Bz", "Kx", "Ky", "Kz"]
        for row in self._buf:
            for k in keys:
                if k in params:
                    row[k] = float(params[k])
        logger.debug(f"M/B/K已写入缓冲区 {len(self._buf)} 帧")

    # ── Episode结束 ──────────────────────────────────
    def end_episode(self, comfort_label: int = -1) -> str | None:
        """
        结束episode：
        1. S-G微分 → TCP线速度/加速度
        2. 打舒适度标签
        3. 写CSV
        """
        if len(self._buf) < 10:
            logger.warning("数据点不足10，丢弃此episode")
            self._active = False
            return None

        t_arr = np.array([r["t"] for r in self._buf])
        pos   = np.array([[r["x"], r["y"], r["z"]] for r in self._buf])

        vel   = smooth_differentiate(pos, t_arr)
        accel = compute_acceleration(pos, t_arr)

        for i, row in enumerate(self._buf):
            row["vx"] = round(float(vel[i, 0]),   4)
            row["vy"] = round(float(vel[i, 1]),   4)
            row["vz"] = round(float(vel[i, 2]),   4)
            row["ax"] = round(float(accel[i, 0]), 4)
            row["ay"] = round(float(accel[i, 1]), 4)
            row["az"] = round(float(accel[i, 2]), 4)
            row["comfort"] = comfort_label

        self._ep_count += 1
        fname    = os.path.join(self.out_dir, f"episode_{self._ep_count:04d}.csv")
        tmp_name = f"{fname}.tmp"
        with open(tmp_name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._buf)
        os.replace(tmp_name, fname)

        self._active = False
        if self._sample_errors:
            logger.warning(f"Episode {self._ep_count} 采样跳过 {self._sample_errors} 帧")
        logger.info(f"Episode {self._ep_count} 已保存: {fname} ({len(self._buf)} 行)")
        return fname


def label_episodes(data_dir: str):
    """事后标注工具：遍历所有comfort=-1的episode，逐个打标签"""
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    count = 0
    for fpath in files:
        with open(fpath) as f:
            rows = list(csv.DictReader(f))
        if not rows or rows[0].get("comfort") != "-1":
            continue

        print(f"\n[{count+1}] {fpath}  ({len(rows)} 行, mode={rows[0]['mode']})")
        label = input("舒适度标签 (0/1/2，跳过按Enter): ").strip()
        if label in ("0", "1", "2"):
            fieldnames = list(rows[0].keys())
            for r in rows:
                r["comfort"] = label
            with open(fpath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  → 已标注为 {label}")
            count += 1

    print(f"\n共标注 {count} 个episode")