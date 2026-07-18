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
    """
    采集器
    - 按 COLLECT_HZ 节奏同步读取机械臂状态 + 力传感器
    - 按 episode 存储 CSV
    - episode 结束后统一计算加速度（S-G滤波微分）
    """

    FIELDNAMES = [
        "t",
        "x", "y", "z", "rx", "ry", "rz",       # 末端笛卡尔位姿
        "vx", "vy", "vz",                         # 末端线速度
        "ax", "ay", "az",                         # 末端线加速度（后处理计算）
        "j1", "j2", "j3", "j4", "j5", "j6",       #joint angles
        "dj1", "dj2", "dj3", "dj4", "dj5", "dj6",  # 关节速度（由机械臂适配层提供）
        "fx", "fy", "fz", "tx", "ty", "tz",      # 力/力矩
        "mode",                                    # passive / active
        "comfort",                                 # 舒适度标注，-1=未标注
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

        # 创建存储目录
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

    @staticmethod
    def _as_six(values, name: str):
        if values is None or len(values) < 6:
            raise ValueError(f"{name} 长度不足6")
        return [float(v) for v in values[:6]]

    def record_sample(self) -> bool:
        """采集一个时间点的样本，在主控制循环中按频率调用。成功返回 True。"""
        if not self._active:
            raise RuntimeError("请先调用 start_episode() 再采样")

        try:
            t_rel = time.perf_counter() - self._t0
            pose = self._as_six(self.robot.get_cartesian_pose(), "cartesian_pose")
            jnt = self._as_six(self.robot.get_joint_angles(), "joint_angles")
            djnt = self._as_six(self.robot.get_actual_joint_speeds(), "joint_speeds")
            force = self.force.get()

            row = {
                "t": round(t_rel, 4),
                # 笛卡尔位姿（位置单位 mm，姿态单位 度）
                "x": pose[0], "y": pose[1], "z": pose[2],
                "rx": pose[3], "ry": pose[4], "rz": pose[5],
                # TCP线速度：占位，end_episode 中由 S-G 微分填充
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
                # TCP线加速度：占位，end_episode 中由 S-G 二阶微分填充
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                # 关节角度
                "j1": jnt[0], "j2": jnt[1], "j3": jnt[2],
                "j4": jnt[3], "j5": jnt[4], "j6": jnt[5],
                # 关节速度（xCoreSDK适配层由连续关节状态差分得到）
                "dj1": djnt[0], "dj2": djnt[1], "dj3": djnt[2],
                "dj4": djnt[3], "dj5": djnt[4], "dj6": djnt[5],
                # 力/力矩
                "fx": force["fx"], "fy": force["fy"], "fz": force["fz"],
                "tx": force["tx"], "ty": force["ty"], "tz": force["tz"],
                "mode": self.mode,
                "comfort": -1,
            }
        except Exception as e:
            self._sample_errors += 1
            logger.warning(f"采样失败，已跳过本帧: {e}")
            return False

        self._buf.append(row)
        return True

    def end_episode(self, comfort_label: int = -1) -> str | None:
        """
        结束episode：
        1. 对位置序列做 S-G 一阶微分 → TCP线速度
        2. 对位置序列做 S-G 二阶微分 → TCP线加速度
        3. 打舒适度标签
        4. 写CSV
        Returns: 保存的文件路径，失败返回None
        """
        if len(self._buf) < 10:
            logger.warning("数据点不足10，丢弃此episode")
            self._active = False
            return None

        t_arr = np.array([r["t"] for r in self._buf])
        # 仅对 x/y/z 做微分（单位 mm），rx/ry/rz 为姿态，不用于线速度/加速度
        pos = np.array([[r["x"], r["y"], r["z"]] for r in self._buf])  # (N, 3)

        # compute_acceleration 内部应先一阶微分得速度，再对速度微分得加速度
        # 如果 smooth_differentiate 支持阶数参数，也可直接调两次
        vel = smooth_differentiate(pos, t_arr)      # (N, 3)  mm/s
        accel = compute_acceleration(pos, t_arr)    # (N, 3)  mm/s^2

        for i, row in enumerate(self._buf):
            row["vx"] = round(float(vel[i, 0]),   4)
            row["vy"] = round(float(vel[i, 1]),   4)
            row["vz"] = round(float(vel[i, 2]),   4)
            row["ax"] = round(float(accel[i, 0]), 4)
            row["ay"] = round(float(accel[i, 1]), 4)
            row["az"] = round(float(accel[i, 2]), 4)
            row["comfort"] = comfort_label

        # 写CSV
        self._ep_count += 1
        fname = os.path.join(self.out_dir, f"episode_{self._ep_count:04d}.csv")
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
    """
    事后标注工具：遍历所有comfort=-1的episode，逐个打标签
    标签：0=舒适  1=轻微不适  2=危险
    """
    import glob
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
