# collection/collector.py
# 数据采集主逻辑：同步读传感器、存CSV、计算加速度

import csv
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
        "j1", "j2", "j3", "j4", "j5", "j6",      # 关节角度
        "fx", "fy", "fz", "tx", "ty", "tz",      # 力/力矩
        "mode",                                    # passive / active
        "comfort",                                 # 舒适度标注，-1=未标注
    ]

    def __init__(self, robot, force_sensor,
                 subject_id: str, session_id: str,
                 mode: str = "passive"):
        self.robot    = robot
        self.force    = force_sensor
        self.mode     = mode
        self._buf     = []
        self._t0      = 0.0

        # 创建存储目录
        self.out_dir = os.path.join(settings.DATA_DIR, subject_id, session_id)
        os.makedirs(self.out_dir, exist_ok=True)

        self._ep_count = self._count_existing()
        logger.info(f"存储路径: {self.out_dir}，已有 {self._ep_count} 个episode")

    def _count_existing(self) -> int:
        files = [f for f in os.listdir(self.out_dir) if f.endswith(".csv")]
        return len(files)

    # ── Episode控制 ──────────────────────────────────
    def start_episode(self):
        self._buf = []
        self._t0  = time.time()
        logger.info(f"Episode {self._ep_count + 1} 开始")

    def record_sample(self):
        """采集一个时间点的样本，在主控制循环中按频率调用"""
        t_rel = time.time() - self._t0
        state = self.robot.get_state()
        force = self.force.get()
        pose  = state["cartesian_pose"]
        vel   = state["tcp_speed"]
        jnt   = state["joint_angles"]

        self._buf.append({
            "t":   round(t_rel, 4),
            "x":   pose[0], "y":  pose[1], "z":  pose[2],
            "rx":  pose[3], "ry": pose[4], "rz": pose[5],
            "vx":  vel[0],  "vy": vel[1],  "vz": vel[2],
            "ax":  0.0, "ay": 0.0, "az": 0.0,   # 占位，后处理填充
            "j1":  jnt[0], "j2": jnt[1], "j3": jnt[2],
            "j4":  jnt[3], "j5": jnt[4], "j6": jnt[5],
            "fx":  force["fx"], "fy": force["fy"], "fz": force["fz"],
            "tx":  force["tx"], "ty": force["ty"], "tz": force["tz"],
            "mode":    self.mode,
            "comfort": -1,
        })

    def end_episode(self, comfort_label: int = -1) -> str | None:
        """
        结束episode：
        1. 后处理计算加速度
        2. 打舒适度标签
        3. 写CSV
        Returns: 保存的文件路径，失败返回None
        """
        if len(self._buf) < 10:
            logger.warning("数据点不足10，丢弃此episode")
            return None

        # 计算加速度
        t_arr = np.array([r["t"]  for r in self._buf])
        poses = np.array([[r["x"], r["y"], r["z"],
                           r["rx"],r["ry"],r["rz"]] for r in self._buf])
        accel = compute_acceleration(poses, t_arr)

        for i, row in enumerate(self._buf):
            row["ax"]      = round(float(accel[i, 0]), 4)
            row["ay"]      = round(float(accel[i, 1]), 4)
            row["az"]      = round(float(accel[i, 2]), 4)
            row["comfort"] = comfort_label

        # 写CSV
        self._ep_count += 1
        fname = os.path.join(self.out_dir, f"episode_{self._ep_count:04d}.csv")
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._buf)

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
