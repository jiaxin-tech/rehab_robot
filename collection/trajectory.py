# collection/trajectory.py
# 持续激励轨迹生成：多频正弦叠加

import numpy as np
import time
from dataclasses import dataclass
from config import settings


@dataclass(frozen=True)
class TrajectoryProjection:
    """Nearest valid path projection in the robot base frame."""

    trajectory_s: float
    arc_length_m: float
    tangent_base: tuple[float, float, float]


class TrajectoryGeometry:
    """Arc-length and tangent helper for a Cartesian waypoint path.

    Consecutive duplicate waypoints are skipped for projection rather than
    normalized into a bogus direction.  A path containing no non-zero segment
    remains representable but all projections return an explicit invalid reason.
    """

    def __init__(self, waypoints: np.ndarray):
        values = np.asarray(waypoints, dtype=float)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("Trajectory must be an (N, >=3) finite waypoint array")
        if len(values) == 0 or not np.all(np.isfinite(values[:, :3])):
            raise ValueError("Trajectory positions must be non-empty and finite")
        self.waypoints = values
        self.positions_m = values[:, :3].copy()
        deltas = np.diff(self.positions_m, axis=0)
        self.segment_length_m = np.linalg.norm(deltas, axis=1)
        self._valid_segment = self.segment_length_m > 1e-12
        self.arc_at_waypoint_m = np.concatenate(
            ([0.0], np.cumsum(self.segment_length_m))
        )
        self.total_arc_length_m = float(self.arc_at_waypoint_m[-1])

    def project(
        self,
        position_m,
        *,
        reference_arc_length_m: float | None = None,
        continuity_tolerance_m: float = 0.003,
    ) -> tuple[TrajectoryProjection | None, str]:
        """Project a base-frame position, optionally preserving path phase.

        A rehabilitation trajectory can intentionally retrace the same physical
        arc.  Geometry alone then has more than one valid tangent/arc result.
        Supplying the preceding chronological arc length resolves candidates
        within a small spatial tolerance by continuity instead of silently
        mapping every return pass to the first segment.
        """
        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None, "trajectory_projection_invalid_position"
        if self.total_arc_length_m <= 1e-12 or not np.any(self._valid_segment):
            return None, "trajectory_zero_length"
        if not np.isfinite(continuity_tolerance_m) or continuity_tolerance_m < 0.0:
            return None, "trajectory_projection_continuity_tolerance_invalid"
        reference_arc: float | None = None
        if reference_arc_length_m is not None:
            reference_arc = float(reference_arc_length_m)
            if not np.isfinite(reference_arc):
                return None, "trajectory_reference_arc_invalid"
            if reference_arc < -1e-12 or reference_arc > self.total_arc_length_m + 1e-12:
                return None, "trajectory_reference_arc_out_of_range"
            reference_arc = min(self.total_arc_length_m, max(0.0, reference_arc))

        candidates: list[tuple[float, int, float, float]] = []
        for index, length in enumerate(self.segment_length_m):
            if not self._valid_segment[index]:
                continue
            start = self.positions_m[index]
            delta = self.positions_m[index + 1] - start
            alpha = float(np.dot(position - start, delta) / (length * length))
            alpha = min(1.0, max(0.0, alpha))
            candidate = start + alpha * delta
            distance_sq = float(np.dot(position - candidate, position - candidate))
            arc_length_m = float(
                self.arc_at_waypoint_m[index] + alpha * self.segment_length_m[index]
            )
            candidates.append((distance_sq, index, alpha, arc_length_m))

        if not candidates:
            return None, "trajectory_zero_length"
        min_distance_sq = min(item[0] for item in candidates)
        if reference_arc is None:
            # Stable ordering makes the no-phase case deterministic, while the
            # caller can choose the continuity-aware path whenever it has a
            # previous projection/command phase.
            _, best_index, best_alpha, arc_length_m = min(
                candidates, key=lambda item: (item[0], item[1])
            )
        else:
            tolerance_sq = float(continuity_tolerance_m) ** 2
            phase_candidates = [
                item for item in candidates if item[0] <= min_distance_sq + tolerance_sq
            ]
            _, best_index, best_alpha, arc_length_m = min(
                phase_candidates,
                key=lambda item: (abs(item[3] - reference_arc), item[0], item[1]),
            )
        delta = self.positions_m[best_index + 1] - self.positions_m[best_index]
        tangent = delta / self.segment_length_m[best_index]
        return (
            TrajectoryProjection(
                trajectory_s=arc_length_m / self.total_arc_length_m,
                arc_length_m=arc_length_m,
                tangent_base=tuple(float(value) for value in tangent),
            ),
            "",
        )

    def pose_at_normalized_s(self, trajectory_s: float) -> np.ndarray:
        """Interpolate a full waypoint pose at a clipped arc-length parameter."""
        if not np.isfinite(trajectory_s):
            raise ValueError("trajectory_s must be finite")
        if self.total_arc_length_m <= 1e-12 or not np.any(self._valid_segment):
            raise ValueError("Cannot sample a zero-length trajectory")
        arc_m = min(1.0, max(0.0, float(trajectory_s))) * self.total_arc_length_m
        index = int(np.searchsorted(self.arc_at_waypoint_m, arc_m, side="right") - 1)
        index = min(max(0, index), len(self.segment_length_m) - 1)
        if not self._valid_segment[index]:
            candidates = np.flatnonzero(self._valid_segment)
            index = int(candidates[np.argmin(np.abs(candidates - index))])
        segment_start_m = self.arc_at_waypoint_m[index]
        alpha = (arc_m - segment_start_m) / self.segment_length_m[index]
        alpha = min(1.0, max(0.0, float(alpha)))
        return (1.0 - alpha) * self.waypoints[index] + alpha * self.waypoints[index + 1]


def project_along_tangent(vector_base, tangent_base) -> float | None:
    """Project a finite base-frame Cartesian vector onto a unit tangent."""
    vector = np.asarray(vector_base, dtype=float)
    tangent = np.asarray(tangent_base, dtype=float)
    if vector.shape != (3,) or tangent.shape != (3,):
        return None
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(tangent)):
        return None
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return None
    value = float(np.dot(vector, tangent / norm))
    return value if np.isfinite(value) else None


def generate_excitation_trajectory(
    duration: float = settings.EXCITATION_DURATION,
    dt: float       = settings.COLLECT_DT,
    center: list    = settings.JOINT_CENTER,
    radius: float   = settings.JOINT_RADIUS,
    angle_min: float = settings.JOINT_ANGLE_MIN,
    angle_max: float = settings.JOINT_ANGLE_MAX,
    neutral: float  = settings.JOINT_NEUTRAL,
) -> np.ndarray:
    """
    生成持续激励轨迹（多频正弦叠加）
    确保同时激励 M（加速度变化）、B（速度变化）、K（位置变化）

    Returns:
        waypoints: (N, 6) base-frame 笛卡尔路径点 [x,y,z,rx,ry,rz]，单位 m/rad
    """
    t = np.arange(0, duration, dt)

    # 多频正弦叠加，生成关节角度变化量
    delta_angle = np.zeros_like(t)
    for arc_amplitude_m, freq_rad_s in settings.EXCITATION_PARAMS:
        amp_rad = arc_amplitude_m / radius  # 弧长转角度（小角度近似）
        delta_angle += amp_rad * np.sin(freq_rad_s * t)

    # 限制在关节活动范围内
    angle = neutral + delta_angle
    angle = np.clip(angle, angle_min, angle_max)

    # 转换为笛卡尔坐标（在xz平面内的弧线运动）
    cx, cy, cz = center
    # 末端姿态固定（根据实际安装方式调整 rx,ry,rz）
    rx, ry, rz = 0.0, np.pi / 2.0, 0.0

    waypoints = np.zeros((len(t), 6))
    waypoints[:, 0] = cx + radius * np.cos(angle)   # x
    waypoints[:, 1] = cy                              # y（运动平面内保持不变）
    waypoints[:, 2] = cz + radius * np.sin(angle)   # z
    waypoints[:, 3] = rx
    waypoints[:, 4] = ry
    waypoints[:, 5] = rz

    return waypoints


def generate_slow_sweep(n_points: int = 100,
                        center=settings.JOINT_CENTER,
                        radius=settings.JOINT_RADIUS,
                        angle_min=settings.JOINT_ANGLE_MIN,
                        angle_max=settings.JOINT_ANGLE_MAX) -> np.ndarray:
    """
    慢速全程扫描轨迹（用于ROM探测和K辨识）
    从最小角度线性扫到最大角度，再返回
    """
    angles_go   = np.linspace(angle_min, angle_max, n_points)
    angles_back = np.linspace(angle_max, angle_min, n_points)
    angles      = np.concatenate([angles_go, angles_back])

    cx, cy, cz = center
    waypoints = np.zeros((len(angles), 6))
    waypoints[:, 0] = cx + radius * np.cos(angles)
    waypoints[:, 1] = cy
    waypoints[:, 2] = cz + radius * np.sin(angles)
    waypoints[:, 3] = 0.0
    waypoints[:, 4] = np.pi / 2.0
    waypoints[:, 5] = 0.0
    return waypoints


def _read_cartesian_position(robot) -> list:
    """读取 base 坐标系下的当前末端 xyz，单位 m。"""
    if hasattr(robot, "get_cartesian_pose"):
        return list(robot.get_cartesian_pose()[:3])
    if hasattr(robot, "get_state"):
        return list(robot.get_state()["cartesian_pose"][:3])
    raise AttributeError("robot 缺少 get_cartesian_pose/get_state 位姿读取接口")


def _start_drag(robot):
    if not hasattr(robot, "start_drag"):
        raise AttributeError("robot 缺少 start_drag 拖拽接口")
    return robot.start_drag()


def _stop_drag(robot):
    if not hasattr(robot, "stop_drag"):
        raise AttributeError("robot 缺少 stop_drag 拖拽接口")
    return robot.stop_drag()


def calibrate_joint_center(robot, use_drag: bool = True) -> tuple:
    """
    标定关节旋转中心和半径
    操作：进入拖拽模式，手动将机械臂引导到3个关节角度位置，记录末端坐标，用圆拟合
    Returns: (center [x,y,z], radius float)
    """
    print("标定模式：程序会自动进入拖拽模式，请拖动末端到对应人体关节位置后按回车。")
    print("提示：这里的伸直/中立/弯曲指的是人体目标关节状态，不是机械臂自身关节。")
    points = []
    for i, desc in enumerate(["关节伸直位", "关节中立位", "关节弯曲位"]):
        print(f"\n  [{i+1}/3] 准备记录【{desc}】")
        drag_started = False
        try:
            if use_drag:
                _start_drag(robot)
                drag_started = True
                time.sleep(0.3)
                print("    已进入拖拽模式。请手动拖动机械臂末端到目标位置。")
            input("    到位后按回车记录坐标...")
        finally:
            if drag_started:
                _stop_drag(robot)
                time.sleep(0.3)

        p = _read_cartesian_position(robot)
        points.append(p)
        print(f"    记录坐标: x={p[0]:.4f}, y={p[1]:.4f}, z={p[2]:.4f} m")

    points = np.array(points)

    # 在 xz 平面内拟合圆（假设运动在 xz 平面）。三行都相对 p1，
    # 否则左/右两侧参考点不一致会产生错误圆心。
    # 三点确定圆心：联立方程组
    x1,_,z1 = points[0]
    x2,_,z2 = points[1]
    x3,_,z3 = points[2]

    A = np.array([
        [2*(x2-x1), 2*(z2-z1)],
        [2*(x3-x1), 2*(z3-z1)],
    ])
    b = np.array([
        x2**2 - x1**2 + z2**2 - z1**2,
        x3**2 - x1**2 + z3**2 - z1**2,
    ])
    try:
        cx, cz = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("三个位姿过于接近或近似共线，无法拟合关节圆心，请重新标定") from e
    cy     = points[:, 1].mean()
    radius = np.mean(np.sqrt((points[:, 0]-cx)**2 + (points[:, 2]-cz)**2))

    print(f"\n标定结果：")
    print(f"  旋转中心: [{cx:.4f}, {cy:.4f}, {cz:.4f}] m (base)")
    print(f"  运动半径: {radius:.4f} m")
    print(f"  请将以上值更新到 config/settings.py 中的 JOINT_CENTER 和 JOINT_RADIUS")

    return [cx, cy, cz], radius
