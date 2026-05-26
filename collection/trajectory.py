# collection/trajectory.py
# 轨迹生成/加载：数学激励轨迹 + 真实CSV轨迹

import csv
import os
import numpy as np
import time
from config import settings


def _resolve_tool_orientation(orientation=None) -> tuple:
    """Return the fixed TCP orientation [rx, ry, rz] used for all waypoints."""
    if orientation is None:
        orientation = getattr(settings, "TOOL_DOWN_ORIENTATION", None)
    if orientation is None:
        orientation = [0.0, 90.0, 0.0]
    if len(orientation) < 3:
        raise ValueError("tool orientation must contain rx, ry, rz")
    return tuple(float(v) for v in orientation[:3])


def _resolve_project_path(path: str) -> str:
    """解析相对工程根目录的文件路径。"""
    if path is None:
        raise ValueError("CSV路径不能为空")
    path = os.fspath(path)
    if os.path.isabs(path):
        return path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.abspath(path),
        os.path.join(repo_root, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _read_csv_rows(csv_path: str) -> tuple[list[dict], list[str]]:
    resolved = _resolve_project_path(csv_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"真实轨迹CSV不存在: {resolved}")

    with open(resolved, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"真实轨迹CSV为空: {resolved}")
    return rows, fieldnames


def _actual_csv_columns(fieldnames: list[str], required: list[str], csv_path: str) -> list[str]:
    lookup = {name.lower(): name for name in fieldnames}
    missing = [name for name in required if name.lower() not in lookup]
    if missing:
        raise ValueError(f"CSV缺少列 {missing}: {csv_path}")
    return [lookup[name.lower()] for name in required]


def _to_float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _read_xyz_columns(rows: list[dict], fieldnames: list[str], prefix: str,
                      csv_path: str) -> np.ndarray:
    cols = _actual_csv_columns(
        fieldnames,
        [f"{prefix}_X", f"{prefix}_Y", f"{prefix}_Z"],
        csv_path,
    )
    values = np.array(
        [[_to_float_or_nan(row[col]) for col in cols] for row in rows],
        dtype=float,
    )
    if not np.any(np.all(np.isfinite(values), axis=1)):
        raise ValueError(f"CSV中 {prefix} 三维坐标均无效: {csv_path}")
    return values


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    window = int(window)
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window < 3:
        return values
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _smooth_nd(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return _smooth_1d(values, window)
    smoothed = np.zeros_like(values, dtype=float)
    for i in range(values.shape[1]):
        smoothed[:, i] = _smooth_1d(values[:, i], window)
    return smoothed


def _setting_vector(name: str, default, size: int = 3) -> np.ndarray:
    value = getattr(settings, name, default)
    if value is None:
        value = default
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1 and size > 1:
        arr = np.full(size, float(arr[0]))
    if arr.size != size:
        raise ValueError(f"{name}必须包含{size}个数值")
    return arr


def _setting_int_vector(name: str, default, size: int = 3) -> np.ndarray:
    arr = np.asarray(getattr(settings, name, default), dtype=int).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name}必须包含{size}个整数")
    if np.any(arr < 0) or np.any(arr >= size):
        raise ValueError(f"{name}索引必须在0到{size - 1}之间")
    return arr


def _source_dt_from_settings(source_dt=None, source_hz=None):
    if source_dt is None:
        source_dt = getattr(settings, "REAL_TRAJECTORY_SOURCE_DT", None)
    if source_hz is None:
        source_hz = getattr(settings, "REAL_TRAJECTORY_SOURCE_HZ", None)
    if source_dt is None and source_hz not in (None, 0):
        source_dt = 1.0 / float(source_hz)
    if source_dt is None:
        return None
    source_dt = float(source_dt)
    if source_dt <= 0.0:
        raise ValueError("REAL_TRAJECTORY_SOURCE_DT必须大于0")
    return source_dt


def _resample_sequence(values: np.ndarray,
                       source_dt: float | None,
                       target_dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if source_dt is None or len(values) < 2 or abs(source_dt - target_dt) < 1e-9:
        return values

    src_t = np.arange(len(values), dtype=float) * source_dt
    dst_t = np.arange(0.0, src_t[-1] + target_dt * 0.5, target_dt)
    dst_t = dst_t[dst_t <= src_t[-1] + 1e-9]
    if values.ndim == 1:
        return np.interp(dst_t, src_t, values)

    out = np.zeros((len(dst_t), values.shape[1]), dtype=float)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, values[:, i])
    return out


def _extract_joint_angle(rows: list[dict],
                         fieldnames: list[str],
                         joints: tuple[str, str, str],
                         csv_path: str) -> np.ndarray:
    proximal_name, joint_name, distal_name = joints
    proximal = _read_xyz_columns(rows, fieldnames, proximal_name, csv_path)
    joint = _read_xyz_columns(rows, fieldnames, joint_name, csv_path)
    distal = _read_xyz_columns(rows, fieldnames, distal_name, csv_path)

    n = min(len(proximal), len(joint), len(distal))
    proximal = proximal[:n]
    joint = joint[:n]
    distal = distal[:n]

    v1 = proximal - joint
    v2 = distal - joint
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    valid = (denom > 1e-9) & np.all(np.isfinite(v1), axis=1) & np.all(np.isfinite(v2), axis=1)
    if not np.any(valid):
        raise ValueError(f"无法从CSV计算关节角，骨段长度无效: {csv_path}")

    cos_angle = np.sum(v1[valid] * v2[valid], axis=1) / denom[valid]
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def _extract_point_xyz(rows: list[dict],
                       fieldnames: list[str],
                       point_name: str,
                       csv_path: str) -> np.ndarray:
    values = _read_xyz_columns(rows, fieldnames, point_name, csv_path)
    valid = np.all(np.isfinite(values), axis=1)
    values = values[valid]
    if len(values) < 2:
        raise ValueError(f"CSV中 {point_name} 有效轨迹点不足2帧: {csv_path}")
    return values


def _fallback_point_anchor(center=None, radius=None) -> np.ndarray:
    center = settings.JOINT_CENTER if center is None else center
    radius = settings.JOINT_RADIUS if radius is None else radius
    neutral = float(getattr(settings, "JOINT_NEUTRAL", settings.JOINT_ANGLE_MIN))
    cx, cy, cz = center
    return np.array([
        float(cx) + float(radius) * np.cos(neutral),
        float(cy),
        float(cz) + float(radius) * np.sin(neutral),
    ], dtype=float)


def _resolve_point_anchor(anchor_xyz=None, center=None, radius=None) -> np.ndarray:
    if anchor_xyz is not None:
        anchor = np.asarray(anchor_xyz, dtype=float).reshape(-1)
    else:
        configured = getattr(settings, "REAL_TRAJECTORY_POINT_ANCHOR_XYZ", None)
        if configured is None:
            anchor = _fallback_point_anchor(center, radius)
        else:
            anchor = np.asarray(configured, dtype=float).reshape(-1)
    if anchor.size != 3 or not np.all(np.isfinite(anchor)):
        raise ValueError("wrist轨迹锚点必须是有效的[x,y,z]")
    return anchor


def _point_xyz_to_waypoints(source_xyz: np.ndarray,
                            orientation,
                            anchor_xyz=None,
                            center=None,
                            radius=None) -> np.ndarray:
    axis_map = _setting_int_vector("REAL_TRAJECTORY_POINT_AXIS_MAP", (0, 1, 2))
    axis_sign = _setting_vector("REAL_TRAJECTORY_POINT_AXIS_SIGN", (1.0, 1.0, 1.0))
    axis_scale = _setting_vector("REAL_TRAJECTORY_POINT_SCALE", (1.0, 1.0, 1.0))
    offset = _setting_vector("REAL_TRAJECTORY_POINT_OFFSET_XYZ", (0.0, 0.0, 0.0))

    use_absolute = bool(getattr(settings, "REAL_TRAJECTORY_POINT_USE_ABSOLUTE", False))
    mapped_xyz = source_xyz[:, axis_map] * axis_sign
    if use_absolute:
        robot_xyz = mapped_xyz * axis_scale + offset
    else:
        anchor = _resolve_point_anchor(anchor_xyz, center, radius)
        delta = (mapped_xyz - mapped_xyz[0]) * axis_scale
        max_delta = getattr(settings, "REAL_TRAJECTORY_POINT_MAX_DELTA_MM", None)
        if max_delta is not None:
            max_delta_vec = _setting_vector("REAL_TRAJECTORY_POINT_MAX_DELTA_MM", max_delta)
            delta = np.clip(delta, -max_delta_vec, max_delta_vec)
        robot_xyz = anchor + delta + offset

    if not np.all(np.isfinite(robot_xyz)):
        raise ValueError("wrist轨迹映射后包含无效机器人坐标")

    rx, ry, rz = _resolve_tool_orientation(orientation)
    waypoints = np.zeros((len(robot_xyz), 6), dtype=float)
    waypoints[:, :3] = robot_xyz
    waypoints[:, 3] = rx
    waypoints[:, 4] = ry
    waypoints[:, 5] = rz
    return waypoints


def _validate_rehab_motion_limits(range_scale: float, cycles: float, duration: float):
    """真实机器人康复轨迹的参数护栏，危险样本只能走离线合成。"""
    max_range = float(getattr(settings, "REHAB_MAX_REAL_RANGE_SCALE", 1.0))
    max_freq = float(getattr(settings, "REHAB_MAX_REAL_CYCLES_PER_SEC", 0.30))
    min_duration = float(getattr(settings, "REHAB_MIN_REAL_DURATION", 0.0))

    if range_scale < 0.0:
        raise ValueError(f"range_scale不能为负: {range_scale}")
    if range_scale > max_range:
        raise ValueError(
            "拒绝生成真实机器人康复轨迹："
            f"range_scale={range_scale:.2f} > 安全上限{max_range:.2f}。"
            "越界/危险负样本请使用 --synthetic-risk-only 离线生成CSV。"
        )
    if duration <= 0.0:
        raise ValueError("duration必须大于0")
    if duration < min_duration:
        raise ValueError(
            "拒绝生成真实机器人康复轨迹："
            f"duration={duration:.2f}s < 最小时长{min_duration:.2f}s，动作过快风险较高。"
        )

    cycles_per_sec = cycles / duration
    if cycles_per_sec > max_freq:
        raise ValueError(
            "拒绝生成真实机器人康复轨迹："
            f"cycles/duration={cycles_per_sec:.3f}Hz > 安全上限{max_freq:.3f}Hz。"
            "高加速度危险负样本请使用离线合成CSV。"
        )


def generate_real_trajectory_from_csv(
    csv_path: str = None,
    dt: float = None,
    center: list = None,
    radius: float = None,
    angle_min: float = None,
    angle_max: float = None,
    orientation: list = None,
    anchor_xyz: list = None,
    mode: str = None,
    point_name: str = None,
    joints: tuple[str, str, str] = None,
    invert: bool = None,
    smooth_window: int = None,
    source_dt: float = None,
    source_hz: float = None,
) -> np.ndarray:
    """
    从真实骨架CSV生成机器人康复参考轨迹。

    默认使用 point/wrist 模式：读取 RWrist_X/Y/Z，生成机器人末端相对轨迹。
    旧的 joint_angle 模式仍可通过 REAL_TRAJECTORY_MODE = "joint_angle" 使用。

    Returns:
        waypoints: (N, 6) 笛卡尔路径点数组 [x,y,z,rx,ry,rz]
    """
    csv_path = getattr(settings, "REAL_TRAJECTORY_CSV_PATH", None) if csv_path is None else csv_path
    dt = settings.COLLECT_DT if dt is None else float(dt)
    center = settings.JOINT_CENTER if center is None else center
    radius = settings.JOINT_RADIUS if radius is None else radius
    angle_min = settings.JOINT_ANGLE_MIN if angle_min is None else angle_min
    angle_max = settings.JOINT_ANGLE_MAX if angle_max is None else angle_max
    mode = getattr(settings, "REAL_TRAJECTORY_MODE", "point") if mode is None else mode
    point_name = getattr(settings, "REAL_TRAJECTORY_POINT", "RWrist") if point_name is None else point_name
    joints = getattr(settings, "REAL_TRAJECTORY_JOINTS", None) if joints is None else joints
    invert = getattr(settings, "REAL_TRAJECTORY_INVERT", False) if invert is None else invert
    smooth_window = (
        getattr(settings, "REAL_TRAJECTORY_SMOOTH_WINDOW", 11)
        if smooth_window is None else smooth_window
    )

    if dt <= 0.0:
        raise ValueError("dt必须大于0")
    rows, fieldnames = _read_csv_rows(csv_path)
    mode = str(mode).strip().lower()

    if mode in ("point", "wrist", "xyz"):
        source_xyz = _extract_point_xyz(rows, fieldnames, str(point_name), csv_path)
        source_xyz = _smooth_nd(source_xyz, int(smooth_window))
        source_xyz = _resample_sequence(
            source_xyz,
            _source_dt_from_settings(source_dt, source_hz),
            dt,
        )
        return _point_xyz_to_waypoints(
            source_xyz,
            orientation,
            anchor_xyz=anchor_xyz,
            center=center,
            radius=radius,
        )

    if mode not in ("joint_angle", "angle", "arc"):
        raise ValueError(f"未知真实轨迹模式: {mode}")
    if joints is None or len(joints) != 3:
        raise ValueError("REAL_TRAJECTORY_JOINTS必须包含三个骨架点名，例如 ('RShoulder','RElbow','RWrist')")

    source_angles = _extract_joint_angle(rows, fieldnames, tuple(joints), csv_path)
    source_angles = _smooth_1d(source_angles, int(smooth_window))
    source_angles = _resample_sequence(
        source_angles,
        _source_dt_from_settings(source_dt, source_hz),
        dt,
    )

    low_q = float(getattr(settings, "REAL_TRAJECTORY_RANGE_LOW_PERCENTILE", 1.0))
    high_q = float(getattr(settings, "REAL_TRAJECTORY_RANGE_HIGH_PERCENTILE", 99.0))
    src_min, src_max = np.percentile(source_angles, [low_q, high_q])
    min_range = float(getattr(settings, "REAL_TRAJECTORY_MIN_SOURCE_RANGE_RAD", 0.05))
    if src_max - src_min < min_range:
        src_min = float(np.min(source_angles))
        src_max = float(np.max(source_angles))
    if src_max - src_min < min_range:
        raise ValueError(
            "真实CSV中的关节角变化太小，无法生成有效康复轨迹: "
            f"range={src_max - src_min:.4f}rad"
        )

    normalized = (source_angles - src_min) / (src_max - src_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized

    robot_angles = float(angle_min) + normalized * (float(angle_max) - float(angle_min))

    cx, cy, cz = center
    rx, ry, rz = _resolve_tool_orientation(orientation)
    waypoints = np.zeros((len(robot_angles), 6), dtype=float)
    waypoints[:, 0] = cx + float(radius) * np.cos(robot_angles)
    waypoints[:, 1] = cy
    waypoints[:, 2] = cz + float(radius) * np.sin(robot_angles)
    waypoints[:, 3] = rx
    waypoints[:, 4] = ry
    waypoints[:, 5] = rz
    return waypoints


def generate_excitation_trajectory(
    duration: float = None,
    dt: float = None,
    center: list = None,
    radius: float = None,
    angle_min: float = None,
    angle_max: float = None,
    neutral: float = None,
    orientation: list = None,
) -> np.ndarray:
    """
    生成持续激励轨迹（多频正弦叠加）
    确保同时激励 M（加速度变化）、B（速度变化）、K（位置变化）

    Returns:
        waypoints: (N, 6) 笛卡尔路径点数组 [x,y,z,rx,ry,rz]
    """
    duration = settings.EXCITATION_DURATION if duration is None else duration
    dt = settings.COLLECT_DT if dt is None else dt
    center = settings.JOINT_CENTER if center is None else center
    radius = settings.JOINT_RADIUS if radius is None else radius
    angle_min = settings.JOINT_ANGLE_MIN if angle_min is None else angle_min
    angle_max = settings.JOINT_ANGLE_MAX if angle_max is None else angle_max
    neutral = settings.JOINT_NEUTRAL if neutral is None else neutral

    t = np.arange(0, duration, dt)

    # 多频正弦叠加，生成关节角度变化量
    delta_angle = np.zeros_like(t)
    for amp_mm, freq in settings.EXCITATION_PARAMS:
        amp_rad = amp_mm / radius   # 线位移转角度（小角度近似）
        delta_angle += amp_rad * np.sin(freq * t)

    # 限制在关节活动范围内
    angle = neutral + delta_angle
    angle = np.clip(angle, angle_min, angle_max)

    # 转换为笛卡尔坐标（在xz平面内的弧线运动）
    cx, cy, cz = center
    # 末端姿态固定，保证夹爪始终保持向下。
    rx, ry, rz = _resolve_tool_orientation(orientation)

    waypoints = np.zeros((len(t), 6))
    waypoints[:, 0] = cx + radius * np.cos(angle)   # x
    waypoints[:, 1] = cy                              # y（运动平面内保持不变）
    waypoints[:, 2] = cz + radius * np.sin(angle)   # z
    waypoints[:, 3] = rx
    waypoints[:, 4] = ry
    waypoints[:, 5] = rz

    return waypoints


def generate_rehab_trajectory(
    duration: float = None,
    dt: float = None,
    center: list = None,
    radius: float = None,
    angle_min: float = None,
    angle_max: float = None,
    range_scale: float = None,
    cycles: float = None,
    orientation: list = None,
    enforce_safety: bool = True,
) -> np.ndarray:
    """
    生成康复训练轨迹：在标定得到的活动范围内做平滑往复运动。

    与 generate_excitation_trajectory 不同，这条轨迹用于真实康复动作和舒适度判断；
    它不会刻意叠加高频激励，也默认只使用活动范围的一部分，避免贴边。
    """
    duration = settings.REHAB_DURATION if duration is None else duration
    dt = settings.COLLECT_DT if dt is None else dt
    center = settings.JOINT_CENTER if center is None else center
    radius = settings.JOINT_RADIUS if radius is None else radius
    angle_min = settings.JOINT_ANGLE_MIN if angle_min is None else angle_min
    angle_max = settings.JOINT_ANGLE_MAX if angle_max is None else angle_max
    range_scale = settings.REHAB_RANGE_SCALE if range_scale is None else range_scale
    cycles = settings.REHAB_CYCLES if cycles is None else cycles
    range_scale = float(range_scale)
    cycles = float(cycles)
    duration = float(duration)
    dt = float(dt)

    if dt <= 0.0:
        raise ValueError("dt必须大于0")

    if enforce_safety:
        _validate_rehab_motion_limits(range_scale, cycles, duration)

    t = np.arange(0, duration, dt)
    angle_center = 0.5 * (angle_min + angle_max)
    angle_amp = 0.5 * (angle_max - angle_min) * range_scale
    angle = angle_center + angle_amp * np.sin(2.0 * np.pi * cycles * t / duration)
    angle = np.clip(angle, angle_min, angle_max)

    cx, cy, cz = center
    rx, ry, rz = _resolve_tool_orientation(orientation)
    waypoints = np.zeros((len(t), 6))
    waypoints[:, 0] = cx + radius * np.cos(angle)
    waypoints[:, 1] = cy
    waypoints[:, 2] = cz + radius * np.sin(angle)
    waypoints[:, 3] = rx
    waypoints[:, 4] = ry
    waypoints[:, 5] = rz
    return waypoints


def _read_cartesian_position(robot) -> list:
    """读取当前末端 xyz，单位 mm，兼容不同机器人封装。"""
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
    Returns: (center [x,y,z], radius, angle_min, angle_max, neutral)
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

        p = [float(v) for v in _read_cartesian_position(robot)]
        points.append(p)
        print(f"    记录坐标: x={p[0]:.1f}, y={p[1]:.1f}, z={p[2]:.1f}")

    points = np.array(points)

    # 在xz平面内拟合圆（假设运动在xz平面）
    # 三点确定圆心：联立方程组
    x1,_,z1 = points[0]
    x2,_,z2 = points[1]
    x3,_,z3 = points[2]

    A = np.array([
        [2*(x2-x1), 2*(z2-z1)],
        [2*(x3-x2), 2*(z3-z2)],
    ])
    b = np.array([
        x2**2 - x1**2 + z2**2 - z1**2,
        x3**2 - x2**2 + z3**2 - z2**2,
    ])
    try:
        cx, cz = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("三个位姿过于接近或近似共线，无法拟合关节圆心，请重新标定") from e
    cy     = points[:, 1].mean()
    radius = np.mean(np.sqrt((points[:, 0]-cx)**2 + (points[:, 2]-cz)**2))
    angles = np.unwrap(np.arctan2(points[:, 2] - cz, points[:, 0] - cx))
    angle_min = min(angles[0], angles[2])
    angle_max = max(angles[0], angles[2])
    neutral = angles[1]

    print(f"\n标定结果：")
    print(f"  旋转中心: [{cx:.1f}, {cy:.1f}, {cz:.1f}] mm")
    print(f"  运动半径: {radius:.1f} mm")
    print(f"  活动角度范围: [{angle_min:.4f}, {angle_max:.4f}] rad")
    print(f"  中立位角度: {neutral:.4f} rad")
    print("  请将以上值更新到 config/settings.py 中的 JOINT_CENTER / JOINT_RADIUS / JOINT_ANGLE_*")

    return (
        [float(cx), float(cy), float(cz)],
        float(radius),
        float(angle_min),
        float(angle_max),
        float(neutral),
    )
