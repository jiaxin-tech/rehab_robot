# collection/trajectory.py
# 持续激励轨迹生成：多频正弦叠加

import numpy as np
import time
from config import settings


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
        waypoints: (N, 6) 笛卡尔路径点数组 [x,y,z,rx,ry,rz]
    """
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
    # 末端姿态固定（根据实际安装方式调整 rx,ry,rz）
    rx, ry, rz = 0.0, 90.0, 0.0

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
    waypoints[:, 4] = 90.0
    waypoints[:, 5] = 0.0
    return waypoints


def _read_cartesian_position(robot) -> list:
    """读取当前末端 xyz，兼容不同机器人封装。"""
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

    print(f"\n标定结果：")
    print(f"  旋转中心: [{cx:.1f}, {cy:.1f}, {cz:.1f}] mm")
    print(f"  运动半径: {radius:.1f} mm")
    print(f"  请将以上值更新到 config/settings.py 中的 JOINT_CENTER 和 JOINT_RADIUS")

    return [cx, cy, cz], radius
