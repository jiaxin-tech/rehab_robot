# scripts/run_collection.py
# 入口：数据采集

import argparse
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from hardware.dobot_cr5 import DobotCR5
from hardware.force_sensor import ForceSensor
from collection.safety_guard import SafetyGuard
from collection.trajectory import (
    generate_excitation_trajectory,
    generate_rehab_trajectory,
    generate_slow_sweep,
    calibrate_joint_center,
)
from collection.collector import DataCollector, label_episodes
from utils.logger import get_logger

logger = get_logger("RunCollection")


def _make_robot(robot_ip: str) -> DobotCR5:
    return DobotCR5(
        ip_address=robot_ip,
        dashboard_port=settings.ROBOT_DASH_PORT,
        move_port=settings.ROBOT_MOVE_PORT,
        feedback_port=settings.ROBOT_FEED_PORT,
    )


def _set_speed(robot, ratio: int):
    if hasattr(robot, "set_speed"):
        return robot.set_speed(ratio)
    return robot.set_speed_ratio(ratio)


def _move_l(robot, pose):
    if hasattr(robot, "move_l"):
        return robot.move_l(pose)
    return robot.mov_l(*[float(v) for v in pose[:6]])


def _stop_robot(robot):
    if hasattr(robot, "stop"):
        return robot.stop()
    return robot.stop_move()


def _wait_idle(robot, timeout: float = 30.0):
    if hasattr(robot, "wait_idle"):
        return robot.wait_idle()

    time.sleep(0.1)
    deadline = time.time() + timeout
    running_modes = {"RUNNING", "JOG", "RECORDING"}
    idle_seen_at = None
    while time.time() < deadline:
        if getattr(robot, "robot_mode", "") not in running_modes:
            if idle_seen_at is None:
                idle_seen_at = time.time()
            elif time.time() - idle_seen_at >= 0.2:
                return True
        else:
            idle_seen_at = None
        time.sleep(0.05)
    logger.warning(f"等待机器人空闲超时 {timeout:.1f}s，继续执行")
    return False


def _ask_comfort_label() -> int:
    label = input("本段康复轨迹舒适度标签 (0=舒适, 1=轻微不适, 2=危险/明显不适, 回车=跳过): ").strip()
    if label in ("0", "1", "2"):
        return int(label)
    return -1


def _run_waypoints(robot, guard, collector, waypoints, dt):
    collector.start_episode()
    next_tick = time.perf_counter()
    for wp in waypoints:
        guard.check()
        _move_l(robot, wp)
        collector.record_sample()
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))


def run(robot_ip, sensor_ip, subject_id, session_id, do_calibrate,
        n_sweeps: int = 5, n_excitations: int = 20,
        collect_kind: str = "pinn", n_rehab: int = 5):
    robot = _make_robot(robot_ip)
    force = ForceSensor(ip=sensor_ip)

    # ── 连接 ─────────────────────────────────────────
    robot.connect()
    force.connect()
    logger.info("等待反馈稳定 2s...")
    time.sleep(2.0)

    # ── 使能 ─────────────────────────────────────────
    robot.clear_error()
    robot.enable(load=0.0)
    _set_speed(robot, settings.INIT_SPEED_RATIO)
    time.sleep(1.0)

    # ── 力传感器准备 ─────────────────────────────────
    force.start_streaming()
    time.sleep(0.5)
    force.set_bias()
    time.sleep(0.3)

    # ── 可选：标定关节中心 ────────────────────────────
    if do_calibrate:
        center, radius, angle_min, angle_max, neutral = calibrate_joint_center(robot)
        settings.JOINT_CENTER = center
        settings.JOINT_RADIUS = radius
        settings.JOINT_ANGLE_MIN = angle_min
        settings.JOINT_ANGLE_MAX = angle_max
        settings.JOINT_NEUTRAL = neutral
        logger.info(
            "标定完成，请更新settings.py: "
            f"center={center}, radius={radius:.1f}, "
            f"angle_min={angle_min:.4f}, angle_max={angle_max:.4f}, neutral={neutral:.4f}"
        )
        input("本次运行已使用新标定值；请也更新settings.py用于下次运行。按回车继续...")

    # ── 安全守卫 ─────────────────────────────────────
    guard = SafetyGuard(force, robot)
    guard.start()

    # ── 采集器 ────────────────────────────────────────
    collector = DataCollector(robot, force, subject_id, session_id, mode="passive")
    dt = settings.COLLECT_DT

    logger.info("=== 开始采集 ===  (Ctrl+C 随时中止)")

    try:
        # ── 阶段1：慢速扫描（ROM探测 + K辨识）─────────
        if collect_kind in ("pinn", "both"):
            logger.info(f"▶ PINN阶段1：慢速全程扫描 ({n_sweeps}次，不做舒适度标签)")
            sweep_wps = generate_slow_sweep()
            _set_speed(robot, 3)   # 极慢

            for rep in range(n_sweeps):
                guard.check()
                logger.info(f"  慢速扫描 {rep+1}/{n_sweeps}")
                _move_l(robot, sweep_wps[0])
                _wait_idle(robot)
                time.sleep(0.3)

                collector.start_episode()
                for wp in sweep_wps[1:]:
                    guard.check()
                    _move_l(robot, wp)
                    # 等待运动中持续采集
                    deadline = time.perf_counter() + 0.5  # 每段最多等0.5s
                    next_tick = time.perf_counter()
                    while time.perf_counter() < deadline:
                        collector.record_sample()
                        next_tick += dt
                        time.sleep(max(0.0, next_tick - time.perf_counter()))
                collector.end_episode(comfort_label=-1)
                time.sleep(0.3)

        # ── 阶段2：持续激励轨迹（主力训练数据）────────
            logger.info(f"▶ PINN阶段2：持续激励轨迹 ({n_excitations}次，不做舒适度标签)")
            _set_speed(robot, settings.INIT_SPEED_RATIO)
            excit_wps = generate_excitation_trajectory()

            for rep in range(n_excitations):
                guard.check()
                logger.info(f"  激励轨迹 {rep+1}/{n_excitations}")

                # 先回起点
                _move_l(robot, excit_wps[0])
                _wait_idle(robot)
                time.sleep(0.3)

                _run_waypoints(robot, guard, collector, excit_wps, dt)
                collector.end_episode(comfort_label=-1)
                time.sleep(0.3)

        # ── 阶段3：康复轨迹（用于舒适度标签）───────────
        if collect_kind in ("comfort", "both"):
            logger.info(f"▶ 舒适度阶段：康复轨迹 ({n_rehab}次，需要人工标签)")
            _set_speed(robot, settings.INIT_SPEED_RATIO)
            rehab_wps = generate_rehab_trajectory()

            for rep in range(n_rehab):
                guard.check()
                logger.info(f"  康复轨迹 {rep+1}/{n_rehab}")

                _move_l(robot, rehab_wps[0])
                _wait_idle(robot)
                time.sleep(0.3)

                _run_waypoints(robot, guard, collector, rehab_wps, dt)
                comfort_label = _ask_comfort_label()
                collector.end_episode(comfort_label=comfort_label)
                time.sleep(0.3)

    except KeyboardInterrupt:
        logger.info("用户中止采集")
        _stop_robot(robot)
    except RuntimeError as e:
        logger.error(f"安全停止: {e}")
    finally:
        guard.stop()
        force.disconnect()
        robot.disable()
        robot.disconnect()
        logger.info("=== 采集结束 ===")


def main():
    parser = argparse.ArgumentParser(description="康复机器人数据采集")
    parser.add_argument("--robot-ip",   default=settings.ROBOT_IP)
    parser.add_argument("--sensor-ip",  default=settings.SENSOR_IP)
    parser.add_argument("--subject",    default="subject_001")
    parser.add_argument("--session",    default="session_01")
    parser.add_argument("--calibrate",  action="store_true", help="采集前做关节中心标定")
    parser.add_argument("--label-only", action="store_true", help="只做标注，不采集")
    parser.add_argument("--sweeps", type=int, default=5, help="慢速扫描episode数量")
    parser.add_argument("--excitations", type=int, default=20, help="激励轨迹episode数量")
    parser.add_argument("--collect-kind", choices=("pinn", "comfort", "both"),
                        default="pinn", help="pinn=只采PINN数据，comfort=只采康复舒适度数据，both=都采")
    parser.add_argument("--rehab-episodes", type=int, default=5, help="康复轨迹舒适度episode数量")
    args = parser.parse_args()

    if args.label_only:
        label_episodes(settings.DATA_DIR)
    else:
        run(args.robot_ip, args.sensor_ip,
            args.subject, args.session, args.calibrate,
            n_sweeps=args.sweeps, n_excitations=args.excitations,
            collect_kind=args.collect_kind, n_rehab=args.rehab_episodes)


if __name__ == "__main__":
    main()
