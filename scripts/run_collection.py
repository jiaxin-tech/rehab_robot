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


def run(robot_ip, sensor_ip, subject_id, session_id, do_calibrate,
        n_sweeps: int = 5, n_excitations: int = 20):
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
        center, radius = calibrate_joint_center(robot)
        logger.info(f"标定完成，请更新settings.py: center={center}, radius={radius:.1f}")
        input("更新settings.py后按回车继续...")

    # ── 安全守卫 ─────────────────────────────────────
    guard = SafetyGuard(force, robot)
    guard.start()

    # ── 采集器 ────────────────────────────────────────
    collector = DataCollector(robot, force, subject_id, session_id, mode="passive")
    dt = settings.COLLECT_DT

    logger.info("=== 开始采集 ===  (Ctrl+C 随时中止)")

    try:
        # ── 阶段1：慢速扫描（ROM探测 + K辨识）─────────
        logger.info(f"▶ 阶段1：慢速全程扫描 ({n_sweeps}次)")
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
        logger.info(f"▶ 阶段2：持续激励轨迹 ({n_excitations}次)")
        _set_speed(robot, settings.INIT_SPEED_RATIO)
        excit_wps = generate_excitation_trajectory()

        for rep in range(n_excitations):
            guard.check()
            logger.info(f"  激励轨迹 {rep+1}/{n_excitations}")

            # 先回起点
            _move_l(robot, excit_wps[0])
            _wait_idle(robot)
            time.sleep(0.3)

            collector.start_episode()
            wp_idx = 0
            next_tick = time.perf_counter()

            while wp_idx < len(excit_wps):
                guard.check()

                # 这里发送笛卡尔路径点；如后续加入IK，可在此替换成 ServoJ。
                _move_l(robot, excit_wps[wp_idx])
                collector.record_sample()
                next_tick += dt
                time.sleep(max(0.0, next_tick - time.perf_counter()))
                wp_idx += 1

            collector.end_episode(comfort_label=-1)
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
    args = parser.parse_args()

    if args.label_only:
        label_episodes(settings.DATA_DIR)
    else:
        run(args.robot_ip, args.sensor_ip,
            args.subject, args.session, args.calibrate,
            n_sweeps=args.sweeps, n_excitations=args.excitations)


if __name__ == "__main__":
    main()
