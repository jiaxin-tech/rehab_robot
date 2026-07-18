# scripts/run_collection.py
# 入口：数据采集

import argparse
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import settings
from hardware.windows import RokaeForceSensor, RokaeRobot
from collection.safety_guard import SafetyGuard
from collection.trajectory import (
    generate_excitation_trajectory,
    generate_slow_sweep,
    calibrate_joint_center,
)
from collection.collector import DataCollector, label_episodes
from utils.logger import get_logger

logger = get_logger("RunCollection")


def _make_robot(robot_ip: str) -> RokaeRobot:
    return RokaeRobot(
        ip_address=robot_ip,
        local_ip=settings.ROBOT_LOCAL_IP,
        robot_class=settings.ROBOT_CLASS,
        state_interval_ms=settings.ROBOT_STATE_MS,
        max_linear_speed_mm_s=settings.ROBOT_MAX_SPEED,
        command_cache_size=settings.ROBOT_CMD_CACHE,
        rt_network_tolerance_percent=settings.ROBOT_RT_NETWORK_TOLERANCE,
        rt_filter_hz=settings.ROBOT_RT_FILTER_HZ,
    )


def run(robot_ip, subject_id, session_id, do_calibrate,
        n_sweeps: int = 5, n_excitations: int = 20):
    robot = _make_robot(robot_ip)
    force = RokaeForceSensor(robot)

    # ── 连接 ─────────────────────────────────────────
    robot.connect()
    force.connect()
    logger.info("等待反馈稳定 2s...")
    time.sleep(2.0)

    # ── 使能 ─────────────────────────────────────────
    robot.clear_error()
    robot.enable(load=0.0)
    robot.set_speed(settings.INIT_SPEED_RATIO)
    time.sleep(1.0)

    # ── 力传感器准备 ─────────────────────────────────
    try:
        force.start_streaming()
        time.sleep(0.5)
        force.set_bias()
        time.sleep(0.3)
    except Exception:
        force.disconnect()
        robot.disable()
        robot.disconnect()
        raise

    guard = None
    try:
        # ── 可选：标定关节中心 ────────────────────────
        if do_calibrate:
            center, radius = calibrate_joint_center(robot)
            logger.info(
                f"标定完成，请更新settings.py: center={center}, radius={radius:.1f}"
            )
            input("更新settings.py后按回车继续...")

        # ── 安全守卫与采集器 ──────────────────────────
        guard = SafetyGuard(force, robot)
        guard.start()
        collector = DataCollector(
            robot, force, subject_id, session_id, mode="passive"
        )
        dt = settings.COLLECT_DT
        logger.info("=== 开始采集 ===  (Ctrl+C 随时中止)")

        # ── 阶段1：慢速扫描（ROM探测 + K辨识）─────────
        logger.info(f"▶ 阶段1：慢速全程扫描 ({n_sweeps}次)")
        sweep_wps = generate_slow_sweep()
        robot.set_speed(3)   # 极慢

        for rep in range(n_sweeps):
            guard.check()
            logger.info(f"  慢速扫描 {rep+1}/{n_sweeps}")
            robot.move_l(sweep_wps[0])
            robot.wait_idle()
            time.sleep(0.3)

            collector.start_episode()
            for wp in sweep_wps[1:]:
                guard.check()
                robot.move_l(wp)
                # 等待运动中持续采集
                deadline = time.perf_counter() + 0.5  # 每段最多等0.5s
                next_tick = time.perf_counter()
                while time.perf_counter() < deadline:
                    collector.record_sample()
                    next_tick += dt
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
            collector.end_episode(comfort_label=-1)
            time.sleep(0.3)

        # ── 阶段2：实时持续激励轨迹（主力训练数据）────
        logger.info(f"▶ 阶段2：持续激励轨迹 ({n_excitations}次)")
        excit_wps = generate_excitation_trajectory()

        if n_excitations > 0:
            # NRT 只负责安全地到达起点；连续 50 Hz 路径点交给 0.7.0
            # 实时回调，避免把 MoveLCommand 当作伺服指令堆入路径队列。
            robot.move_l(excit_wps[0])
            robot.wait_idle()
            time.sleep(0.3)
            robot.disable()
            robot.enable_realtime(load=0.0)
            robot.start_realtime_cartesian(excit_wps[0])

        for rep in range(n_excitations):
            guard.check()
            logger.info(f"  激励轨迹 {rep+1}/{n_excitations}")

            if rep > 0:
                # 用 1 秒线性过渡返回轨迹起点，避免 episode 之间目标跳变。
                transition_start = np.asarray(
                    robot.get_cartesian_pose(), dtype=float
                )
                transition_target = np.asarray(excit_wps[0], dtype=float)
                transition_steps = max(1, int(round(1.0 / dt)))
                next_tick = time.perf_counter()
                for index in range(1, transition_steps + 1):
                    guard.check()
                    alpha = index / transition_steps
                    pose = (
                        (1.0 - alpha) * transition_start
                        + alpha * transition_target
                    )
                    robot.set_realtime_cartesian_target(pose)
                    next_tick += dt
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
                time.sleep(0.3)

            collector.start_episode()
            next_tick = time.perf_counter()
            for waypoint in excit_wps:
                guard.check()
                robot.set_realtime_cartesian_target(waypoint)
                collector.record_sample()
                next_tick += dt
                time.sleep(max(0.0, next_tick - time.perf_counter()))

            collector.end_episode(comfort_label=-1)
            time.sleep(0.3)

        if n_excitations > 0:
            robot.stop_realtime()

    except KeyboardInterrupt:
        logger.info("用户中止采集")
        robot.stop()
    except RuntimeError as e:
        logger.error(f"安全停止: {e}")
        robot.stop()
    finally:
        if guard is not None:
            guard.stop()
        try:
            robot.stop_realtime()
        except Exception as exc:
            logger.warning(f"停止实时模式失败: {exc}")
        force.disconnect()
        try:
            robot.disable()
        except Exception as exc:
            logger.warning(f"机械臂下电失败: {exc}")
        robot.disconnect()
        logger.info("=== 采集结束 ===")


def main():
    parser = argparse.ArgumentParser(description="康复机器人数据采集")
    parser.add_argument("--robot-ip",   default=settings.ROBOT_IP)
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
        run(args.robot_ip, args.subject, args.session, args.calibrate,
            n_sweeps=args.sweeps, n_excitations=args.excitations)


if __name__ == "__main__":
    main()
