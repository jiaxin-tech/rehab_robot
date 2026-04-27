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


def run(robot_ip, sensor_ip, subject_id, session_id, do_calibrate):
    robot = DobotCR5(ip=robot_ip)
    force = ForceSensor(ip=sensor_ip)

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
        logger.info("▶ 阶段1：慢速全程扫描 (5次)")
        sweep_wps = generate_slow_sweep()
        robot.set_speed(3)   # 极慢

        for rep in range(5):
            guard.check()
            logger.info(f"  慢速扫描 {rep+1}/5")
            robot.move_j(sweep_wps[0])
            robot.wait_idle()
            time.sleep(0.3)

            collector.start_episode()
            for wp in sweep_wps[1:]:
                guard.check()
                robot.move_l(wp)
                # 等待运动中持续采集
                deadline = time.time() + 0.5  # 每段最多等0.5s
                while time.time() < deadline:
                    collector.record_sample()
                    time.sleep(dt)
            collector.end_episode(comfort_label=-1)
            time.sleep(0.3)

        # ── 阶段2：持续激励轨迹（主力训练数据）────────
        logger.info("▶ 阶段2：持续激励轨迹 (20次)")
        robot.set_speed(settings.INIT_SPEED_RATIO)
        excit_wps = generate_excitation_trajectory()

        for rep in range(20):
            guard.check()
            logger.info(f"  激励轨迹 {rep+1}/20")

            # 先回起点
            robot.move_j(excit_wps[0])
            robot.wait_idle()
            time.sleep(0.3)

            collector.start_episode()
            t_start = time.time()
            wp_idx  = 0

            while wp_idx < len(excit_wps):
                guard.check()

                # ServoJ发下一个路径点
                # 注意：ServoJ需要关节坐标，这里用move_l代替（简化）
                # 真实部署时换成robot.servo_j(ik(excit_wps[wp_idx]))
                robot.move_l(excit_wps[wp_idx])
                collector.record_sample()
                time.sleep(dt)
                wp_idx += 1

            collector.end_episode(comfort_label=-1)
            time.sleep(0.3)

    except KeyboardInterrupt:
        logger.info("用户中止采集")
        robot.stop()
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
    args = parser.parse_args()

    if args.label_only:
        label_episodes(settings.DATA_DIR)
    else:
        run(args.robot_ip, args.sensor_ip,
            args.subject, args.session, args.calibrate)


if __name__ == "__main__":
    main()
