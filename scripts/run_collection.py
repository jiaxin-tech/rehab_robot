# scripts/run_collection.py
# 入口：数据采集

import argparse
import time
import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from hardware.dobot_cr5 import DobotCR5
from hardware.force_sensor import ForceSensor
from collection.safety_guard import SafetyGuard
from collection.trajectory import (
    generate_excitation_trajectory,
    generate_rehab_trajectory,
    calibrate_joint_center,
)
from collection.collector import DataCollector, label_episodes
from collection.synthetic_risk import write_synthetic_risk_episodes
from models.pinn import OnlinePINN
from utils.logger import get_logger

logger = get_logger("RunCollection")


# ── 清理 __pycache__ ─────────────────────────────────
def _clean_pycache(root: str = None):
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    removed = 0
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == "__pycache__":
                full = os.path.join(dirpath, d)
                shutil.rmtree(full, ignore_errors=True)
                removed += 1
    logger.info(f"已清理 {removed} 个 __pycache__ 目录")


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


def _servo_p(robot, pose):
    if hasattr(robot, "servo_p"):
        return robot.servo_p(*[float(v) for v in pose[:6]])
    return _move_l(robot, pose)


def _get_tool_orientation(robot):
    configured = getattr(settings, "TOOL_DOWN_ORIENTATION", None)
    if configured is not None:
        orientation = [float(v) for v in configured[:3]]
        logger.info(
            "使用 settings.TOOL_DOWN_ORIENTATION 锁定末端姿态: "
            f"rx={orientation[0]:.2f}, ry={orientation[1]:.2f}, rz={orientation[2]:.2f}"
        )
        return orientation

    pose = robot.get_cartesian_pose()
    if pose is None or len(pose) < 6:
        raise RuntimeError("无法读取当前末端姿态，请检查机器人反馈连接")

    orientation = [float(v) for v in pose[3:6]]
    logger.info(
        "已将当前末端姿态锁定为夹爪向下姿态: "
        f"rx={orientation[0]:.2f}, ry={orientation[1]:.2f}, rz={orientation[2]:.2f}"
    )
    return orientation


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
    """执行一段轨迹，同时把每帧数据存入collector缓冲区，供后续infer_mbk使用"""
    collector.start_episode()
    next_tick = time.perf_counter()
    for wp in waypoints:
        guard.check()
        _servo_p(robot, wp)
        collector.record_sample()
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.perf_counter()))


def _infer_and_write_mbk(collector, pinn: OnlinePINN, epochs: int = 300):
    """
    从collector的当前episode缓冲区取数据，调用PINN推M/B/K，
    然后把结果写回collector的最后一个episode（追加列）。
    """
    buf = collector.get_current_episode_buffer()
    if buf is None or len(buf["t"]) < 30:
        logger.warning("infer_mbk: 当前episode数据不足30帧，跳过M/B/K推理")
        return

    params = pinn.infer_mbk(
        t_buf=buf["t"],
        xyz_buf=buf["xyz"],
        F_buf=buf["F"],
        epochs=epochs,
    )
    if params is None:
        return

    collector.write_mbk_to_episode(params)
    logger.info(
        f"M/B/K已写入episode | "
        f"M=({params['Mx']:.3f},{params['My']:.3f},{params['Mz']:.3f}) | "
        f"B=({params['Bx']:.3f},{params['By']:.3f},{params['Bz']:.3f}) | "
        f"K=({params['Kx']:.3f},{params['Ky']:.3f},{params['Kz']:.3f})"
    )


def _default_rehab_variant() -> dict:
    return {
        "name": "default",
        "range_scale": settings.REHAB_RANGE_SCALE,
        "cycles": settings.REHAB_CYCLES,
        "duration": settings.REHAB_DURATION,
    }


def _rehab_variant(rep: int, profile: str = "mixed") -> dict:
    normal = list(getattr(settings, "REHAB_VARIANTS", None) or [_default_rehab_variant()])
    discomfort = list(getattr(settings, "REHAB_UNCOMFORTABLE_VARIANTS", None) or [])

    if profile == "normal":
        variants = normal
    elif profile == "safe-discomfort":
        variants = discomfort or normal
    else:
        variants = normal + discomfort

    return variants[rep % len(variants)]


def run(robot_ip, sensor_ip, subject_id, session_id, do_calibrate,
        n_excitations: int = 20,
        collect_kind: str = "pinn", n_rehab: int = 5,
        rehab_profile: str = "mixed"):

    robot = _make_robot(robot_ip)
    force = ForceSensor(ip=sensor_ip)
    pinn  = OnlinePINN()

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
    tool_orientation = _get_tool_orientation(robot)
    guard = SafetyGuard(force, robot)
    guard.start()

    # ── 采集器 ────────────────────────────────────────
    collector = DataCollector(robot, force, subject_id, session_id, mode="passive")
    dt = settings.COLLECT_DT

    logger.info("=== 开始采集 ===  (Ctrl+C 随时中止)")

    try:
        if collect_kind in ("pinn", "both"):
            logger.info(f"▶ PINN阶段：持续激励轨迹 ({n_excitations}次，不做舒适度标签)")
            _set_speed(robot, settings.INIT_SPEED_RATIO)
            excit_wps = generate_excitation_trajectory(orientation=tool_orientation)

            for rep in range(n_excitations):
                guard.check()
                logger.info(f"  激励轨迹 {rep+1}/{n_excitations}")
                _move_l(robot, excit_wps[0])
                _wait_idle(robot)
                time.sleep(0.3)

                collector.set_trajectory_type("excitation", "multi_sine")
                _run_waypoints(robot, guard, collector, excit_wps, dt)
                collector.end_episode(comfort_label=-1)
                time.sleep(0.3)

        # ── 阶段3：康复轨迹（舒适度标签 + M/B/K推理）──
        if collect_kind in ("comfort", "both"):
            logger.info(f"▶ 舒适度阶段：康复轨迹 ({n_rehab}次，需要人工标签, profile={rehab_profile})")
            _set_speed(robot, settings.INIT_SPEED_RATIO)

            for rep in range(n_rehab):
                guard.check()
                variant = _rehab_variant(rep, rehab_profile)
                variant_name = str(variant.get("name", f"variant_{rep+1}"))
                logger.info(
                    f"  康复轨迹 {rep+1}/{n_rehab} | {variant_name} "
                    f"range={variant.get('range_scale', settings.REHAB_RANGE_SCALE)} "
                    f"cycles={variant.get('cycles', settings.REHAB_CYCLES)} "
                    f"duration={variant.get('duration', settings.REHAB_DURATION)}s"
                )
                rehab_wps = generate_rehab_trajectory(
                    duration=float(variant.get("duration", settings.REHAB_DURATION)),
                    range_scale=float(variant.get("range_scale", settings.REHAB_RANGE_SCALE)),
                    cycles=float(variant.get("cycles", settings.REHAB_CYCLES)),
                    orientation=tool_orientation,
                )
                _move_l(robot, rehab_wps[0])
                _wait_idle(robot)
                time.sleep(0.3)

                collector.set_trajectory_type("rehab", variant_name)
                _run_waypoints(robot, guard, collector, rehab_wps, dt)

                # 先推M/B/K，写入CSV，再打comfort标签
                # 顺序很重要：infer_mbk在end_episode之前，这样数据还在缓冲区
                logger.info("  正在推理M/B/K参数...")
                _infer_and_write_mbk(collector, pinn, epochs=300)

                comfort_label = _ask_comfort_label()
                collector.end_episode(comfort_label=comfort_label)
                time.sleep(0.3)

    except KeyboardInterrupt:
        logger.info("用户中止采集")
        _stop_robot(robot)
    except RuntimeError as e:
        logger.error(f"安全停止: {e}")
    finally:
        try:
            _stop_robot(robot)
        except Exception as e:
            logger.warning(f"停止机器人失败: {e}")
        guard.stop()
        force.disconnect()
        robot.disable()
        robot.disconnect()
        logger.info("=== 采集结束 ===")
        _clean_pycache()


def main():
    parser = argparse.ArgumentParser(description="康复机器人数据采集")
    parser.add_argument("--robot-ip",   default=settings.ROBOT_IP)
    parser.add_argument("--sensor-ip",  default=settings.SENSOR_IP)
    parser.add_argument("--subject",    default="subject_001")
    parser.add_argument("--session",    default="session_01")
    parser.add_argument("--calibrate",  action="store_true", help="采集前做关节中心标定")
    parser.add_argument("--label-only", action="store_true", help="只做标注，不采集")
    parser.add_argument("--excitations",type=int, default=3,  help="激励轨迹episode数量")
    parser.add_argument("--collect-kind", choices=("pinn", "comfort", "both"),
                        default="pinn", help="pinn=只采PINN数据，comfort=只采康复舒适度数据，both=都采")
    parser.add_argument("--rehab-episodes", type=int, default=1, help="康复轨迹舒适度episode数量")
    parser.add_argument("--rehab-profile", choices=("normal", "mixed", "safe-discomfort"),
                        default="mixed",
                        help="normal=原康复轨迹，mixed=原轨迹+安全不适轨迹，safe-discomfort=只采安全不适轨迹")
    parser.add_argument("--synthetic-risk-episodes", type=int, default=0,
                        help="追加离线危险/明显不适episode数量；只写CSV，不执行机器人，comfort=2")
    parser.add_argument("--synthetic-risk-only", action="store_true",
                        help="只生成离线危险/明显不适CSV，不连接机器人")
    args = parser.parse_args()

    if args.synthetic_risk_only:
        n_synth = args.synthetic_risk_episodes
        if n_synth <= 0:
            n_synth = len(getattr(settings, "SYNTHETIC_RISK_VARIANTS", [])) or 3
        write_synthetic_risk_episodes(args.subject, args.session, n_synth)
    elif args.label_only:
        label_episodes(settings.DATA_DIR)
    else:
        run(args.robot_ip, args.sensor_ip,
            args.subject, args.session, args.calibrate,
            n_excitations=args.excitations,
            collect_kind=args.collect_kind, n_rehab=args.rehab_episodes,
            rehab_profile=args.rehab_profile)
        if args.synthetic_risk_episodes > 0:
            write_synthetic_risk_episodes(args.subject, args.session, args.synthetic_risk_episodes)


if __name__ == "__main__":
    main()
