"""ROKAE episode collection entry point with explicit state/wrench validity."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from collection.collector import DataCollector, label_episodes
from collection.safety_guard import SafetyGuard
from collection.trajectory import (
    calibrate_joint_center,
    generate_excitation_trajectory,
    generate_slow_sweep,
)
from config import settings
from hardware.windows import RokaeInternalWrenchSource, RokaeRobot
from utils.logger import get_logger


logger = get_logger("RunCollection")


def _make_robot(robot_ip: str) -> RokaeRobot:
    return RokaeRobot(
        ip_address=robot_ip,
        local_ip=settings.ROBOT_LOCAL_IP,
        robot_class=settings.ROBOT_CLASS,
        state_interval_ms=settings.ROBOT_STATE_MS,
        max_linear_speed_m_s=settings.ROBOT_MAX_LINEAR_SPEED_M_S,
        command_cache_size=settings.ROBOT_CMD_CACHE,
        rt_network_tolerance_percent=settings.ROBOT_RT_NETWORK_TOLERANCE,
        rt_filter_hz=settings.ROBOT_RT_FILTER_HZ,
    )


def _speed_norm(robot: RokaeRobot) -> float | None:
    frame = robot.get_state_frame()
    velocity = frame.tcp_linear_velocity_mps
    if velocity is None:
        return None
    return math.sqrt(sum(value * value for value in velocity))


def _wait_stable(
    robot: RokaeRobot,
    timeout_s: float = 10.0,
    *,
    require_idle: bool = True,
) -> None:
    """Require a valid, low-speed state for a sustained period.

    NRT moves must reach ``IDLE``.  During the SDK-owned realtime Cartesian
    callback, however, the adapter correctly reports ``RT_CONTROLLING`` even
    while the held target is physically stable, so only the velocity/state gate
    applies there.
    """
    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    while time.monotonic() < deadline:
        frame = robot.get_state_frame()
        speed = _speed_norm(robot)
        operation_ready = (
            frame.operation_state == "IDLE"
            if require_idle
            else frame.operation_state not in {"DISCONNECTED", "UNKNOWN"}
        )
        ready = (
            frame.valid
            and operation_ready
            and speed is not None
            and speed <= settings.STABLE_TCP_SPEED_MPS
        )
        if ready:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= settings.STABLE_DURATION_S:
                return
        else:
            stable_since = None
        time.sleep(0.02)
    raise TimeoutError(
        "Robot did not remain stable below "
        f"{settings.STABLE_TCP_SPEED_MPS} m/s for {settings.STABLE_DURATION_S}s"
    )


def _move_and_verify(robot: RokaeRobot, target: np.ndarray, guard: SafetyGuard | None = None) -> None:
    """Move with an explicit completion/position/stability check, never a fixed sleep."""
    current = np.asarray(robot.get_cartesian_pose(), dtype=float)
    if guard is not None:
        guard.check()
        guard.check_target(current, target)
    robot.move_l(target)
    robot.wait_idle()
    frame = robot.get_state_frame()
    if frame.tcp_position_m is None:
        raise RuntimeError("No valid TCP pose after MoveL completion")
    position_error_m = float(
        np.linalg.norm(np.asarray(frame.tcp_position_m) - np.asarray(target[:3]))
    )
    if position_error_m > settings.ARRIVAL_POSITION_TOLERANCE_M:
        raise RuntimeError(
            f"Arrival error {position_error_m:.6f}m exceeds "
            f"{settings.ARRIVAL_POSITION_TOLERANCE_M:.6f}m"
        )
    _wait_stable(robot)


def _verify_operator_setup(robot: RokaeRobot) -> None:
    """Record what can be checked in software and require human setup confirmation."""
    metadata = robot.get_robot_metadata()
    logger.info(
        "xCoreSDK=%s, robot=%s, controller=%s, serial=%s",
        metadata.get("xcore_sdk_version"),
        metadata.get("robot_model"),
        metadata.get("controller_version"),
        metadata.get("robot_serial_number"),
    )
    logger.warning(
        "请在 HMI 上确认 tool=%r, workpiece=%r, payload=%r kg, COM=%r m；"
        "xCoreSDK v0.7.0 当前适配层不伪造这些配置的读取校验。",
        settings.TOOL_NAME,
        settings.WORKPIECE_NAME,
        settings.PAYLOAD_MASS_KG,
        settings.PAYLOAD_COM_M,
    )


def _collect_slow_sweep(
    robot: RokaeRobot,
    guard: SafetyGuard,
    collector: DataCollector,
    waypoints: np.ndarray,
    repeats: int,
) -> None:
    logger.info("▶ 阶段 1：慢速全程扫描（%d episodes）", repeats)
    robot.set_speed(3)
    for index in range(repeats):
        guard.check()
        logger.info("慢速扫描 %d/%d", index + 1, repeats)
        _move_and_verify(robot, waypoints[0], guard)
        collector.start_episode(waypoints)
        collector.start_background_sampling()
        try:
            for waypoint in waypoints[1:]:
                guard.check()
                _move_and_verify(robot, waypoint, guard)
        except BaseException as exc:
            collector.abort_episode(f"slow_sweep_error:{type(exc).__name__}:{exc}")
            raise
        else:
            collector.end_episode(comfort_label=-1, completed=True)


def _collect_realtime_excitation(
    robot: RokaeRobot,
    guard: SafetyGuard,
    collector: DataCollector,
    waypoints: np.ndarray,
    repeats: int,
) -> None:
    if repeats <= 0:
        return
    logger.info("▶ 阶段 2：连续激励轨迹（%d episodes）", repeats)
    _move_and_verify(robot, waypoints[0], guard)
    robot.disable()
    robot.enable_realtime(load=0.0)
    robot.start_realtime_cartesian(waypoints[0])
    previous_target = np.asarray(waypoints[0], dtype=float)

    def transition_to_episode_start(
        current_target: np.ndarray, episode_start: np.ndarray
    ) -> np.ndarray:
        """Return to a repeated trajectory's start with bounded RT increments."""
        displacement_m = float(np.linalg.norm(episode_start[:3] - current_target[:3]))
        steps = max(1, int(math.ceil(displacement_m / settings.MAX_TARGET_STEP_M)))
        next_tick = time.monotonic()
        last = current_target
        for step in range(1, steps + 1):
            guard.check()
            target = current_target + (episode_start - current_target) * (step / steps)
            guard.check_target(last, target)
            robot.set_realtime_cartesian_target(target)
            last = target
            next_tick += settings.COLLECT_DT
            delay_s = next_tick - time.monotonic()
            if delay_s > 0:
                time.sleep(delay_s)
        _wait_stable(robot, require_idle=False)
        return last

    try:
        for repeat in range(repeats):
            logger.info("连续激励 %d/%d", repeat + 1, repeats)
            # The excitation is not necessarily closed.  Transition between
            # episodes outside the CSV with small, guarded RT increments rather
            # than issuing one unsafe start-point jump on the next repeat.
            previous_target = transition_to_episode_start(
                previous_target, np.asarray(waypoints[0], dtype=float)
            )
            collector.start_episode(waypoints)
            collector.start_background_sampling()
            next_tick = time.monotonic()
            try:
                for waypoint in waypoints:
                    guard.check()
                    target = np.asarray(waypoint, dtype=float)
                    # Check inter-command increment, rather than actual lagging
                    # TCP pose, so a normal servo tracking lag is not mistaken
                    # for a requested target jump.
                    guard.check_target(previous_target, target)
                    robot.set_realtime_cartesian_target(target)
                    previous_target = target
                    next_tick += settings.COLLECT_DT
                    delay_s = next_tick - time.monotonic()
                    if delay_s > 0:
                        time.sleep(delay_s)
                    elif -delay_s > settings.COLLECT_DT:
                        logger.warning("激励目标更新超期 %.1f ms", -delay_s * 1000.0)
                        next_tick = time.monotonic()

                # Hold the last target while a separate collection thread saves
                # a short static tail.  No motion arrival is assumed from sleep.
                tail_end = time.monotonic() + settings.POST_MOTION_RECORD_S
                while time.monotonic() < tail_end:
                    guard.check()
                    time.sleep(min(0.02, tail_end - time.monotonic()))
                _wait_stable(robot, require_idle=False)
            except BaseException as exc:
                collector.abort_episode(f"realtime_excitation_error:{type(exc).__name__}:{exc}")
                raise
            else:
                collector.end_episode(comfort_label=-1, completed=True)
    finally:
        robot.stop_realtime()


def run(
    robot_ip: str,
    subject_id: str,
    session_id: str,
    do_calibrate: bool,
    n_sweeps: int = 5,
    n_excitations: int = 20,
) -> None:
    robot = _make_robot(robot_ip)
    wrench_source = RokaeInternalWrenchSource(robot)
    guard: SafetyGuard | None = None
    collector: DataCollector | None = None
    stop_reason: str | None = None
    try:
        robot.connect()
        _verify_operator_setup(robot)
        robot.clear_error()
        robot.enable(load=0.0)
        robot.set_speed(settings.INIT_SPEED_RATIO)

        if do_calibrate:
            center, radius = calibrate_joint_center(robot)
            logger.info("标定完成：center=%s m, radius=%.4fm", center, radius)
            input("确认已将标定结果写入 settings.py 后按 Enter 继续...")

        # Start controller-owned wrench queries.  This is not an external sensor
        # stream and no robot force hardware calibration is invoked.
        wrench_source.connect()
        wrench_source.start_streaming()

        sweep_waypoints = generate_slow_sweep()
        _move_and_verify(robot, sweep_waypoints[0])
        if settings.FORCE_BIAS_REQUIRE_CONFIRMATION:
            input(
                "机器人已到固定参考姿态且静止。确认工具/负载正确、无人接触后按 Enter 进行软件参考偏置..."
            )
        _wait_stable(robot)
        bias = wrench_source.set_bias(settings.FORCE_BIAS_DURATION_S)
        logger.info("软件参考偏置已保存: %s", [round(value, 5) for value in bias])

        guard = SafetyGuard(wrench_source, robot)
        # Fail before motion if workspace/soft-limit configuration or live
        # snapshots are not acceptable for a human-facing collection session.
        guard.check()
        guard.start()
        collector = DataCollector(robot, wrench_source, subject_id, session_id, mode="passive")
        _collect_slow_sweep(robot, guard, collector, sweep_waypoints, n_sweeps)
        _collect_realtime_excitation(
            robot,
            guard,
            collector,
            generate_excitation_trajectory(),
            n_excitations,
        )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        logger.warning("用户中止采集")
        if collector is not None:
            collector.abort_episode(stop_reason)
        robot.stop()
    except BaseException as exc:
        stop_reason = f"collection_error:{type(exc).__name__}:{exc}"
        logger.exception("采集异常，执行安全停止")
        if collector is not None:
            collector.abort_episode(stop_reason)
        robot.stop()
        raise
    finally:
        if guard is not None:
            guard.stop()
        try:
            robot.stop_realtime()
        except Exception as exc:
            logger.warning("停止实时模式失败: %s", exc)
        wrench_source.disconnect()
        try:
            robot.disable()
        except Exception as exc:
            logger.warning("机械臂下电失败: %s", exc)
        robot.disconnect()
        logger.info("=== 采集结束%s ===", f" ({stop_reason})" if stop_reason else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROKAE 内部力估计数据采集")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--subject", default="subject_001")
    parser.add_argument("--session", default="session_01")
    parser.add_argument("--calibrate", action="store_true", help="采集前拖拽标定关节中心")
    parser.add_argument("--label-only", action="store_true", help="只做舒适度标注")
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--excitations", type=int, default=20)
    args = parser.parse_args()
    if args.sweeps < 0 or args.excitations < 0:
        parser.error("--sweeps and --excitations must be non-negative")
    if args.label_only:
        label_episodes(settings.DATA_DIR)
        return
    run(
        args.robot_ip,
        args.subject,
        args.session,
        args.calibrate,
        n_sweeps=args.sweeps,
        n_excitations=args.excitations,
    )


if __name__ == "__main__":
    main()
