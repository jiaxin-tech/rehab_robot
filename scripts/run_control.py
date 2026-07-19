"""Base-frame, trajectory-tangent MPC control using valid robot snapshots only."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from collection.safety_guard import SafetyGuard
from collection.snapshot import read_live_robot_state_sample
from collection.trajectory import TrajectoryGeometry, project_along_tangent, generate_excitation_trajectory
from config import settings
from control.mpc_controller import MPCController
from hardware.windows import RokaeInternalWrenchSource, RokaeRobot
from models.comfort_net import ComfortPredictor
from models.pinn import OnlineTangentialPINN
from utils.logger import get_logger


logger = get_logger("RunControl")
PINN_WINDOW = 200


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


def _wait_stable(robot: RokaeRobot, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    stable_since: float | None = None
    while time.monotonic() < deadline:
        state = robot.get_state_frame()
        velocity = state.tcp_linear_velocity_mps
        speed = (
            math.sqrt(sum(value * value for value in velocity))
            if velocity is not None
            else None
        )
        if state.valid and state.operation_state == "IDLE" and speed is not None and speed <= settings.STABLE_TCP_SPEED_MPS:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= settings.STABLE_DURATION_S:
                return
        else:
            stable_since = None
        time.sleep(0.02)
    raise TimeoutError("Robot did not reach a stable idle state")


def run(robot_ip: str, subject_id: str) -> None:
    if not settings.BASE_WRENCH_ROTATION_VERIFIED:
        raise RuntimeError(
            "Refusing MPC: set BASE_WRENCH_ROTATION_VERIFIED=True only after the "
            "empty-load and known-direction validation procedure succeeds."
        )
    comfort_predictor = ComfortPredictor(settings.COMFORT_MODEL_PATH)
    trajectory = TrajectoryGeometry(generate_excitation_trajectory())
    ref_arc_m = trajectory.arc_at_waypoint_m
    ref_velocity_mps = np.gradient(ref_arc_m, settings.MPC_DT)
    ref_states = np.stack([ref_arc_m, ref_velocity_mps], axis=1)
    mpc = MPCController(control_axis=settings.CONTROL_AXIS)
    mpc.set_comfort_predictor(comfort_predictor)

    def scalar_arc_to_base_context(
        arc_length_m: float, velocity_tangent_mps: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map MPC's scalar arc coordinate to full base-frame ComfortNet input."""
        total = trajectory.total_arc_length_m
        normalized_s = min(1.0, max(0.0, float(arc_length_m) / total))
        pose = trajectory.pose_at_normalized_s(normalized_s)
        projection, reason = trajectory.project(
            pose[:3], reference_arc_length_m=arc_length_m
        )
        if projection is None:
            raise RuntimeError(f"Cannot map tangent MPC state to trajectory: {reason}")
        velocity = np.asarray(projection.tangent_base, dtype=float) * float(
            velocity_tangent_mps
        )
        return np.asarray(pose[:3], dtype=float), velocity

    mpc.set_scalar_state_mapper(scalar_arc_to_base_context)
    mpc.set_equilibrium_position(float(ref_arc_m[0]))

    robot = _make_robot(robot_ip)
    wrench_source = RokaeInternalWrenchSource(robot)
    guard: SafetyGuard | None = None
    pinn = OnlineTangentialPINN()
    pinn_lock = threading.Lock()
    identification_thread: threading.Thread | None = None
    previous_sample = None
    projection_reference_arc_m: float | None = None
    try:
        robot.connect()
        robot.clear_error()
        robot.enable(load=0.0)
        robot.set_speed(settings.INIT_SPEED_RATIO)
        wrench_source.connect()
        wrench_source.start_streaming()

        robot.move_l(trajectory.waypoints[0])
        robot.wait_idle()
        _wait_stable(robot)
        input("确认机器人在参考姿态、工具/负载正确且无人接触后按 Enter 进行软件偏置...")
        wrench_source.set_bias(settings.FORCE_BIAS_DURATION_S)
        guard = SafetyGuard(wrench_source, robot)
        guard.check()
        guard.start()

        robot.disable()
        robot.enable_realtime(load=0.0)
        robot.start_realtime_cartesian(trajectory.waypoints[0])
        last_target = np.asarray(trajectory.waypoints[0], dtype=float)
        time_buffer: list[float] = []
        arc_buffer: list[float] = []
        force_buffer: list[float] = []
        t0: float | None = None
        warm_start = None
        logger.info("=== 切向 MPC 控制启动: subject=%s ===", subject_id)

        for step in range(len(ref_states) - settings.MPC_HORIZON - 1):
            loop_started_s = time.monotonic()
            guard.check()
            sample = read_live_robot_state_sample(
                robot, wrench_source, previous_sample=previous_sample
            )
            previous_sample = sample
            if not sample.valid:
                raise RuntimeError(f"Invalid control snapshot: {sample.invalid_reason}")
            projection, projection_reason = trajectory.project(
                sample.tcp_position_m,
                reference_arc_length_m=projection_reference_arc_m,
            )
            if projection is None:
                raise RuntimeError(f"Invalid trajectory projection: {projection_reason}")
            projection_reference_arc_m = projection.arc_length_m
            tangent_velocity = project_along_tangent(
                sample.tcp_linear_velocity_mps, projection.tangent_base
            )
            tangent_force = project_along_tangent(
                sample.cartesian_force_base_n, projection.tangent_base
            )
            if tangent_velocity is None or tangent_force is None:
                raise RuntimeError("Tangent velocity/force unavailable for MPC")
            pose = np.asarray([*sample.tcp_position_m, *sample.tcp_orientation_rad], dtype=float)
            linear_velocity = np.asarray(sample.tcp_linear_velocity_mps, dtype=float)
            force_base = np.asarray(sample.cartesian_force_base_n, dtype=float)
            x0 = np.asarray([projection.arc_length_m, tangent_velocity], dtype=float)
            if t0 is None:
                t0 = sample.sample_time_s
            time_buffer.append(sample.sample_time_s - t0)
            arc_buffer.append(projection.arc_length_m)
            force_buffer.append(tangent_force)
            if len(time_buffer) > PINN_WINDOW:
                time_buffer.pop(0)
                arc_buffer.pop(0)
                force_buffer.pop(0)

            # Identification is deliberately off the 50 Hz command path.  A
            # stale or failed background fit leaves the last safe parameters in
            # place rather than blocking the controller with hundreds of epochs.
            if (
                step % 100 == 0
                and len(time_buffer) >= 50
                and (identification_thread is None or not identification_thread.is_alive())
            ):
                fit_inputs = (list(time_buffer), list(arc_buffer), list(force_buffer))

                def update_pinn() -> None:
                    nonlocal pinn
                    # Train an isolated candidate. Holding the shared lock for
                    # 100 epochs would make the 50 Hz control loop block despite
                    # the background-thread intent.
                    candidate = OnlineTangentialPINN()
                    try:
                        candidate.update(*fit_inputs, epochs=100)
                        if candidate.is_ready:
                            with pinn_lock:
                                pinn = candidate
                    except Exception as exc:
                        logger.warning("后台切向 PINN 更新失败: %s", exc)

                identification_thread = threading.Thread(
                    target=update_pinn, name="tangential-pinn-fit", daemon=True
                )
                identification_thread.start()

            with pinn_lock:
                if pinn.is_ready:
                    params = pinn.get_params()
                    mpc.set_patient_params(params["M"], params["B"], params["K"])
                    equilibrium_arc_m = params.get("equilibrium_arc_m")
                    if equilibrium_arc_m is not None and math.isfinite(
                        float(equilibrium_arc_m)
                    ):
                        mpc.set_equilibrium_position(float(equilibrium_arc_m))

            comfort_score = comfort_predictor.predict(
                fx=force_base[0], fy=force_base[1], fz=force_base[2],
                x=pose[0], y=pose[1], z=pose[2],
                vx=linear_velocity[0], vy=linear_velocity[1], vz=linear_velocity[2],
                tactile=None,
            )
            reference = ref_states[step : step + settings.MPC_HORIZON + 1]
            acceleration, warm_start = mpc.solve(
                x0,
                reference,
                force_base,
                warm_start,
                pose_context=pose,
                velocity_context=linear_velocity,
            )
            next_target, _, _ = mpc.acceleration_to_trajectory_pose(
                trajectory,
                projection.arc_length_m,
                tangent_velocity,
                acceleration,
            )
            guard.check_target(last_target, next_target)
            robot.set_realtime_cartesian_target(next_target)
            last_target = next_target

            if step % 50 == 0:
                logger.info(
                    "step=%d arc=%.4fm Ft=%.2fN comfort=%.3f pinn=%s",
                    step, projection.arc_length_m, tangent_force, comfort_score, pinn.is_ready,
                )
            elapsed_s = time.monotonic() - loop_started_s
            if elapsed_s < settings.MPC_DT:
                time.sleep(settings.MPC_DT - elapsed_s)
            elif elapsed_s > settings.MPC_DT * 1.5:
                logger.warning("控制周期超期: %.1fms", elapsed_s * 1000.0)
    except KeyboardInterrupt:
        logger.warning("用户中止控制")
        robot.stop()
    except BaseException:
        logger.exception("控制异常，执行安全停止")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ROKAE 切向 PINN + MPC 控制")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--subject", default="subject_001")
    args = parser.parse_args()
    run(args.robot_ip, args.subject)


if __name__ == "__main__":
    main()
