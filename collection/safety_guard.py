"""Fail-closed safety checks for time-qualified robot state and wrench data."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Sequence

from config import settings
from utils.logger import get_logger


logger = get_logger("SafetyGuard")


class SafetyGuard:
    """Continuously reject stale, invalid, excessive, or collision state.

    This guard supplements—not replaces—the ROKAE controller's hardware safety,
    collision settings, emergency stop, workspace configuration, and trained
    operator supervision.  Internal force estimates depend on the configured
    tool/load and are not an independent safety sensor.
    """

    def __init__(
        self,
        wrench_source: Any,
        robot: Any,
        threshold: float = settings.MAX_FORCE_N,
        max_torque_nm: float = settings.MAX_CARTESIAN_TORQUE_NM,
        max_joint_external_torque_nm: Sequence[float] = settings.MAX_JOINT_EXTERNAL_TORQUE_NM,
    ) -> None:
        self.wrench_source = wrench_source
        self.robot = robot
        self.force_threshold_n = float(threshold)
        self.torque_threshold_nm = float(max_torque_nm)
        self.joint_external_threshold_nm = tuple(
            float(value) for value in max_joint_external_torque_nm
        )
        if self.force_threshold_n <= 0 or self.torque_threshold_nm <= 0:
            raise ValueError("Safety force and torque thresholds must be positive")
        if len(self.joint_external_threshold_nm) != 6 or any(
            value <= 0 for value in self.joint_external_threshold_nm
        ):
            raise ValueError("MAX_JOINT_EXTERNAL_TORQUE_NM must contain six positive values")

        self.triggered = False
        self.trigger_reason: str | None = None
        self._active = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_collision_poll_s = float("-inf")
        self._joint_soft_limits_rad: tuple[tuple[float, float], ...] | None = None
        self._joint_soft_limit_error: str | None = None
        self._load_joint_soft_limits()

    def _load_joint_soft_limits(self) -> None:
        """Read controller soft limits once; never poll this normal API at 200 Hz."""
        if not hasattr(self.robot, "get_joint_soft_limits_rad"):
            self._joint_soft_limit_error = "joint_soft_limit_api_unavailable"
            return
        try:
            limits = self.robot.get_joint_soft_limits_rad()
            if limits is None or len(limits) != 6:
                raise ValueError("expected six [lower, upper] joint limits")
            parsed = tuple((float(pair[0]), float(pair[1])) for pair in limits)
            if any(
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower >= upper
                for lower, upper in parsed
            ):
                raise ValueError("joint soft limits are non-finite or inverted")
            self._joint_soft_limits_rad = parsed
            self._joint_soft_limit_error = None
        except Exception as exc:
            self._joint_soft_limits_rad = None
            self._joint_soft_limit_error = (
                f"joint_soft_limit_query_error:{type(exc).__name__}:{exc}"
            )

    @staticmethod
    def _norm(values: Sequence[float] | None) -> float | None:
        if values is None or len(values) != 3:
            return None
        if not all(math.isfinite(float(value)) for value in values):
            return None
        return math.sqrt(sum(float(value) ** 2 for value in values))

    def start(self) -> None:
        if self._active:
            return
        self.triggered = False
        self.trigger_reason = None
        self._active = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="robot-safety-guard", daemon=True
        )
        self._thread.start()
        logger.info(
            "安全守卫已启动: |F|<=%.2fN, |T|<=%.2fNm",
            self.force_threshold_n,
            self.torque_threshold_nm,
        )

    def _loop(self) -> None:
        period_s = 1.0 / settings.SAFETY_CHECK_HZ
        while self._active and not self._stop_event.is_set():
            reason = self._evaluate()
            if reason:
                self._trigger(reason)
                break
            self._stop_event.wait(period_s)

    def _evaluate(self) -> str | None:
        try:
            if not settings.CONTROLLER_COLLISION_CONFIGURATION_CONFIRMED:
                return "controller_collision_configuration_unconfirmed"
            if not hasattr(self.robot, "get_state_frame"):
                return "robot_state_snapshot_api_missing"
            state = self.robot.get_state_frame()
            # Obtain the host check time only after reading the cached state;
            # otherwise a concurrently published frame can falsely look future.
            now_s = time.monotonic()
            if not state.valid:
                return state.invalid_reason or "robot_state_invalid"
            if state.host_monotonic_time_s is None:
                return "robot_state_missing_time"
            if state.host_monotonic_time_s > now_s:
                return "robot_state_time_in_future"
            if now_s - state.host_monotonic_time_s > settings.MAX_ROBOT_STATE_AGE_S:
                return "robot_state_stale"
            if state.controller_error:
                return f"controller_error:{state.controller_error}"
            if state.collision_state is True:
                return "robot_collision"
            if state.operation_state in {"DISCONNECTED", "UNKNOWN"}:
                return f"robot_operation_state:{state.operation_state}"

            # Safety event is a separate, unsynchronized SDK query.  Poll at a
            # bounded rate to avoid starving the state/wrench SDK lock.
            if (
                hasattr(self.robot, "get_collision_state")
                and now_s - self._last_collision_poll_s >= 1.0 / 20.0
            ):
                self._last_collision_poll_s = now_s
                collision = self.robot.get_collision_state()
                if collision is True:
                    return "robot_collision"
                if collision is None and settings.REQUIRE_COLLISION_STATE_QUERY:
                    return "collision_state_unavailable"

            if not hasattr(self.wrench_source, "snapshot"):
                return "robot_wrench_snapshot_api_missing"
            wrench = self.wrench_source.snapshot(now_s)
            wrench_check_time_s = time.monotonic()
            if not wrench.valid:
                return wrench.invalid_reason or "robot_wrench_invalid"
            if wrench.host_monotonic_time_s is None:
                return "robot_wrench_missing_time"
            if wrench.host_monotonic_time_s > wrench_check_time_s:
                return "robot_wrench_time_in_future"
            if wrench_check_time_s - wrench.host_monotonic_time_s > settings.MAX_FORCE_SAMPLE_AGE_S:
                return "robot_wrench_stale"
            force_norm = self._norm(wrench.cartesian_force_base_n)
            torque_norm = self._norm(wrench.cartesian_torque_base_nm)
            if force_norm is None or torque_norm is None:
                return "base_corrected_wrench_unavailable"
            if force_norm > self.force_threshold_n:
                return f"cartesian_force_limit:{force_norm:.3f}N"
            if torque_norm > self.torque_threshold_nm:
                return f"cartesian_torque_limit:{torque_norm:.3f}Nm"
            external = wrench.joint_external_torque_nm
            if external is None or len(external) < 6:
                return "joint_external_torque_unavailable"
            for index, (value, limit) in enumerate(
                zip(external[:6], self.joint_external_threshold_nm), start=1
            ):
                if not math.isfinite(value):
                    return "joint_external_torque_nonfinite"
                if abs(value) > limit:
                    return f"joint_external_torque_limit:q{index}:{value:.3f}Nm"
            speed_norm = self._norm(state.tcp_linear_velocity_mps)
            # First state frame lacks a numerical derivative.  It cannot be
            # used for motion, but it is not by itself an emergency condition.
            if speed_norm is not None and speed_norm > settings.MAX_TCP_SPEED_MPS:
                return f"tcp_speed_limit:{speed_norm:.4f}mps"
            joint_limit_reason = self._joint_soft_limit_reason(state.joint_position_rad)
            if joint_limit_reason:
                return joint_limit_reason
            return self._workspace_reason(state.tcp_position_m)
        except Exception as exc:
            return f"safety_check_error:{type(exc).__name__}:{exc}"

    @staticmethod
    def _workspace_reason(position_m: Sequence[float] | None) -> str | None:
        lower = settings.WORKSPACE_MIN_M
        upper = settings.WORKSPACE_MAX_M
        if lower is None or upper is None:
            return (
                "workspace_configuration_missing"
                if settings.REQUIRE_WORKSPACE_LIMITS
                else None
            )
        if position_m is None or len(position_m) != 3:
            return "workspace_position_unavailable"
        if len(lower) != 3 or len(upper) != 3:
            return "workspace_configuration_invalid"
        for axis, value in enumerate(position_m):
            if value < lower[axis] or value > upper[axis]:
                return f"workspace_limit:axis{axis}"
        return None

    def _joint_soft_limit_reason(
        self, joint_position_rad: Sequence[float] | None
    ) -> str | None:
        limits = self._joint_soft_limits_rad
        if limits is None:
            return (
                self._joint_soft_limit_error or "joint_soft_limit_unavailable"
                if settings.REQUIRE_JOINT_SOFT_LIMITS
                else None
            )
        if joint_position_rad is None or len(joint_position_rad) < 6:
            return "joint_position_unavailable"
        margin = float(settings.JOINT_SOFT_LIMIT_MARGIN_RAD)
        if not math.isfinite(margin) or margin < 0.0:
            return "joint_soft_limit_margin_invalid"
        for index, ((lower, upper), value) in enumerate(
            zip(limits, joint_position_rad[:6]), start=1
        ):
            if not math.isfinite(float(value)):
                return f"joint_position_nonfinite:q{index}"
            if float(value) <= lower + margin or float(value) >= upper - margin:
                return f"joint_soft_limit_margin:q{index}"
        return None

    def check_target(self, current_pose: Sequence[float], target_pose: Sequence[float]) -> None:
        """Validate a commanded Cartesian target before handing it to xCoreSDK."""
        if len(current_pose) < 3 or len(target_pose) < 3:
            raise ValueError("Cartesian target safety check requires xyz pose")
        displacement = math.sqrt(
            sum((float(target_pose[index]) - float(current_pose[index])) ** 2 for index in range(3))
        )
        if not math.isfinite(displacement) or displacement > settings.MAX_TARGET_STEP_M:
            self._trigger(f"target_step_limit:{displacement:.6f}m")
            raise RuntimeError(self.trigger_reason)
        reason = self._workspace_reason(target_pose[:3])
        if reason:
            self._trigger(reason)
            raise RuntimeError(self.trigger_reason)

    def _trigger(self, reason: str) -> None:
        if self.triggered:
            return
        self.triggered = True
        self.trigger_reason = reason
        self._active = False
        self._stop_event.set()
        logger.error("安全停止: %s", reason)
        try:
            self._stop_robot()
        except Exception as exc:
            logger.error("安全停止命令失败: %s", exc)

    def check(self) -> None:
        """Use from the motion/control loop; raise once any safety rule fails."""
        if self.triggered:
            raise RuntimeError(f"SafetyGuard triggered: {self.trigger_reason}")
        reason = self._evaluate()
        if reason:
            self._trigger(reason)
            raise RuntimeError(f"SafetyGuard triggered: {reason}")

    def stop(self) -> None:
        self._active = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        logger.info("安全守卫已停止")

    def _stop_robot(self) -> None:
        if hasattr(self.robot, "stop"):
            self.robot.stop()
            return
        if hasattr(self.robot, "stop_move"):
            self.robot.stop_move()
            return
        raise AttributeError("robot lacks stop/stop_move safety interface")
