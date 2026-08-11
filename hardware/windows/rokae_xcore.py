"""Rokae robot adapter backed by xCoreSDK 0.7.0 on Windows.

The realtime stream provides TCP pose and joint position in the robot *base*
frame.  It does not provide an SDK device timestamp, velocity, wrench, or
collision field.  This adapter therefore exposes receipt times and explicitly
labels finite-difference velocity estimates; it never advertises them as an
SDK-synchronous wrench/state packet.
"""

from __future__ import annotations

from datetime import timedelta
import importlib
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

from collection.state import (
    KinematicStateFrame,
    as_float_tuple,
    as_vec3,
    finite_vector,
    rpy_euler_xyz_rotation_matrix,
    transpose_rotation,
    utc_now_iso,
)


logger = logging.getLogger(__name__)

_SDK_DIR = Path(__file__).resolve().parent / "xcoresdk"
_EXPECTED_SDK_VERSION = "0.7.0"
_DLL_DIRECTORY_HANDLE = None


def _load_sdk():
    """Load the bundled Windows extension and its dependent DLL."""
    global _DLL_DIRECTORY_HANDLE

    if sys.platform != "win32":
        raise ImportError("The bundled xCoreSDK in hardware/windows only supports Windows")

    sdk_path = str(_SDK_DIR)
    if hasattr(os, "add_dll_directory") and _DLL_DIRECTORY_HANDLE is None:
        _DLL_DIRECTORY_HANDLE = os.add_dll_directory(sdk_path)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    try:
        return importlib.import_module("xCoreSDK_python")
    except ImportError as exc:
        raise ImportError(
            "Unable to load xCoreSDK_python. This repository bundles the CPython "
            "3.12 64-bit Windows build; run it with 64-bit Python 3.12."
        ) from exc


class RokaeRobot:
    """Project-facing interface for a Rokae robot controlled by xCoreSDK.

    Public pose units are ``[m, m, m, rad, rad, rad]`` in the robot base frame.
    Joint positions are SDK realtime values in rad.  TCP and joint velocities
    are host-time finite differences between consecutive realtime frames.
    """

    def __init__(
        self,
        ip_address: str,
        local_ip: str = "",
        robot_class: str = "xMateRobot",
        state_interval_ms: int = 8,
        max_linear_speed_m_s: float = 1.0,
        command_cache_size: int = 1,
        rt_network_tolerance_percent: int = 20,
        rt_filter_hz: float = 50.0,
    ):
        if state_interval_ms not in (1, 2, 4, 8, 1000):
            raise ValueError("state_interval_ms must be one of 1, 2, 4, 8, or 1000")
        if max_linear_speed_m_s <= 0:
            raise ValueError("max_linear_speed_m_s must be positive")
        if not 1 <= command_cache_size <= 300:
            raise ValueError("command_cache_size must be between 1 and 300")
        if not 0 <= rt_network_tolerance_percent <= 100:
            raise ValueError("rt_network_tolerance_percent must be between 0 and 100")
        if not 1.0 <= rt_filter_hz <= 1000.0:
            raise ValueError("rt_filter_hz must be between 1 and 1000 Hz")

        self.ip_address = ip_address
        self.local_ip = local_ip
        self.robot_class = robot_class
        self.state_interval_ms = state_interval_ms
        self.max_linear_speed_m_s = float(max_linear_speed_m_s)
        self.command_cache_size = int(command_cache_size)
        self.rt_network_tolerance_percent = int(rt_network_tolerance_percent)
        self.rt_filter_hz = float(rt_filter_hz)

        self._sdk = None
        self._robot = None
        self._robot_info = None
        self._force_control = None
        self._rt_controller = None
        self._rt_callback = None
        self._rt_target_native: tuple[float, ...] | None = None
        self._rt_active = False
        self._sdk_lock = threading.RLock()
        # xCoreSDK state reception and getEndTorque are independent producer
        # loops.  Keep their host-side serialization domains separate so a
        # slow wrench query cannot take the state-cache or command-target lock.
        # Native thread-safety and physical timing still require Windows/robot
        # validation; this is not evidence of controller-side synchronization.
        self._state_sdk_lock = threading.RLock()
        self._wrench_sdk_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._state_thread: threading.Thread | None = None
        self._state_running = False
        self._state_streaming = False
        self._state_ready = threading.Event()
        self._last_state_time: float | None = None
        self._state_sequence_id = 0
        self._state_error: str | None = None
        self._state_keypads: tuple[bool, ...] | None = None
        self._state_wall_time_iso: str | None = None
        self._sdk_version: str | None = None
        self._collision_state: bool | None = None
        self._collision_error: str | None = None
        self._joint_soft_limits_rad: tuple[tuple[float, float], ...] | None = None
        self._joint_soft_limit_error: str | None = None

        self.is_connected = False
        self.robot_mode = "DISCONNECTED"
        self.current_speed_ratio = 0
        self.cartesian_pose: list[float] | None = None
        self.tcp_speed: list[float] | None = None
        self.joint_angles: list[float] | None = None
        self.actual_joint_speeds: list[float] | None = None
        self._pre_drag_powered = False
        self._pre_drag_automatic = False

    @staticmethod
    def _check_ec(action: str, ec: dict[str, Any]) -> None:
        code = int(ec.get("ec", 0))
        if code != 0:
            message = ec.get("message", "unknown xCoreSDK error")
            raise RuntimeError(f"xCoreSDK {action} failed ({code}): {message}")

    def _call(self, action: str, method, *args):
        ec: dict[str, Any] = {}
        with self._sdk_lock:
            result = method(*args, ec)
        self._check_ec(action, ec)
        return result

    def _require_connected(self) -> None:
        if not self.is_connected or self._robot is None:
            raise ConnectionError("Robot is not connected")

    def connect(self) -> None:
        """Connect and start observation-only state feedback.

        This path deliberately does not select a motion-control mode, change
        command-cache settings, clear alarms, power servos, or issue motion.
        The SDK connection/disconnection implementation may still have
        controller-session side effects and therefore needs a supervised
        Windows validation before being treated as an observation session.
        """
        if self.is_connected:
            return

        self._sdk = _load_sdk()
        sdk_version = str(self._sdk.BaseRobot.sdkVersion())
        if sdk_version != _EXPECTED_SDK_VERSION:
            raise RuntimeError(
                f"Expected xCoreSDK {_EXPECTED_SDK_VERSION}, but loaded {sdk_version}. "
                "Check hardware/windows/xcoresdk and the Python DLL search path."
            )
        self._sdk_version = sdk_version
        robot_type = getattr(self._sdk, self.robot_class, None)
        if robot_type is None:
            raise ValueError(f"Unknown xCoreSDK robot class: {self.robot_class}")

        try:
            # Use one unambiguous vendor-supported connection style.  The
            # alternate xMateRobot(remoteIP[, localIP]) constructor is shown in
            # examples as already usable without connectToRobot(), so combining
            # both styles risks a duplicate connection.
            self._robot = robot_type()
            with self._sdk_lock:
                if self.local_ip:
                    self._robot.connectToRobot(self.ip_address, self.local_ip)
                else:
                    self._robot.connectToRobot(self.ip_address)
            self.is_connected = True
            self._robot_info = self._call("robotInfo", self._robot.robotInfo)
            if int(self._robot_info.joint_num) != 6:
                raise RuntimeError(
                    "This project adapter currently requires a 6-axis robot, but "
                    f"the controller reported {self._robot_info.joint_num} axes"
                )
            logger.info(
                "Connected to Rokae %s (controller %s, xCoreSDK %s)",
                self._robot_info.type,
                self._robot_info.version,
                sdk_version,
            )
            force_control_factory = getattr(self._robot, "forceControl", None)
            if force_control_factory is None:
                raise RuntimeError(
                    f"xCoreSDK robot class {self.robot_class} does not expose "
                    "built-in force sensing"
                )
            self._force_control = force_control_factory()
            try:
                self._joint_soft_limits_rad = self._read_joint_soft_limits_rad()
                self._joint_soft_limit_error = None
            except Exception as exc:
                # The safety layer decides whether unavailable soft limits are
                # acceptable for the current operation; never invent defaults.
                self._joint_soft_limits_rad = None
                self._joint_soft_limit_error = (
                    f"joint_soft_limit_query_error:{type(exc).__name__}:{exc}"
                )
                logger.warning("Unable to read xCoreSDK joint soft limits: %s", exc)
            self.start_state_stream()
            self._refresh_operation_state()
        except Exception as connection_exc:
            try:
                self._cleanup_failed_connection()
            except BaseException as cleanup_exc:
                if hasattr(connection_exc, "add_note"):
                    connection_exc.add_note(
                        "xCoreSDK failed-connection cleanup was not confirmed; "
                        "native handles were retained: "
                        f"{type(cleanup_exc).__name__}:{cleanup_exc}"
                    )
            raise

    def _cleanup_failed_connection(self) -> None:
        """Strict cleanup that retains retryable handles on any uncertainty."""

        self._stop_realtime_impl(raise_on_error=True)
        self.stop_state_stream()
        if self._robot is not None:
            self._call(
                "disconnectFromRobot",
                self._robot.disconnectFromRobot,
            )
        self._clear_after_confirmed_disconnect()

    def _clear_after_confirmed_disconnect(self) -> None:
        """Clear local handles only after native disconnect has succeeded."""

        self._robot = None
        self._robot_info = None
        self._force_control = None
        self._rt_controller = None
        self._joint_soft_limits_rad = None
        self._joint_soft_limit_error = None
        self._clear_cached_state()
        self.is_connected = False
        self.robot_mode = "DISCONNECTED"
        self._state_ready.clear()

    def _clear_cached_state(self) -> None:
        """Discard a previous connection's RT cache instead of reusing it."""
        with self._state_lock:
            self.cartesian_pose = None
            self.tcp_speed = None
            self.joint_angles = None
            self.actual_joint_speeds = None
            self._last_state_time = None
            self._state_sequence_id = 0
            self._state_wall_time_iso = None
            self._state_keypads = None
            self._state_error = "robot_state_disconnected"

    def disconnect(self) -> None:
        """Stop feedback and disconnect. Calling this repeatedly is safe."""
        if self._robot is None:
            self._clear_cached_state()
            self.is_connected = False
            self.robot_mode = "DISCONNECTED"
            self._state_ready.clear()
            return

        # A failed realtime stop or a producer still inside updateRobotState
        # leaves native activity uncertain.  Preserve every handle and refuse
        # disconnect so the caller can retry or escalate to supervised manual
        # recovery without a false local "DISCONNECTED" state.
        self._stop_realtime_impl(raise_on_error=True)
        self.stop_state_stream()
        self._call("disconnectFromRobot", self._robot.disconnectFromRobot)
        self._clear_after_confirmed_disconnect()

    def start_state_stream(self) -> None:
        """Start the receive-only RT state cache; calling repeatedly is safe."""
        self._require_connected()
        if self._state_streaming and self._state_running:
            return
        from xCoreSDK_python import RtSupportedFields

        interval = timedelta(milliseconds=self.state_interval_ms)
        # v0.7.0 exposes these three fields only; velocity/torque/wrench and
        # controller time are intentionally not requested because they are not
        # part of RtSupportedFields.
        fields = [
            RtSupportedFields.tcpPoseAbc_m,
            RtSupportedFields.jointPos_m,
            RtSupportedFields.keypads,
        ]
        with self._state_sdk_lock:
            self._robot.startReceiveRobotState(interval, fields)
        self._state_streaming = True
        self._state_running = True
        self._state_ready.clear()
        self._last_state_time = None
        self._state_sequence_id = 0
        self._state_error = None
        self._state_thread = threading.Thread(
            target=self._state_loop,
            name="rokae-xcore-state",
            daemon=True,
        )
        self._state_thread.start()
        if not self._state_ready.wait(timeout=3.0):
            raise TimeoutError("Timed out waiting for the first xCoreSDK state frame")

    # Kept as a private compatibility alias for older offline tests/tools.
    _start_state_stream = start_state_stream

    def stop_state_stream(self) -> None:
        """Stop only receive-side state resources; calling repeatedly is safe."""
        self._state_running = False
        thread = self._state_thread
        if thread is threading.current_thread():
            self._state_ready.clear()
            raise RuntimeError(
                "xCoreSDK state stream cannot be stopped from its producer thread"
            )
        if thread is not None:
            thread.join(timeout=1.0)
            if thread.is_alive():
                # updateRobotState may still own _state_sdk_lock.  Never block
                # on that same lock or lose the only live-thread handle.
                self._state_ready.clear()
                raise RuntimeError(
                    "xCoreSDK state producer did not stop within 1.0s; "
                    "refusing stopReceiveRobotState and disconnect"
                )
            self._state_thread = None
        if self._state_streaming and self._robot is not None:
            try:
                with self._state_sdk_lock:
                    self._robot.stopReceiveRobotState()
            except Exception as exc:
                logger.warning("Unable to stop xCoreSDK state stream: %s", exc)
                # Keep the streaming flag true so a subsequent cleanup attempt
                # retries the native stop instead of claiming success.
                self._state_ready.clear()
                raise RuntimeError(
                    "xCoreSDK stopReceiveRobotState was not confirmed"
                ) from exc
            self._state_streaming = False
        self._state_ready.clear()

    def _state_loop(self) -> None:
        from xCoreSDK_python import RtSupportedFields

        timeout = timedelta(milliseconds=max(1, self.state_interval_ms * 2))
        while self._state_running:
            try:
                with self._state_sdk_lock:
                    updated = self._robot.updateRobotState(timeout)
                    if not updated:
                        continue
                    # The SDK queue does not overwrite old frames. Drain every
                    # immediately available frame before reading fields so the
                    # application observes the newest controller state.
                    while self._robot.updateRobotState(timedelta(milliseconds=0)):
                        pass
                    tcp = self._sdk.PyTypeVectorDouble()
                    joints = self._sdk.PyTypeVectorDouble()
                    keypads = self._sdk.PyTypeVectorBool()
                    self._robot.getStateData(RtSupportedFields.tcpPoseAbc_m, tcp, 6)
                    self._robot.getStateData(RtSupportedFields.jointPos_m, joints, 6)
                    self._robot.getStateData(RtSupportedFields.keypads, keypads)
                    tcp_native = list(tcp.content())
                    joints_native = list(joints.content())
                    keypad_native = tuple(bool(value) for value in keypads.content())
                self._accept_state(
                    tcp_native,
                    joints_native,
                    time.perf_counter_ns() / 1_000_000_000.0,
                    keypad_native,
                )
            except Exception as exc:
                if self._state_running:
                    self._mark_state_error(exc)
                    logger.warning("xCoreSDK state update failed: %s", exc)
                    time.sleep(0.05)

    def _accept_state(
        self,
        tcp_native: Sequence[float],
        joints_native: Sequence[float],
        now: float,
        keypads: Sequence[bool] | None = None,
    ) -> None:
        if len(tcp_native) < 6 or len(joints_native) < 6:
            raise ValueError("xCoreSDK returned an incomplete state frame")

        pose = self._pose_from_sdk(tcp_native)
        joints = [float(value) for value in joints_native[:6]]
        if not finite_vector(joints, 6):
            raise ValueError("xCoreSDK returned non-finite joint positions")
        with self._state_lock:
            if self._last_state_time is None:
                # The first frame has no predecessor.  Leave velocity absent
                # rather than encoding an unmeasured zero velocity.
                tcp_speed = None
                joint_speed = None
            else:
                dt = now - self._last_state_time
                if dt <= 0:
                    raise ValueError("Non-increasing host time in state stream")
                tcp_speed = [
                    (pose[i] - self.cartesian_pose[i]) / dt for i in range(6)
                ]
                joint_speed = [
                    (joints[i] - self.joint_angles[i]) / dt for i in range(6)
                ]
            self.cartesian_pose = pose
            self.tcp_speed = tcp_speed
            self.joint_angles = joints
            self.actual_joint_speeds = joint_speed
            self._last_state_time = now
            self._state_wall_time_iso = utc_now_iso()
            self._state_keypads = tuple(bool(value) for value in keypads) if keypads else ()
            self._state_sequence_id += 1
            self._state_error = None
        self._state_ready.set()

    def _mark_state_error(self, exc: BaseException) -> None:
        """Make stream faults visible to collector and SafetyGuard immediately."""
        with self._state_lock:
            self._state_error = f"robot_state_stream_error:{type(exc).__name__}:{exc}"

    @staticmethod
    def _pose_to_sdk(pose: Sequence[float]) -> list[float]:
        if len(pose) < 6:
            raise ValueError("Cartesian pose must contain 6 values")
        values = [float(value) for value in pose[:6]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Cartesian pose must contain only finite values")
        return values

    @staticmethod
    def _pose_from_sdk(pose: Sequence[float]) -> list[float]:
        if len(pose) < 6:
            raise ValueError("Cartesian pose must contain 6 values")
        values = [float(value) for value in pose[:6]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Cartesian pose must contain only finite values")
        return values

    def clear_error(self) -> None:
        self._require_connected()
        self._call("clearServoAlarm", self._robot.clearServoAlarm)

    def reset(self) -> None:
        self._require_connected()
        self._call("moveReset", self._robot.moveReset)

    def enable(self, load: float = 0.0) -> None:
        """Switch to automatic mode, power on, and prepare NRT commands.

        Payload configuration remains the controller's configured toolset. The
        ``load`` argument is retained for the project interface and must be zero.
        """
        self._require_connected()
        if self._rt_active:
            raise RuntimeError("Stop realtime control before enabling NRT mode")
        if self._rt_controller is not None:
            self.stop_realtime(switch_to_nrt=True)
        if load != 0.0:
            raise ValueError(
                "Set the Rokae payload in the controller toolset; enable(load=...) "
                "only supports load=0.0"
            )
        self._call(
            "setOperateMode",
            self._robot.setOperateMode,
            self._sdk.OperateMode.automatic,
        )
        self._call(
            "setMotionControlMode",
            self._robot.setMotionControlMode,
            self._sdk.MotionControlMode.NrtCommandMode,
        )
        self._call("setPowerState", self._robot.setPowerState, True)
        self._call("setDefaultZone", self._robot.setDefaultZone, 0.0)
        self.reset()
        self._refresh_operation_state()

    def enable_realtime(self, load: float = 0.0) -> None:
        """Power on in xCoreSDK 0.7.0 realtime command mode.

        Realtime traffic uses a second network channel, so ``local_ip`` must be
        the address of the Windows network adapter connected to the robot.
        """
        self._require_connected()
        if not self.local_ip:
            raise ValueError(
                "ROBOT_LOCAL_IP is required for xCoreSDK realtime control; set it "
                "to this Windows PC's address on the robot subnet"
            )
        if load != 0.0:
            raise ValueError(
                "Set the Rokae payload in the controller toolset; "
                "enable_realtime(load=...) only supports load=0.0"
            )
        if self._rt_active:
            return

        self._call(
            "setOperateMode",
            self._robot.setOperateMode,
            self._sdk.OperateMode.automatic,
        )
        # The SDK explicitly requires network tolerance to be configured before
        # switching to RtCommandMode.
        self._call(
            "setRtNetworkTolerance",
            self._robot.setRtNetworkTolerance,
            self.rt_network_tolerance_percent,
        )
        self._call(
            "setMotionControlMode",
            self._robot.setMotionControlMode,
            self._sdk.MotionControlMode.RtCommandMode,
        )
        self._call("setPowerState", self._robot.setPowerState, True)

        try:
            with self._sdk_lock:
                self._rt_controller = self._robot.getRtMotionController()
            self._call(
                "setFilterFrequency",
                self._rt_controller.setFilterFrequency,
                self.rt_filter_hz,
                self.rt_filter_hz,
                self.rt_filter_hz,
            )
        except Exception:
            self._stop_realtime_best_effort()
            try:
                self._call("setPowerState", self._robot.setPowerState, False)
            except Exception as power_exc:
                logger.warning("Unable to power down after realtime setup failure: %s", power_exc)
            raise
        self._refresh_operation_state()

    def attach_externally_prepared_realtime(
        self,
        *,
        reviewed_filter_hz: float,
    ) -> None:
        """Attach to realtime Cartesian control without changing mode or power.

        The operator must have prepared automatic mode, servo power, realtime
        command mode, and network tolerance outside this method.  This method
        verifies the queryable mode/power state, obtains the SDK's confirmed
        realtime controller, and configures only its command filter.  It never
        clears errors, powers on, or changes a motion-control mode.
        """
        self._require_connected()
        if (
            isinstance(reviewed_filter_hz, bool)
            or not isinstance(reviewed_filter_hz, (int, float))
            or not math.isfinite(float(reviewed_filter_hz))
            or not 1.0 <= float(reviewed_filter_hz) <= 1000.0
        ):
            raise ValueError(
                "reviewed_filter_hz must be an explicitly reviewed value in "
                "[1, 1000] Hz"
            )
        if not self.local_ip:
            raise ValueError(
                "ROBOT_LOCAL_IP is required for xCoreSDK realtime control"
            )
        if self._rt_controller is not None:
            return
        operation_state = self._refresh_operation_state()
        if operation_state != self._sdk.OperationState.idle:
            raise RuntimeError("robot must be idle before realtime attachment")
        operate_mode = self._call("operateMode", self._robot.operateMode)
        power_state = self._call("powerState", self._robot.powerState)
        if operate_mode != self._sdk.OperateMode.automatic:
            raise RuntimeError(
                "operator must prepare automatic mode outside the program"
            )
        if power_state != self._sdk.PowerState.on:
            raise RuntimeError(
                "operator must prepare servo power outside the program"
            )
        try:
            with self._sdk_lock:
                controller = self._robot.getRtMotionController()
            self._call(
                "setFilterFrequency",
                controller.setFilterFrequency,
                float(reviewed_filter_hz),
                float(reviewed_filter_hz),
                float(reviewed_filter_hz),
            )
        except Exception as exc:
            raise RuntimeError(
                "unable to attach realtime controller; verify external RtCommandMode "
                "and network-tolerance preparation"
            ) from exc
        self._rt_controller = controller
        operation_state = self._refresh_operation_state()
        if operation_state != self._sdk.OperationState.idle:
            self._rt_controller = None
            raise RuntimeError("robot left idle state during realtime attachment")

    def start_realtime_cartesian(
        self, initial_pose: Sequence[float] | None = None
    ) -> None:
        """Start the SDK's 1 ms Cartesian callback scheduler.

        The application may update the held target at 50 Hz with
        :meth:`set_realtime_cartesian_target`; xCoreSDK itself invokes the
        callback at the controller's required 1 ms period.
        """
        self._require_connected()
        if self._rt_controller is None:
            raise RuntimeError(
                "Call attach_externally_prepared_realtime(reviewed_filter_hz=...) "
                "before starting "
                "gated realtime motion"
            )
        if self._rt_active:
            raise RuntimeError("Realtime Cartesian control is already active")

        if initial_pose is None:
            if not self._state_ready.is_set():
                raise RuntimeError("No robot state is available for the realtime hold pose")
            initial_pose = self.get_cartesian_pose()
        self._rt_target_native = tuple(self._pose_to_sdk(initial_pose))
        sdk = self._sdk

        def cartesian_callback():
            target = self._rt_target_native
            if target is None:
                raise RuntimeError("Realtime Cartesian target is unavailable")
            return sdk.CartesianPosition(target)

        self._rt_callback = cartesian_callback
        move_attempted = False
        try:
            with self._sdk_lock:
                self._rt_controller.setControlLoopCar(
                    self._rt_callback,
                    0,
                    False,
                )
                move_attempted = True
                self._rt_controller.startMove(
                    self._sdk.RtControllerMode.cartesianPosition
                )
                # Non-blocking starts the SDK-owned periodic task. It must later
                # be paired with stopLoop(), as required by the 0.7.0 API.
                self._rt_controller.startLoop(False)
                # Publish active state before releasing the same lock used by
                # stop, so a concurrent request cannot observe a started loop
                # as inactive and skip stopLoop/stopMove.
                self._rt_active = True
        except Exception as start_exc:
            cleanup_failures: list[str] = []
            # startMove/startLoop may have a controller-side effect before an
            # exception reaches Python.  Once startMove was attempted, always
            # try both paired stop calls instead of guessing how far native
            # startup progressed.
            if move_attempted:
                try:
                    self._rt_controller.stopLoop()
                except Exception as stop_exc:
                    cleanup_failures.append(
                        f"stopLoop:{type(stop_exc).__name__}:{stop_exc}"
                    )
                    logger.warning(
                        "Unable to stop partially started realtime loop: %s",
                        stop_exc,
                    )
                try:
                    self._rt_controller.stopMove()
                except Exception as stop_exc:
                    cleanup_failures.append(
                        f"stopMove:{type(stop_exc).__name__}:{stop_exc}"
                    )
                    logger.warning(
                        "Unable to stop partially started realtime move: %s",
                        stop_exc,
                    )
            if cleanup_failures:
                # Keep a valid hold callback/target published.  Clearing it
                # while a native loop may still be alive would turn a stop
                # failure into repeated callback exceptions.
                self._rt_active = True
                self.robot_mode = "RT_START_FAILED_STOP_UNCONFIRMED"
                raise RuntimeError(
                    "realtime start failed and cleanup was not confirmed: "
                    + ";".join(cleanup_failures)
                ) from start_exc
            self._rt_active = False
            self._rt_callback = None
            self._rt_target_native = None
            raise
        self.robot_mode = "RT_CONTROLLING"

    def set_realtime_cartesian_target(self, pose: Sequence[float]) -> None:
        """Atomically replace the target held by the 1 ms SDK callback.

        This hot path performs no blocking SDK query.  Motion-error polling is
        intentionally a separate health-monitor responsibility so a delayed
        state or wrench producer cannot directly serialize command scheduling.
        """
        if not self._rt_active or self._rt_controller is None:
            raise RuntimeError("Realtime Cartesian control is not active")
        target = tuple(self._pose_to_sdk(pose))
        if not all(math.isfinite(value) for value in target):
            raise ValueError("Realtime Cartesian target must contain finite values")
        self._rt_target_native = target

    def realtime_motion_error(self) -> bool:
        """Poll the SDK motion-error flag outside the command update hot path."""
        if self._rt_controller is None:
            return False
        with self._sdk_lock:
            return bool(self._rt_controller.hasMotionError())

    def _stop_realtime_impl(self, *, raise_on_error: bool) -> None:
        controller = self._rt_controller
        if controller is None:
            self._rt_active = False
            self._rt_callback = None
            self._rt_target_native = None
            return

        failures: list[str] = []
        with self._sdk_lock:
            if self._rt_active:
                try:
                    controller.stopLoop()
                except Exception as exc:
                    failures.append(f"stopLoop:{type(exc).__name__}:{exc}")
                    logger.warning("Unable to stop xCoreSDK realtime loop: %s", exc)
                try:
                    controller.stopMove()
                except Exception as exc:
                    failures.append(f"stopMove:{type(exc).__name__}:{exc}")
                    logger.warning("Unable to stop xCoreSDK realtime motion: %s", exc)
        if failures:
            # The callback may still be running when stopLoop reports failure.
            # Preserve its finite hold target and keep the state retryable.
            self._rt_active = True
            self.robot_mode = "RT_STOP_FAILED_UNCONFIRMED"
            if raise_on_error:
                raise RuntimeError(
                    "xCoreSDK realtime stop failed: " + ";".join(failures)
                )
            return
        self._rt_active = False
        self._rt_callback = None
        self._rt_target_native = None

    def _stop_realtime_best_effort(self) -> None:
        self._stop_realtime_impl(raise_on_error=False)

    def stop_realtime(self, switch_to_nrt: bool = True) -> None:
        """Stop the callback and realtime motion, optionally returning to NRT."""
        if self._robot is None:
            return
        self._stop_realtime_impl(raise_on_error=True)
        if switch_to_nrt and self.is_connected:
            self._call(
                "setMotionControlMode",
                self._robot.setMotionControlMode,
                self._sdk.MotionControlMode.NrtCommandMode,
            )
            self._rt_controller = None
            self._refresh_operation_state()

    def disable(self) -> None:
        if not self.is_connected or self._robot is None:
            return
        self._stop_realtime_best_effort()
        self._call("setPowerState", self._robot.setPowerState, False)
        self._refresh_operation_state()

    def set_speed(self, ratio: int) -> None:
        """Set default Cartesian speed from a 1..100 percentage.

        ``setDefaultSpeed`` is one of the SDK's non-SI calls and accepts mm/s.
        """
        self._require_connected()
        if not 1 <= int(ratio) <= 100:
            raise ValueError("Speed ratio must be between 1 and 100")
        speed_m_s = max(0.005, self.max_linear_speed_m_s * int(ratio) / 100.0)
        self._call(
            "setDefaultSpeed", self._robot.setDefaultSpeed, speed_m_s * 1000.0
        )
        self.current_speed_ratio = int(ratio)

    set_speed_ratio = set_speed

    def move_l(self, pose: Sequence[float]) -> str:
        """Append and start a non-realtime Cartesian linear command."""
        self._require_connected()
        if self._rt_controller is not None:
            raise RuntimeError("MoveLCommand is unavailable while realtime mode is selected")
        target = self._sdk.CartesianPosition(self._pose_to_sdk(pose))
        command = self._sdk.MoveLCommand(target)
        command_id = self._sdk.PyString()
        self._call("moveAppend", self._robot.moveAppend, [command], command_id)
        self._call("moveStart", self._robot.moveStart)
        # The RT stream has no operation-state field.  Do not keep reporting a
        # stale IDLE value while an NRT command is known to be pending.
        self.robot_mode = "NRT_COMMAND_PENDING"
        return str(command_id.content())

    def move_j(self, joints_rad: Sequence[float]) -> str:
        self._require_connected()
        if self._rt_controller is not None:
            raise RuntimeError(
                "MoveAbsJCommand is unavailable while realtime mode is selected"
            )
        if len(joints_rad) < 6:
            raise ValueError("Joint target must contain 6 values")
        target = [float(value) for value in joints_rad[:6]]
        command = self._sdk.MoveAbsJCommand(target)
        command_id = self._sdk.PyString()
        self._call("moveAppend", self._robot.moveAppend, [command], command_id)
        self._call("moveStart", self._robot.moveStart)
        self.robot_mode = "NRT_COMMAND_PENDING"
        return str(command_id.content())

    def stop(self) -> None:
        if not self.is_connected or self._robot is None:
            return
        if self._rt_controller is not None:
            was_active = self._rt_active
            self._stop_realtime_best_effort()
            if not was_active:
                self._call("setPowerState", self._robot.setPowerState, False)
            self._refresh_operation_state()
            return
        self._call("stop", self._robot.stop)
        self._refresh_operation_state()

    stop_move = stop

    def wait_idle(self, timeout: float = 30.0) -> bool:
        self._require_connected()
        started_at = time.monotonic()
        deadline = started_at + timeout
        idle_since = None
        while time.monotonic() < deadline:
            state = self._refresh_operation_state()
            if state == self._sdk.OperationState.idle:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif (
                    time.monotonic() - idle_since >= 0.2
                    and time.monotonic() - started_at >= 0.5
                ):
                    return True
            else:
                idle_since = None
            time.sleep(0.05)
        raise TimeoutError(f"Robot did not become idle within {timeout:.1f}s")

    def _refresh_operation_state(self):
        if not self.is_connected or self._robot is None:
            self.robot_mode = "DISCONNECTED"
            return None
        state = self._call("operationState", self._robot.operationState)
        self.robot_mode = state.name.upper()
        return state

    def get_robot_mode(self) -> str:
        self._refresh_operation_state()
        return self.robot_mode

    def get_cartesian_pose(self) -> list[float]:
        with self._state_lock:
            if self.cartesian_pose is None:
                raise RuntimeError("No xCoreSDK realtime pose frame is available")
            return list(self.cartesian_pose)

    def get_joint_angles(self) -> list[float]:
        with self._state_lock:
            if self.joint_angles is None:
                raise RuntimeError("No xCoreSDK realtime joint frame is available")
            return list(self.joint_angles)

    def get_actual_joint_speeds(self) -> list[float]:
        """Compatibility accessor for host-difference joint velocity estimates."""
        with self._state_lock:
            if self.actual_joint_speeds is None:
                raise RuntimeError("Joint velocity needs two realtime state frames")
            return list(self.actual_joint_speeds)

    def get_state_frame(self) -> KinematicStateFrame:
        """Return an immutable latest realtime frame without re-querying the SDK."""
        with self._state_lock:
            pose = list(self.cartesian_pose) if self.cartesian_pose is not None else None
            tcp_speed = list(self.tcp_speed) if self.tcp_speed is not None else None
            joints = list(self.joint_angles) if self.joint_angles is not None else None
            joint_speed = (
                list(self.actual_joint_speeds)
                if self.actual_joint_speeds is not None
                else None
            )
            error = self._state_error
            state_time = self._last_state_time
            sequence_id = self._state_sequence_id
            wall_time_iso = self._state_wall_time_iso
            keypads = self._state_keypads

        valid = (
            self.is_connected
            and error is None
            and pose is not None
            and finite_vector(pose, 6)
            and joints is not None
            and finite_vector(joints, 6)
            and state_time is not None
        )
        reason = "" if valid else (error or "robot_state_not_ready")
        velocity_available = finite_vector(tcp_speed, 6) and finite_vector(joint_speed, 6)
        return KinematicStateFrame(
            sequence_id=sequence_id,
            host_monotonic_time_s=state_time,
            wall_time_iso=wall_time_iso,
            robot_device_time_s=None,  # v0.7.0 RT fields expose no device time.
            valid=valid,
            invalid_reason=reason,
            tcp_position_m=as_vec3(pose[:3]) if pose is not None else None,
            tcp_orientation_rad=as_vec3(pose[3:6]) if pose is not None else None,
            tcp_linear_velocity_mps=as_vec3(tcp_speed[:3]) if velocity_available else None,
            # This is an Euler-angle rate, not a guaranteed physical angular
            # velocity vector.  Keep it separate and label its provenance.
            tcp_angular_velocity_radps=as_vec3(tcp_speed[3:6]) if velocity_available else None,
            velocity_source=(
                "numerical_difference_realtime_frame_rpy"
                if velocity_available
                else "unavailable_first_realtime_frame"
            ),
            joint_position_rad=as_float_tuple(joints, 6),
            joint_velocity_radps=as_float_tuple(joint_speed, 6),
            pose_time_s=state_time,
            joint_time_s=state_time,
            velocity_time_s=state_time if velocity_available else None,
            operation_state=self.robot_mode,
            collision_state=self._collision_state,
            controller_error=error,
            keypad_state=keypads,
        )

    def get_state(self) -> dict[str, Any]:
        """Compatibility dictionary; new code should consume ``get_state_frame``."""
        frame = self.get_state_frame()
        return {
            "cartesian_pose": list(frame.tcp_position_m or ())
            + list(frame.tcp_orientation_rad or ()),
            "tcp_speed": list(frame.tcp_linear_velocity_mps or ())
            + list(frame.tcp_angular_velocity_radps or ()),
            "joint_angles": list(frame.joint_position_rad or ()),
            "joint_speeds": list(frame.joint_velocity_radps or ()),
            "robot_mode": frame.operation_state,
            "state_time_s": frame.host_monotonic_time_s,
            "valid": frame.valid,
            "invalid_reason": frame.invalid_reason,
        }

    def get_robot_metadata(self) -> dict[str, Any]:
        """Return only values actually supplied by xCoreSDK/the adapter."""
        info = self._robot_info
        try:
            tool_payload = self.get_tool_payload_metadata()
        except Exception as exc:
            tool_payload = {
                "read_error": f"tool_payload_query_error:{type(exc).__name__}:{exc}"
            }
        return {
            "robot_ip": self.ip_address,
            "local_ip": self.local_ip or None,
            "robot_class": self.robot_class,
            "robot_model": getattr(info, "type", None) if info is not None else None,
            "robot_serial_number": getattr(info, "id", None) if info is not None else None,
            "controller_version": getattr(info, "version", None) if info is not None else None,
            "xcore_sdk_version": self._sdk_version,
            "state_interval_ms": self.state_interval_ms,
            "joint_soft_limits_rad": self._joint_soft_limits_rad,
            "joint_soft_limit_read_error": self._joint_soft_limit_error,
            "sdk_tool_payload": tool_payload,
        }

    def get_tool_payload_metadata(self) -> dict[str, Any]:
        """Read the SDK toolset/load configuration without changing it.

        ``Toolset`` is documented as the SDK motion-control tool/workobject
        configuration, not proof of a separate HMI/RL project's active setup.
        The returned provenance therefore records available names and load data
        while explicitly leaving the active HMI tool/workobject unverified.
        """
        self._require_connected()
        toolset = self._call("toolset", self._robot.toolset)
        load = toolset.load
        result: dict[str, Any] = {
            "toolset_load_mass_kg": float(load.mass),
            "toolset_load_cog_m": [float(value) for value in load.cog],
            "toolset_load_inertia_kg_m2": [float(value) for value in load.inertia],
            "toolset_end_translation_m": [float(value) for value in toolset.end.trans],
            "toolset_end_rpy_rad": [float(value) for value in toolset.end.rpy],
            "toolset_ref_translation_m": [float(value) for value in toolset.ref.trans],
            "toolset_ref_rpy_rad": [float(value) for value in toolset.ref.rpy],
            "active_hmi_tool_workobject_verified": False,
        }
        tools = self._call("toolsInfo", self._robot.toolsInfo)
        wobjs = self._call("wobjsInfo", self._robot.wobjsInfo)
        result["sdk_available_tool_names"] = [str(item.name) for item in tools]
        result["sdk_available_workobject_names"] = [str(item.name) for item in wobjs]
        return result

    def _read_joint_soft_limits_rad(self) -> tuple[tuple[float, float], ...]:
        """Read configured controller soft limits through the normal SDK API."""
        self._require_connected()
        limits_native = self._sdk.PyTypeVectorArrayDouble2()
        self._call("getSoftLimit", self._robot.getSoftLimit, limits_native)
        content = list(limits_native.content())
        if len(content) < 6:
            raise RuntimeError("xCoreSDK returned fewer than six joint soft limits")
        limits: list[tuple[float, float]] = []
        for index, pair in enumerate(content[:6], start=1):
            if len(pair) != 2:
                raise RuntimeError(f"xCoreSDK soft limit q{index} is not [lower, upper]")
            lower, upper = float(pair[0]), float(pair[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise RuntimeError(f"xCoreSDK soft limit q{index} is invalid")
            limits.append((lower, upper))
        return tuple(limits)

    def get_joint_soft_limits_rad(
        self, *, refresh: bool = False
    ) -> tuple[tuple[float, float], ...] | None:
        """Return cached controller soft limits in radians, optionally refreshing."""
        if refresh:
            try:
                self._joint_soft_limits_rad = self._read_joint_soft_limits_rad()
                self._joint_soft_limit_error = None
            except Exception as exc:
                self._joint_soft_limits_rad = None
                self._joint_soft_limit_error = (
                    f"joint_soft_limit_query_error:{type(exc).__name__}:{exc}"
                )
        return self._joint_soft_limits_rad

    def get_base_frame_pose_in_world(self) -> list[float]:
        """Read ``^world T_base`` from xCoreSDK's normal query interface."""
        self._require_connected()
        pose = self._call("baseFrame", self._robot.baseFrame)
        values = self._pose_from_sdk(pose)
        return values

    def get_world_to_base_rotation(self) -> tuple[tuple[float, float, float], ...]:
        """Build ``R_base_from_world`` from SDK ``baseFrame`` orientation.

        The SDK documents baseFrame as the base frame relative to world and its
        Frame RPY as XYZ Euler.  This is an expression rotation only; callers
        must not infer an unverified wrench reference-point translation from it.
        """
        base_pose_in_world = self.get_base_frame_pose_in_world()
        world_from_base = rpy_euler_xyz_rotation_matrix(base_pose_in_world[3:6])
        return transpose_rotation(world_from_base)

    def get_collision_state(self) -> bool | None:
        """Poll the separate safety event when the controller supports it.

        Collision is not contained in the realtime state frame and has no
        synchronized timestamp in v0.7.0.  ``None`` means unavailable rather
        than a safe/false collision state.
        """
        self._require_connected()
        try:
            event = self._sdk.Event.safety
            info = self._call("queryEventInfo(safety)", self._robot.queryEventInfo, event)
        except Exception as exc:
            self._collision_error = f"collision_query_error:{type(exc).__name__}:{exc}"
            return None

        collided: bool | None = None
        if isinstance(info, dict):
            for key, value in info.items():
                if str(key).lower().endswith("collided") or str(key).lower() == "collided":
                    if type(value) is bool:
                        collided = value
                    else:
                        self._collision_error = (
                            "collision_query_invalid_boolean_type:"
                            f"{type(value).__name__}"
                        )
                        self._collision_state = None
                        return None
                    break
        self._collision_state = collided
        self._collision_error = (
            None if collided is not None else "collision_query_missing_collided_field"
        )
        return collided

    def get_end_wrench(self, reference_frame: str = "world") -> dict[str, Any]:
        """Query ``forceControl().getEndTorque`` once and retain its provenance.

        The SDK documents the returned joint arrays as measured joint torque and
        model-derived external joint torque (both N·m), and the Cartesian arrays
        as force (N) / torque (N·m) in the requested ``world``, ``flange``, or
        ``tool`` expression frame.  It does *not* document compensation state,
        device time, or synchronization with the realtime pose packet; callers
        must keep the returned host query time and validate those properties on
        the physical robot.
        """
        self._require_connected()
        if self._force_control is None:
            raise RuntimeError("xCoreSDK force-control interface is unavailable")

        frames = {
            "world": self._sdk.FrameType.world,
            "flange": self._sdk.FrameType.flange,
            "tool": self._sdk.FrameType.tool,
        }
        try:
            frame = frames[reference_frame.lower()]
        except KeyError as exc:
            raise ValueError("reference_frame must be world, flange, or tool") from exc

        joint_measured = self._sdk.PyTypeVectorDouble()
        joint_external = self._sdk.PyTypeVectorDouble()
        cart_torque = self._sdk.PyTypeVectorDouble()
        cart_force = self._sdk.PyTypeVectorDouble()
        query_started_s = time.perf_counter_ns() / 1_000_000_000.0
        ec: dict[str, Any] = {}
        with self._wrench_sdk_lock:
            self._force_control.getEndTorque(
                frame,
                joint_measured,
                joint_external,
                cart_torque,
                cart_force,
                ec,
            )
        query_finished_s = time.perf_counter_ns() / 1_000_000_000.0
        self._check_ec("getEndTorque", ec)

        force = [float(value) for value in cart_force.content()]
        torque = [float(value) for value in cart_torque.content()]
        measured = [float(value) for value in joint_measured.content()]
        external = [float(value) for value in joint_external.content()]
        if (
            len(force) < 3
            or len(torque) < 3
            or len(measured) < 6
            or len(external) < 6
            or not all(math.isfinite(value) for value in [*force[:3], *torque[:3], *measured[:6], *external[:6]])
        ):
            raise RuntimeError("xCoreSDK returned an incomplete Cartesian wrench")
        midpoint_s = (query_started_s + query_finished_s) / 2.0
        return {
            "joint_measured_torque_nm": measured[:6],
            "joint_external_torque_nm": external[:6],
            "cartesian_force_raw_n": force[:3],
            "cartesian_torque_raw_nm": torque[:3],
            "raw_force_frame": reference_frame.lower(),
            "force_query_started_s": query_started_s,
            "force_query_finished_s": query_finished_s,
            "host_monotonic_time_s": midpoint_s,
            "wall_time_iso": utc_now_iso(),
            # Compatibility keys.  New collection code consumes the explicit
            # unit/provenance names above.
            "force": force[:3],
            "torque": torque[:3],
            "joint_torque_measured": measured[:6],
            "joint_torque_external": external[:6],
            "reference_frame": reference_frame.lower(),
            "ts": midpoint_s,
        }

    def calibrate_force_sensors(self) -> None:
        """Start calibration of all joint torque sensors.

        The controller toolset/load must already be configured correctly and
        the robot must be unloaded. Normal session startup should use software
        biasing instead of calling this method.
        """
        self._require_connected()
        self._call("calibrateForceSensor", self._robot.calibrateForceSensor, True, 0)

    def start_drag(self) -> None:
        """Enter free Cartesian drag mode, preserving the previous power/mode."""
        self._require_connected()
        power = self._call("powerState", self._robot.powerState)
        mode = self._call("operateMode", self._robot.operateMode)
        self._pre_drag_powered = power == self._sdk.PowerState.on
        self._pre_drag_automatic = mode == self._sdk.OperateMode.automatic

        if self._pre_drag_powered:
            self._call("setPowerState", self._robot.setPowerState, False)
        self._call(
            "setOperateMode", self._robot.setOperateMode, self._sdk.OperateMode.manual
        )
        self.reset()
        ec: dict[str, Any] = {}
        with self._sdk_lock:
            self._robot.enableDrag(1, 2, ec, True)
        self._check_ec("enableDrag", ec)
        self.robot_mode = "DRAG"

    def stop_drag(self) -> None:
        self._require_connected()
        self._call("disableDrag", self._robot.disableDrag)
        if self._pre_drag_automatic:
            self._call(
                "setOperateMode",
                self._robot.setOperateMode,
                self._sdk.OperateMode.automatic,
            )
        if self._pre_drag_powered:
            self._call(
                "setMotionControlMode",
                self._robot.setMotionControlMode,
                self._sdk.MotionControlMode.NrtCommandMode,
            )
            self._call("setPowerState", self._robot.setPowerState, True)
            self.reset()
        self._refresh_operation_state()
