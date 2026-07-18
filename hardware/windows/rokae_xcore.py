"""Rokae robot adapter backed by xCoreSDK 0.7.0 on Windows.

The rest of this project uses millimetres and degrees. xCoreSDK uses metres and
radians for Cartesian poses, and radians for joint positions. All unit
conversions are kept in this adapter so the collection and control layers do
not depend on vendor-specific conventions.
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

    Public pose units are ``[mm, mm, mm, deg, deg, deg]``. Joint positions and
    velocities are returned in degrees and degrees/s. The native SDK keeps its
    own conventions internally (m and rad).
    """

    def __init__(
        self,
        ip_address: str,
        local_ip: str = "",
        robot_class: str = "xMateRobot",
        state_interval_ms: int = 8,
        max_linear_speed_mm_s: float = 1000.0,
        command_cache_size: int = 1,
        rt_network_tolerance_percent: int = 20,
        rt_filter_hz: float = 50.0,
    ):
        if state_interval_ms not in (1, 2, 4, 8, 1000):
            raise ValueError("state_interval_ms must be one of 1, 2, 4, 8, or 1000")
        if max_linear_speed_mm_s <= 0:
            raise ValueError("max_linear_speed_mm_s must be positive")
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
        self.max_linear_speed_mm_s = float(max_linear_speed_mm_s)
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
        self._state_lock = threading.Lock()
        self._state_thread: threading.Thread | None = None
        self._state_running = False
        self._state_streaming = False
        self._state_ready = threading.Event()
        self._last_state_time: float | None = None

        self.is_connected = False
        self.robot_mode = "DISCONNECTED"
        self.current_speed_ratio = 0
        self.cartesian_pose = [0.0] * 6
        self.tcp_speed = [0.0] * 6
        self.joint_angles = [0.0] * 6
        self.actual_joint_speeds = [0.0] * 6
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
        """Connect, select non-realtime command mode, and start state feedback."""
        if self.is_connected:
            return

        self._sdk = _load_sdk()
        sdk_version = str(self._sdk.BaseRobot.sdkVersion())
        if sdk_version != _EXPECTED_SDK_VERSION:
            raise RuntimeError(
                f"Expected xCoreSDK {_EXPECTED_SDK_VERSION}, but loaded {sdk_version}. "
                "Check hardware/windows/xcoresdk and the Python DLL search path."
            )
        robot_type = getattr(self._sdk, self.robot_class, None)
        if robot_type is None:
            raise ValueError(f"Unknown xCoreSDK robot class: {self.robot_class}")

        try:
            self._robot = (
                robot_type(self.ip_address, self.local_ip)
                if self.local_ip
                else robot_type(self.ip_address)
            )
            self._call("connectToRobot", self._robot.connectToRobot)
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
            self._call(
                "setMotionControlMode",
                self._robot.setMotionControlMode,
                self._sdk.MotionControlMode.NrtCommandMode,
            )
            self._call(
                "setMaxCacheSize",
                self._robot.setMaxCacheSize,
                self.command_cache_size,
            )
            self._start_state_stream()
            self._refresh_operation_state()
        except Exception:
            self._cleanup_failed_connection()
            raise

    def _cleanup_failed_connection(self) -> None:
        self._stop_realtime_best_effort()
        self._state_running = False
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
        if self._robot is not None:
            if self._state_streaming:
                try:
                    with self._sdk_lock:
                        self._robot.stopReceiveRobotState()
                except Exception:
                    pass
                self._state_streaming = False
            try:
                self._robot.disconnectFromRobot({})
            except Exception:
                pass
        self._robot = None
        self._robot_info = None
        self._force_control = None
        self._rt_controller = None
        self.is_connected = False
        self.robot_mode = "DISCONNECTED"

    def disconnect(self) -> None:
        """Stop feedback and disconnect. Calling this repeatedly is safe."""
        if self._robot is None:
            self.is_connected = False
            self.robot_mode = "DISCONNECTED"
            return

        self._stop_realtime_best_effort()
        self._state_running = False
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
        if self._state_streaming:
            try:
                with self._sdk_lock:
                    self._robot.stopReceiveRobotState()
            except Exception as exc:
                logger.warning("Unable to stop xCoreSDK state stream: %s", exc)
            finally:
                self._state_streaming = False

        try:
            if self.is_connected:
                self._call("disconnectFromRobot", self._robot.disconnectFromRobot)
        finally:
            self._robot = None
            self._robot_info = None
            self._force_control = None
            self._rt_controller = None
            self.is_connected = False
            self.robot_mode = "DISCONNECTED"
            self._state_ready.clear()

    def _start_state_stream(self) -> None:
        from xCoreSDK_python import RtSupportedFields

        interval = timedelta(milliseconds=self.state_interval_ms)
        fields = [RtSupportedFields.tcpPoseAbc_m, RtSupportedFields.jointPos_m]
        with self._sdk_lock:
            self._robot.startReceiveRobotState(interval, fields)
        self._state_streaming = True
        self._state_running = True
        self._state_ready.clear()
        self._last_state_time = None
        self._state_thread = threading.Thread(
            target=self._state_loop,
            name="rokae-xcore-state",
            daemon=True,
        )
        self._state_thread.start()
        if not self._state_ready.wait(timeout=3.0):
            raise TimeoutError("Timed out waiting for the first xCoreSDK state frame")

    def _state_loop(self) -> None:
        from xCoreSDK_python import RtSupportedFields

        timeout = timedelta(milliseconds=max(1, self.state_interval_ms * 2))
        while self._state_running:
            try:
                with self._sdk_lock:
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
                    self._robot.getStateData(RtSupportedFields.tcpPoseAbc_m, tcp, 6)
                    self._robot.getStateData(RtSupportedFields.jointPos_m, joints, 6)
                    tcp_native = list(tcp.content())
                    joints_native = list(joints.content())
                self._accept_state(tcp_native, joints_native, time.perf_counter())
            except Exception as exc:
                if self._state_running:
                    logger.warning("xCoreSDK state update failed: %s", exc)
                    time.sleep(0.05)

    def _accept_state(
        self,
        tcp_native: Sequence[float],
        joints_native: Sequence[float],
        now: float,
    ) -> None:
        if len(tcp_native) < 6 or len(joints_native) < 6:
            raise ValueError("xCoreSDK returned an incomplete state frame")

        pose = self._pose_from_sdk(tcp_native)
        joints = [math.degrees(float(value)) for value in joints_native[:6]]
        with self._state_lock:
            if self._last_state_time is None:
                tcp_speed = [0.0] * 6
                joint_speed = [0.0] * 6
            else:
                dt = now - self._last_state_time
                if dt <= 0:
                    return
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
        self._state_ready.set()

    @staticmethod
    def _pose_to_sdk(pose: Sequence[float]) -> list[float]:
        if len(pose) < 6:
            raise ValueError("Cartesian pose must contain 6 values")
        values = [float(value) for value in pose[:6]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Cartesian pose must contain only finite values")
        return [
            values[0] / 1000.0,
            values[1] / 1000.0,
            values[2] / 1000.0,
            math.radians(values[3]),
            math.radians(values[4]),
            math.radians(values[5]),
        ]

    @staticmethod
    def _pose_from_sdk(pose: Sequence[float]) -> list[float]:
        values = [float(value) for value in pose[:6]]
        return [
            values[0] * 1000.0,
            values[1] * 1000.0,
            values[2] * 1000.0,
            math.degrees(values[3]),
            math.degrees(values[4]),
            math.degrees(values[5]),
        ]

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
            except Exception:
                pass
            raise
        self._refresh_operation_state()

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
            raise RuntimeError("Call enable_realtime() before starting realtime motion")
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
        move_started = False
        try:
            with self._sdk_lock:
                self._rt_controller.setControlLoopCar(
                    self._rt_callback,
                    0,
                    False,
                )
                self._rt_controller.startMove(
                    self._sdk.RtControllerMode.cartesianPosition
                )
                move_started = True
                # Non-blocking starts the SDK-owned periodic task. It must later
                # be paired with stopLoop(), as required by the 0.7.0 API.
                self._rt_controller.startLoop(False)
        except Exception:
            if move_started:
                try:
                    self._rt_controller.stopMove()
                except Exception:
                    pass
            self._rt_callback = None
            self._rt_target_native = None
            raise
        self._rt_active = True
        self.robot_mode = "RT_CONTROLLING"

    def set_realtime_cartesian_target(self, pose: Sequence[float]) -> None:
        """Atomically replace the target held by the 1 ms SDK callback."""
        if not self._rt_active or self._rt_controller is None:
            raise RuntimeError("Realtime Cartesian control is not active")
        target = tuple(self._pose_to_sdk(pose))
        if not all(math.isfinite(value) for value in target):
            raise ValueError("Realtime Cartesian target must contain finite values")
        with self._sdk_lock:
            if self._rt_controller.hasMotionError():
                raise RuntimeError("xCoreSDK reported a realtime motion error")
        self._rt_target_native = target

    def _stop_realtime_best_effort(self) -> None:
        controller = self._rt_controller
        if controller is None:
            self._rt_active = False
            self._rt_callback = None
            self._rt_target_native = None
            return

        with self._sdk_lock:
            if self._rt_active:
                try:
                    controller.stopLoop()
                except Exception as exc:
                    logger.warning("Unable to stop xCoreSDK realtime loop: %s", exc)
                try:
                    controller.stopMove()
                except Exception as exc:
                    logger.warning("Unable to stop xCoreSDK realtime motion: %s", exc)
        self._rt_active = False
        self._rt_callback = None
        self._rt_target_native = None

    def stop_realtime(self, switch_to_nrt: bool = True) -> None:
        """Stop the callback and realtime motion, optionally returning to NRT."""
        if self._robot is None:
            return
        self._stop_realtime_best_effort()
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
        """Set default Cartesian speed from a 1..100 percentage."""
        self._require_connected()
        if not 1 <= int(ratio) <= 100:
            raise ValueError("Speed ratio must be between 1 and 100")
        speed = max(5.0, self.max_linear_speed_mm_s * int(ratio) / 100.0)
        self._call("setDefaultSpeed", self._robot.setDefaultSpeed, speed)
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
        return str(command_id.content())

    def move_j(self, joints_deg: Sequence[float]) -> str:
        self._require_connected()
        if self._rt_controller is not None:
            raise RuntimeError(
                "MoveAbsJCommand is unavailable while realtime mode is selected"
            )
        if len(joints_deg) < 6:
            raise ValueError("Joint target must contain 6 values")
        target = [math.radians(float(value)) for value in joints_deg[:6]]
        command = self._sdk.MoveAbsJCommand(target)
        command_id = self._sdk.PyString()
        self._call("moveAppend", self._robot.moveAppend, [command], command_id)
        self._call("moveStart", self._robot.moveStart)
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
            return list(self.cartesian_pose)

    def get_joint_angles(self) -> list[float]:
        with self._state_lock:
            return list(self.joint_angles)

    def get_actual_joint_speeds(self) -> list[float]:
        with self._state_lock:
            return list(self.actual_joint_speeds)

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "cartesian_pose": list(self.cartesian_pose),
                "tcp_speed": list(self.tcp_speed),
                "joint_angles": list(self.joint_angles),
                "joint_speeds": list(self.actual_joint_speeds),
                "robot_mode": self.robot_mode,
            }

    def get_end_wrench(self, reference_frame: str = "tool") -> dict[str, Any]:
        """Read the controller-estimated external wrench.

        The collaborative robot measures joint torques and uses its dynamics
        model to estimate Cartesian force/torque. Returned Cartesian values are
        expressed in ``world``, ``flange``, or ``tool`` coordinates.
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
        ec: dict[str, Any] = {}
        with self._sdk_lock:
            self._force_control.getEndTorque(
                frame,
                joint_measured,
                joint_external,
                cart_torque,
                cart_force,
                ec,
            )
        self._check_ec("getEndTorque", ec)

        force = [float(value) for value in cart_force.content()]
        torque = [float(value) for value in cart_torque.content()]
        if len(force) < 3 or len(torque) < 3:
            raise RuntimeError("xCoreSDK returned an incomplete Cartesian wrench")
        return {
            "force": force[:3],
            "torque": torque[:3],
            "joint_torque_measured": [
                float(value) for value in joint_measured.content()
            ],
            "joint_torque_external": [
                float(value) for value in joint_external.content()
            ],
            "reference_frame": reference_frame.lower(),
            "ts": time.time(),
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
