"""Offline contract tests for the read-only ROKAE adapter."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from collection.state import KinematicStateFrame
from hardware.rokae_adapter import RokaeRobotAdapter
from hardware.windows.rokae_xcore import RokaeRobot


class FakeReadOnlyNative:
    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[str] = []
        self._state_thread = threading.Thread()

    def connect(self) -> None:
        self.calls.append("connect")
        self.is_connected = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.is_connected = False

    def start_state_stream(self) -> None:
        self.calls.append("start_state_stream")

    def stop_state_stream(self) -> None:
        self.calls.append("stop_state_stream")

    def get_cartesian_pose(self):
        self.calls.append("get_cartesian_pose")
        return [0.4, -0.1, 0.3, 0.1, 0.2, 0.3]

    def get_joint_angles(self):
        self.calls.append("get_joint_angles")
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def get_state_frame(self):
        self.calls.append("get_state_frame")
        return KinematicStateFrame(
            sequence_id=3,
            host_monotonic_time_s=1.0,
            wall_time_iso="2026-08-09T00:00:00+00:00",
            robot_device_time_s=None,
            valid=True,
            invalid_reason="",
            tcp_position_m=(0.4, -0.1, 0.3),
            tcp_orientation_rad=(0.1, 0.2, 0.3),
            tcp_linear_velocity_mps=None,
            tcp_angular_velocity_radps=None,
            velocity_source="unavailable_first_realtime_frame",
            joint_position_rad=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            joint_velocity_radps=None,
            pose_time_s=1.0,
            joint_time_s=1.0,
            velocity_time_s=None,
            operation_state="IDLE",
            collision_state=None,
            controller_error=None,
        )

    def get_end_wrench(self, frame):
        self.calls.append(f"get_end_wrench:{frame}")
        return {
            "cartesian_force_raw_n": [1.0, 2.0, 3.0],
            "cartesian_torque_raw_nm": [0.1, 0.2, 0.3],
            "joint_measured_torque_nm": [1.0] * 6,
            "joint_external_torque_nm": [0.5] * 6,
            "raw_force_frame": frame,
            "force_query_started_s": 10.0,
            "force_query_finished_s": 10.004,
            "host_monotonic_time_s": 10.002,
            "wall_time_iso": "2026-08-09T00:00:00+00:00",
        }

    def get_robot_metadata(self):
        self.calls.append("get_robot_metadata")
        return {
            "robot_model": "xMate6-Pro",
            "robot_serial_number": "SN-ADAPTER",
            "controller_version": "controller-test",
            "xcore_sdk_version": "0.7.0",
            "joint_soft_limits_rad": [[-2.0, 2.0]] * 6,
            "sdk_tool_payload": {
                "toolset_load_mass_kg": 1.25,
                "toolset_load_cog_m": [0.0, 0.0, 0.08],
                "toolset_load_inertia_kg_m2": [0.01, 0.02, 0.03],
                "sdk_available_tool_names": ["rehab_cuff"],
                "sdk_available_workobject_names": ["rehab_fixture"],
                "active_hmi_tool_workobject_verified": False,
            },
        }

    def get_collision_state(self):
        self.calls.append("get_collision_state")
        return False


def test_adapter_exposes_only_read_contract_and_preserves_units():
    native = FakeReadOnlyNative()
    adapter = RokaeRobotAdapter(native_robot=native)
    adapter.connect()
    adapter.start_state_stream()
    assert adapter.is_connected()
    assert adapter.read_tcp_pose() == (0.4, -0.1, 0.3, 0.1, 0.2, 0.3)
    assert adapter.read_joint_positions() == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    adapter.stop_state_stream()
    adapter.disconnect()
    assert not adapter.is_connected()
    assert native.calls == [
        "connect",
        "start_state_stream",
        "get_cartesian_pose",
        "get_joint_angles",
        "stop_state_stream",
        "disconnect",
    ]
    for forbidden in ("enable", "power", "move", "drag", "calibrate", "clear_error"):
        assert not hasattr(adapter, forbidden)


def test_wrench_frame_keeps_query_and_publish_times():
    native = FakeReadOnlyNative()
    adapter = RokaeRobotAdapter(native_robot=native)
    frame = adapter.read_internal_wrench("world")
    assert frame.valid
    assert frame.timestamp_source == "host_query_time_no_robot_device_timestamp"
    assert frame.host_query_start_s == 10.0
    assert frame.host_query_end_s == 10.004
    assert abs(frame.query_duration_ms - 4.0) < 1e-9
    assert frame.host_publish_s > 0.0
    assert frame.cartesian_force_raw_n == (1.0, 2.0, 3.0)


def test_summary_labels_host_time_and_does_not_invent_device_time():
    native = FakeReadOnlyNative()
    native.is_connected = True
    adapter = RokaeRobotAdapter(native_robot=native)
    summary = adapter.get_robot_state_summary()
    assert summary["connected"] is True
    assert summary["timestamp_source"] == "host_receive_time_no_robot_device_timestamp"
    assert summary["robot_metadata"]["xcore_sdk_version"] == "0.7.0"
    assert summary["collision_state"] is False
    assert summary["collision_state_query_valid"] is True
    assert summary["joint_soft_limits_valid"] is True
    assert summary["sdk_tool_payload_read_valid"] is True
    assert "robot_device_time_s" not in summary


def test_summary_latches_observed_collision_across_later_false_query():
    native = FakeReadOnlyNative()
    values = iter((True, False))
    native.get_collision_state = lambda: next(values)
    adapter = RokaeRobotAdapter(native_robot=native)
    assert adapter.get_robot_state_summary()["collision_state"] is True
    assert adapter.get_robot_state_summary()["collision_state"] is True


def test_native_connect_does_not_change_motion_mode_or_command_cache():
    class Native:
        def __init__(self, *_args):
            self.calls = []

        def connectToRobot(self, *args):
            self.calls.append(("connectToRobot", *args))

        def robotInfo(self, ec):
            self.calls.append("robotInfo")
            return SimpleNamespace(joint_num=6, type="fake", version="fake")

        def forceControl(self):
            self.calls.append("forceControl")
            return object()

        def setMotionControlMode(self, *_args):
            raise AssertionError("read-only connect changed motion mode")

        def setMaxCacheSize(self, *_args):
            raise AssertionError("read-only connect changed command cache")

    native_holder = {}

    def make_native(*args):
        native_holder["robot"] = Native(*args)
        return native_holder["robot"]

    sdk = SimpleNamespace(
        BaseRobot=SimpleNamespace(sdkVersion=lambda: "0.7.0"),
        xMateRobot=make_native,
    )
    robot = RokaeRobot("192.0.2.1")
    with (
        patch("hardware.windows.rokae_xcore._load_sdk", return_value=sdk),
        patch.object(robot, "_read_joint_soft_limits_rad", return_value=None),
        patch.object(robot, "start_state_stream") as start_stream,
        patch.object(robot, "_refresh_operation_state"),
    ):
        robot.connect()

    assert robot.is_connected
    assert native_holder["robot"].calls == [
        ("connectToRobot", "192.0.2.1"),
        "robotInfo",
        "forceControl",
    ]
    start_stream.assert_called_once_with()


def test_command_target_update_does_not_take_wrench_query_lock():
    robot = RokaeRobot("192.0.2.1")
    robot._rt_active = True
    robot._rt_controller = object()
    with robot._wrench_sdk_lock:
        robot.set_realtime_cartesian_target([0.3, 0.0, 0.4, 0.0, 0.0, 0.0])
    assert robot._rt_target_native == (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)


def test_external_realtime_attach_checks_but_never_sets_power_or_mode():
    automatic = object()
    powered_on = object()
    idle = SimpleNamespace(name="idle")

    class Controller:
        def __init__(self, calls):
            self.calls = calls

        def setFilterFrequency(self, joint, cartesian, torque, ec):
            self.calls.append(("setFilterFrequency", joint, cartesian, torque))

    class Native:
        def __init__(self):
            self.calls = []
            self.controller = Controller(self.calls)

        def operateMode(self, ec):
            self.calls.append("operateMode")
            return automatic

        def powerState(self, ec):
            self.calls.append("powerState")
            return powered_on

        def getRtMotionController(self):
            self.calls.append("getRtMotionController")
            return self.controller

        def operationState(self, ec):
            self.calls.append("operationState")
            return idle

        def setOperateMode(self, *_args):
            raise AssertionError("attach set operate mode")

        def setPowerState(self, *_args):
            raise AssertionError("attach set power state")

        def setMotionControlMode(self, *_args):
            raise AssertionError("attach set motion mode")

    robot = RokaeRobot("192.0.2.1", local_ip="192.0.2.2")
    robot._sdk = SimpleNamespace(
        OperateMode=SimpleNamespace(automatic=automatic),
        PowerState=SimpleNamespace(on=powered_on),
        OperationState=SimpleNamespace(idle=idle),
    )
    robot._robot = Native()
    robot.is_connected = True
    robot.attach_externally_prepared_realtime(reviewed_filter_hz=25.0)
    assert robot._robot.calls == [
        "operationState",
        "operateMode",
        "powerState",
        "getRtMotionController",
        ("setFilterFrequency", 25.0, 25.0, 25.0),
        "operationState",
    ]


def test_native_stop_failure_calls_both_stops_and_preserves_hold_target():
    class Controller:
        def __init__(self):
            self.calls = []

        def stopLoop(self):
            self.calls.append("stopLoop")
            raise RuntimeError("loop did not confirm stop")

        def stopMove(self):
            self.calls.append("stopMove")

    robot = RokaeRobot("192.0.2.1")
    controller = Controller()
    robot._robot = object()
    robot._rt_controller = controller
    robot._rt_active = True
    robot._rt_callback = lambda: None
    robot._rt_target_native = (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)
    robot.is_connected = True
    with pytest.raises(RuntimeError, match="stopLoop"):
        robot.stop_realtime(switch_to_nrt=False)
    assert controller.calls == ["stopLoop", "stopMove"]
    assert robot._rt_active is True
    assert robot._rt_callback is not None
    assert robot._rt_target_native == (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)


def test_state_stream_stop_retains_live_thread_and_refuses_native_stop():
    class StuckThread:
        def __init__(self):
            self.join_calls = []

        def join(self, timeout):
            self.join_calls.append(timeout)

        def is_alive(self):
            return True

    class Native:
        def __init__(self):
            self.stop_receive_calls = 0

        def stopReceiveRobotState(self):
            self.stop_receive_calls += 1

    robot = RokaeRobot("192.0.2.1")
    thread = StuckThread()
    native = Native()
    robot._robot = native
    robot.is_connected = True
    robot._state_thread = thread
    robot._state_running = True
    robot._state_streaming = True
    with pytest.raises(RuntimeError, match="refusing.*disconnect"):
        robot.stop_state_stream()
    assert thread.join_calls == [1.0]
    assert robot._state_thread is thread
    assert robot._state_streaming is True
    assert native.stop_receive_calls == 0


def test_disconnect_error_preserves_native_handle_and_connection_state():
    class Native:
        def __init__(self):
            self.fail = True
            self.calls = 0

        def disconnectFromRobot(self, ec):
            self.calls += 1
            if self.fail:
                ec.update(ec=42, message="controller did not confirm disconnect")

    robot = RokaeRobot("192.0.2.1")
    native = Native()
    robot._robot = native
    robot._robot_info = object()
    robot._force_control = object()
    robot.is_connected = True
    with pytest.raises(RuntimeError, match="disconnectFromRobot failed"):
        robot.disconnect()
    assert native.calls == 1
    assert robot._robot is native
    assert robot._robot_info is not None
    assert robot._force_control is not None
    assert robot.is_connected is True

    native.fail = False
    robot.disconnect()
    assert native.calls == 2
    assert robot._robot is None
    assert robot.is_connected is False
    assert robot.robot_mode == "DISCONNECTED"


def test_start_and_stop_share_one_atomic_native_lifecycle_lock():
    start_entered = threading.Event()
    allow_start_return = threading.Event()

    class Controller:
        def __init__(self):
            self.calls = []

        def setControlLoopCar(self, callback, priority, use_state):
            self.calls.append("setControlLoopCar")

        def startMove(self, mode):
            self.calls.append("startMove")

        def startLoop(self, blocking):
            self.calls.append("startLoop")
            start_entered.set()
            assert allow_start_return.wait(timeout=1.0)

        def stopLoop(self):
            self.calls.append("stopLoop")

        def stopMove(self):
            self.calls.append("stopMove")

    robot = RokaeRobot("192.0.2.1")
    controller = Controller()
    robot._sdk = SimpleNamespace(
        CartesianPosition=lambda target: tuple(target),
        RtControllerMode=SimpleNamespace(cartesianPosition="cartesianPosition"),
    )
    robot._robot = object()
    robot._rt_controller = controller
    robot.is_connected = True
    errors = []

    starter = threading.Thread(
        target=lambda: _capture_thread_error(
            errors,
            robot.start_realtime_cartesian,
            (0.3, 0.0, 0.4, 0.0, 0.0, 0.0),
        )
    )
    stopper = threading.Thread(
        target=lambda: _capture_thread_error(
            errors,
            robot.stop_realtime,
            False,
        )
    )
    starter.start()
    assert start_entered.wait(timeout=1.0)
    stopper.start()
    allow_start_return.set()
    starter.join(timeout=1.0)
    stopper.join(timeout=1.0)
    assert not starter.is_alive()
    assert not stopper.is_alive()
    assert errors == []
    assert controller.calls == [
        "setControlLoopCar",
        "startMove",
        "startLoop",
        "stopLoop",
        "stopMove",
    ]
    assert robot._rt_active is False


def _capture_thread_error(errors, function, *args):
    try:
        function(*args)
    except BaseException as exc:
        errors.append(exc)
