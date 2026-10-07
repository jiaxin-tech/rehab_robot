"""Offline tests for the supervised ROKAE commissioning layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.rokae_commissioning import (
    COMMISSIONING_CONFIRMATION,
    CommissioningError,
    CommissioningLimits,
    RokaeCommissioningConsole,
)

SOFT_LIMITS = (
    (-math.pi, math.pi),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
)


class FakeNative:
    """Records every call so the tests can prove which commands were issued."""

    def __init__(self, *, collision=False, soft_limits=SOFT_LIMITS, state_ok=True):
        self.calls = []
        self.ip_address = "192.0.2.10"
        self.local_ip = "192.0.2.20"
        self.rt_network_tolerance_percent = 20
        self._collision = collision
        self._soft_limits = soft_limits
        self._state_ok = state_ok
        self._joints = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self._pose = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]
        self.powered = False

    def get_robot_metadata(self):
        self.calls.append("get_robot_metadata")
        return {
            "robot_model": "xMate3",
            "xcore_sdk_version": "0.7.0",
            "joint_soft_limits_rad": self._soft_limits,
        }

    def get_cartesian_pose(self):
        return list(self._pose)

    def get_joint_angles(self):
        return list(self._joints)

    def enable(self, load=0.0):
        self.calls.append(("enable", load))
        self.powered = True

    def disable(self):
        self.calls.append("disable")
        self.powered = False

    def set_speed(self, ratio):
        self.calls.append(("set_speed", ratio))

    def get_robot_mode(self):
        return "IDLE" if self.powered else "DISCONNECTED"

    def clear_error(self):
        self.calls.append("clear_error")

    def reset(self):
        self.calls.append("reset")

    def move_j(self, joints):
        self.calls.append(("move_j", [float(v) for v in joints]))
        self._joints = [float(v) for v in joints]

    def move_l(self, pose):
        self.calls.append(("move_l", [float(v) for v in pose]))
        self._pose = [float(v) for v in pose]

    def wait_idle(self, timeout=30.0):
        self.calls.append(("wait_idle", timeout))
        return True

    def stop(self):
        self.calls.append("stop")

    def enable_realtime(self, load=0.0):
        self.calls.append(("enable_realtime", load))
        self.powered = True

    def stop_realtime(self, switch_to_nrt=True):
        self.calls.append(("stop_realtime", switch_to_nrt))
        self.powered = False

    def _realtime_prepared_state(self):
        return {
            "operate_mode": "automatic",
            "power_state": "on" if self.powered else "off",
            "rt_controller_obtained": any(
                isinstance(call, tuple) and call[0] == "enable_realtime"
                for call in self.calls
            ),
        }


class FakeAdapter:
    def __init__(self, native):
        self._native = native
        self.calls = []
        self.connected = False
        self.streaming = False

    @property
    def native_robot(self):
        return self._native

    def connect(self):
        self.calls.append("connect")
        self.connected = True

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False

    def start_state_stream(self):
        self.calls.append("start_state_stream")
        self.streaming = True

    def stop_state_stream(self):
        self.calls.append("stop_state_stream")
        self.streaming = False

    def is_connected(self):
        return self.connected

    @property
    def state_thread_alive(self):
        return self.streaming

    def read_tcp_pose(self):
        return tuple(self._native.get_cartesian_pose())

    def read_joint_positions(self):
        return tuple(self._native.get_joint_angles())

    def read_internal_wrench(self):
        return SimpleNamespace(valid=True)

    def get_robot_state_summary(self):
        return {
            "connected": self.connected,
            "state_stream_thread_alive": self.streaming,
            "state_valid": True,
            "state_invalid_reason": "",
            "state_age_ms": 1.0,
            "tcp_pose_base_m_rad": list(self._native.get_cartesian_pose()),
            "joint_position_rad": list(self._native.get_joint_angles()),
            "operation_state": self._native.get_robot_mode(),
            "collision_state": self._native._collision,
            "collision_state_query_valid": True,
            "collision_state_invalid_reason": "",
            "joint_soft_limits_valid": True,
            "sdk_tool_payload_read_valid": True,
            "robot_metadata": self._native.get_robot_metadata(),
        }


def make_console(**native_kwargs):
    native = FakeNative(**native_kwargs)
    adapter = FakeAdapter(native)
    console = RokaeCommissioningConsole(adapter, limits=CommissioningLimits())
    return console, adapter, native


def test_power_on_requires_exact_confirmation_and_connect():
    console, adapter, native = make_console()
    with pytest.raises(CommissioningError, match="connect before power_on"):
        console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    console.connect()
    with pytest.raises(CommissioningError, match="requires --confirm"):
        console.power_on(confirmation="yes")
    assert native.calls.count("enable") == 0
    result = console.power_on(confirmation=COMMISSIONING_CONFIRMATION, speed_ratio=5)
    assert result["powered"] is True
    assert ("enable", 0.0) in native.calls
    assert ("set_speed", 5) in native.calls


def test_joint_jog_is_bounded_by_step_and_soft_limits():
    console, adapter, native = make_console()
    console.connect()
    console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    with pytest.raises(CommissioningError, match="exceeds per-command limit"):
        console.jog_joint(1, math.radians(30.0), confirmation=COMMISSIONING_CONFIRMATION)
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in native.calls)

    console.jog_joint(1, math.radians(1.0), confirmation=COMMISSIONING_CONFIRMATION)
    issued = [call for call in native.calls if isinstance(call, tuple) and call[0] == "move_j"]
    assert len(issued) == 1
    assert issued[0][1][1] == pytest.approx(0.1 + math.radians(1.0))

    native._joints = [0.1, 1.99, 0.1, 0.1, 0.1, 0.1]
    with pytest.raises(CommissioningError, match="outside"):
        console.jog_joint(1, math.radians(1.0), confirmation=COMMISSIONING_CONFIRMATION)


def test_collision_state_blocks_motion_before_any_command():
    console, adapter, native = make_console(collision=True)
    console.connect()
    # A collision latch is a stop condition; it blocks power-on as well as jog.
    with pytest.raises(CommissioningError, match="collision"):
        console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    assert "enable" not in native.calls

    clean, clean_adapter, clean_native = make_console()
    clean.connect()
    clean.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    clean_native._collision = True
    with pytest.raises(CommissioningError, match="collision"):
        clean.jog_joint(1, math.radians(1.0), confirmation=COMMISSIONING_CONFIRMATION)
    assert not any(isinstance(call, tuple) and call[0] == "move_j" for call in clean_native.calls)


def test_missing_soft_limits_block_motion():
    console, adapter, native = make_console(soft_limits=None)
    console.connect()
    console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    with pytest.raises(CommissioningError, match="soft limits"):
        console.jog_joint(1, math.radians(1.0), confirmation=COMMISSIONING_CONFIRMATION)


def test_cartesian_jog_respects_step_limit_and_workspace():
    limits = CommissioningLimits(
        max_cartesian_step_m=0.01,
        workspace_min_base_m=(0.2, -0.5, 0.0),
        workspace_max_base_m=(0.8, 0.5, 0.6),
    )
    native = FakeNative()
    adapter = FakeAdapter(native)
    console = RokaeCommissioningConsole(adapter, limits=limits)
    console.connect()
    console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    with pytest.raises(CommissioningError, match="must not exceed"):
        console.jog_cartesian((0.05, 0.0, 0.0), confirmation=COMMISSIONING_CONFIRMATION)

    console.jog_cartesian((0.005, 0.0, 0.01), confirmation=COMMISSIONING_CONFIRMATION)
    issued = [call for call in native.calls if isinstance(call, tuple) and call[0] == "move_l"]
    assert issued[-1][1][0] == pytest.approx(0.405)
    assert issued[-1][1][2] == pytest.approx(0.31)

    # Raise the step ceiling so this case exercises the workspace gate, not the step gate.
    wide = CommissioningLimits(
        max_cartesian_step_m=1.0,
        workspace_min_base_m=(0.2, -0.5, 0.0),
        workspace_max_base_m=(0.8, 0.5, 0.6),
    )
    wide_native = FakeNative()
    wide_console = RokaeCommissioningConsole(FakeAdapter(wide_native), limits=wide)
    wide_console.connect()
    wide_console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    with pytest.raises(CommissioningError, match="outside the reviewed workspace"):
        wide_console.jog_cartesian((0.0, 0.0, 0.6), confirmation=COMMISSIONING_CONFIRMATION)


def test_stop_motion_needs_no_confirmation_and_is_audited(tmp_path: Path):
    audit = tmp_path / "commission.jsonl"
    native = FakeNative()
    adapter = FakeAdapter(native)
    console = RokaeCommissioningConsole(adapter, audit_path=audit)
    console.connect()
    console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    result = console.stop_motion("operator_pressed_estop")
    assert result["stopped"] is True
    assert "stop" in native.calls
    assert ("stop_realtime", True) in native.calls
    records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert any(record["action"] == "stop_motion" for record in records)
    assert any(record["action"] == "power_on" for record in records)


def test_self_test_connects_runs_read_only_checks_then_disconnects():
    native = FakeNative()
    adapter = FakeAdapter(native)
    console = RokaeCommissioningConsole(adapter)
    result = console.self_test()
    assert result["motion_commanded"] is False
    assert result["power_commanded"] is False
    assert result["cleanup"]["opened_by_self_test"] is True
    assert adapter.calls[:2] == ["connect", "start_state_stream"]
    assert "disconnect" in adapter.calls
    assert "stop_state_stream" in adapter.calls
    assert "enable" not in native.calls
    assert result["success"] is True


def test_disconnect_powers_down_and_stops_stream():
    native = FakeNative()
    adapter = FakeAdapter(native)
    console = RokaeCommissioningConsole(adapter)
    console.connect()
    console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    result = console.disconnect()
    assert result["disconnected"] is True
    assert "disable" in native.calls
    assert "stop_state_stream" in adapter.calls
    assert "disconnect" in adapter.calls


def test_cli_requires_confirmation_for_motion(monkeypatch, capsys):
    from scripts import rokae_commission as cli

    code = cli.main(
        ["--ip", "192.0.2.10", "power-on"],
        adapter_factory=lambda ip: FakeAdapter(FakeNative()),
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "requires --confirm" in captured.err


def test_cli_status_runs_and_reports_connected_state(capsys):
    from scripts import rokae_commission as cli

    code = cli.main(
        ["--ip", "192.0.2.10", "status"],
        adapter_factory=lambda ip: FakeAdapter(FakeNative()),
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["status"]["connected"] is True

def test_prepare_realtime_selects_mode_without_starting_motion():
    console, adapter, native = make_console()
    console.connect()
    with pytest.raises(CommissioningError, match="requires --confirm"):
        console.prepare_realtime(confirmation="nope")
    assert not any(isinstance(call, tuple) and call[0] == "enable_realtime" for call in native.calls)

    result = console.prepare_realtime(
        confirmation=COMMISSIONING_CONFIRMATION,
        network_tolerance_percent=10,
    )
    assert result["realtime_prepared"] is True
    assert result["motion_loop_started"] is False
    assert ("enable_realtime", 0.0) in native.calls
    assert native.rt_network_tolerance_percent == 10
    # Preparing realtime never issues a target or starts the RT loop.
    assert not any(isinstance(call, tuple) and call[0] in ("move_j", "move_l") for call in native.calls)
    assert "start_realtime_cartesian" not in native.calls


def test_prepare_realtime_refuses_power_already_on_and_bad_tolerance():
    console, adapter, native = make_console()
    console.connect()
    console.power_on(confirmation=COMMISSIONING_CONFIRMATION)
    with pytest.raises(CommissioningError, match="power_off before"):
        console.prepare_realtime(confirmation=COMMISSIONING_CONFIRMATION)

    clean, clean_adapter, clean_native = make_console()
    clean.connect()
    with pytest.raises(CommissioningError, match="network_tolerance_percent"):
        clean.prepare_realtime(
            confirmation=COMMISSIONING_CONFIRMATION,
            network_tolerance_percent=999,
        )
    assert "enable_realtime" not in clean_native.calls


def test_end_realtime_returns_to_non_realtime_mode():
    console, adapter, native = make_console()
    console.connect()
    console.prepare_realtime(confirmation=COMMISSIONING_CONFIRMATION)
    assert console.realtime_prepared is True
    result = console.end_realtime(confirmation=COMMISSIONING_CONFIRMATION)
    assert result["realtime_prepared"] is False
    assert console.powered is False
    assert ("stop_realtime", True) in native.calls
    assert "disable" in native.calls


def test_prepare_realtime_requires_a_local_interface():
    console, adapter, native = make_console()
    native.local_ip = ""
    console.connect()
    with pytest.raises(CommissioningError, match="local_ip|subnet"):
        console.prepare_realtime(confirmation=COMMISSIONING_CONFIRMATION)


def test_audit_records_a_failed_motion_wait(tmp_path: Path):
    audit = tmp_path / "commission.jsonl"
    native = FakeNative()
    adapter = FakeAdapter(native)
    console = RokaeCommissioningConsole(adapter, audit_path=audit)
    console.connect()
    console.power_on(confirmation=COMMISSIONING_CONFIRMATION)

    def failing_wait(timeout=30.0):
        raise TimeoutError("robot did not become idle")

    native.wait_idle = failing_wait
    with pytest.raises(TimeoutError):
        console.jog_joint(1, math.radians(1.0), confirmation=COMMISSIONING_CONFIRMATION)

    records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    entries = [r for r in records if r["action"] == "jog_joint"]
    assert len(entries) == 1
    assert entries[0]["command_issued"] is True
    assert entries[0]["ok"] is False
    assert "TimeoutError" in entries[0]["failure"]


def test_network_probe_validates_and_separates_network_from_sdk(monkeypatch):
    from hardware import network_probe

    assert network_probe.validate_ipv4("192.168.50.103") == "192.168.50.103"
    with pytest.raises(ValueError, match="not an IPv4"):
        network_probe.validate_ipv4("192.168.50")
    with pytest.raises(ValueError, match="not an IPv4"):
        network_probe.validate_ipv4("192.168.50.300")

    monkeypatch.setattr(network_probe, "icmp_ping", lambda ip, timeout_s=1.0: {"attempted": True, "reachable": False})
    monkeypatch.setattr(network_probe, "tcp_reachable", lambda ip, port, timeout_s=0.5: port == 8055)
    result = network_probe.probe_controller("192.0.2.10", ports=(8055,), timeout_s=0.1).to_dict()
    assert result["reachable"] is True
    assert result["robot_commands_sent"] is False


def test_network_probe_reports_inconclusive_when_ping_blocked(monkeypatch):
    from hardware import network_probe

    monkeypatch.setattr(network_probe, "icmp_ping", lambda ip, timeout_s=1.0: {"attempted": True, "reachable": False})
    monkeypatch.setattr(network_probe, "tcp_reachable", lambda ip, port, timeout_s=0.5: False)
    result = network_probe.probe_controller("192.0.2.10", ports=(8055,), timeout_s=0.1).to_dict()
    assert result["reachable"] is False
    assert "inconclusive" in result["operator_hint"] or "subnet" in result["operator_hint"]


def test_cli_network_command_sends_no_robot_commands(monkeypatch, capsys):
    from scripts import rokae_commission as cli
    from hardware import network_probe

    monkeypatch.setattr(network_probe, "icmp_ping", lambda ip, timeout_s=1.0: {"attempted": True, "reachable": True})
    monkeypatch.setattr(network_probe, "tcp_reachable", lambda ip, port, timeout_s=0.5: True)
    monkeypatch.setattr(cli, "probe_controller", network_probe.probe_controller)

    code = cli.main(["--ip", "192.0.2.10", "network"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["robot_commands_sent"] is False


def test_cli_rejects_bad_confirmation_before_opening_a_session():
    from scripts import rokae_commission as cli

    opened = []

    def factory(ip):
        opened.append(ip)
        return FakeAdapter(FakeNative())

    code = cli.main(["--ip", "192.0.2.10", "--confirm", "wrong", "power-on"], adapter_factory=factory)
    assert code == 2
    assert opened == []


def test_cli_accepts_shared_option_after_subcommand():
    from scripts import rokae_commission as cli

    captured = {}

    def factory(ip):
        captured["ip"] = ip
        return FakeAdapter(FakeNative())

    code = cli.main(
        ["--ip", "192.0.2.10", "power-on", "--confirm", COMMISSIONING_CONFIRMATION],
        adapter_factory=factory,
    )
    assert code == 0
    assert captured["ip"] == "192.0.2.10"
