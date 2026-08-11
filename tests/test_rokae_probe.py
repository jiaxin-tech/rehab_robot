"""Offline safety coverage for the read-only ROKAE probe."""

from __future__ import annotations

import json
import math

import pytest

from scripts.rokae_probe import (
    PROJECT_ACTION_STATEMENT,
    VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE,
    main,
    probe_adapter,
    run_probe,
)


class FakeReadOnlyAdapter:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.connected = False
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.state_summary: object = {
            "operation_state": "IDLE",
            "sdk_version": "fake",
            "state_valid": True,
            "state_stream_thread_alive": True,
        }
        self.tcp_pose: object = {
            "position_base_m": [0.3, -0.2, 0.4],
            "orientation_rad": [0.0, 0.0, 0.0],
        }
        self.joint_positions: object = [0.0] * 6
        self.internal_wrench: object = {
            "cartesian_force_n": [1.0, 2.0, 3.0],
            "cartesian_torque_nm": [0.1, 0.2, 0.3],
            "host_query_start_s": 10.0,
            "host_query_end_s": 10.01,
            "host_monotonic_time_s": 10.005,
            "valid": True,
        }

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"simulated {name} failure")

    def connect(self) -> None:
        self._call("connect")
        self.connected = True

    def disconnect(self) -> None:
        self._call("disconnect")
        self.connected = False

    def is_connected(self) -> bool:
        self._call("is_connected")
        return self.connected

    def start_state_stream(self) -> None:
        self._call("start_state_stream")

    def stop_state_stream(self) -> None:
        self._call("stop_state_stream")

    def read_tcp_pose(self) -> dict[str, object]:
        self._call("read_tcp_pose")
        return self.tcp_pose  # type: ignore[return-value]

    def read_joint_positions(self) -> list[float]:
        self._call("read_joint_positions")
        return self.joint_positions  # type: ignore[return-value]

    def read_internal_wrench(self) -> dict[str, object]:
        self._call("read_internal_wrench")
        return self.internal_wrench  # type: ignore[return-value]

    def get_robot_state_summary(self) -> dict[str, object]:
        self._call("get_robot_state_summary")
        return self.state_summary  # type: ignore[return-value]

    # If the probe ever expands into a motion/power workflow these methods make
    # the regression fail immediately, in addition to the exact call-list check.
    def clear_error(self) -> None:
        raise AssertionError("probe called clear_error")

    def enable(self) -> None:
        raise AssertionError("probe called enable")

    def set_power_state(self, _enabled: bool) -> None:
        raise AssertionError("probe called set_power_state")

    def move_l(self, _target: object) -> None:
        raise AssertionError("probe called move_l")

    def move_j(self, _target: object) -> None:
        raise AssertionError("probe called move_j")

    def enable_realtime(self) -> None:
        raise AssertionError("probe called enable_realtime")

    def start_realtime_cartesian(self, _target: object) -> None:
        raise AssertionError("probe called start_realtime_cartesian")


EXPECTED_SUCCESS_CALLS = [
    "connect",
    "is_connected",
    "start_state_stream",
    "get_robot_state_summary",
    "read_tcp_pose",
    "read_joint_positions",
    "read_internal_wrench",
    "stop_state_stream",
    "disconnect",
    "is_connected",
]


def test_probe_uses_only_read_only_contract_and_cleans_up() -> None:
    adapter = FakeReadOnlyAdapter()

    result = probe_adapter(adapter, robot_ip="192.0.2.10")

    assert result["success"] is True
    assert "declared_read_only" not in result
    assert result["project_action_statement"] == PROJECT_ACTION_STATEMENT
    assert result["vendor_session_side_effects_possible"] is True
    assert (
        result["vendor_session_side_effect_disclosure"]
        == VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE
    )
    assert result["checks"]["tcp_pose"]["value"]["position_base_m"] == [0.3, -0.2, 0.4]
    assert result["checks"]["tcp_pose"]["semantic_valid"] is True
    assert result["checks"]["internal_wrench"]["value"]["valid"] is True
    assert result["cleanup"]["is_disconnected"]["value"] is True
    assert adapter.connected is False
    assert adapter.calls == EXPECTED_SUCCESS_CALLS


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "check_name", "message"),
    [
        (
            "state_summary",
            {"state_valid": False, "state_stream_thread_alive": True},
            "robot_state_summary",
            "state_valid",
        ),
        (
            "state_summary",
            {"state_valid": True, "state_stream_thread_alive": False},
            "robot_state_summary",
            "thread_alive",
        ),
        ("tcp_pose", [0.0] * 5, "tcp_pose", "exactly 6"),
        ("tcp_pose", [0.0, 0.0, 0.0, 0.0, math.nan, 0.0], "tcp_pose", "finite"),
        ("joint_positions", [0.0] * 5, "joint_positions", "exactly 6"),
        (
            "internal_wrench",
            {
                "valid": False,
                "host_query_start_s": 1.0,
                "host_query_end_s": 2.0,
            },
            "internal_wrench",
            "valid",
        ),
        (
            "internal_wrench",
            {
                "valid": True,
                "host_query_start_s": 2.0,
                "host_query_end_s": 1.0,
            },
            "internal_wrench",
            "must not precede",
        ),
        (
            "internal_wrench",
            {
                "valid": True,
                "host_query_start_s": math.nan,
                "host_query_end_s": 2.0,
            },
            "internal_wrench",
            "must be finite",
        ),
    ],
)
def test_success_requires_semantically_valid_measurements(
    field_name: str,
    invalid_value: object,
    check_name: str,
    message: str,
) -> None:
    adapter = FakeReadOnlyAdapter()
    setattr(adapter, field_name, invalid_value)

    result = probe_adapter(adapter)

    check = result["checks"][check_name]
    assert result["success"] is False
    assert check["transport_ok"] is True
    assert check["semantic_valid"] is False
    assert check["error"]["type"] == "ProbeSemanticError"
    assert message in check["error"]["message"]
    assert adapter.connected is False


def test_probe_read_failure_is_reported_and_cleanup_still_runs() -> None:
    adapter = FakeReadOnlyAdapter(fail_on="read_internal_wrench")

    result = probe_adapter(adapter)

    assert result["success"] is False
    assert result["checks"]["internal_wrench"]["error"] == {
        "type": "RuntimeError",
        "message": "simulated read_internal_wrench failure",
    }
    assert adapter.calls[-3:] == ["stop_state_stream", "disconnect", "is_connected"]
    assert adapter.connected is False


def test_partial_stream_start_failure_is_cleaned_up() -> None:
    adapter = FakeReadOnlyAdapter(fail_on="start_state_stream")

    result = probe_adapter(adapter)

    assert result["success"] is False
    assert result["checks"]["start_state_stream"]["ok"] is False
    assert "stop_state_stream" in adapter.calls
    assert adapter.calls[-2:] == ["disconnect", "is_connected"]
    assert adapter.connected is False


def test_disconnect_must_be_semantically_confirmed() -> None:
    class StickyConnection(FakeReadOnlyAdapter):
        def disconnect(self) -> None:
            self._call("disconnect")
            # Simulate a transport call that returned without actually changing
            # the adapter's observable connection state.

    adapter = StickyConnection()
    result = probe_adapter(adapter)
    assert result["success"] is False
    assert result["cleanup"]["disconnect"]["transport_ok"] is True
    assert result["cleanup"]["is_disconnected"]["semantic_valid"] is False
    assert "must be true" in result["cleanup"]["is_disconnected"]["error"]["message"]


def test_cli_prints_machine_readable_json_with_injected_factory(capsys) -> None:
    adapter = FakeReadOnlyAdapter()

    exit_code = main(
        ["--ip", "192.0.2.20"],
        adapter_factory=lambda robot_ip: adapter,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["robot_ip"] == "192.0.2.20"
    assert adapter.calls == EXPECTED_SUCCESS_CALLS


def test_factory_failure_is_json_reportable_without_importing_sdk() -> None:
    def unavailable(_robot_ip: str) -> object:
        raise ImportError("simulated missing xCoreSDK adapter")

    payload = run_probe("192.0.2.30", adapter_factory=unavailable)

    assert payload["success"] is False
    assert payload["checks"]["adapter_factory"]["error"]["type"] == "ImportError"
    assert payload["cleanup"] == {}
