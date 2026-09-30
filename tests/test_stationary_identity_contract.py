"""Offline contract tests: real metadata shape, no SDK or robot connection."""
from types import SimpleNamespace

import pytest

from collection.real_robot_acquisition import RealRobotAcquisition
from hardware.windows.rokae_xcore import RokaeRobot
from scripts import run_wrench_stationary_validation as runner
from scripts.prepare_wrench_stationary_validation import prepare


def metadata_from_native_contract(sdk_version):
    native = SimpleNamespace(
        ip_address="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE",
        _robot_info=SimpleNamespace(type="fixture-model", id="fixture-serial", version="fixture-controller"),
        _sdk_version=sdk_version, state_interval_ms=8,
        _joint_soft_limits_rad=None, _joint_soft_limit_error=None,
        get_tool_payload_metadata=lambda: {},
    )
    # Exercise the actual producer of the metadata key, without constructing SDK objects.
    result = RokaeRobot.get_robot_metadata(native)
    assert "xcore_sdk_version" in result and "sdk_version" not in result
    return result


def expected_identity():
    return dict(robot_model="fixture-model", robot_serial="fixture-serial",
                controller_version="fixture-controller", sdk_version="fixture-sdk")


@pytest.fixture(autouse=True)
def forbid_native_sdk_and_network(monkeypatch):
    import socket
    import sys
    from hardware.windows import rokae_xcore
    def forbidden(*args, **kwargs):
        raise AssertionError("native SDK/network forbidden in identity contract tests")
    monkeypatch.setattr(rokae_xcore, "_load_sdk", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert "xCoreSDK_python" not in sys.modules
    yield
    assert "xCoreSDK_python" not in sys.modules


@pytest.mark.parametrize("sdk_version,accepted", [("fixture-sdk", True), ("wrong-sdk", False), (None, False)])
def test_stationary_parent_uses_native_sdk_metadata(tmp_path, monkeypatch, sdk_version, accepted):
    calls = []

    class FakeAdapter(runner.OfflineStateAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.native_robot = SimpleNamespace(
                _rt_controller=None, get_robot_mode=lambda: "IDLE",
                _call=lambda *args: SimpleNamespace(name="off"),
                _robot=SimpleNamespace(powerState=lambda: None),
            )
        def read_robot_metadata(self):
            return metadata_from_native_contract(sdk_version)

    def stop_before_acquisition(self, **kwargs):
        calls.append("identity_verified")
        raise RuntimeError("OFFLINE_FIXTURE_STOP_BEFORE_ACQUISITION")

    original_hashes = runner.source_hashes()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_hashes", lambda: original_hashes)
    # Only this offline fixture bypasses authorization; the real hardware constructor is replaced.
    monkeypatch.setattr(runner, "authorize", lambda *args, **kwargs: None)
    monkeypatch.setattr("hardware.rokae_adapter.RokaeRobotAdapter", FakeAdapter)
    monkeypatch.setattr(RealRobotAcquisition, "start", stop_before_acquisition)
    request = prepare(tmp_path / "request", 50.)
    request.update(
        connection=dict(robot_ip="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE"),
        expected_identity=expected_identity(), allowed_power_states=["off"],
        budgets=dict.fromkeys(("startup_s", "query_s", "progress_s", "sample_age_s", "ack_s", "log_s",
                               "shutdown_s", "terminate_s", "state_age_s", "skew_s"), 2.),
        lifecycle_budgets=dict.fromkeys(("connect_s", "stop_s", "end_state_s", "disconnect_s", "logger_s"), 2.),
    )
    request["budgets"]["provenance"] = "NOT_SAFETY_THRESHOLDS"
    _, summary = runner.run_case(request, "A", tmp_path / "outputs", mode="LIVE_STATIONARY",
        authorization={"authorization_id": "OFFLINE_METADATA_CONTRACT_FIXTURE"})
    assert bool(calls) is accepted
    failures = ";".join(summary["failures"])
    assert ("parent_identity_mismatch:sdk_version" not in failures) is accepted
    if accepted:
        assert "OFFLINE_FIXTURE_STOP_BEFORE_ACQUISITION" in failures
    assert not summary["completed_900s"]


@pytest.mark.parametrize("sdk_version,accepted", [("fixture-sdk", True), ("wrong-sdk", False), (None, False)])
def test_acquisition_parent_child_uses_native_sdk_metadata(sdk_version, accepted):
    calls = []
    def stop_before_stream():
        calls.append("identity_verified")
        raise RuntimeError("OFFLINE_FIXTURE_STOP_BEFORE_STREAM")
    adapter = SimpleNamespace(
        native_robot=SimpleNamespace(ip_address="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE"),
        is_connected=lambda: True,
        read_robot_metadata=lambda: metadata_from_native_contract(sdk_version),
        start_state_stream=stop_before_stream, stop_state_stream=lambda: None, disconnect=lambda: None,
    )
    provider = SimpleNamespace(
        config=dict(mode="live", rate_hz=50., robot_ip="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE",
                    expected_identity=expected_identity()), process=None,
    )
    logger = SimpleNamespace(assert_healthy=lambda: None, signal_failure=lambda *args, **kwargs: None,
                             mark_failed=lambda *args, **kwargs: None)
    acquisition = RealRobotAcquisition(adapter, logger, wrench_provider=provider)
    error = RuntimeError if accepted else PermissionError
    reason = "OFFLINE_FIXTURE_STOP_BEFORE_STREAM" if accepted else "parent_child_identity_binding_mismatch:sdk_version"
    with pytest.raises(error, match=reason):
        acquisition.start(manage_connection=False)
    assert bool(calls) is accepted
