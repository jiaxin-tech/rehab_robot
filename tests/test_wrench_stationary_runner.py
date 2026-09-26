"""Offline integration only. Real spawned production provider; no live SDK."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from scripts.prepare_wrench_stationary_validation import prepare
from scripts import run_wrench_stationary_validation as runner


@pytest.fixture
def request_record(tmp_path):
    return prepare(tmp_path / "request", 50.)


@pytest.fixture
def outputs(tmp_path, request):
    root = Path(os.environ.get("STATIONARY_TEST_OUTPUT", tmp_path)) / request.node.name
    root.mkdir(parents=True, exist_ok=False)
    return root


@pytest.mark.parametrize("case", ["A", "B"])
def test_normal_case(request_record, outputs, case, monkeypatch):
    # A hardware constructor or any native wrench call in parent is a test failure.
    from hardware.rokae_adapter import RokaeRobotAdapter
    def forbidden(*args, **kwargs): raise AssertionError("hardware forbidden")
    monkeypatch.setattr(RokaeRobotAdapter, "__init__", forbidden)
    path, summary = runner.run_case(request_record, case, outputs, offline_seconds=1.2)
    assert summary["offline_dry_run_pass"], summary["failures"]
    assert summary["requested_duration_s"] == 900
    assert not summary["completed_900s"]
    assert summary["intended_wrench_rate_hz"] == 50
    assert not summary["robot_connected"]
    assert summary["rt_sample_count"] > 0
    assert summary["timing_acceptance"] == "UNDEFINED"
    for name in ("protocol_snapshot.json", "rt_state.csv", "wrench_query_events.jsonl", "wrench_samples.csv",
                 "health_timeline.jsonl", "process_lifecycle.jsonl", "cleanup_outcome.json", "end_state.json", "case_summary.json"):
        assert (path / name).exists()
    if case == "A":
        assert summary["wrench_status"] == "N/A_INTENTIONALLY_DISABLED"
        assert all(not r["valid"] for r in runner.rows(path / "health_timeline.jsonl"))
    else:
        assert summary["wrench_successful_queries"] > 0
        assert summary["cleanup"]["child"]["GRACEFUL_SDK_DISCONNECT"] is None
        assert summary["cleanup"]["child"]["process_exited"]


@pytest.mark.parametrize("behavior,expected", [("error263", "263"), ("never", None)])
def test_faults_preserved(request_record, outputs, behavior, expected):
    path, summary = runner.run_case(request_record, "B", outputs, offline_seconds=3., behavior=behavior)
    assert summary["failures"]
    assert not summary["completed_900s"]
    assert summary["requested_duration_s"] == 900
    assert summary["actual_duration_s"] < 3
    if expected:
        assert summary["sdk_error_histogram"] == {"263": 1}
        assert summary["wrench_started_queries"] == 2
    else:
        assert summary["pending_queries"]
        assert summary["wrench_completed_queries"] == 0
        assert summary["cleanup"]["child"]["FORCED_WORKER_TERMINATION"]
        assert summary["cleanup"]["child"]["GRACEFUL_SDK_DISCONNECT"] is False


def test_missing_final_state(request_record, outputs):
    path, summary = runner.run_case(request_record, "A", outputs, offline_seconds=.1, final_state=False)
    end = json.loads((path / "end_state.json").read_text())
    assert end["final_operation_state"] is None and end["reason"]
    assert "final_operation_state_missing" in summary["failures"]


def test_state_failure(request_record, outputs):
    _, summary = runner.run_case(request_record, "A", outputs, offline_seconds=1., state_failure=True)
    assert "state_unhealthy" in summary["failures"]


@pytest.mark.parametrize("authorization", [None, {}, {"reviewed": True}])
def test_live_missing_auth_before_sdk(request_record, tmp_path, authorization, monkeypatch):
    from hardware.rokae_adapter import RokaeRobotAdapter
    calls = []
    monkeypatch.setattr(RokaeRobotAdapter, "__init__", lambda *a, **k: calls.append("SDK"))
    with pytest.raises(PermissionError):
        runner.run_case(request_record, "B", tmp_path / "unused", mode="LIVE_STATIONARY", authorization=authorization)
    assert not calls and not (tmp_path / "unused").exists()
    assert "xCoreSDK_python" not in sys.modules


@pytest.mark.parametrize("key,value", [("schema_version", 2), ("duration_s", 1), ("intended_wrench_hz", 0), ("retry", True)])
def test_request_rejection(request_record, tmp_path, key, value):
    request_record[key] = value
    with pytest.raises(ValueError): runner.run_case(request_record, "A", tmp_path)


def test_comparison_incomplete(request_record, outputs):
    _, a = runner.run_case(request_record, "A", outputs, offline_seconds=.1)
    _, b = runner.run_case(request_record, "B", outputs, offline_seconds=2., behavior="error263")
    result = runner.compare(a, b)
    assert result["status"] == "INCOMPLETE"
    assert result["causality"] == "NOT_ESTABLISHED"
    runner.write_json(outputs / "ab_comparison.json", result)


def test_runner_has_no_control_dependency():
    import ast
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("control", "safety"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"execute", "servo", "setPowerState", "setOperateMode", "setMotionControlMode", "move", "movej", "movel"}


def complete_authorization(request_record):
    # Deliberately fictional OFFLINE artifact; this helper NEVER invokes run_case live.
    request_record.update(budgets=dict(startup_s=5., query_s=.5, progress_s=2., sample_age_s=2.,
        ack_s=2., log_s=2., shutdown_s=.5, terminate_s=.5, state_age_s=2., skew_s=2.,
        provenance="OFFLINE_AUTHORIZATION_VALIDATOR_FIXTURE"),
        lifecycle_budgets=dict.fromkeys(("connect_s", "stop_s", "end_state_s", "disconnect_s", "logger_s"), 1.),
        connection=dict(robot_ip="OFFLINE", local_ip="OFFLINE", robot_class="OFFLINE"),
        expected_identity=dict(robot_model="OFFLINE", robot_serial="OFFLINE", controller_version="OFFLINE", sdk_version="OFFLINE"),
        allowed_power_states=["off"], criteria=dict(max_supervisor_interval_s=.5))
    a = dict(schema_version=1, scope="LIVE_STATIONARY_NO_MOTION", reviewed=True,
        request_sha256=runner.digest(request_record), case="A", source_hashes=runner.source_hashes(),
        previous_session_recovery_confirmed=True, operator_present=True, stop_procedure_reviewed=True,
        parent_native_lifecycle_reviewed=True, end_state_path="existing_session_operationState",
        expires_at="2099-01-01T00:00:00Z")
    for key in ("authorization_id", "reviewer", "operator", "stop_procedure", "recovery_record", "setup_record", "vendor_session_record"):
        a[key] = "OFFLINE_FIXTURE"
    return a


@pytest.mark.parametrize("tamper", ["request", "source", "expiry", "ack", "budget", "case"])
def test_authorization_binding(request_record, tamper):
    a = complete_authorization(request_record)
    ack = "OFFLINE_FIXTURE:OFFLINE_FIXTURE"
    runner.authorize(request_record, "A", a, ack)  # No adapter construction.
    if tamper == "request": request_record["intended_wrench_hz"] = 20.
    elif tamper == "source": a["source_hashes"] = {}
    elif tamper == "expiry": a["expires_at"] = "2000-01-01T00:00:00Z"
    elif tamper == "ack": ack = "yes"
    elif tamper == "budget":
        request_record["budgets"] = None
        a["request_sha256"] = runner.digest(request_record)
    else: a["case"] = "B"
    with pytest.raises(PermissionError): runner.authorize(request_record, "A", a, ack)


def test_b_requires_reviewed_a(request_record):
    a = complete_authorization(request_record)
    a["case"] = "B"
    with pytest.raises(PermissionError, match="requires reviewed successful A"):
        runner.authorize(request_record, "B", a, "OFFLINE_FIXTURE:OFFLINE_FIXTURE")


def test_failed_stop_does_not_retry_disconnect(request_record, outputs, monkeypatch):
    calls = []
    def stop(self): raise RuntimeError("injected_stop_failure")
    def disconnect(self): calls.append("disconnect")
    monkeypatch.setattr(runner.OfflineStateAdapter, "stop_state_stream", stop)
    monkeypatch.setattr(runner.OfflineStateAdapter, "disconnect", disconnect)
    _, summary = runner.run_case(request_record, "A", outputs, offline_seconds=.1)
    assert not calls
    assert not summary["cleanup"]["parent"]["normal_cleanup"]
    assert summary["final_robot_operation_state"] is None
    assert summary["failures"]


def test_telemetry_failure_aborts(request_record, outputs, monkeypatch):
    monkeypatch.setattr(runner.Evidence, "check", lambda self: "injected_telemetry_failure")
    _, summary = runner.run_case(request_record, "A", outputs, offline_seconds=1.)
    assert summary["failures"] and not summary["completed_900s"]


def test_incomplete_jsonl_retained(tmp_path):
    path = tmp_path / "torn.jsonl"
    path.write_text('{"type":"good"}\n{"type":', encoding="utf-8")
    failures = []
    assert runner.rows(path, failures) == [{"type": "good"}]
    assert failures and "telemetry_corrupt" in failures[0]
    assert path.read_text().endswith('{"type":')


def test_bounded_call_does_not_claim_cancel():
    value, unresolved = runner.bounded_call(lambda: time.sleep(.05), .001)
    assert unresolved and "UNRESOLVED_PARENT_CALL" in value["error"]
    time.sleep(.06)
