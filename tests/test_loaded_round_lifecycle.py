"""SYNTHETIC HARDWARE ONLY. No SDK imports, connection or laboratory evidence."""
from dataclasses import replace
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from collection.twincat_leg import atomic_json
from control.robot_trajectory_executor import RokaeMotionExecutor
from safety.robot_trial_release import read_json, sha256_file
from tests.test_robot_leg_round import prepared_round, SyntheticCapture
from tests.test_robot_trajectory_executor import Acquisition as BaseAcquisition, Motion
import scripts.run_rehab_experiment as runner
import scripts.run_robot_leg_round as rounds


def install_hardware(monkeypatch, directory, fault=None):
    calls = []
    capture_instances = []
    anchor = read_json(directory / "anchor.json")["tcp_pose_base"]

    class Native:
        def get_robot_mode(self):
            return "MOVING" if fault == "busy" else "IDLE"
        def enable_realtime(self, *, load):
            calls.append("prepare_rt")
            assert load == 0.0
            if fault == "prepare":
                raise RuntimeError("SYNTHETIC_PARTIAL_PREPARATION_FAILURE")
        def stop_realtime(self, *, switch_to_nrt):
            calls.append("stop_rt")
        def disable(self):
            calls.append("power_off")
            if fault == "cleanup":
                raise RuntimeError("SYNTHETIC_POWER_OFF_FAILURE")

    summary = BaseAcquisition().summary
    summary["robot_metadata"].update(robot_serial_number="SN-RUNNER", xcore_sdk_version="0.7.0")
    summary.update(collision_state_source="events", safety_event_status=dict(max_age_s=.5))
    if fault in ("unknown_collision", "collision", "259"):
        summary.update(collision_state_query_valid=fault == "collision",
                       collision_state=True if fault == "collision" else None,
                       collision_state_invalid_reason="259" if fault == "259" else "unknown")

    class Adapter:
        native_robot = Native()
        def connect(self): calls.append("connect")
        def disconnect(self): calls.append("disconnect")
        def is_connected(self): return True
        def get_robot_state_summary(self): return summary
        def read_tcp_pose(self): return anchor

    class Acquisition(BaseAcquisition):
        live_producer_names = ()
        def __init__(self, adapter, logger, **kwargs):
            super().__init__()
            self.adapter = adapter
        def start(self, *, manage_connection):
            assert manage_connection is False
            calls.append("streams_start")
        def wait_until_healthy(self, **kwargs): return True
        def latest_state_frame(self):
            return SimpleNamespace(valid=True, tcp_position_m=anchor[:3], tcp_orientation_rad=anchor[3:])
        def latest_health(self):
            if "hold" in calls and fault == "disconnect":
                return replace(self.health, valid=False, invalid_reason="SYNTHETIC_DISCONNECT")
            if "hold" in calls and fault == "query_hang":
                return replace(self.health, valid=False, invalid_reason="query_pending_too_long")
            return self.health
        def stop(self): calls.append("streams_stop")

    class FakeMotion(Motion):
        def start_cartesian_hold(self, pose):
            calls.append("hold")
            if fault == "cancel":
                raise KeyboardInterrupt("SYNTHETIC_OPERATOR_CANCEL")
            super().start_cartesian_hold(pose)
        def send_cartesian_target(self, pose):
            calls.append("send")
            super().send_cartesian_target(pose)
        def request_stop(self, reason):
            calls.append("motion_stop")
            super().request_stop(reason)

    class VirtualExecutor(RokaeMotionExecutor):
        """Only scheduler waiting is virtual; real release and gates still run."""
        def __init__(self, *args, **kwargs):
            self.test_time = time.perf_counter()
            super().__init__(*args, clock=SimpleNamespace(now_s=lambda: self.test_time), **kwargs)
        def _wait_until(self, deadline_s):
            self.test_time = deadline_s
            reasons = self._runtime_reasons()
            if reasons:
                raise RuntimeError("runtime safety gate: " + ";".join(reasons))

    class Capture(SyntheticCapture):
        stopped = False
        def __init__(self, *args):
            super().__init__(*args)
            capture_instances.append(self)
        def start(self):
            super().start()
            self.raw_sha256 = sha256_file(self.directory / "leg.csv")
        def stop(self):
            self.stopped = True
            calls.append("leg_stop")

    monkeypatch.setattr(runner, "RealRobotAcquisition", Acquisition)
    monkeypatch.setattr(runner, "RokaeMotionExecutor", VirtualExecutor)
    monkeypatch.setattr(runner, "build_live_wrench_provider", lambda *a, **k: object())
    monkeypatch.setattr(rounds, "_configured_adapter_factory", lambda **k: lambda ip: Adapter())
    if fault == "disk":
        def disk_failure(self, **kwargs):
            raise OSError("SYNTHETIC_DISK_FULL")
        monkeypatch.setattr(runner.EpisodeLogger, "append_trajectory_command_bounded", disk_failure)

    def execute(**kwargs):
        assert kwargs["prepare_realtime"] is True
        original_observer = kwargs["dispatch_observer"]
        synthetic_dispatch = pd.read_csv(directory / "leg/dispatch.csv").itertuples()
        def observer(t, start, end):
            # Virtual time is not a hardware alignment measurement.
            row = next(synthetic_dispatch)
            assert t == pytest.approx(row.trajectory_time_s)
            original_observer(t, row.dispatch_start_ns, row.dispatch_end_ns)
        kwargs["dispatch_observer"] = observer
        return runner.run_execute(**kwargs, motion_factory=lambda adapter: FakeMotion())

    return Capture, execute, calls, capture_instances


@pytest.mark.parametrize("fault", [None, "disconnect", "query_hang", "disk", "cancel", "training_timeout",
                                  "prepare", "cleanup", "unknown_collision", "collision", "259", "busy"])
def test_full_loaded_round_lifecycle(tmp_path, monkeypatch, fault):
    directory = prepared_round(tmp_path)
    state = read_json(directory / "round_status.json")
    state["software_verification_only"] = True
    rounds.report(directory, state)
    Capture, execute, calls, captures = install_hardware(monkeypatch, directory, fault)
    real_worker = rounds.run_algorithm_worker
    def worker(*args, **kwargs):
        if not kwargs.get("validate_only"):
            assert calls.index("motion_stop") < calls.index("stop_rt") < calls.index("power_off") < calls.index("disconnect")
            assert captures[0].stopped
            calls.append("train")
            if fault == "training_timeout":
                # Exercise subprocess.run's actual timeout/kill/wait path.
                return real_worker(*args[:3], .001, **kwargs)
        return real_worker(*args, **kwargs)
    monkeypatch.setattr(rounds, "run_algorithm_worker", worker)
    if fault is None:
        result = rounds.run(directory, confirmation=rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
        assert result["status"] == "ROUND_COMPLETE"
        assert result["training_status"] == "MODEL_UPDATED"
    else:
        with pytest.raises((Exception, KeyboardInterrupt)):
            rounds.run(directory, confirmation=rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
        result = read_json(directory / "round_status.json")
        assert result["status"] == "ROUND_FAILED"
        assert result["training_status"] == "NOT_UPDATED"
        assert not (directory / "model_update.json").exists()
        assert ("train" in calls) == (fault == "training_timeout")
    assert captures[0].stopped
    assert calls.count("connect") == 1 and calls.count("disconnect") == 1
    assert calls.index("stop_rt") < calls.index("power_off") < calls.index("disconnect")
    assert (directory / "leg/leg.csv").is_file()
    assert sha256_file(directory / "leg/leg.csv") == captures[0].raw_sha256
    assert "SYNTHETIC HARDWARE" in (directory / "report.md").read_text(encoding="utf-8")
    if fault in ("prepare", "unknown_collision", "collision", "259", "busy"):
        assert "hold" not in calls
    if fault in ("unknown_collision", "collision", "259", "busy"):
        assert "prepare_rt" not in calls
    if fault in ("disconnect", "query_hang", "disk", "cancel"):
        assert "motion_stop" in calls and "send" not in calls


def test_capture_cleanup_precedes_failure_reporting_when_storage_fails(tmp_path, monkeypatch, capsys):
    directory = prepared_round(tmp_path, capture_only=True)
    Capture, execute, calls, captures = install_hardware(monkeypatch, directory, "disconnect")
    original_report = rounds.report
    original_atomic = rounds.atomic_json
    def report_with_failed_storage(path, state):
        if state["status"] == "ROUND_FAILED":
            assert captures[0].stopped
            def unavailable(*args, **kwargs):
                raise OSError("SYNTHETIC_STATUS_DISK_FAILURE")
            monkeypatch.setattr(rounds, "atomic_json", unavailable)
        try:
            return original_report(path, state)
        finally:
            monkeypatch.setattr(rounds, "atomic_json", original_atomic)
    monkeypatch.setattr(rounds, "report", report_with_failed_storage)
    with pytest.raises(RuntimeError, match="SYNTHETIC_DISCONNECT") as caught:
        rounds.run(directory, confirmation=rounds.CONFIRMATION, capture_factory=Capture, execute_fn=execute)
    assert "disconnect" in calls and captures[0].stopped
    assert (directory / "leg/leg.csv").is_file()
    assert "ROUND_FAILED" in (directory / "report.md").read_text(encoding="utf-8")
    assert "ROUND_REPORT_WRITE_FAILED" in capsys.readouterr().err
    assert any("could not be fully saved" in note for note in caught.value.__notes__)


def test_old_loaded_config_does_not_authorize_automatic_power_preparation(tmp_path):
    prepared_round(tmp_path, capture_only=True)
    p = tmp_path / "round.json"
    c = read_json(p)
    del c["realtime_preparation"]
    atomic_json(p, c)
    assert any("same_session" in reason for reason in rounds.check_config(p))


def test_idle_only_wrench_config_is_an_offline_blocker(tmp_path):
    prepared_round(tmp_path, capture_only=True)
    p = tmp_path / "wrench.json"
    c = read_json(p)
    c["allowed_operation_states"] = ["idle"]
    atomic_json(p, c)
    assert any("rtcontrolling" in reason for reason in rounds.check_config(tmp_path / "round.json"))
