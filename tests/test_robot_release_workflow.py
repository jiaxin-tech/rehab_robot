"""Real release/runner/preflight/executor chain with synthetic hardware only."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from control.execution_preflight import OPERATOR_CONFIRMATION, evaluate_offline_execution_request
from control.robot_trajectory_executor import RokaeMotionExecutor
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import build_start_anchored_relative_trajectory, load_rehab_frame_config
from lower_limb_sim.run_robot_trajectory_export import DEFAULT_REFERENCE_PATH
from safety.experiment_safety import load_experiment_safety_config
from safety.robot_trial_release import (
    APPROVAL_CONFIRMATION, EVIDENCE_KINDS, load_and_validate_robot_trial_release,
    read_json, sha256_file,
)
from scripts.prepare_robot_trial_release import main as release_main
from scripts.acquire_robot_data import load_live_wrench_config
import scripts.run_rehab_experiment as runner
from tests.test_robot_trial_release import _wrench_config
from tests.test_run_rehab_experiment import _reviewed_files
from tests.test_robot_trajectory_executor import Acquisition, Motion


def workflow(tmp_path):
    anchor, frame, safety = _reviewed_files(tmp_path)
    wrench = _wrench_config(tmp_path / "wrench_config.json")
    draft, approved = tmp_path / "draft.json", tmp_path / "releases/approved.json"
    expires = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    argv = ["--reference", str(DEFAULT_REFERENCE_PATH), "--anchor", str(anchor),
            "--frame-config", str(frame), "--safety-config", str(safety), "--wrench-config", str(wrench),
            "--robot-model", "xMate6-Pro", "--robot-serial-number", "SN-RUNNER",
            "--controller-version", "controller-test", "--sdk-version", "0.7.0",
            "--reviewer", "safety", "--operator", "operator", "--expires-at", expires,
            "--collision-source", "events", "--collision-event-max-age-s", "0.5", "--wrench-hz", "50"]
    assert release_main([*argv, "--output", str(draft)]) == 0
    payload = read_json(draft)
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    records = {}
    for kind in EVIDENCE_KINDS:
        path = site_dir / (kind + ".json")
        path.write_text('{"synthetic_test_evidence": true}', encoding="utf-8")
        records[kind] = dict(passed=True, path=path.name, sha256=sha256_file(path))
    site = site_dir / "review.json"
    site.write_text(json.dumps(dict(scope=payload["scope"], decision="approved", reviewer="safety",
                        reviewed_at=datetime.now(timezone.utc).isoformat(),
                        bundle_sha256=payload["bundle_sha256"], records=records)), encoding="utf-8")
    approve_args = [*argv, "--output", str(approved), "--approve", "--input-draft", str(draft),
                    "--site-review", str(site), "--confirmation", APPROVAL_CONFIRMATION]
    inputs = dict(reference_path=DEFAULT_REFERENCE_PATH, anchor_path=anchor,
                  frame_config_path=frame, safety_config_path=safety, wrench_config_path=wrench)
    return SimpleNamespace(anchor=anchor, frame=frame, safety=safety, wrench=wrench,
                           draft=draft, approved=approved, site=site, approve_args=approve_args, inputs=inputs)


def approve(w):
    assert release_main(w.approve_args) == 0
    return load_and_validate_robot_trial_release(w.approved, **w.inputs)


def runner_args(w, tmp_path):
    return dict(episode_dir=tmp_path / "episode", trajectory_id="reference_measured_asymmetric_closed_slow",
                anchor_path=w.anchor, frame_config_path=w.frame,
                safety_config_path=w.safety, requested_anchor_id="anchor_runner_test",
                enable_motion=True, operator_confirmation=OPERATOR_CONFIRMATION,
                robot_ip="192.0.2.1", local_ip="192.0.2.2", wrench_config_path=w.wrench,
                live_wrench_config=load_live_wrench_config(w.wrench), wrench_hz=50.,
                collision_source="events", collision_event_max_age_s=.5, release_approval_path=w.approved)


class FakeAdapter:
    def __init__(self):
        self.summary = Acquisition().summary
        self.summary.update(collision_state_source="events", safety_event_status={"max_age_s": .5})
        self.summary["robot_metadata"].update(robot_serial_number="SN-RUNNER", xcore_sdk_version="0.7.0")

    def is_connected(self):
        return True

    def get_robot_state_summary(self):
        return deepcopy(self.summary)

    def read_tcp_pose(self):
        return (.4, 0., .3, .1, .2, .3)


class FakeAcquisition(Acquisition):
    def __init__(self, adapter, logger, **kwargs):
        super().__init__()
        self.adapter = adapter
        self.stopped = False
        self.live_producer_names = ()

    def start(self, **kwargs):
        pass

    def stop(self):
        self.stopped = True

    def wait_until_healthy(self, **kwargs):
        return True

    def latest_state_frame(self):
        return SimpleNamespace(valid=True, tcp_position_m=(.4, 0., .3), tcp_orientation_rad=(.1, .2, .3))


def install_fakes(monkeypatch, *, motion=None):
    adapter = FakeAdapter()
    motion = motion or Motion()
    # Accelerate only the scheduler clock/wait; keep trajectory audit, bindings,
    # actual logger and command dispatch logic intact.
    class Clock:
        t = 0.
        def now_s(self):
            return self.t
    class FastExecutor(RokaeMotionExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, clock=Clock())
        def _wait_until(self, deadline_s):
            self.clock.t = deadline_s
    monkeypatch.setattr(runner, "RealRobotAcquisition", FakeAcquisition)
    monkeypatch.setattr(runner, "build_live_wrench_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "RokaeMotionExecutor", FastExecutor)
    return adapter, motion


def test_cli_to_real_runner_preflight_and_executor_success(tmp_path, monkeypatch):
    w = workflow(tmp_path)
    validated = approve(w)
    assert read_json(w.draft)["approved"] is False
    assert validated.payload["wrench_hz"] == 50
    adapter, motion = install_fakes(monkeypatch)
    result = runner.run_execute(**runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                                motion_factory=lambda a: motion)
    assert result["completed"] is True
    assert result["commands_dispatched"] > 2
    assert [call[0] for call in motion.calls].count("start") == 1
    assert motion.calls[-1] == ("stop", "trajectory_completed")
    metadata = read_json(tmp_path / "episode/metadata.json")
    assert metadata["trial_release"]["approval"]["scope"] == "FIRST_ROBOT_TRIAL_EMPTY_LOAD"
    assert metadata["runtime_preflight"]["allowed"] is True


@pytest.mark.parametrize("change", ["missing_evidence", "failed_evidence", "changed_input", "expired",
                                   "wrong_scope", "changed_code", "bad_frequency", "bad_reviewer"])
def test_invalid_approval_is_never_published(tmp_path, change):
    w = workflow(tmp_path)
    site = read_json(w.site)
    draft = read_json(w.draft)
    if change == "missing_evidence":
        site["records"].pop("setup_and_stop")
    elif change == "failed_evidence":
        site["records"]["controller_safety"]["passed"] = False
    elif change == "changed_input":
        w.safety.write_text(w.safety.read_text() + "\n")
    elif change == "expired":
        draft["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    elif change == "wrong_scope":
        draft["scope"] = site["scope"] = "FIRST_ROBOT_TRIAL"
    elif change == "changed_code":
        draft["code_sha256"]["control/robot_trajectory_executor.py"] = "bad"
    elif change == "bad_frequency":
        draft["wrench_hz"] = -1
    elif change == "bad_reviewer":
        site["reviewer"] = "other"
    w.site.write_text(json.dumps(site), encoding="utf-8")
    w.draft.write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises((ValueError, PermissionError)):
        release_main(w.approve_args)
    assert not w.approved.exists()


@pytest.mark.parametrize("change", ["frequency", "source", "event_age", "wrench_object", "identity", "reviewer"])
def test_runner_or_approval_mismatch_never_connects(tmp_path, change):
    w = workflow(tmp_path)
    approve(w)
    kwargs = runner_args(w, tmp_path)
    if change == "frequency":
        kwargs["wrench_hz"] = 25
    elif change == "source":
        kwargs.update(collision_source="query", collision_event_max_age_s=None)
    elif change == "event_age":
        kwargs["collision_event_max_age_s"] = .7
    elif change == "wrench_object":
        kwargs["live_wrench_config"]["budgets"]["query_s"] = .6
    elif change in ("identity", "reviewer"):
        args = w.approve_args.copy()
        index = args.index("--robot-serial-number" if change == "identity" else "--reviewer") + 1
        args[index] = "other"
        # Another publication target proves CLI rejection before writing.
        args[args.index("--output") + 1] = str(tmp_path / "other.json")
        with pytest.raises(PermissionError):
            release_main(args)
        assert not (tmp_path / "other.json").exists()
        return
    calls = []
    with pytest.raises(PermissionError):
        runner.run_execute(**kwargs, adapter_factory=lambda ip: calls.append(ip))
    assert calls == []
    assert not (tmp_path / "episode").exists()


def test_plain_dict_cannot_approve_preflight(tmp_path):
    w = workflow(tmp_path)
    validated = approve(w)
    frame, anchor = load_rehab_frame_config(w.frame), load_start_anchor(w.anchor)
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        DEFAULT_REFERENCE_PATH, current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    args = dict(mode="execute", enable_motion=True, operator_confirmation=OPERATOR_CONFIRMATION,
                requested_anchor_id=anchor.anchor_id, frame=frame, anchor=anchor,
                safety=load_experiment_safety_config(w.safety), trajectory=trajectory, audit=audit)
    assert evaluate_offline_execution_request(**args, release_approval=validated).allowed
    assert not evaluate_offline_execution_request(**args, release_approval=validated.payload).allowed


@pytest.mark.parametrize("change", ["sdk", "event_age", "serial"])
def test_runtime_identity_or_source_budget_blocks_before_attachment(tmp_path, monkeypatch, change):
    w = workflow(tmp_path)
    approve(w)
    adapter, motion = install_fakes(monkeypatch)
    if change == "sdk":
        adapter.summary["robot_metadata"]["xcore_sdk_version"] = "other"
    elif change == "serial":
        adapter.summary["robot_metadata"]["robot_serial_number"] = "other"
    else:
        adapter.summary["safety_event_status"]["max_age_s"] = 9
    with pytest.raises(PermissionError):
        runner.run_execute(**runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                           motion_factory=lambda a: motion)
    assert motion.calls == []


def test_expiry_during_execution_stops_before_next_target(tmp_path, monkeypatch):
    import safety.robot_trial_release as release_module
    w = workflow(tmp_path)
    approve(w)
    class ExpiringMotion(Motion):
        def start_cartesian_hold(self, pose):
            super().start_cartesian_hold(pose)
            real_datetime = datetime
            class FutureDateTime:
                @staticmethod
                def now(tz):
                    return real_datetime.now(tz) + timedelta(days=1)
                fromisoformat = staticmethod(real_datetime.fromisoformat)
            monkeypatch.setattr(release_module, "datetime", FutureDateTime)
    adapter, motion = install_fakes(monkeypatch, motion=ExpiringMotion())
    with pytest.raises(RuntimeError, match="expired"):
        runner.run_execute(**runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                           motion_factory=lambda a: motion)
    assert not any(call[0] == "send" for call in motion.calls)
    assert any(call[0] == "stop" for call in motion.calls)


def test_actual_execute_cli_forwards_release_and_acquisition_options(tmp_path, monkeypatch):
    w = workflow(tmp_path)
    approve(w)
    adapter, motion = install_fakes(monkeypatch)
    assert runner.main([
        "--mode", "execute", "--ip", "192.0.2.1", "--local-ip", "192.0.2.2",
        "--episode-dir", str(tmp_path / "episode"), "--anchor", str(w.anchor),
        "--anchor-id", "anchor_runner_test", "--frame-config", str(w.frame),
        "--safety-config", str(w.safety), "--wrench-config", str(w.wrench), "--wrench-hz", "50",
        "--release-approval", str(w.approved), "--enable-motion",
        "--operator-confirmation", OPERATOR_CONFIRMATION, "--collision-source", "events",
        "--collision-event-max-age-s", "0.5"],
        adapter_factory=lambda ip: adapter, motion_factory=lambda a: motion) == 0
    assert motion.calls[-1] == ("stop", "trajectory_completed")


@pytest.mark.parametrize("change", ["evidence", "approval", "trajectory", "safety"])
def test_changes_after_loading_block_preflight(tmp_path, change):
    w = workflow(tmp_path)
    validated = approve(w)
    frame, anchor = load_rehab_frame_config(w.frame), load_start_anchor(w.anchor)
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        DEFAULT_REFERENCE_PATH, current_tcp_start_pose=anchor.tcp_pose_base, rehab_frame=frame)
    safety = load_experiment_safety_config(w.safety)
    if change == "evidence":
        (w.site.parent / "setup_and_stop.json").write_text("changed")
    elif change == "approval":
        w.approved.write_text(w.approved.read_text() + "\n")
    elif change == "trajectory":
        trajectory.loc[1, "tcp_x_base"] += .001
    else:
        from dataclasses import replace
        safety = replace(safety, max_force_n=99.)
    preflight = evaluate_offline_execution_request(
        mode="execute", enable_motion=True, operator_confirmation=OPERATOR_CONFIRMATION,
        requested_anchor_id=anchor.anchor_id, frame=frame, anchor=anchor,
        safety=safety, trajectory=trajectory, audit=audit, release_approval=validated)
    assert not preflight.allowed
    assert any("release_validation_failed" in r for r in preflight.reasons)


def test_evidence_change_after_attachment_blocks_first_hold(tmp_path, monkeypatch):
    w = workflow(tmp_path)
    approve(w)
    class TamperingMotion(Motion):
        def attach_externally_prepared(self, **kwargs):
            super().attach_externally_prepared(**kwargs)
            (w.site.parent / "controller_safety.json").write_text("changed")
    adapter, motion = install_fakes(monkeypatch, motion=TamperingMotion())
    with pytest.raises(PermissionError, match="reviewed file changed"):
        runner.run_execute(**runner_args(w, tmp_path), adapter_factory=lambda ip: adapter,
                           motion_factory=lambda a: motion)
    assert not any(c[0] in ("start", "send") for c in motion.calls)
    assert any(c[0] == "stop" for c in motion.calls)


def test_draft_cannot_execute_and_missing_frequency_cannot_generate_draft(tmp_path):
    w = workflow(tmp_path)
    kwargs = runner_args(w, tmp_path)
    kwargs["release_approval_path"] = w.draft
    calls = []
    with pytest.raises(PermissionError, match="not approved"):
        runner.run_execute(**kwargs, adapter_factory=lambda ip: calls.append(ip))
    assert calls == []
    args = w.approve_args.copy()
    index = args.index("--wrench-hz")
    del args[index:index + 2]
    with pytest.raises(SystemExit):
        release_main(args)
