from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.lab_session as lab
from scripts.prepare_empty_load_commissioning_release import bounded_expiry


@pytest.fixture
def profile():
    return deepcopy(lab.load(lab.ROOT / "config/lab_next_visit.json"))


@pytest.mark.parametrize("action", ["diagnostics", "events", "capture", "observe", "empty", "round"])
def test_device_actions_require_explicit_live_before_subprocess(tmp_path, profile, action):
    with pytest.raises(PermissionError, match="--live"):
        lab.execute(action, profile, tmp_path, runner=lambda *a, **k: pytest.fail("device command"))


def test_diagnostics_retains_independent_failures_without_retry(tmp_path, profile):
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        assert "--enable-motion" not in command
        assert "--prepare-realtime" not in command
        assert "--operator-confirmation" not in command
        return SimpleNamespace(returncode=1 if command[4] == "scripts.diagnose_collision_events" else 0)
    result = lab.execute("diagnostics", profile, tmp_path, live=True, runner=runner)
    assert len(commands) == 3
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert [s["returncode"] for s in result["steps"]] == [1, 0, 0]
    assert all((tmp_path / name / "session_status.json").exists() for name in ("events", "capture", "observe"))


def test_missing_empty_release_never_reaches_motion_or_confirmation(tmp_path, profile):
    result = lab.execute("empty", profile, tmp_path, live=True,
        runner=lambda *a, **k: pytest.fail("motion command"),
        confirm=lambda *_: pytest.fail("invalid release should fail first"))
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert "required path" in result["reason"]


def test_empty_precheck_failure_prevents_motion(tmp_path, profile, monkeypatch):
    monkeypatch.setattr(lab, "empty_release", lambda _: SimpleNamespace(approval_sha256="test", payload={}))
    profile["empty_load"]["release"] = "test-release.json"
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    result = lab.execute("empty", profile, tmp_path, live=True, runner=runner,
                         confirm=lambda *_: pytest.fail("precheck did not pass"))
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert len(calls) == 1 and "scripts.check_empty_load_commissioning_release" in calls[0]


@pytest.mark.parametrize("seconds,allowed", [(600, True), (1800, True), (1801, False)])
def test_empty_release_lifetime_is_enforced(profile, monkeypatch, seconds, allowed):
    import safety.empty_load_commissioning_release as releases
    now = datetime.now(timezone.utc)
    payload = dict(issued_at=now.isoformat(), expires_at=(now+timedelta(seconds=seconds)).isoformat(),
                   single_cycle=True, training_allowed=False)
    monkeypatch.setattr(releases, "load_and_validate_empty_load_commissioning_release",
                        lambda *a, **k: SimpleNamespace(payload=payload))
    profile["empty_load"]["release"] = "test-release.json"
    if allowed:
        assert lab.empty_release(profile).payload == payload
    else:
        with pytest.raises(PermissionError, match="lifetime"):
            lab.empty_release(profile)


@pytest.mark.parametrize("confirmed", [False, True])
def test_empty_uses_only_existing_release_after_precheck_and_manual_confirmation(tmp_path, profile, monkeypatch, confirmed):
    from control import start_anchor
    profile["empty_load"]["release"] = "test-release.json"
    payload = dict(expires_at="future-for-test-only", wrench_hz=50)
    monkeypatch.setattr(lab, "empty_release", lambda _: SimpleNamespace(approval_sha256="test", payload=payload))
    monkeypatch.setattr(start_anchor, "load_start_anchor", lambda _: SimpleNamespace(anchor_id="test-anchor"))
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    result = lab.execute("empty", profile, tmp_path, live=True, runner=runner,
        confirm=lambda _: "I CONFIRM EMPTY LOAD ONE CYCLE" if confirmed else "cancel")
    assert len(calls) == (2 if confirmed else 1)
    assert result["status"] == ("COMPLETE" if confirmed else "BLOCKED_OR_FAILED")
    if confirmed:
        command = calls[1]
        assert command[4] == "scripts.run_rehab_experiment"
        assert "--prepare-realtime" in command and "--enable-motion" in command
        assert command[command.index("--collision-source")+1] == "query"
        assert "scripts.train_robot_leg_round" not in command


def test_capture_only_config_cannot_masquerade_as_training(tmp_path, profile, monkeypatch):
    import scripts.run_robot_leg_round as rounds
    monkeypatch.setattr(rounds, "check_config", lambda _: [])
    monkeypatch.setattr(rounds, "paths", lambda _: dict(trial_kind="dummy_leg_capture_only"))
    result = lab.execute("round", profile, tmp_path, live=True,
                         runner=lambda *a, **k: pytest.fail("motion command"))
    assert result["status"] == "BLOCKED_OR_FAILED"
    assert result["blockers"] == ["selected configuration does not enable real training"]


@pytest.mark.parametrize("confirmed", [False, True])
def test_complete_round_requires_preview_confirmation_and_real_update(tmp_path, profile, monkeypatch, confirmed):
    import scripts.run_robot_leg_round as rounds
    monkeypatch.setattr(rounds, "check_config", lambda _: [])
    monkeypatch.setattr(rounds, "paths", lambda _: dict(trial_kind="dummy_leg_reference"))
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        directory = Path(command[command.index("--session") + 1])
        directory.mkdir(exist_ok=True)
        if "run" in command:
            lab.write(directory / "round_status.json", dict(status="ROUND_COMPLETE", training_status="MODEL_UPDATED"))
        return SimpleNamespace(returncode=0)
    result = lab.execute("round", profile, tmp_path, live=True, runner=runner,
        confirm=lambda _: rounds.CONFIRMATION if confirmed else "cancel")
    assert len(calls) == (2 if confirmed else 1)
    assert result["status"] == ("COMPLETE" if confirmed else "BLOCKED_OR_FAILED")
    if confirmed:
        assert result["round_report"].endswith("report.md")


def test_offline_training_rejects_incomplete_round(tmp_path, profile):
    original = tmp_path / "original"
    original.mkdir()
    lab.write(original / "round_status.json", dict(status="ROUND_FAILED", training_status="NOT_UPDATED"))
    result = lab.execute("train", profile, tmp_path, session=original,
        runner=lambda *a, **k: pytest.fail("must not train failed data"))
    assert result["status"] == "BLOCKED_OR_FAILED"


def test_initialization_has_no_measurements_and_no_device_command(tmp_path, profile, monkeypatch):
    import scripts.run_robot_leg_round as rounds
    config = rounds.paths(lab.path(profile["round_config"]))
    reference = lab.ROOT / "reference_release/reference_measured_asymmetric_closed_slow.csv"
    config["candidates"][0]["path"] = str(reference)
    monkeypatch.setattr(rounds, "paths", lambda _: config)
    def runner(command, **kwargs):
        assert "--validate-only" in command
        assert command[4] == "scripts.train_robot_leg_round"
        lab.write(Path(command[command.index("--output")+1]), dict(status="ALGORITHM_INITIALIZED"))
        return SimpleNamespace(returncode=0)
    result = lab.execute("initialize", profile, tmp_path, runner=runner)
    assert result["status"] == "COMPLETE"
    assert lab.load(tmp_path / "initialization_request.json")["history"] == []


@pytest.mark.parametrize("minutes", [0, -1, 31, float("nan"), float("inf")])
def test_invalid_short_release_lifetime(minutes):
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        bounded_expiry((now+timedelta(days=1)).isoformat(), minutes, now)


def test_release_expiry_never_extends_reviewed_period():
    now = datetime.now(timezone.utc)
    assert bounded_expiry((now+timedelta(days=1)).isoformat(), 30, now) == now+timedelta(minutes=30)
    assert bounded_expiry((now+timedelta(minutes=5)).isoformat(), 30, now) == now+timedelta(minutes=5)
    with pytest.raises(PermissionError):
        bounded_expiry((now-timedelta(seconds=1)).isoformat(), 30, now)


def test_release_requires_explicit_confirmation_not_replayed_request(tmp_path, profile):
    result = lab.execute("release", profile, tmp_path,
        runner=lambda *a, **k: pytest.fail("unconfirmed approval generated"), confirm=lambda _: "")
    assert result["status"] == "BLOCKED_OR_FAILED"
