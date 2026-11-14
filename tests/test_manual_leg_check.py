"""Local operator CLI tests; all acquisition is fake and no network is used."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from scripts import manual_leg_check as manual
from scripts import readonly_lab_client as readonly


def fake_offline(path):
    return {}, dict(ok=True, checks=dict(windows=True, python_64bit=True, dll_exists=True, configuration_valid=True),
                    config_path=str(path), config_sha256="a"*64, hostname="OLD_TEST", device_connected=False)


def fake_executor(task, config, output):
    output.mkdir(exist_ok=False)
    status = dict(ok=True, status="COMPLETE", actual_local_ams_net_id=readonly.EXPECTED_MAPPING["ams_net_id"],
                  toggle_changes=3, completed_utc=datetime.now(timezone.utc).isoformat())
    readonly.write_json(output / "segment_status.json", status)
    if task["action"] == "capture":
        (output / "leg.csv").write_text("sample_index,input_toggle\n0,0\n1,1\n", encoding="utf-8")
    return dict(ok=True, result=dict(rows=4, reads=20), error=None)


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(manual, "offline_check", fake_offline)
    monkeypatch.setattr(manual, "git_head", lambda: "test_commit")
    monkeypatch.setattr(manual.socket, "gethostname", lambda: "OLD_TEST")


def test_check_does_not_construct_or_call_acquisition(tmp_path, environment):
    def forbidden(*a, **k): pytest.fail("offline check must not read ADS")
    output, result = manual.run("check", config_path=tmp_path / "config", output_dir=tmp_path / "check", executor=forbidden)
    assert result["ok"] and result["git_head"] == "test_commit"
    assert not result["remote_helper_used"] and not result["motion_commanded"]
    assert (output / "manifest.json").is_file() and not (output / "segment_status.json").exists()


def test_live_flag_and_probe_required_no_automatic_retry(tmp_path, environment):
    calls = []
    def forbidden(*a): calls.append(True); pytest.fail("acquisition must be blocked")
    for action, live in (("probe", False), ("capture", True)):
        _, result = manual.run(action, config_path=tmp_path / "config", output_dir=tmp_path / action,
                               live=live, duration_s=10, executor=forbidden)
        assert not result["ok"]
    assert not calls


def test_successful_same_host_recent_probe_allows_bounded_capture(tmp_path, environment):
    probe, p = manual.run("probe", config_path=tmp_path / "config", output_dir=tmp_path / "probe",
                          live=True, executor=fake_executor)
    capture, c = manual.run("capture", config_path=tmp_path / "config", output_dir=tmp_path / "capture",
                            live=True, duration_s=10, probe_session=probe, executor=fake_executor)
    assert p["ok"] and c["ok"] and c["probe_session"] == str(probe)
    assert c["duration_s"] == 10 and (capture / "segment_status.json").is_file()
    assert c["output_files_sha256"] == {name: manual.sha256_file(capture / name)
        for name in ("leg.csv", "segment_status.json")}
    assert c["segment_status_sha256"] == c["output_files_sha256"]["segment_status.json"]


@pytest.mark.parametrize("kind", ["host", "config", "old", "failed", "source", "status_hash"])
def test_probe_binding_refuses_mixed_or_stale_or_failed_session(tmp_path, environment, kind):
    probe, _ = manual.run("probe", config_path=tmp_path / "config", output_dir=tmp_path / "probe",
                           live=True, executor=fake_executor)
    path = probe / "manifest.json"
    manifest = readonly.read_json(path)
    if kind == "host": manifest["hostname"] = "OTHER"
    if kind == "config": manifest["config_sha256"] = "b"*64
    if kind == "failed": manifest["ok"] = False
    if kind == "source": manifest["source_sha256"]["readonly_lab_client.py"] = "b"*64
    if kind == "status_hash": manifest["segment_status_sha256"] = "b"*64
    if kind == "old":
        status_path = probe / "segment_status.json"
        status = readonly.read_json(status_path)
        status["completed_utc"] = (datetime.now(timezone.utc)-timedelta(minutes=11)).isoformat()
        readonly.write_json(status_path, status)
        manifest["segment_status_sha256"] = manual.sha256_file(status_path)
    readonly.write_json(path, manifest)
    with pytest.raises(ValueError):
        manual.validate_probe_session(probe, config_sha256="a"*64, hostname="OLD_TEST")


def test_failed_acquisition_preserves_manifest_and_nonzero_main(tmp_path, environment):
    calls = []
    def failing(task, config, output):
        calls.append(task)
        output.mkdir()
        readonly.write_json(output / "segment_status.json", dict(status="FAILED", ok=False))
        return dict(ok=False, result=None, error="axis1 fault=True")
    output, result = manual.run("probe", config_path=tmp_path / "config", output_dir=tmp_path / "failed",
                                live=True, executor=failing)
    assert not result["ok"] and len(calls) == 1 and result["segment_status_sha256"]
    assert readonly.read_json(output / "manifest.json")["error"] == "axis1 fault=True"
    with pytest.raises(FileExistsError):
        manual.run("probe", config_path=tmp_path / "config", output_dir=output, live=True, executor=failing)


def test_cli_failure_returns_nonzero_and_prints_directory(tmp_path, environment, capsys):
    result = manual.main(["probe", "--config", str(tmp_path / "config"), "--output-dir", str(tmp_path / "blocked")])
    assert result == 1
    output = capsys.readouterr()
    assert str(tmp_path / "blocked") in output.out and "--live" in output.err
