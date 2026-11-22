"""Standalone parent orchestration with fake capture processes; no native ADS."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from collection.twincat_leg import FIELDS
from config.bundle import BUNDLE_PATH, ROOT
from scripts import diagnose_leg_capture as diagnostic


class FakeClock:
    def __init__(self):
        self.now = 5.0
        self.sleeps = []

    def perf_counter(self):
        return self.now

    def sleep(self, duration):
        assert 0 < duration <= .010
        self.sleeps.append(duration)
        self.now += duration


class FakeCapture:
    def __init__(self, config_path, directory, clock, *, start_error=None,
                 health_error=None, cleanup_error=None, telemetry=True,
                 final_state="completed", declared_rows=2, diagnostics_overrides=None,
                 progress_telemetry=None):
        self.config_path, self.directory, self.clock = Path(config_path), Path(directory), clock
        self.start_error, self.health_error, self.cleanup_error = start_error, health_error, cleanup_error
        self.telemetry, self.final_state, self.declared_rows = telemetry, final_state, declared_rows
        self.diagnostics_overrides = diagnostics_overrides or {}
        self.progress_telemetry = progress_telemetry
        self.start_calls, self.stop_calls, self.health_calls = 0, 0, 0
        self.status = {}

    def _row(self, index, stamp):
        row = dict.fromkeys(FIELDS, "")
        row.update(sample_index=index, host_monotonic_ns=round(stamp * 1e9),
                   query_start_ns=round((stamp - .001) * 1e9), query_end_ns=round(stamp * 1e9),
                   host_timestamp_utc=f"synthetic-{index}")
        return row

    def start(self):
        self.start_calls += 1
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = self.clock.now
        self.status = dict(state="recording", rows=1)
        with (self.directory / "leg.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(self._row(0, self.started))
        if self.start_error:
            raise self.start_error

    def health_reason(self):
        self.health_calls += 1
        if self.health_error and self.clock.now - self.started >= .025:
            self.status = dict(state="failed", error=self.health_error)
            return self.health_error
        return None

    def stop(self):
        self.stop_calls += 1
        with (self.directory / "leg.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(self._row(1, max(self.clock.now, self.started + .001)))
        if self.telemetry:
            telemetry = dict(
                completed=not self.health_error and not self.start_error and not self.cleanup_error,
                successful_reads=12, accepted_samples=2, same_toggle_image_changes=3,
                maximum_query_duration_ms=1.2, error=self.health_error,
                cleanup_error=str(self.cleanup_error) if self.cleanup_error else None)
            telemetry.update(self.diagnostics_overrides)
            (self.directory / "leg_capture_diagnostics.json").write_text(json.dumps(telemetry), encoding="utf-8")
        if self.progress_telemetry is not None:
            (self.directory / "progress_delivery_diagnostics.json").write_text(
                json.dumps(self.progress_telemetry), encoding="utf-8")
        if self.cleanup_error:
            raise self.cleanup_error
        if self.status["state"] != "failed" and not self.start_error:
            self.status = dict(state=self.final_state, rows=self.declared_rows)

    def snapshot(self):
        return dict(self.status)


@pytest.fixture
def environment(monkeypatch):
    clock, captures = FakeClock(), []
    monkeypatch.setattr(diagnostic, "git_metadata", lambda: dict(head="fake-git", tracked_worktree_dirty=False))

    def factory(**options):
        def create(config_path, directory):
            capture = FakeCapture(config_path, directory, clock, **options)
            captures.append(capture)
            return capture
        return create

    return clock, captures, factory


def execute(tmp_path, environment, **options):
    clock, _, factory = environment
    return diagnostic.run(BUNDLE_PATH, tmp_path / "new-session", duration_s=.05,
                          live=True, capture_factory=factory(**options), clock=clock)


def test_success_freezes_config_records_real_span_and_consumes_child_diagnostics(tmp_path, environment):
    directory, manifest = execute(tmp_path, environment)
    clock, captures, _ = environment
    assert manifest["ok"] and manifest["status"] == "CAPTURE_COMPLETE"
    assert manifest["csv_complete_acknowledged"]
    assert manifest["monitor_duration_s"] == pytest.approx(.05)
    assert manifest["csv_summary"]["recorded_span_s"] == pytest.approx(.05)
    assert manifest["csv_summary"]["rows"] == 2
    assert max(clock.sleeps) <= .010
    assert captures[0].stop_calls == 1
    assert captures[0].config_path == directory / "config_snapshot.json"
    assert captures[0].config_path.read_bytes() == BUNDLE_PATH.read_bytes()
    assert manifest["config_sha256"] == hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
    assert manifest["git"]["head"] == "fake-git"
    assert manifest["runtime"]["python_version"] == sys.version
    assert manifest["runtime"]["perf_counter"]["monotonic"] is True
    assert set(manifest["source_sha256"]) == set(diagnostic.SOURCE_FILES)
    assert manifest["child_diagnostics"]["same_toggle_image_changes"] == 3
    assert manifest["output_files_sha256"]["leg/leg_capture_diagnostics.json"] == diagnostic.sha256_file(directory / "leg/leg_capture_diagnostics.json")
    assert not manifest["robot_connected"] and not manifest["motion_commanded"]
    assert not manifest["training_authorized"] and not manifest["clinical_use_authorized"]
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["status"] == "CAPTURE_COMPLETE"


def test_failure_preserves_primary_cleanup_and_partial_evidence(tmp_path, environment):
    directory, manifest = execute(tmp_path, environment,
        health_error="leg_capture_failed:TimeoutError: input toggle stopped changing",
        cleanup_error=RuntimeError("CSV not acknowledged; partial CSV excluded"))
    assert not manifest["ok"] and manifest["status"] == "FAILED"
    assert "input toggle stopped changing" in manifest["error"]
    assert "partial CSV excluded" in manifest["cleanup_error"]
    assert not manifest["csv_complete_acknowledged"]
    assert manifest["monitor_duration_s"] == pytest.approx(.03)
    assert manifest["csv_summary"]["rows"] == 2
    assert not manifest["child_diagnostics"]["completed"]
    assert (directory / "leg/leg.csv").is_file()
    assert "关闭错误" in (directory / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("error", [RuntimeError("ADS startup failed"), KeyboardInterrupt()])
def test_start_failure_is_cleaned_and_never_claims_completed_csv(tmp_path, environment, error):
    _, manifest = execute(tmp_path, environment, start_error=error)
    assert not manifest["ok"]
    assert manifest["status"] == ("CANCELLED" if isinstance(error, KeyboardInterrupt) else "FAILED")
    assert manifest["monitor_duration_s"] is None
    assert environment[1][0].stop_calls == 1
    assert not manifest["csv_complete_acknowledged"]


def test_cleanup_failure_after_full_duration_still_fails(tmp_path, environment):
    _, manifest = execute(tmp_path, environment, cleanup_error=TimeoutError("worker did not close CSV"))
    assert not manifest["ok"] and manifest["status"] == "FAILED"
    assert manifest["error"] is None
    assert "worker did not close CSV" in manifest["cleanup_error"]
    assert manifest["monitor_duration_s"] == pytest.approx(.05)


@pytest.mark.parametrize("options,reason", [({"final_state": "recording"}, "did not acknowledge"),
                                          ({"declared_rows": 3}, "row count differs")])
def test_success_requires_closed_acknowledged_csv_with_matching_rows(tmp_path, environment, options, reason):
    _, manifest = execute(tmp_path, environment, **options)
    assert not manifest["ok"]
    assert reason in manifest["error"]


def test_missing_child_diagnostics_is_reported_without_inventing_timing(tmp_path, environment):
    _, manifest = execute(tmp_path, environment, telemetry=False)
    assert manifest["ok"]
    assert manifest["child_diagnostics_available"] is False
    assert "child_diagnostics" not in manifest


@pytest.mark.parametrize("updates", [{"completed": False}, {"completed": None},
                                     {"completed": "true"}, {"error": "read failed"},
                                     {"cleanup_error": "ADS close failed"}])
def test_child_diagnostic_failure_cannot_be_overridden_by_completed_ack(tmp_path, environment, updates):
    _, manifest = execute(tmp_path, environment, diagnostics_overrides=updates)
    assert manifest["csv_complete_acknowledged"] is True
    assert manifest["child_final_status"]["state"] == "completed"
    assert manifest["status"] == "FAILED" and not manifest["ok"]
    assert "child diagnostics do not confirm" in manifest["error"]
    assert manifest["child_diagnostics"] | updates == manifest["child_diagnostics"]


def test_progress_delivery_diagnostics_are_consumed_after_stop_and_hashed(tmp_path, environment):
    telemetry = {"submitted": 8, "delivered": 6, "coalesced": 2, "writer_alive": False}
    directory, manifest = execute(tmp_path, environment, progress_telemetry=telemetry)
    assert manifest["ok"]
    assert manifest["progress_delivery_diagnostics_available"]
    assert manifest["progress_delivery_diagnostics"] == telemetry
    assert manifest["output_files_sha256"]["leg/progress_delivery_diagnostics.json"] == diagnostic.sha256_file(
        directory / "leg/progress_delivery_diagnostics.json")


@pytest.mark.parametrize("duration", [0, -1, 30.01, float("nan"), float("inf"), True, None])
def test_invalid_duration_cannot_create_session_or_start_capture(tmp_path, environment, duration):
    clock, captures, factory = environment
    with pytest.raises(ValueError, match="at most 30"):
        diagnostic.run(BUNDLE_PATH, tmp_path / "new", duration_s=duration, live=True,
                       capture_factory=factory(), clock=clock)
    assert not captures and not (tmp_path / "new").exists()


def test_live_flag_is_required_before_any_io(tmp_path, environment):
    with pytest.raises(PermissionError, match="--live"):
        diagnostic.run(tmp_path / "missing.json", tmp_path / "new", live=False)
    assert not (tmp_path / "new").exists()


def test_existing_session_is_preserved_and_not_started(tmp_path, environment):
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "keep.txt").write_text("original", encoding="utf-8")
    clock, captures, factory = environment
    with pytest.raises(FileExistsError):
        diagnostic.run(BUNDLE_PATH, directory, live=True, capture_factory=factory(), clock=clock)
    assert not captures
    assert list(directory.iterdir()) == [directory / "keep.txt"]
    assert (directory / "keep.txt").read_text(encoding="utf-8") == "original"


def test_cli_has_single_step_no_probe_and_uses_production_capture_by_default(tmp_path, environment, monkeypatch):
    clock, captures, factory = environment
    monkeypatch.setattr(diagnostic, "LegCaptureProcess", factory())
    real_run = diagnostic.run
    monkeypatch.setattr(diagnostic, "run", lambda *a, **kw: real_run(*a, **kw, clock=clock))
    assert diagnostic.main(["--config", str(BUNDLE_PATH), "--session", str(tmp_path / "cli"),
                            "--duration-s", ".05", "--live"]) == 0
    assert len(captures) == 1


def test_clean_import_loads_no_robot_or_analysis_modules():
    script = """
import builtins
original = builtins.__import__
forbidden = ('hardware', 'control', 'scipy', 'numpy', 'pandas', 'measurement_validation', 'scripts.lab_session')
def guard(name, *args, **kwargs):
    if any(name == item or name.startswith(item + '.') for item in forbidden):
        raise AssertionError('forbidden robot/analysis import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from scripts import diagnose_leg_capture
assert diagnose_leg_capture.HEALTH_POLL_S <= .01
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr
