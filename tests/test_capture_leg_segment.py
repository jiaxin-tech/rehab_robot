"""Bounded segment lifecycle, with test-only hardware substitute."""
import pytest

from collection.twincat_leg import atomic_json
from safety.robot_trial_release import read_json
from scripts.capture_leg_segment import record
from tests.test_robot_leg_round import measured_fixture


@pytest.mark.parametrize("failure", [None, "start", "health", "close"])
def test_segment_closes_and_retains_failure_report(tmp_path, failure):
    config = tmp_path / "config.json"
    atomic_json(config, {})
    calls = []
    class Worker:
        def __init__(self, config, directory): self.directory = directory
        def start(self):
            calls.append("start")
            if failure == "start": raise RuntimeError("startup failed")
            measured_fixture(self.directory)
        def health_reason(self): return "stale" if failure == "health" else None
        def stop(self):
            calls.append("stop")
            if failure == "close": raise RuntimeError("unclosed CSV")
    directory = tmp_path / "segment"
    if failure:
        with pytest.raises(RuntimeError): record(config, directory, .01, factory=Worker)
        assert read_json(directory / "segment_status.json")["status"] == "CAPTURE_FAILED"
        assert not (directory / "quality.json").exists()
    else:
        result = record(config, directory, .01, factory=Worker)
        assert result["status"] == "CAPTURE_COMPLETE" and not result["motion_commanded"]
        assert (directory / "quality.json").exists()
        with pytest.raises(FileExistsError): record(config, directory, .01, factory=Worker)
    assert calls[0] == "start" and "stop" in calls


@pytest.mark.parametrize("duration", [0, -1, float('nan'), 3601])
def test_bad_duration_never_creates_source(tmp_path, duration):
    def forbidden(*a): pytest.fail("must validate before constructing source")
    with pytest.raises(ValueError): record(tmp_path / "config", tmp_path / "segment", duration, factory=forbidden)
