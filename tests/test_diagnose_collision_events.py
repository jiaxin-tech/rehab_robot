import json

import pytest

from hardware.safety_events import SafetyEventMonitor
from scripts.diagnose_collision_events import diagnose


@pytest.mark.parametrize("behavior", ["silent", "clear", "periodic", "collision", "disconnect", "registration_failure"])
def test_callback_diagnostic_preserves_unknown_faults_and_cleanup(tmp_path, behavior):
    class Clock:
        now = 10.

        def __call__(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds
            if behavior == "periodic":
                monitor.receive(robot.generation, dict(collided=False))
            if behavior == "disconnect":
                robot.is_connected = False
                monitor.invalidate("safety_event_disconnected")

    clock = Clock()
    monitor = SafetyEventMonitor(.05, clock=clock)
    calls = []

    class CallbackRobot:
        is_connected = False

        def connect(self):
            calls.append("connect")
            self.is_connected = True
            generation = monitor.begin()
            self.generation = generation
            if behavior == "registration_failure":
                monitor.invalidate("safety_event_registration_failed")
                raise RuntimeError("registration failure")
            monitor.registered(generation)
            if behavior in ("clear", "periodic"):
                monitor.receive(generation, dict(collided=False))
            if behavior == "collision":
                monitor.receive(generation, dict(collided=True))
                monitor.receive(generation, dict(collided=False))

        def get_safety_event_status(self):
            calls.append("cached_status")
            return monitor.snapshot().to_dict()

        def disconnect(self):
            calls.append("disconnect")
            self.is_connected = False
            monitor.invalidate("safety_event_disconnected")

    robot = CallbackRobot()

    def factory(ip, **kwargs):
        assert ip == "192.0.2.1"
        assert kwargs == dict(local_ip="192.0.2.2", collision_source="events", collision_event_max_age_s=.05)
        return robot

    path = tmp_path / "events.json"
    report = diagnose(robot_ip="192.0.2.1", local_ip="192.0.2.2", output=path,
                      duration_s=.1, max_age_s=.05, robot_factory=factory,
                      clock=clock, sleep=clock.sleep)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert report["cleanup"]["disconnected_confirmed"]
    assert report["cleanup"]["status_after_disconnect"]["valid"] is False
    assert report["execution_authorized"] is False
    assert report["assessment"]["final_status_valid"] is (behavior == "periodic")
    assert calls[-2:] == ["disconnect", "cached_status"]
    if behavior == "silent":
        assert report["assessment"]["observed_event_count"] == 0
        assert report["assessment"]["unresolved_reasons"] == ["safety_event_initial_state_unknown"]
    if behavior == "clear":
        assert report["assessment"]["observed_clear"]
        assert "safety_event_expired_or_clock_invalid" in report["assessment"]["unresolved_reasons"]
    if behavior == "collision":
        assert report["assessment"]["observed_collision"]
    if behavior == "registration_failure":
        assert report["errors"]
    if behavior == "periodic":
        assert report["assessment"]["observed_clear"]
        assert report["assessment"]["observed_event_count"] > 1
        assert not report["assessment"]["unresolved_reasons"]


@pytest.mark.parametrize("age,duration", [(None, 1), (0, 1), (float("nan"), 1), (.1, -1)])
def test_invalid_parameters_fail_before_connection_and_output(tmp_path, age, duration):
    path = tmp_path / "events.json"
    with pytest.raises(ValueError):
        diagnose(robot_ip="192.0.2.1", local_ip="192.0.2.2", output=path,
                 max_age_s=age, duration_s=duration,
                 robot_factory=lambda *_args, **_kwargs: pytest.fail("connection forbidden"))
    assert not path.exists()


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "events.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnose(robot_ip="192.0.2.1", local_ip="192.0.2.2", output=path,
                 max_age_s=.1, duration_s=1,
                 robot_factory=lambda *_args, **_kwargs: pytest.fail("connection forbidden"))
    assert path.read_text() == "original"
