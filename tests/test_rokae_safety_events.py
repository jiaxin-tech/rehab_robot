"""Safety-event lifecycle regression with a fake SDK; never opens a socket."""

from types import SimpleNamespace
import threading

import pytest

from hardware.safety_events import SafetyEventMonitor, validate_collision_source
from hardware.rokae_adapter import RokaeRobotAdapter
from hardware.windows import rokae_xcore


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class Native:
    def __init__(self):
        self.callbacks = []
        self.fail_registration = False
        self.fail_disconnect = False
        self.initial_event = None
        self.calls = []

    def connectToRobot(self, *args):
        self.calls.append("connect")

    def robotInfo(self, ec):
        return SimpleNamespace(joint_num=6, type="synthetic", version="test", id="fake")

    def forceControl(self):
        return object()

    def setEventWatcher(self, event, callback, ec):
        assert event == "safety"
        self.calls.append("subscribe")
        self.callbacks.append(callback)
        if self.initial_event is not None:
            callback(self.initial_event)
        if self.fail_registration:
            ec.update(ec=259, message="synthetic registration failure")

    def disconnectFromRobot(self, ec):
        self.calls.append("disconnect")
        if self.fail_disconnect:
            ec.update(ec=42, message="synthetic disconnect failure")

    def queryEventInfo(self, *_args):
        raise AssertionError("event mode must not call the failing active query")


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    native = Native()
    robot = rokae_xcore.RokaeRobot("192.0.2.1", collision_source="events",
                                  collision_event_max_age_s=0.5)
    robot._safety_events = SafetyEventMonitor(0.5, clock=clock)
    sdk = SimpleNamespace(BaseRobot=SimpleNamespace(sdkVersion=lambda: "0.7.0"),
                          xMateRobot=lambda: native, Event=SimpleNamespace(safety="safety"))
    monkeypatch.setattr(rokae_xcore, "_load_sdk", lambda: sdk)
    monkeypatch.setattr(robot, "start_state_stream", lambda: None)
    monkeypatch.setattr(robot, "_refresh_operation_state", lambda: None)
    monkeypatch.setattr(robot, "_read_joint_soft_limits_rad", lambda: ((-2., 2.),) * 6)
    return robot, native, clock


def test_subscription_without_initial_event_is_unknown(rig):
    robot, native, _ = rig
    robot.connect()
    assert native.calls == ["connect", "subscribe"]
    assert robot.get_collision_state() is None
    status = robot.get_safety_event_status()
    assert status["registered"]
    assert not status["valid"]
    assert status["invalid_reason"] == "safety_event_initial_state_unknown"


def test_actual_boolean_false_makes_event_source_valid(rig):
    robot, native, _ = rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is False
    status = robot.get_safety_event_status()
    assert status["valid"] and status["sequence"] == 1
    assert status["received_host_time_s"] == 10.0
    adapter = RokaeRobotAdapter(native_robot=robot)
    summary = adapter.get_robot_state_summary()
    assert summary["collision_state_query_valid"]
    assert summary["collision_state_source"] == "events"
    assert summary["collision_state_timestamp_source"].startswith("host_callback")


@pytest.mark.parametrize("payload", [{}, {"collided": "false"}, {"collided": 0},
                                     {"collided": None}, None, {"not_collided": False}])
def test_bad_payload_cannot_become_clearance_and_fault_stays_latched(rig, payload):
    robot, native, _ = rig
    robot.connect()
    native.callbacks[-1](payload)
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    assert "invalid_collided_boolean" in robot.get_safety_event_status()["invalid_reason"]


def test_collision_survives_false_event_and_reconnect(rig):
    robot, native, _ = rig
    robot.connect()
    native.callbacks[-1]({"collided": True})
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is True
    robot.disconnect()
    robot.connect()
    assert robot.get_collision_state() is None
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is True


def test_expiry_and_late_event_do_not_automatically_restore_clearance(rig):
    robot, native, clock = rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    clock.now += 0.6
    # The receive path must detect the gap even without an intervening reader.
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    assert "expired" in robot.get_safety_event_status()["invalid_reason"]


def test_reconnect_needs_new_initial_event_and_ignores_old_callback(rig):
    robot, native, _ = rig
    robot.connect()
    old_callback = native.callbacks[-1]
    old_callback({"collided": False})
    robot.disconnect()
    old_callback({"collided": False})
    assert robot.get_collision_state() is None
    robot.connect()
    old_callback({"collided": False})
    assert robot.get_collision_state() is None
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is False


def test_failed_disconnect_invalidates_events_and_prevents_resubscription(rig):
    robot, native, _ = rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    native.fail_disconnect = True
    with pytest.raises(RuntimeError, match="disconnectFromRobot"):
        robot.disconnect()
    assert robot.is_connected
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    with pytest.raises(RuntimeError, match="disconnect is pending"):
        robot.start_safety_events()


def test_registration_failure_discards_even_synchronous_initial_false(rig):
    robot, native, _ = rig
    native.initial_event = {"collided": False}
    native.fail_registration = True
    with pytest.raises(RuntimeError, match="setEventWatcher"):
        robot.connect()
    assert not robot.is_connected
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None


def test_synchronous_initial_event_waits_for_registration_confirmation():
    monitor = SafetyEventMonitor(1.)
    generation = monitor.begin()
    monitor.receive(generation, {"collided": False})
    assert not monitor.snapshot().valid
    monitor.registered(generation)
    assert monitor.snapshot().valid


def test_stream_fault_does_not_recover_from_next_callback(rig):
    robot, native, _ = rig
    robot.connect()
    native.callbacks[-1]({"collided": False})
    robot._mark_state_error(RuntimeError("lost state transport"))
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is None
    assert "stream_error" in robot.get_safety_event_status()["invalid_reason"]


def test_explicit_resubscription_starts_unknown_and_rejects_old_callbacks(rig):
    robot, native, clock = rig
    robot.connect()
    old_callback = native.callbacks[-1]
    old_callback({"collided": False})
    clock.now += 1.
    assert robot.get_collision_state() is None
    robot.start_safety_events()
    old_callback({"collided": False})
    assert robot.get_collision_state() is None
    native.callbacks[-1]({"collided": False})
    assert robot.get_collision_state() is False
    robot._rt_active = True
    with pytest.raises(RuntimeError, match="during realtime motion"):
        robot.start_safety_events()


def test_callback_never_takes_sdk_or_state_lock(rig):
    robot, native, _ = rig
    robot.connect()
    finished = threading.Event()
    with robot._sdk_lock, robot._state_lock:
        thread = threading.Thread(target=lambda: (native.callbacks[-1]({"collided": False}), finished.set()))
        thread.start()
        assert finished.wait(1.0)
    thread.join(1.0)
    assert robot.get_collision_state() is False


@pytest.mark.parametrize("age", [None, 0., -1., float("inf"), float("nan"), True])
def test_event_mode_requires_explicit_finite_freshness_budget(age):
    with pytest.raises(ValueError, match="explicit finite positive"):
        validate_collision_source("events", age)


def test_query_remains_default_and_event_only_budget_is_rejected():
    robot = rokae_xcore.RokaeRobot("192.0.2.1")
    assert robot.get_safety_event_status() is None
    with pytest.raises(ValueError, match="requires collision_source=events"):
        validate_collision_source("query", .5)


def test_disconnect_during_registration_cannot_confirm_clearance(rig, monkeypatch):
    robot, native, _ = rig
    original = native.setEventWatcher
    native.initial_event = {"collided": False}

    def disconnect_pending(*args):
        original(*args)
        robot._disconnect_requested = True

    monkeypatch.setattr(native, "setEventWatcher", disconnect_pending)
    with pytest.raises(RuntimeError, match="disconnect requested during"):
        robot.connect()
    assert robot.get_collision_state() is None


def test_events_mode_is_forwarded_by_both_default_adapter_factories(monkeypatch):
    import hardware.rokae_adapter as adapter_module
    from scripts.acquire_robot_data import _configured_adapter_factory
    from scripts.run_rehab_experiment import _default_adapter_factory

    calls = []
    monkeypatch.setattr(adapter_module, "RokaeRobotAdapter",
                        lambda ip, **kwargs: calls.append((ip, kwargs)))
    _configured_adapter_factory(local_ip="192.0.2.2", robot_class=None,
                                state_interval_ms=None, collision_source="events",
                                collision_event_max_age_s=.2)("192.0.2.1")
    _default_adapter_factory("192.0.2.1", local_ip="192.0.2.2",
                             collision_source="events", collision_event_max_age_s=.2)
    for ip, kwargs in calls:
        assert ip == "192.0.2.1"
        assert kwargs["collision_source"] == "events"
        assert kwargs["collision_event_max_age_s"] == .2
    assert len(calls) == 2


def test_execute_cli_passes_events_configuration_to_runner(monkeypatch, capsys):
    import scripts.run_rehab_experiment as module

    calls = []
    monkeypatch.setattr(module, "run_execute", lambda **kwargs: calls.append(kwargs) or {})
    assert module.main(["--mode", "execute", "--ip", "192.0.2.1",
                        "--episode-dir", "unused", "--anchor", "unused.json",
                        "--anchor-id", "unused", "--collision-source", "events",
                        "--collision-event-max-age-s", "0.2"]) == 0
    assert calls[0]["collision_source"] == "events"
    assert calls[0]["collision_event_max_age_s"] == .2


def test_state_only_cli_can_subscribe_with_explicit_local_ip(monkeypatch, capsys):
    import scripts.acquire_robot_data as module
    import hardware.rokae_adapter as adapter_module

    calls = []
    monkeypatch.setattr(adapter_module, "RokaeRobotAdapter",
                        lambda ip, **kwargs: calls.append((ip, kwargs)))

    def acquire(**kwargs):
        assert kwargs["mode"] == "state-only"
        kwargs["adapter_factory"](kwargs["robot_ip"])
        return {}

    monkeypatch.setattr(module, "run_acquisition", acquire)
    assert module.main(["--ip", "192.0.2.1", "--local-ip", "192.0.2.2",
                        "--episode-dir", "unused", "--duration-s", "1",
                        "--collision-source", "events", "--collision-event-max-age-s", "0.2"]) == 0
    assert calls[0][1]["collision_source"] == "events"
    assert calls[0][1]["local_ip"] == "192.0.2.2"


def test_missing_event_budget_fails_before_sdk_or_files(monkeypatch, tmp_path):
    import scripts.acquire_robot_data as module

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid event configuration reached acquisition")

    monkeypatch.setattr(module, "run_acquisition", forbidden)
    episode = tmp_path / "episode"
    with pytest.raises(ValueError, match="explicit finite positive"):
        module.main(["--ip", "192.0.2.1", "--episode-dir", str(episode),
                     "--duration-s", "1", "--collision-source", "events"])
    assert not episode.exists()
