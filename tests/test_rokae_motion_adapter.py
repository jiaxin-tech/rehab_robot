"""Offline tests for the externally prepared motion boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hardware.rokae_motion import RokaeCartesianMotionAdapter


class FakeNativeMotion:
    def __init__(self):
        self.calls = []

    def attach_externally_prepared_realtime(self, *, reviewed_filter_hz):
        self.calls.append(("attach_externally_prepared_realtime", reviewed_filter_hz))

    def start_realtime_cartesian(self, pose):
        self.calls.append(("start_realtime_cartesian", tuple(pose)))

    def set_realtime_cartesian_target(self, pose):
        self.calls.append(("set_realtime_cartesian_target", tuple(pose)))

    def realtime_motion_error(self):
        self.calls.append("realtime_motion_error")
        return False

    def stop_realtime(self, *, switch_to_nrt):
        self.calls.append(("stop_realtime", switch_to_nrt))


def test_motion_adapter_requires_explicit_attach_and_has_one_stop_path():
    native = FakeNativeMotion()
    adapter = RokaeCartesianMotionAdapter(SimpleNamespace(native_robot=native))
    pose = (0.3, 0.0, 0.4, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="not attached"):
        adapter.start_cartesian_hold(pose)
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.start_cartesian_hold(pose)
    adapter.send_cartesian_target(pose)
    assert not adapter.has_motion_error()
    adapter.request_stop("offline test")
    adapter.request_stop("duplicate ignored")
    assert native.calls == [
        ("attach_externally_prepared_realtime", 25.0),
        ("start_realtime_cartesian", pose),
        ("set_realtime_cartesian_target", pose),
        "realtime_motion_error",
        ("stop_realtime", False),
    ]
    for forbidden in ("enable", "power_on", "clear_error", "move_l", "move_j"):
        assert not hasattr(adapter, forbidden)


def test_stop_failure_is_not_reported_as_success():
    class FailingStop(FakeNativeMotion):
        def stop_realtime(self, *, switch_to_nrt):
            self.calls.append(("stop_realtime", switch_to_nrt))
            raise RuntimeError("simulated stopLoop failure")

    native = FailingStop()
    adapter = RokaeCartesianMotionAdapter(SimpleNamespace(native_robot=native))
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.start_cartesian_hold((0.3, 0.0, 0.4, 0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match="stopLoop failure"):
        adapter.request_stop("fault")
    assert adapter.stop_reason == "fault"
    assert adapter.stop_confirmed is False
    assert adapter.active is True
    with pytest.raises(RuntimeError, match="stopLoop failure"):
        adapter.request_stop("retry keeps original reason")
    assert native.calls.count(("stop_realtime", False)) == 2


def test_stop_failure_can_be_retried_and_then_becomes_idempotent():
    class FailOnce(FakeNativeMotion):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def stop_realtime(self, *, switch_to_nrt):
            self.calls.append(("stop_realtime", switch_to_nrt))
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("first stop failed")

    native = FailOnce()
    adapter = RokaeCartesianMotionAdapter(SimpleNamespace(native_robot=native))
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.start_cartesian_hold((0.3, 0.0, 0.4, 0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match="first stop failed"):
        adapter.request_stop("operator")
    adapter.request_stop("retry")
    adapter.request_stop("idempotent")
    assert native.calls.count(("stop_realtime", False)) == 2
    assert adapter.stop_reason == "operator"
    assert adapter.stop_confirmed is True
    assert adapter.active is False


def test_stop_intent_permanently_blocks_later_attach_or_start():
    native = FakeNativeMotion()
    adapter = RokaeCartesianMotionAdapter(SimpleNamespace(native_robot=native))
    adapter.request_stop("prestart operator stop")
    with pytest.raises(RuntimeError, match="permanently stopped"):
        adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    assert native.calls == [("stop_realtime", False)]

    native = FakeNativeMotion()
    adapter = RokaeCartesianMotionAdapter(SimpleNamespace(native_robot=native))
    adapter.attach_externally_prepared(reviewed_filter_hz=25.0)
    adapter.request_stop("stop after attach")
    with pytest.raises(RuntimeError, match="permanently stopped"):
        adapter.start_cartesian_hold((0.3, 0.0, 0.4, 0.0, 0.0, 0.0))
    assert native.calls == [
        ("attach_externally_prepared_realtime", 25.0),
        ("stop_realtime", False),
    ]
