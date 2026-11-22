"""Exercise the real collector publication and passive guard with fake ADS images."""
from __future__ import annotations

import csv
import struct
import threading
from types import SimpleNamespace

import pytest

from collection import twincat_leg as collector
from config.bundle import configuration
from scripts import lab_session as lab


class FakeClock:
    def __init__(self):
        self.ns = 10_000_000_000

    def perf_counter_ns(self):
        return self.ns

    def perf_counter(self):
        return self.ns / 1e9

    def sleep(self, seconds):
        self.ns += round(seconds * 1e9)


def image_for(c, *, toggle, angle_deg=30.0, changed_axis=None, change=None):
    image = bytearray(c["leg"]["image_bytes"])
    for axis, offset in enumerate(c["leg"]["axis_offsets"], 1):
        name = {1: "hip", 4: "knee"}.get(axis)
        position = 123 * axis
        if name:
            cal = c["calibration"][name]
            position = round(cal["zero_counts"] + angle_deg * cal["counts_per_degree"] / cal["flexion_sign"])
        status, error, torque = 0x0040, 0, float(axis)
        if axis == changed_axis:
            if change == "fault":
                status |= 8
            elif change == "error":
                error = 0x0022
            elif change == "enabled":
                status |= 4
            elif change == "torque":
                torque = c["leg"]["guards"][f"axis{axis}"]["max_abs_torque_nm"] + 1
        struct.pack_into("<HiihiHbf", image, offset, status, position, 0, 0, 0, error, 8, torque)
    image[c["leg"]["toggle_byte"]] = toggle << c["leg"]["toggle_bit"]
    return bytes(image)


class FakeSource:
    def __init__(self, images, clock):
        self.images = iter(images)
        self.clock = clock
        self.closed = False

    def read(self):
        self.clock.ns += 2_000_000
        return next(self.images)

    def close(self):
        self.closed = True


def process_view(c):
    """Use the production snapshot and health methods without spawning a process."""
    view = object.__new__(collector.LegCaptureProcess)
    view.config = c["leg"]
    view.status = {}
    view.process = SimpleNamespace(poll=lambda: None)
    view._lock = threading.Lock()
    view._error = None
    return view


def test_real_capture_publishes_snapshot_consumed_by_passive_guard(tmp_path, monkeypatch):
    c, clock, stop = configuration(), FakeClock(), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    monkeypatch.setattr(lab, "time", clock)
    source = FakeSource([image_for(c, toggle=0), image_for(c, toggle=1)], clock)
    view, messages, reasons = process_view(c), [], []
    origin = {"axes": {name: {"physical_angle_deg": 30.0} for name in ("hip", "knee")}}

    def notify(record):
        messages.append(record)
        view.status = record
        if record["state"] == "recording":
            reasons.append(lab._passive_leg_guard(view, c, origin, before_motion=True))
            stop.set()

    path = tmp_path / "leg.csv"
    assert collector.capture(c["leg"], path, stop_event=stop, notify=notify,
                             source_factory=lambda config: source) == 2
    assert source.closed
    assert reasons == [None]
    assert [record["state"] for record in messages] == ["recording", "completed"]
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    latest = messages[0]["latest"]
    assert messages[0]["last_sample_ns"] == int(rows[-1]["host_monotonic_ns"])
    for axis in (1, 4):
        assert latest[f"axis{axis}_position_raw"] == int(rows[-1][f"axis{axis}_position_raw"])
        assert latest[f"axis{axis}_torque_3c1a_nm"] == float(axis)
        assert latest[f"axis{axis}_fault"] is False
        assert latest[f"axis{axis}_operation_enabled"] is False
        assert latest[f"axis{axis}_error_code_hex"] == "0x0000"


@pytest.mark.parametrize("axis", [1, 4])
@pytest.mark.parametrize("change", ["fault", "error", "enabled", "torque"])
def test_worker_rejects_bad_image_before_publishing_it(tmp_path, monkeypatch, axis, change):
    c, clock = configuration(), FakeClock()
    monkeypatch.setattr(collector, "time", clock)
    source = FakeSource([image_for(c, toggle=0), image_for(c, toggle=1),
                         image_for(c, toggle=0, changed_axis=axis, change=change)], clock)
    messages = []
    path = tmp_path / "leg.csv"
    with pytest.raises(RuntimeError, match=f"axis{axis} leg state/torque guard failed"):
        collector.capture(c["leg"], path, stop_event=threading.Event(), notify=messages.append,
                          source_factory=lambda config: source)
    assert source.closed
    assert len(messages) == 1
    assert messages[0]["state"] == "recording"
    assert messages[0]["rows"] == 2
    assert messages[0]["latest"][f"axis{axis}_fault"] is False
    assert messages[0]["latest"][f"axis{axis}_error_code_hex"] == "0x0000"
    assert messages[0]["latest"][f"axis{axis}_operation_enabled"] is False
    assert messages[0]["latest"][f"axis{axis}_torque_3c1a_nm"] == float(axis)
    with path.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
