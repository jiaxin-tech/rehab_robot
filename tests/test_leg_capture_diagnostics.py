"""Exercise production ADS capture timing with synthetic images and host clocks."""
import csv
import hashlib
import json
import struct
import threading

import pytest

from collection import twincat_leg as collector
from config.bundle import configuration


class Clock:
    def __init__(self, *, pause_after_first_read_ns=0):
        self.ns = 10_000_000_000
        self.sleeps = 0
        self.pause_ns = pause_after_first_read_ns

    def perf_counter_ns(self):
        return self.ns

    def perf_counter(self):
        return self.ns / 1e9

    def sleep(self, seconds):
        self.sleeps += 1
        self.ns += round(seconds * 1e9)
        if self.sleeps == 1:
            self.ns += self.pause_ns


def image_for(config, toggle, sequence):
    image = bytearray(config["image_bytes"])
    for axis, offset in enumerate(config["axis_offsets"], 1):
        struct.pack_into("<HiihiHbf", image, offset,
                         0x0231, axis * 100, 0, 0, 0, 0, 8, float(axis))
    image[config["toggle_byte"]] = toggle << config["toggle_bit"]
    # An independent synthetic counter outside the four decoded records.
    # It lets a healthy simulated source change while a sampled toggle aliases.
    struct.pack_into("<I", image, 100, sequence)
    return bytes(image)


class Source:
    def __init__(self, config, clock, stop, *, latency_ns=1_500_000,
                 bit=None, stop_after=None, fail_at=None, close_error=None):
        self.config, self.clock, self.stop = config, clock, stop
        self.latency_ns, self.bit, self.stop_after = latency_ns, bit, stop_after
        self.fail_at, self.close_error = fail_at, close_error
        self.reads = 0
        self.closed = False
        self.primary = OSError("synthetic native ADS read failed: code=1811")

    def read(self):
        self.reads += 1
        self.clock.ns += self.latency_ns
        if self.reads == self.fail_at:
            raise self.primary
        toggle = (self.reads - 1) % 2 if self.bit is None else self.bit(self)
        if self.stop_after is not None and self.reads >= self.stop_after:
            self.stop.set()
        if self.reads > 1000:
            pytest.fail("synthetic capture exceeded its bounded read budget")
        return image_for(self.config, toggle, self.reads)

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def report_for(path):
    return json.loads(path.with_name(path.stem + "_capture_diagnostics.json").read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == collector.FIELDS
        return list(reader)


def verify_raw_records(records, config):
    for record in records:
        image = bytes.fromhex(record["raw_image_hex"])
        assert len(image) == config["image_bytes"]
        assert hashlib.sha256(image).hexdigest() == record["image_sha256"]
        assert collector.decode_image(image, config)["input_toggle"] == record["input_toggle"]
        assert record["query_end_ns"] >= record["query_start_ns"]


def test_healthy_two_ms_toggle_aliases_at_four_ms_and_still_times_out(tmp_path, monkeypatch):
    config, clock, stop = configuration()["leg"], Clock(), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    source = Source(config, clock, stop, latency_ns=3_500_000,
                    bit=lambda reader: (reader.clock.ns // 2_000_000) % 2)
    path, messages = tmp_path / "leg.csv", []
    with pytest.raises(TimeoutError, match="input toggle stopped changing"):
        collector.capture(config, path, stop_event=stop, notify=messages.append,
                          source_factory=lambda _: source)
    report = report_for(path)
    assert source.closed and report["completed"] is False
    assert report["accepted_samples"] == len(csv_rows(path)) == 1
    assert report["successful_reads"] == len(report["recent_reads"]) >= 26
    assert report["same_toggle_image_changes"] == report["successful_reads"] - 1
    assert report["maximum_sample_age_observed_before_accept_ms"] == pytest.approx(104.)
    assert 103. < (report["finished_host_ns"] - report["last_accepted_host_ns"]) / 1e6 < 109.
    assert len({row["input_toggle"] for row in report["recent_reads"]}) == 1
    assert len({row["image_sha256"] for row in report["recent_reads"]}) == report["successful_reads"]
    assert all(not row["accepted"] for row in report["recent_reads"][1:])
    assert not any(message["state"] == "completed" for message in messages)
    assert report["thresholds"]["no_new_sample_timeout_s"] == .1
    assert report["training_authorized"] is False
    verify_raw_records(report["recent_reads"], config)


@pytest.mark.parametrize("next_bit_changes", [False, True])
def test_host_pause_remains_visible_even_when_next_toggle_resets_age(tmp_path, monkeypatch, next_bit_changes):
    config, clock, stop = configuration()["leg"], Clock(pause_after_first_read_ns=150_000_000), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    source = Source(config, clock, stop, stop_after=2,
                    bit=None if next_bit_changes else lambda _: 0)
    path = tmp_path / "leg.csv"
    invoke = lambda: collector.capture(config, path, stop_event=stop, notify=lambda record: None,
                                        source_factory=lambda _: source)
    if next_bit_changes:
        assert invoke() == 2  # Existing acceptance semantics are intentionally unchanged.
    else:
        with pytest.raises(TimeoutError, match="input toggle stopped changing"):
            invoke()
    report = report_for(path)
    assert report["completed"] is next_bit_changes
    assert report["maximum_between_reads_ms"] == pytest.approx(150.5)
    assert report["maximum_sample_age_observed_before_accept_ms"] == pytest.approx(152.)
    second = report["recent_reads"][1]
    assert second["sample_age_before_accept_ns"] == 152_000_000
    assert second["accepted"] is next_bit_changes
    assert second["between_reads_ns"] == 150_500_000


def test_blocked_recording_notify_is_timed_without_relaxing_capture(tmp_path, monkeypatch):
    config, clock, stop = configuration()["leg"], Clock(), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    source = Source(config, clock, stop, stop_after=3)
    path, messages = tmp_path / "leg.csv", []
    def notify(record):
        messages.append(record)
        if record["state"] == "recording" and record["rows"] == 2:
            clock.ns += 125_000_000
    assert collector.capture(config, path, stop_event=stop, notify=notify,
                             source_factory=lambda _: source) == 3
    report = report_for(path)
    assert report["maximum_notify_duration_ms"] == pytest.approx(125.)
    assert report["maximum_between_reads_ms"] == pytest.approx(125.5)
    assert report["maximum_sample_age_observed_before_accept_ms"] >= 127.
    assert report["completed"] is True and messages[-1]["state"] == "completed"


def test_recent_raw_read_history_is_bounded_to_256_and_hashes_verify(tmp_path, monkeypatch):
    config, clock, stop = configuration()["leg"], Clock(), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    source = Source(config, clock, stop, stop_after=600)
    path = tmp_path / "leg.csv"
    assert collector.capture(config, path, stop_event=stop, notify=lambda record: None,
                             source_factory=lambda _: source) == 600
    report = report_for(path)
    assert report["successful_reads"] == report["accepted_samples"] == 600
    assert report["recent_read_capacity"] == len(report["recent_reads"]) == 256
    assert [row["read_index"] for row in report["recent_reads"]] == list(range(345, 601))
    assert report["recent_reads"][-1]["query_end_ns"] == report["last_accepted_host_ns"]
    assert all(row["accepted"] for row in report["recent_reads"])
    verify_raw_records(report["recent_reads"], config)


def test_native_read_failure_preserves_primary_when_close_also_fails(tmp_path, monkeypatch):
    config, clock, stop = configuration()["leg"], Clock(), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    source = Source(config, clock, stop, fail_at=3,
                    close_error=RuntimeError("synthetic ADS port close failed"))
    path, messages = tmp_path / "leg.csv", []
    with pytest.raises(OSError, match="native ADS read failed") as caught:
        collector.capture(config, path, stop_event=stop, notify=messages.append,
                          source_factory=lambda _: source)
    report = report_for(path)
    assert caught.value is source.primary and source.closed
    assert report["completed"] is False
    assert report["successful_reads"] == report["accepted_samples"] == 2
    assert report["error"] == "OSError: synthetic native ADS read failed: code=1811"
    assert report["cleanup_error"] == "RuntimeError: synthetic ADS port close failed"
    assert report["last_read_error"]["error"] == report["error"]
    assert report["last_read_error"]["query_end_ns"] >= report["last_read_error"]["query_start_ns"]
    assert not any(record["state"] == "completed" for record in messages)
    assert len(csv_rows(path)) == 2


@pytest.mark.parametrize("existing", ["leg.csv", "leg_capture_diagnostics.json"])
def test_existing_csv_or_diagnostics_are_never_overwritten(tmp_path, existing):
    path = tmp_path / existing
    path.write_bytes(b"existing laboratory evidence\n")
    calls = []
    def forbidden(config):
        calls.append(config)
        pytest.fail("existing evidence was detected after opening ADS")
    with pytest.raises(FileExistsError):
        collector.capture(configuration()["leg"], tmp_path / "leg.csv", stop_event=threading.Event(),
                          notify=lambda record: None, source_factory=forbidden)
    assert path.read_bytes() == b"existing laboratory evidence\n" and calls == []
    assert not (tmp_path / "leg_capture_diagnostics.json.tmp").exists()


def test_success_records_completed_diagnostics_and_production_status_schema(tmp_path, monkeypatch):
    config, clock, stop = configuration()["leg"], Clock(), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    source = Source(config, clock, stop, stop_after=8)
    path, messages = tmp_path / "leg.csv", []
    assert collector.capture(config, path, stop_event=stop, notify=messages.append,
                             source_factory=lambda _: source) == 8
    report = report_for(path)
    rows = csv_rows(path)
    assert source.closed and report["completed"] is True
    assert report["error"] is report["cleanup_error"] is report["last_read_error"] is None
    assert report["successful_reads"] == report["accepted_samples"] == len(rows) == 8
    assert messages[-1] == {"state": "completed", "rows": 8,
                            "last_sample_ns": int(rows[-1]["host_monotonic_ns"])}
    for message in messages[:-1]:
        assert message["state"] == "recording"
        for axis in (1, 4):
            latest = message["latest"]
            assert latest[f"axis{axis}_position_raw"] == 100 * axis
            assert latest[f"axis{axis}_torque_3c1a_nm"] == float(axis)
            assert latest[f"axis{axis}_fault"] is False
            assert latest[f"axis{axis}_operation_enabled"] is False
            assert latest[f"axis{axis}_error_code_hex"] == "0x0000"
    assert report["motion_commanded"] is False and report["training_authorized"] is False


def test_failed_query_is_included_in_maximum_duration_and_host_gap():
    from collection.leg_capture_diagnostics import CaptureDiagnostics
    config = configuration()["leg"]
    diagnostics = CaptureDiagnostics(config, 0)
    diagnostics.read(0, 1_000_000, bytes(config["image_bytes"]), 0)
    diagnostics.accept(1_000_000)
    diagnostics.read_error(150_000_000, 400_000_000, OSError("synthetic timeout"))
    report = diagnostics.report(400_000_000, completed=False, error="synthetic timeout")
    assert report["maximum_query_duration_ms"] == 250.
    assert report["maximum_between_reads_ms"] == 149.
    assert report["successful_reads"] == 1
    assert report["last_read_error"]["between_reads_ns"] == 149_000_000


def test_production_worker_keeps_all_csv_samples_while_stdout_notify_is_blocked(tmp_path, monkeypatch):
    config, clock, stop = configuration()["leg"], Clock(), threading.Event()
    monkeypatch.setattr(collector, "time", clock)
    entered, release, sampled = threading.Event(), threading.Event(), threading.Event()
    messages, results, errors = [], [], []
    class CoordinatedSource(Source):
        def read(self):
            if self.reads == 2:
                assert entered.wait(1.), "sender never entered the blocked transport"
            return super().read()
        def close(self):
            super().close()
            sampled.set()
    source = CoordinatedSource(config, clock, stop, stop_after=600)
    path = tmp_path / "leg.csv"
    def notify(record):
        if not messages:
            entered.set()
            assert release.wait(2.)
        messages.append(record)
    def run():
        try:
            results.append(collector.capture_worker(config, path, stop_event=stop, notify=notify,
                                                    source_factory=lambda _: source))
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert sampled.wait(1.), "blocked progress prevented ADS sampling/CSV closure"
        assert len(csv_rows(path)) == source.reads == 600
        assert worker.is_alive()  # Final delivery is still waiting for its acknowledgement.
    finally:
        release.set()
        worker.join(2.)
    assert not errors and not worker.is_alive() and results == [600]
    assert messages[-1]["state"] == "completed" and messages[-1]["rows"] == 600
    delivery = json.loads((tmp_path / "progress_delivery_diagnostics.json").read_text(encoding="utf-8"))
    assert delivery["terminal_sent"] is True and delivery["sender_alive"] is False
    assert delivery["samples_coalesced"] is False and delivery["superseded_progress"] > 0
    assert report_for(path)["accepted_samples"] == 600
