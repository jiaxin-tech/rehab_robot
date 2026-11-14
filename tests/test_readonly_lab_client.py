"""Fake-only checks for the bounded outbound read-only lab client."""
import csv
import ctypes as ct
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from scripts import readonly_lab_client as client

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "config/twincat_leg_capture.json").read_text(encoding="utf-8"))


def image(toggle=0, *, status=0, error=0, torque=2.):
    payload = bytearray(1366)
    for axis, offset in enumerate((0, 23, 46, 69), 1):
        struct.pack_into("<HiihiHbf", payload, offset, status, -100*axis, -20, -12, 9, error, 8, torque)
    payload[1365] = toggle << 4
    return bytes(payload)


class Clock:
    def __init__(self): self.ns = 1_000_000_000
    def perf_counter_ns(self): return self.ns
    def sleep(self, seconds): self.ns += int(seconds*1e9)


class Source:
    local_ams_net_id = "10.20.4.89.1.1"
    ads_state = 5
    device_state = 0
    def __init__(self, config, clock, *, repeat=False, fail=False, fault=False):
        self.clock, self.repeat, self.fail, self.fault = clock, repeat, fail, fault
        self.reads, self.closed = 0, False
    def read(self):
        self.reads += 1
        self.clock.ns += 100_000
        if self.fail: raise OSError("synthetic ADS error")
        return image(0 if self.repeat else (self.reads // 2) % 2, status=8 if self.fault else 0)
    def close(self): self.closed = True


def test_decode_layout_signed_fields_and_all_axis_guards():
    c = config()
    row = client.decode_image(image(1), c)
    assert ct.sizeof(client.AmsNetId) == 6 and ct.sizeof(client.AmsAddr) == 8
    assert row["input_toggle"] == 1 and row["axis4_position_raw"] == -400
    assert row["axis1_torque_6077_raw_bits"] == 65524
    assert row["axis1_torque_6077_pct_rated"] is None
    assert row["axis4_torque_3c1a_nm"] == 2.
    client.guard_frame(row, c)
    with pytest.raises(ValueError): client.decode_image(image()[:-1], c)
    with pytest.raises(ValueError): client.decode_image(image(torque=float("nan")), c)


@pytest.mark.parametrize("kwargs", [dict(status=8), dict(status=4), dict(error=1), dict(torque=100.1)])
def test_stationary_guards_refuse_fault_enabled_error_or_torque(kwargs):
    with pytest.raises(RuntimeError): client.guard_frame(client.decode_image(image(**kwargs), config()), config())


def test_probe_has_exactly_twenty_reads_and_rejects_static_toggle(tmp_path):
    clock = Clock()
    source = Source(config(), clock)
    result = client.execute_task(dict(task_id="probe1", action="probe"), config(), tmp_path / "probe",
                                 source_factory=lambda c: source, clock=clock)
    assert result["ok"] and result["result"]["reads"] == 20
    assert result["result"]["toggle_changes"] > 0 and result["result"]["span_s"] > 0
    assert source.closed
    clock = Clock()
    static = Source(config(), clock, repeat=True)
    failed = client.execute_task(dict(task_id="probe2", action="probe"), config(), tmp_path / "static",
                                 source_factory=lambda c: static, clock=clock)
    assert not failed["ok"] and "toggle" in failed["error"] and static.closed


@pytest.mark.parametrize("kind", ["read", "fault", "cancel"])
def test_failure_and_cancel_always_close_and_preserve_status(tmp_path, kind):
    clock = Clock()
    source = Source(config(), clock, fail=kind == "read", fault=kind == "fault")
    if kind == "cancel":
        def cancel(): raise KeyboardInterrupt()
        source.read = cancel
    result = client.execute_task(dict(task_id="test", action="probe"), config(), tmp_path / "task",
                                 source_factory=lambda c: source, clock=clock)
    assert not result["ok"] and source.closed
    assert json.loads((tmp_path / "task/segment_status.json").read_text(encoding="utf-8"))["status"] == "FAILED"


@pytest.mark.parametrize("kwargs, detail", [(dict(status=8), "fault=True"),
    (dict(status=4), "operation_enabled=True"), (dict(error=1), "error=0x0001"),
    (dict(torque=100.1), "torque_3c1a_nm=")])
def test_guard_failure_preserves_complete_raw_and_decoded_evidence(tmp_path, kwargs, detail):
    clock = Clock()
    source = Source(config(), clock)
    payload = image(1, **kwargs)
    def read():
        clock.ns += 100_000
        return payload
    source.read = read
    result = client.execute_task(dict(task_id="guard_evidence", action="probe"), config(), tmp_path / "task",
                                 source_factory=lambda c: source, clock=clock)
    status = json.loads((tmp_path / "task/segment_status.json").read_text(encoding="utf-8"))
    assert not result["ok"] and source.closed
    assert detail in result["error"] and "max_abs_torque_nm=100.0" in result["error"]
    assert status["completed_reads"] == 1 and status["rows"] == 0
    assert bytes.fromhex(status["last_raw_image_hex"]) == payload and len(payload) == 1366
    assert status["last_query_end_ns"] - status["last_query_start_ns"] == 100_000
    assert status["last_observed_frame"] == client.decode_image(payload, config())


def test_capture_uses_actual_query_timestamps_and_never_duplicates_toggle(tmp_path):
    clock = Clock()
    source = Source(config(), clock)
    result = client.execute_task(dict(task_id="capture", action="capture", duration_s=10), config(), tmp_path / "task",
                                 source_factory=lambda c: source, clock=clock)
    assert result["ok"] and source.closed
    with (tmp_path / "task/leg.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) < source.reads and len(rows) == result["result"]["rows"]
    assert all(a["input_toggle"] != b["input_toggle"] for a, b in zip(rows, rows[1:]))
    assert all(int(r["query_end_ns"])-int(r["query_start_ns"]) == 100_000 for r in rows)
    assert all(int(a["host_monotonic_ns"]) < int(b["host_monotonic_ns"]) for a, b in zip(rows, rows[1:]))
    assert result["result"]["span_s"] >= 10
    assert len(client.output_files(tmp_path / "task")) == 2


@pytest.mark.parametrize("task", [dict(task_id="x", action="shell"), dict(task_id="../x", action="probe"),
    dict(task_id=None, action="probe"),
    dict(task_id="x", action="capture", duration_s=9), dict(task_id="x", action="capture", duration_s=True),
    dict(task_id="x", action="probe", command="whoami"), dict(task_id="x", action="capture", duration_s=3600)])
def test_task_action_duration_and_fields_are_fixed(task):
    with pytest.raises(ValueError): client.validate_task(task)


def test_broker_task_timestamps_are_metadata_not_extra_actions():
    task = dict(task_id="a"*32, action="probe", created_at="2026-01-01T00:00:00+00:00", expires_at="2099-01-01T00:00:00+00:00")
    assert client.validate_task(task) is task
    task["expires_at"] = "2000-01-01T00:00:00+00:00"
    with pytest.raises(ValueError): client.validate_task(task)


def test_config_cannot_expand_mapping_or_stationary_limits():
    for key, value in (("ams_net_id", "1.2.3.4.5.6"), ("image_bytes", 100), ("no_new_sample_timeout_s", 1.)):
        c = config(); c[key] = value
        with pytest.raises(ValueError): client.validate_config(c)
    c = config(); c["guards"]["axis1"]["max_abs_torque_nm"] = 101
    with pytest.raises(ValueError): client.validate_config(c)


def test_native_local_ams_mismatch_closes_port_without_reading(monkeypatch):
    calls = []
    class Function:
        def __init__(self, callback): self.callback = callback
        def __call__(self, *args): return self.callback(*args)
    def local(port, address):
        value = ct.cast(address, ct.POINTER(client.AmsAddr)).contents
        value.net_id.bytes[:] = (1, 2, 3, 4, 5, 6)
        return 0
    dll = SimpleNamespace(AdsPortOpenEx=Function(lambda: 1), AdsPortCloseEx=Function(lambda p: calls.append("close") or 0),
        AdsSyncSetTimeoutEx=Function(lambda *a: 0), AdsGetLocalAddressEx=Function(local),
        AdsSyncReadStateReqEx=Function(lambda *a: pytest.fail("must reject AMS before state read")),
        AdsSyncReadReqEx2=Function(lambda *a: pytest.fail("must not read a guessed address")))
    monkeypatch.setattr(client.ct, "WinDLL", lambda path: dll, raising=False)
    monkeypatch.setattr(client.Path, "resolve", lambda self, **kwargs: self)
    monkeypatch.setattr(client.os, "name", "nt")
    with pytest.raises(ValueError, match="actual local AMS"):
        client.AdsReadOnly(config())
    assert calls == ["close"]


def test_upload_is_allowlist_and_size_bounded(tmp_path, monkeypatch):
    (tmp_path / "segment_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("must not upload", encoding="utf-8")
    assert [x["name"] for x in client.output_files(tmp_path)] == ["segment_status.json"]
    monkeypatch.setattr(client, "MAX_UPLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="10 MB"): client.output_files(tmp_path)


def test_broker_client_end_to_end_with_fake_ads_and_fake_time(tmp_path):
    """Exercise real queue/upload schemas without opening any network or ADS."""
    from scripts.readonly_lab_broker import Session, prepare, read_json
    broker_dir = tmp_path / "broker"
    prepare(broker_dir)
    session = Session(broker_dir)
    session.hello(dict(hostname="SYNTHETIC_PROTOCOL_TEST", python="3.12", python_bits=64,
                       dll_available=False, configured_ams_net_id=config()["ams_net_id"],
                       config_path="SYNTHETIC_TEST_CONFIG", motion_available=False))
    assert session.status()["host"]["hostname"] == "SYNTHETIC_PROTOCOL_TEST"
    for action, duration in (("probe", None), ("capture", 10)):
        queued = session.queue(action, duration)
        delivered = session.deliver()
        assert delivered == queued
        assert session.deliver() is None  # No duplicate delivery/acquisition.
        client.validate_task(delivered)  # Includes broker timestamps.
        clock = Clock()
        source = Source(config(), clock)
        local = tmp_path / "client" / delivered["task_id"]
        result = client.execute_task(delivered, config(), local,
                                     source_factory=lambda c: source, clock=clock)
        result["files"] = client.output_files(local)
        assert result["ok"] and source.closed
        received = session.receive(delivered["task_id"], result)
        saved = broker_dir / "tasks" / delivered["task_id"]
        assert received["ok"] and read_json(saved / "segment_status.json")["status"] == "COMPLETE"
        assert (saved / "segment_status.json").read_bytes() == (local / "segment_status.json").read_bytes()
        if action == "probe":
            assert received["result"]["reads"] == 20 and received["result"]["toggle_changes"] > 0
            assert set(received["files"]) == {"segment_status.json"}
        else:
            assert received["result"]["span_s"] >= 10
            assert set(received["files"]) == {"segment_status.json", "leg.csv"}
            assert (saved / "leg.csv").read_bytes() == (local / "leg.csv").read_bytes()
    status = session.status()
    assert len(status["tasks"]) == 2 and not status["failed"]
    assert all(task["result"]["ok"] for task in status["tasks"])
