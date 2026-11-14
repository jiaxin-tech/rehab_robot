"""Fake-only tests: never load a real ADS DLL or contact either device."""
from __future__ import annotations

import csv
import ctypes as ct
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct

import pytest

from config.bundle import BUNDLE_PATH
from scripts import manual_leg_check as diagnostic


def image_for(config, *, toggle=0, axis1_status=0, axis1_error=0, axis1_torque=0.0,
              axis4_status=0, axis4_error=0, axis4_torque=0.0):
    image = bytearray(config["image_bytes"])
    for axis, offset in enumerate(config["axis_offsets"], 1):
        status = axis1_status if axis == 1 else axis4_status if axis == 4 else 0
        error = axis1_error if axis == 1 else axis4_error if axis == 4 else 0
        torque = axis1_torque if axis == 1 else axis4_torque if axis == 4 else 0.0
        struct.pack_into("<HiihiHbf", image, offset, status, 123 * axis, 0, -3, 0,
                         error, 8, torque)
    image[config["toggle_byte"]] = toggle << config["toggle_bit"]
    return bytes(image)


class FakeClock:
    def __init__(self):
        self.ns = 5_000_000_000
        self.sleeps = []

    def perf_counter_ns(self):
        return self.ns

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.ns += round(seconds * 1e9)


class FakeReader:
    def __init__(self, config, clock, *, constant_toggle=False, images=None,
                 read_error=None, close_error=None):
        self.config, self.clock = config, clock
        self.constant_toggle, self.images = constant_toggle, images
        self.read_error, self.close_error = read_error, close_error
        self.actual_local_ams_net_id = config["ams_net_id"]
        self.ads_state = self.device_state = 0
        self.reads, self.closed = 0, False

    def read(self):
        self.clock.ns += 10_000_000
        index, self.reads = self.reads, self.reads + 1
        if self.read_error is not None:
            raise self.read_error
        if self.images is not None:
            return self.images[min(index, len(self.images) - 1)]
        return image_for(self.config, toggle=0 if self.constant_toggle else index % 2)

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    dll = tmp_path / "fake-current-ads.dll"
    dll.write_bytes(b"not a native library; test path only")
    bundle["leg"]["ads_dll_path"] = str(dll)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "git_head", lambda: "fake-only-git-head")
    config, checks = diagnostic.offline_check(path)
    assert checks["ok"], checks
    return path, config


def fake_factory(clock, readers, **options):
    def factory(config):
        reader = FakeReader(config, clock, **options)
        readers.append(reader)
        return reader
    return factory


def make_probe(tmp_path, config_path, **options):
    clock, readers = FakeClock(), []
    directory, manifest = diagnostic.run("probe", config_path=config_path,
        output_dir=tmp_path / "probe", live=True,
        source_factory=fake_factory(clock, readers, **options), clock=clock)
    return directory, manifest, readers, clock


def rewrite_probe_status(directory, **updates):
    status_path = directory / "segment_status.json"
    status = diagnostic.read_json(status_path)
    status.update(updates)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = diagnostic.read_json(manifest_path)
    manifest["output_files_sha256"]["segment_status.json"] = diagnostic.sha256_file(status_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_unified_config_uses_current_dll_and_no_old_mapping_flag():
    config = diagnostic.validate_config(diagnostic.load_section("leg"))
    assert Path(config["ads_dll_path"]) == Path(r"C:\TwinCAT\AdsApi\TcAdsDll\x64\TcAdsDll.dll")
    assert "schema_version" not in config
    assert "mapping_reviewed" not in config


@pytest.mark.parametrize("explicit_offline", [False, True])
def test_default_and_explicit_offline_check_never_open_native_or_network(
        local_config, tmp_path, monkeypatch, explicit_offline):
    path, _ = local_config

    def forbidden(*args, **kwargs):
        pytest.fail("offline check attempted native or network access")

    monkeypatch.setattr(diagnostic.ct, "WinDLL", forbidden)
    monkeypatch.setattr(diagnostic.socket, "socket", forbidden)
    arguments = ["check", "--config", str(path), "--output-dir", str(tmp_path / "check")]
    if explicit_offline:
        arguments.append("--offline")
    assert diagnostic.main(arguments) == 0
    manifest = diagnostic.read_json(tmp_path / "check/manifest.json")
    assert manifest["offline_check"]["ads_library_loaded"] is False
    assert manifest["offline_check"]["device_connected"] is False
    assert manifest["remote_helper_used"] is False
    snapshot = diagnostic.read_json(tmp_path / "check/leg_config_snapshot.json")
    assert snapshot["ads_dll_path"] == diagnostic.load_section("leg", config_path=path)["ads_dll_path"]


class FakeFunction:
    def __init__(self, name, owner):
        self.name, self.owner = name, owner

    def __call__(self, *args):
        owner = self.owner
        owner.calls.append(self.name)
        if self.name == "AdsPortOpenEx":
            return 12345
        if self.name == "AdsGetLocalAddressEx":
            address = ct.cast(args[1], ct.POINTER(diagnostic.AmsAddr)).contents
            address.net_id.bytes[:] = list(map(int, owner.ams.split(".")))
            address.port = 30000
        elif self.name == "AdsSyncReadStateReqEx":
            ct.cast(args[2], ct.POINTER(ct.c_uint16)).contents.value = 0
            ct.cast(args[3], ct.POINTER(ct.c_uint16)).contents.value = 0
        elif self.name == "AdsSyncReadReqEx2":
            ct.memmove(args[5], owner.image, min(len(owner.image), args[4]))
            ct.cast(args[6], ct.POINTER(ct.c_uint32)).contents.value = owner.actual_bytes
            return owner.read_code
        elif self.name == "AdsPortCloseEx":
            return owner.close_code
        return 0


class FakeDll:
    def __init__(self, config, *, ams=None, read_code=0, actual_bytes=None, close_code=0):
        self.ams, self.image = ams or config["ams_net_id"], image_for(config)
        self.actual_bytes = len(self.image) if actual_bytes is None else actual_bytes
        self.read_code, self.close_code = read_code, close_code
        self.calls, self.requested = [], set()
        self.functions = {}

    def __getattr__(self, name):
        self.requested.add(name)
        if name not in self.functions:
            self.functions[name] = FakeFunction(name, self)
        return self.functions[name]


def test_native_client_binds_only_six_session_read_calls(local_config, monkeypatch):
    _, config = local_config
    dll = FakeDll(config)
    loaded_paths = []
    monkeypatch.setattr(diagnostic.ct, "WinDLL", lambda path: loaded_paths.append(path) or dll)
    reader = diagnostic.LocalAdsReader(config)
    assert reader.read() == dll.image
    assert reader.address.port == config["ads_port"]
    assert reader.ads_state == reader.device_state == 0
    reader.close()
    reader.close()
    assert loaded_paths == [config["ads_dll_path"]]
    assert dll.requested == {"AdsPortOpenEx", "AdsPortCloseEx", "AdsSyncSetTimeoutEx",
        "AdsGetLocalAddressEx", "AdsSyncReadStateReqEx", "AdsSyncReadReqEx2"}
    assert dll.calls.count("AdsPortCloseEx") == 1
    assert all(function.argtypes is not None for function in dll.functions.values())


def test_native_actual_ams_mismatch_closes_before_any_read(local_config, monkeypatch):
    _, config = local_config
    dll = FakeDll(config, ams="1.2.3.4.5.6")
    monkeypatch.setattr(diagnostic.ct, "WinDLL", lambda path: dll)
    with pytest.raises(ValueError, match="actual local AMS.*differs"):
        diagnostic.LocalAdsReader(config)
    assert dll.calls[-1] == "AdsPortCloseEx"
    assert "AdsSyncReadReqEx2" not in dll.calls
    assert "AdsSyncReadStateReqEx" not in dll.calls


def test_native_constructor_preserves_cleanup_error(local_config, monkeypatch):
    _, config = local_config
    dll = FakeDll(config, ams="1.2.3.4.5.6", close_code=7)
    monkeypatch.setattr(diagnostic.ct, "WinDLL", lambda path: dll)
    with pytest.raises(ValueError) as caught:
        diagnostic.LocalAdsReader(config)
    assert any("close failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("read_code, actual_bytes", [(1797, 0), (0, 4)])
def test_failed_native_read_preserves_buffer_counts_and_timing(
        local_config, tmp_path, monkeypatch, read_code, actual_bytes):
    _, config = local_config
    dll = FakeDll(config, read_code=read_code, actual_bytes=actual_bytes)
    monkeypatch.setattr(diagnostic.ct, "WinDLL", lambda path: dll)
    status = diagnostic.acquire("probe", config, tmp_path,
        source_factory=diagnostic.LocalAdsReader, clock=FakeClock())
    evidence = status["last_failed_read"]
    assert status["status"] == "FAILED"
    assert status["attempted_reads"] == 1 and status["completed_reads"] == 0
    assert evidence["ads_return_code"] == read_code
    assert evidence["reported_bytes"] == actual_bytes
    assert evidence["raw_buffer_hex"] == dll.image.hex()
    assert evidence["returned_data_valid"] is False
    assert status["last_query_end_ns"] >= status["last_query_start_ns"]
    assert "last_observed_frame" not in status
    assert dll.calls.count("AdsSyncReadReqEx2") == 1
    assert dll.calls[-1] == "AdsPortCloseEx"


def test_first_guard_failure_has_raw_decode_and_exact_values(local_config, tmp_path):
    _, config = local_config
    raw = image_for(config, axis1_status=12, axis1_error=0x1234, axis1_torque=-101.5,
                    axis4_torque=125.0)
    readers, clock = [], FakeClock()
    status = diagnostic.acquire("probe", config, tmp_path,
        source_factory=fake_factory(clock, readers, images=[raw]), clock=clock)
    assert status["ok"] is False and status["rows"] == 0
    assert status["completed_reads"] == status["attempted_reads"] == 1
    assert readers[0].reads == 1 and readers[0].closed
    assert status["last_raw_image_hex"] == raw.hex()
    assert status["last_observed_frame"]["axis1_torque_3c1a_nm"] == -101.5
    assert status["guard_failures"] == ["axis1 fault=True", "axis1 error_code=0x1234",
        "axis1 operation_enabled=True, expected=False",
        "axis1 abs(torque_3c1a_nm)=101.5 exceeds configured 100",
        "axis4 abs(torque_3c1a_nm)=125 exceeds configured 100"]
    assert diagnostic.read_json(tmp_path / "segment_status.json") == status


def test_decode_failure_still_preserves_full_raw_image(local_config, tmp_path):
    _, config = local_config
    raw = image_for(config, axis1_torque=float("nan"))
    readers, clock = [], FakeClock()
    status = diagnostic.acquire("probe", config, tmp_path,
        source_factory=fake_factory(clock, readers, images=[raw]), clock=clock)
    assert "nonfinite torque sensor" in status["error"]
    assert status["last_raw_image_hex"] == raw.hex()
    assert "last_observed_frame" not in status
    assert status["completed_reads"] == 1 and readers[0].closed


def test_probe_is_exactly_twenty_reads_and_retains_raw_zero_service_status(local_config, tmp_path):
    path, _ = local_config
    directory, manifest, readers, clock = make_probe(tmp_path, path)
    status = diagnostic.read_json(directory / "segment_status.json")
    assert manifest["ok"] is True
    assert status["completed_reads"] == readers[0].reads == 20
    assert status["rows"] == 20 and status["toggle_changes"] == 19
    assert status["ads_state"] == status["device_state"] == 0
    assert "not proof of TwinCAT RUN/CONFIG" in status["ads_state_interpretation"]
    assert "ads_state_name" not in status
    assert all(interval >= .005 for interval in clock.sleeps)
    assert readers[0].closed
    assert not (directory / "leg.csv").exists()


def test_constant_toggle_is_failed_and_not_retried(local_config, tmp_path):
    path, _ = local_config
    _, manifest, readers, _ = make_probe(tmp_path, path, constant_toggle=True)
    assert manifest["ok"] is False
    assert "toggle stopped changing" in manifest["error"]
    assert 1 < readers[0].reads < 20 and readers[0].closed


def test_successful_probe_allows_ten_second_fake_capture_with_timestamped_csv(local_config, tmp_path):
    path, config = local_config
    probe, probe_manifest, _, _ = make_probe(tmp_path, path)
    assert probe_manifest["ok"]
    readers, clock = [], FakeClock()
    images = [image_for(config, toggle=index % 2) for index in range(1200)]
    images[1] = images[0]  # Duplicate input snapshots are not separate sample rows.
    directory, manifest = diagnostic.run("capture", config_path=path,
        output_dir=tmp_path / "capture", live=True, duration_s=10, probe_session=probe,
        source_factory=fake_factory(clock, readers, images=images), clock=clock)
    assert manifest["ok"] is True and readers[0].closed
    status = diagnostic.read_json(directory / "segment_status.json")
    assert status["completed_reads"] > status["rows"]
    assert status["rows"] >= 2 and status["toggle_changes"] == status["rows"] - 1
    assert clock.ns - 5_000_000_000 >= 10_000_000_000
    with (directory / "leg.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == diagnostic.FIELDS
        rows = list(reader)
    assert len(rows) == status["rows"]
    assert rows[0]["period_us"] == ""
    assert [int(row["sample_index"]) for row in rows] == list(range(len(rows)))
    assert all(int(row["query_start_ns"]) <= int(row["query_end_ns"]) == int(row["host_monotonic_ns"])
               for row in rows)
    assert all(a["input_toggle"] != b["input_toggle"] for a, b in zip(rows, rows[1:]))
    assert all(datetime.fromisoformat(row["host_timestamp_utc"]).tzinfo for row in rows)
    assert manifest["config_sha256"] == diagnostic.sha256_file(path)
    assert manifest["source_sha256"] == diagnostic.source_hashes()
    assert manifest["output_files_sha256"]["leg.csv"] == diagnostic.sha256_file(directory / "leg.csv")
    assert manifest["output_files_sha256"]["segment_status.json"] == diagnostic.sha256_file(directory / "segment_status.json")
    # The documented, current quality CLI must actually consume this CSV.
    from measurement_validation.leg_episode import main as quality_main
    quality_path = tmp_path / "fake_quality.json"
    assert quality_main([str(directory / "leg.csv"), "--minimum-rate-hz", "500",
                         "--output", str(quality_path)]) == 0
    quality = diagnostic.read_json(quality_path)
    assert quality["rows"] == status["rows"]
    assert quality["usable_for_dynamics"] is False  # Fake reads intentionally run at ~95 Hz.
    assert any(value.startswith("sample_rate_below_500hz") for value in quality["findings"])


@pytest.mark.parametrize("mutation", ["stale", "future", "tampered", "failed", "wrong_ams",
    "wrong_host", "wrong_config", "wrong_source", "incomplete", "no_toggle"])
def test_invalid_probe_blocks_before_any_source_open(local_config, tmp_path, mutation):
    path, _ = local_config
    probe, _, _, _ = make_probe(tmp_path, path)
    manifest_path = probe / "manifest.json"
    manifest = diagnostic.read_json(manifest_path)
    if mutation in ("stale", "future"):
        delta = timedelta(seconds=-601 if mutation == "stale" else 60)
        rewrite_probe_status(probe, completed_utc=(datetime.now(timezone.utc) + delta).isoformat())
    elif mutation == "tampered":
        with (probe / "segment_status.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
    elif mutation == "wrong_ams":
        rewrite_probe_status(probe, actual_local_ams_net_id="1.2.3.4.5.6")
    elif mutation == "incomplete":
        rewrite_probe_status(probe, completed_reads=19)
    elif mutation == "no_toggle":
        rewrite_probe_status(probe, toggle_changes=0)
    else:
        key, value = {"failed": ("ok", False), "wrong_host": ("hostname", "another-computer"),
            "wrong_config": ("config_sha256", "another-config"),
            "wrong_source": ("source_sha256", {})}[mutation]
        manifest[key] = value
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden(config):
        pytest.fail("invalid probe opened a source")

    _, result = diagnostic.run("capture", config_path=path, output_dir=tmp_path / "capture",
        live=True, duration_s=10, probe_session=probe, source_factory=forbidden)
    assert result["ok"] is False
    assert result["status"] == "FAILED"
    assert not (tmp_path / "capture/leg.csv").exists()


@pytest.mark.parametrize("options", [dict(live=False), dict(live=True, duration_s=10),
    dict(live=True, duration_s=1), dict(live=True, duration_s=True)])
def test_capture_requires_live_duration_and_probe_before_open(local_config, tmp_path, options):
    path, _ = local_config

    def forbidden(config):
        pytest.fail("invalid capture opened a source")

    _, result = diagnostic.run("capture", config_path=path, output_dir=tmp_path / "capture",
        source_factory=forbidden, **options)
    assert result["status"] == "FAILED" and result["ok"] is False


@pytest.mark.parametrize("failure", ["close", "cancel", "both"])
def test_cancellation_and_cleanup_failure_are_preserved(local_config, tmp_path, failure):
    _, config = local_config
    clock, readers = FakeClock(), []
    options = dict(close_error=OSError("fake close refused")) if failure != "cancel" else {}
    if failure in ("cancel", "both"):
        options["read_error"] = KeyboardInterrupt()
    status = diagnostic.acquire("probe", config, tmp_path,
        source_factory=fake_factory(clock, readers, **options), clock=clock)
    assert status["ok"] is False and readers[0].closed
    if failure != "close":
        assert "operator cancelled; no automatic retry" in status["error"]
        assert readers[0].reads == 1
    if failure != "cancel":
        assert "ADS close failed: OSError: fake close refused" in status["error"]


def test_existing_output_is_refused_without_overwrite(local_config, tmp_path):
    path, _ = local_config
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "evidence.txt"
    marker.write_bytes(b"preserve these original bytes\r\n")
    with pytest.raises(FileExistsError):
        diagnostic.run("check", config_path=path, output_dir=output)
    assert marker.read_bytes() == b"preserve these original bytes\r\n"
    assert list(output.iterdir()) == [marker]


def test_cli_failure_returns_one_and_writes_failed_manifest(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")
    output = tmp_path / "failed_check"
    assert diagnostic.main(["check", "--config", str(path), "--output-dir", str(output)]) == 1
    assert diagnostic.read_json(output / "manifest.json")["status"] == "FAILED"


@pytest.mark.parametrize("arguments", [["check", "--live"], ["probe", "--offline"],
    ["capture", "--offline", "--live"]])
def test_invalid_offline_live_combinations_rejected_without_output(arguments, monkeypatch):
    monkeypatch.setattr(diagnostic, "run", lambda *a, **k: pytest.fail("invalid CLI reached run"))
    with pytest.raises(SystemExit) as caught:
        diagnostic.main(arguments)
    assert caught.value.code == 2


def test_inspect_retains_twenty_faulted_images_without_authorizing_capture(local_config, tmp_path):
    path, config = local_config
    images = [image_for(config, toggle=index % 2, axis1_status=8, axis1_error=0x1234,
                        axis1_torque=120.+index) for index in range(20)]
    readers, clock = [], FakeClock()
    directory, manifest = diagnostic.run("inspect", config_path=path, output_dir=tmp_path / "inspect",
        live=True, source_factory=fake_factory(clock, readers, images=images), clock=clock)
    status = diagnostic.read_json(directory / "segment_status.json")
    evidence = diagnostic.read_json(directory / "inspection_frames.json")
    assert manifest["ok"] is True and status["ok"] is True
    assert readers[0].reads == status["completed_reads"] == 20 and readers[0].closed
    assert manifest["capture_authorized"] is status["capture_authorized"] is evidence["capture_authorized"] is False
    assert "not hardware healthy or mapping verified" in status["interpretation"]
    assert len(evidence["frames"]) == 20 and evidence["frames"][0]["raw_image_hex"] == images[0].hex()
    assert evidence["frames"][-1]["decoded"]["axis1_torque_3c1a_nm"] == 139.
    assert all(not frame["guard_passed"] and frame["guard_failures"] for frame in evidence["frames"])
    assert all(frame["query_end_ns"] >= frame["query_start_ns"] for frame in evidence["frames"])
    summary = evidence["summary"]
    assert summary["toggle_transitions"] == 19 and summary["unique_image_sha256_count"] == 20
    assert summary["axis_status_distributions"]["1"] == {"0x0008": 20}
    assert summary["axis_error_distributions"]["1"] == {"0x1234": 20}
    assert summary["observed_guard_failures"]["axis1 fault=True"] == 20
    assert summary["raw_byte_change_events"] > 0 and config["toggle_byte"] in summary["changed_byte_indices"]
    assert manifest["output_files_sha256"]["inspection_frames.json"] == diagnostic.sha256_file(directory / "inspection_frames.json")
    assert not (directory / "leg.csv").exists()
    def forbidden(config): pytest.fail("inspect must never authorize capture")
    _, capture = diagnostic.run("capture", config_path=path, output_dir=tmp_path / "blocked_capture",
        live=True, duration_s=10, probe_session=directory, source_factory=forbidden)
    assert not capture["ok"] and "successful complete probe" in capture["error"]


def test_inspect_static_images_finish_diagnosis_and_do_not_claim_refresh(local_config, tmp_path):
    _, config = local_config
    readers, clock = [], FakeClock()
    status = diagnostic.inspect(config, tmp_path,
        source_factory=fake_factory(clock, readers, constant_toggle=True), clock=clock)
    assert status["ok"] and readers[0].reads == 20 and readers[0].closed
    assert status["summary"]["toggle_transitions"] == 0
    assert status["summary"]["unique_image_sha256_count"] == 1
    assert status["summary"]["raw_byte_change_events"] == 0
    assert status["summary"]["mapping_verified"] is False and not status["capture_authorized"]


def test_inspect_native_error_stops_once_and_preserves_partial_frames(local_config, tmp_path):
    _, config = local_config
    readers, clock = [], FakeClock()
    def factory(c):
        reader = FakeReader(c, clock)
        readers.append(reader)
        original = reader.read
        def read():
            if reader.reads == 3:
                reader.reads += 1
                raise diagnostic.AdsReadError(1797, 0, bytes(config["image_bytes"]))
            return original()
        reader.read = read
        return reader
    status = diagnostic.inspect(config, tmp_path, source_factory=factory, clock=clock)
    evidence = diagnostic.read_json(tmp_path / "inspection_frames.json")
    assert not status["ok"] and status["attempted_reads"] == 4 and status["completed_reads"] == 3
    assert readers[0].reads == 4 and readers[0].closed and len(evidence["frames"]) == 4
    assert evidence["frames"][-1]["read_completed"] is False
    assert evidence["frames"][-1]["read_evidence"]["ads_return_code"] == 1797
    assert not evidence["frames"][-1]["read_evidence"]["returned_data_valid"]


def test_inspect_decode_errors_preserve_raw_refresh_evidence(local_config, tmp_path):
    _, config = local_config
    images = [image_for(config, toggle=index % 2, axis1_torque=float("nan")) for index in range(20)]
    readers, clock = [], FakeClock()
    status = diagnostic.inspect(config, tmp_path,
        source_factory=fake_factory(clock, readers, images=images), clock=clock)
    evidence = diagnostic.read_json(tmp_path / "inspection_frames.json")
    assert status["ok"] and readers[0].closed and status["summary"]["decode_error_count"] == 20
    assert status["summary"]["toggle_transitions"] == 19
    assert all(frame["raw_image_hex"] and frame["decode_error"] for frame in evidence["frames"])
    assert not status["capture_authorized"]


def test_inspect_requires_explicit_live_before_native_io(local_config, tmp_path):
    path, _ = local_config
    def forbidden(config): pytest.fail("inspect without --live opened native source")
    _, manifest = diagnostic.run("inspect", config_path=path, output_dir=tmp_path / "blocked", source_factory=forbidden)
    assert not manifest["ok"] and "--live" in manifest["error"]


def test_only_inspect_collects_file_evidence_and_binds_its_hash(local_config, tmp_path, monkeypatch):
    path, _ = local_config
    calls = []
    evidence = dict(evidence_scope="FILE_CONFIGURATION_ONLY", not_live_mapping_verified=True)
    def collect():
        calls.append(True)
        return evidence
    monkeypatch.setattr(diagnostic, "collect_file_evidence", collect)
    diagnostic.run("check", config_path=path, output_dir=tmp_path / "check")
    assert calls == []
    clock, readers = FakeClock(), []
    output, manifest = diagnostic.run("inspect", config_path=path, output_dir=tmp_path / "inspect",
        live=True, source_factory=fake_factory(clock, readers), clock=clock)
    assert calls == [True]
    assert diagnostic.read_json(output / "twincat_file_evidence.json") == evidence
    assert manifest["output_files_sha256"]["twincat_file_evidence.json"] == diagnostic.sha256_file(output / "twincat_file_evidence.json")
