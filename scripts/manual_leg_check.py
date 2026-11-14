"""Local, bounded ADS reads using the unified experiment configuration.

Check never opens ADS. Probe and capture expose read/session functions only;
they cannot change a runtime state, enable an axis, or command either device.
"""
from __future__ import annotations

import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from uuid import uuid4

from config.bundle import BUNDLE_PATH, ROOT, load_section
from collection.twincat_leg import AmsAddr, FIELDS, decode_image, validate_config

PROBE_MAX_AGE_S = 600
SOURCE_FILES = ("scripts/manual_leg_check.py", "collection/twincat_leg.py", "config/bundle.py")


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required: " + str(path))
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def git_head():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", timeout=10, check=False)
    if result.returncode != 0:
        raise RuntimeError("无法记录Git版本，请在完整仓库中运行。")
    return result.stdout.strip()


def offline_check(config_path):
    """Read the current unified leg section without loading a native library."""
    path = Path(config_path).resolve(strict=True)
    config = validate_config(load_section("leg", config_path=path))
    checks = dict(windows=os.name == "nt", python_64bit=struct.calcsize("P") == 8,
                  dll_exists=Path(config["ads_dll_path"]).is_file(), configuration_valid=True)
    return config, dict(ok=all(checks.values()), checks=checks,
        python_version=sys.version.split()[0], python_executable=sys.executable,
        hostname=socket.gethostname(), config_path=str(path), config_sha256=sha256_file(path),
        configured_ams_net_id=config["ams_net_id"], ads_dll_path=config["ads_dll_path"],
        device_connected=False, ads_library_loaded=False)


class AdsReadError(OSError):
    """A failed native read, retaining exactly what the API returned."""
    def __init__(self, code, actual_bytes, buffer):
        super().__init__(f"ADS read failed: code={code}, bytes={actual_bytes}/{len(buffer)}")
        self.read_evidence = dict(ads_return_code=code, reported_bytes=actual_bytes,
            expected_bytes=len(buffer), raw_buffer_hex=bytes(buffer).hex(),
            returned_data_valid=False)


class LocalAdsReader:
    """Six ADS read/session calls; the configured target must be this host."""
    def __init__(self, config):
        self.config = validate_config(config)
        self.port = 0
        if os.name != "nt" or struct.calcsize("P") != 8:
            raise OSError("64-bit Windows Python is required for the installed ADS DLL")
        self.dll = ct.WinDLL(str(Path(config["ads_dll_path"]).resolve(strict=True)))
        signatures = {
            "AdsPortOpenEx": [], "AdsPortCloseEx": [ct.c_long],
            "AdsSyncSetTimeoutEx": [ct.c_long, ct.c_uint32],
            "AdsGetLocalAddressEx": [ct.c_long, ct.POINTER(AmsAddr)],
            "AdsSyncReadStateReqEx": [ct.c_long, ct.POINTER(AmsAddr), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_uint16)],
            "AdsSyncReadReqEx2": [ct.c_long, ct.POINTER(AmsAddr), ct.c_uint32, ct.c_uint32,
                                  ct.c_uint32, ct.c_void_p, ct.POINTER(ct.c_uint32)],
        }
        for name, args in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = args, ct.c_long
        self.port = self.dll.AdsPortOpenEx()
        if self.port <= 0:
            self.port = 0
            raise OSError("AdsPortOpenEx failed")
        try:
            code = self.dll.AdsSyncSetTimeoutEx(self.port, math.ceil(config["ads_timeout_ms"]))
            if code:
                raise OSError(f"AdsSyncSetTimeoutEx failed: {code}")
            local = AmsAddr()
            code = self.dll.AdsGetLocalAddressEx(self.port, ct.byref(local))
            if code:
                raise OSError(f"AdsGetLocalAddressEx failed: {code}")
            self.actual_local_ams_net_id = ".".join(str(x) for x in local.net_id.bytes)
            if self.actual_local_ams_net_id != config["ams_net_id"]:
                raise ValueError(f"actual local AMS {self.actual_local_ams_net_id} differs from configured AMS {config['ams_net_id']}; no target address guessed")
            self.address = AmsAddr(local.net_id, config["ads_port"])
            ads, device = ct.c_uint16(), ct.c_uint16()
            code = self.dll.AdsSyncReadStateReqEx(self.port, ct.byref(self.address), ct.byref(ads), ct.byref(device))
            if code:
                raise OSError(f"AdsSyncReadStateReqEx failed: {code}")
            # This is the queried ADS service's raw status, not a system-mode
            # certificate. In particular, zero must never be called CONFIG.
            self.ads_state, self.device_state = ads.value, device.value
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup:
                if hasattr(exc, "add_note"):
                    exc.add_note("ADS close failed: " + str(cleanup))
            raise

    def read(self):
        c = self.config
        buffer = (ct.c_ubyte * c["image_bytes"])()
        actual = ct.c_uint32()
        code = self.dll.AdsSyncReadReqEx2(self.port, ct.byref(self.address), c["index_group"],
            c["index_offset"], len(buffer), buffer, ct.byref(actual))
        if code or actual.value != len(buffer):
            raise AdsReadError(code, actual.value, buffer)
        return bytes(buffer)

    def close(self):
        if self.port:
            port, self.port = self.port, 0
            code = self.dll.AdsPortCloseEx(port)
            if code:
                raise OSError(f"AdsPortCloseEx failed: {code}")


def guard_failures(row, config):
    failures = []
    for axis in (1, 4):
        prefix, guard = f"axis{axis}", config["guards"][f"axis{axis}"]
        if row[prefix + "_fault"]:
            failures.append(prefix + " fault=True")
        if row[prefix + "_error_code_hex"] != "0x0000":
            failures.append(prefix + " error_code=" + row[prefix + "_error_code_hex"])
        actual = row[prefix + "_operation_enabled"]
        if actual != guard["expected_operation_enabled"]:
            failures.append(f"{prefix} operation_enabled={actual}, expected={guard['expected_operation_enabled']}")
        torque = row[prefix + "_torque_3c1a_nm"]
        if abs(torque) > guard["max_abs_torque_nm"]:
            failures.append(f"{prefix} abs(torque_3c1a_nm)={abs(torque):.9g} exceeds configured {guard['max_abs_torque_nm']:.9g}")
    return failures


def acquire(action, config, directory, *, duration_s=None, source_factory=LocalAdsReader, clock=time):
    """Read once per poll, stopping on the first fault with raw evidence."""
    if action not in ("probe", "capture") or (action == "capture" and duration_s not in (10, 30)):
        raise ValueError("only probe or 10/30-second capture is supported")
    status = dict(action=action, status="STARTING", ok=False, attempted_reads=0, completed_reads=0, rows=0,
        toggle_changes=0, motion_commanded=False, ads_write_used=False,
        ads_state_interpretation="raw queried-service value; not proof of TwinCAT RUN/CONFIG",
        timing_semantics="host query/receive timing; not exact device frame loss or hardware synchronization")
    source, stream, error = None, None, None
    rows, reads, changes, previous_toggle, previous_ns = 0, 0, 0, None, None
    durations = []
    try:
        source = source_factory(config)
        actual = source.actual_local_ams_net_id
        status.update(actual_local_ams_net_id=actual, ads_state=source.ads_state, device_state=source.device_state)
        if actual != config["ams_net_id"]:
            raise ValueError("actual local AMS differs from configured target; no address guessed")
        if action == "capture":
            stream = (directory / "leg.csv").open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
        origin = clock.perf_counter_ns()
        last_change = origin
        deadline = origin + (duration_s or 10) * 1_000_000_000
        while (reads < 20 if action == "probe" else clock.perf_counter_ns() < deadline):
            begin = clock.perf_counter_ns()
            status.update(attempted_reads=status["attempted_reads"] + 1, last_query_start_ns=begin)
            try:
                image = source.read()
            except BaseException as exc:
                status["last_query_end_ns"] = clock.perf_counter_ns()
                if getattr(exc, "read_evidence", None) is not None:
                    status["last_failed_read"] = exc.read_evidence
                raise
            end = clock.perf_counter_ns()
            reads += 1
            status.update(completed_reads=reads, last_query_start_ns=begin,
                          last_query_end_ns=end, last_raw_image_hex=image.hex())
            row = decode_image(image, config)
            status["last_observed_frame"] = dict(row)
            failures = guard_failures(row, config)
            status["guard_failures"] = failures
            if failures:
                raise RuntimeError("stationary guard failed: " + "; ".join(failures))
            durations.append((end-begin)/1e6)
            if row["input_toggle"] != previous_toggle:
                if previous_toggle is not None:
                    changes += 1
                if stream is not None:
                    row.update(sample_index=rows, host_monotonic_ns=end, query_start_ns=begin, query_end_ns=end,
                        host_timestamp_utc=datetime.now(timezone.utc).isoformat(), receive_elapsed_ms=(end-origin)/1e6,
                        period_us="" if previous_ns is None else (end-previous_ns)/1000,
                        synchronization="four_axes_same_process_image_snapshot_unverified_device_timing",
                        acquisition_method="ADS_READ_ONLY_SINGLE_INPUT_IMAGE")
                    writer.writerow(row)
                rows += 1
                previous_toggle, previous_ns, last_change = row["input_toggle"], end, end
            if end-last_change > config["no_new_sample_timeout_s"] * 1e9:
                raise TimeoutError("input toggle stopped changing")
            clock.sleep(max(config["poll_interval_s"], .005 if action == "probe" else 0))
        if not changes or rows < 2:
            raise ValueError("no fresh toggle transitions; observation is not complete")
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        error = ("KeyboardInterrupt: operator cancelled; no automatic retry" if isinstance(exc, KeyboardInterrupt)
                 else f"{type(exc).__name__}: {exc}")
        if getattr(exc, "__notes__", None):
            status["exception_notes"] = list(exc.__notes__)
    finally:
        for name, resource in (("CSV", stream), ("ADS", source)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    error = (error + "; " if error else "") + f"{name} close failed: {type(exc).__name__}: {exc}"
        status.update(status="COMPLETE" if error is None else "FAILED", ok=error is None,
            completed_reads=reads, rows=rows, toggle_changes=changes, error=error,
            max_read_duration_ms=max(durations) if durations else None,
            completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "segment_status.json", status)
    return status


def validate_probe_session(path, *, config_sha256, hostname, expected_ams, now=None):
    path = Path(path).resolve(strict=True)
    manifest, status = read_json(path / "manifest.json"), read_json(path / "segment_status.json")
    if (manifest.get("action") != "probe" or manifest.get("status") != "COMPLETE"
        or manifest.get("ok") is not True or status.get("action") != "probe"
        or status.get("ok") is not True or status.get("status") != "COMPLETE"
        or status.get("completed_reads") != 20 or status.get("rows", 0) < 2
        or status.get("guard_failures")):
        raise ValueError("capture requires a successful complete probe directory")
    if manifest.get("config_sha256") != config_sha256 or manifest.get("hostname") != hostname:
        raise ValueError("probe does not belong to this host and current unified configuration")
    if status.get("actual_local_ams_net_id") != expected_ams or status.get("toggle_changes", 0) <= 0:
        raise ValueError("probe did not confirm the configured local AMS and changing input toggle")
    completed = datetime.fromisoformat(status["completed_utc"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc) if now is None else now
    if completed.tzinfo is None or not 0 <= (now-completed).total_seconds() <= PROBE_MAX_AGE_S:
        raise ValueError("probe is older than ten minutes or its clock is inconsistent")
    if manifest.get("source_sha256") != source_hashes():
        raise ValueError("diagnostic code changed after probe")
    if manifest.get("output_files_sha256", {}).get("segment_status.json") != sha256_file(path / "segment_status.json"):
        raise ValueError("probe status SHA-256 changed")
    return path


def run(action, *, config_path=BUNDLE_PATH, output_dir=None, live=False, duration_s=None,
        probe_session=None, source_factory=LocalAdsReader, clock=time):
    if action not in ("check", "probe", "capture"):
        raise ValueError("only check, probe and capture are supported")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+action+"_"+uuid4().hex[:8]
    output = Path(output_dir).resolve() if output_dir else ROOT / "results/manual_lab" / suffix
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, action=action, status="STARTING", ok=False,
        hostname=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
        config_path=str(Path(config_path).resolve()), live_requested=bool(live),
        motion_commanded=False, ads_write_used=False, remote_helper_used=False,
        duration_s=duration_s, source_sha256=source_hashes())
    try:
        manifest["git_head"] = git_head()
        config, checks = offline_check(config_path)
        manifest.update(config_sha256=checks["config_sha256"], offline_check=checks)
        write_json(output / "leg_config_snapshot.json", config)
        if not checks["ok"]:
            raise RuntimeError("offline check failed: " + json.dumps(checks["checks"]))
        if action == "check":
            manifest.update(status="COMPLETE", ok=True)
        else:
            if live is not True:
                raise PermissionError("--live is required for a real ADS read")
            if action == "capture":
                if type(duration_s) is not int or duration_s not in (10, 30):
                    raise ValueError("capture duration must be exactly 10 or 30 seconds")
                if probe_session is None:
                    raise ValueError("capture requires --probe-session")
                probe = validate_probe_session(probe_session, config_sha256=checks["config_sha256"],
                    hostname=manifest["hostname"], expected_ams=config["ams_net_id"])
                manifest["probe_session"] = str(probe)
            status = acquire(action, config, output, duration_s=duration_s, source_factory=source_factory, clock=clock)
            manifest.update(status=status["status"], ok=status["ok"], error=status.get("error"))
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="FAILED", ok=False, error=f"{type(exc).__name__}: {exc}")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_files_sha256"] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return output, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "probe", "capture"))
    parser.add_argument("--config", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--output-dir", type=Path, help="new directory only; default results/manual_lab/UTC_action")
    parser.add_argument("--offline", action="store_true", help="explicit offline check; valid only with check")
    parser.add_argument("--live", action="store_true", help="allow only this bounded ADS read")
    parser.add_argument("--duration-s", type=int, choices=(10, 30))
    parser.add_argument("--probe-session", type=Path)
    args = parser.parse_args(argv)
    if args.offline and (args.action != "check" or args.live):
        parser.error("--offline is valid only for check without --live")
    if args.action == "check" and args.live:
        parser.error("check is always offline; remove --live")
    output, manifest = run(args.action, config_path=args.config, output_dir=args.output_dir,
        live=args.live, duration_s=args.duration_s, probe_session=args.probe_session)
    print("记录目录：" + str(output))
    print("只读检查完成。" if manifest["ok"] else "本次停止，不自动重试：" + str(manifest.get("error")))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
