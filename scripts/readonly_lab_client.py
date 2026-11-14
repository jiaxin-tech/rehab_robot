"""Single-file, standard-library, short-lived read-only TwinCAT queue client.

Only local ADS reads are supported. There is no shell, executable upload,
ADS write, controller activation, mode change, robot SDK or motion command.
"""
from __future__ import annotations

import argparse
import base64
import csv
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4

MAX_SESSION_S = 1800
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
EXPECTED_MAPPING = dict(ams_net_id="10.20.4.89.1.1", ads_port=11, index_group=50593808,
                        index_offset=2147483807, image_bytes=1366,
                        axis_offsets=[0, 23, 46, 69], toggle_byte=1365, toggle_bit=4)
AXIS_FIELDS = ("statusword_hex", "operation_enabled", "fault", "position_raw", "position_motor_deg",
               "velocity_raw", "torque_6077_pct_rated", "following_error_raw", "error_code_hex",
               "mode_display", "torque_3c1a_nm", "torque_6077_raw_bits")
FIELDS = ("sample_index", "host_monotonic_ns", "query_start_ns", "query_end_ns", "host_timestamp_utc",
          "receive_elapsed_ms", "period_us", "input_toggle", "synchronization", "acquisition_method") + tuple(
              f"axis{axis}_{field}" for axis in range(1, 5) for field in AXIS_FIELDS)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("mapping_reviewed") is not True:
        raise ValueError("existing reviewed mapping required")
    for key, value in EXPECTED_MAPPING.items():
        if config.get(key) != value:
            raise ValueError("read-only client supports only the existing mapping: " + key)
    if config.get("clock_domain") != "LOCAL_HOST_PERF_COUNTER" or config.get("expected_hz") != 500:
        raise ValueError("existing source/clock declaration required")
    if type(config.get("torque_6077_signedness_verified")) is not bool:
        raise ValueError("explicit 6077 signedness declaration required")
    if str(config.get("ads_dll_path", "")).replace("\\", "/").lower() != "c:/twincat/common64/tcadsdll.dll":
        raise ValueError("only the existing TwinCAT DLL path is supported")
    for name, limit in (("ads_timeout_ms", 100), ("poll_interval_s", .001), ("no_new_sample_timeout_s", .1)):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= limit:
            raise ValueError("existing bounded acquisition setting required: " + name)
    for axis in (1, 4):
        guard = config.get("guards", {}).get(f"axis{axis}", {})
        if guard.get("expected_operation_enabled") is not False or guard.get("max_abs_torque_nm") != 100.:
            raise ValueError("unchanged stationary axis guard required")
    return config


class AmsNetId(ct.Structure):
    _pack_ = 1
    _fields_ = [("bytes", ct.c_ubyte * 6)]


class AmsAddr(ct.Structure):
    _pack_ = 1
    _fields_ = [("net_id", AmsNetId), ("port", ct.c_uint16)]


class AdsReadOnly:
    """Bind exactly six read/session functions; reject a different actual local AMS."""
    def __init__(self, config):
        self.config = validate_config(config)
        self.port = 0
        if os.name != "nt" or struct.calcsize("P") != 8:
            raise OSError("64-bit Windows Python is required")
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
            self.local_ams_net_id = ".".join(str(x) for x in local.net_id.bytes)
            if self.local_ams_net_id != config["ams_net_id"]:
                raise ValueError(f"actual local AMS {self.local_ams_net_id} differs from configured AMS {config['ams_net_id']}; no address guessed")
            self.address = AmsAddr(AmsNetId((ct.c_ubyte * 6)(*map(int, config["ams_net_id"].split(".")))), config["ads_port"])
            ads, device = ct.c_uint16(), ct.c_uint16()
            code = self.dll.AdsSyncReadStateReqEx(self.port, ct.byref(self.address), ct.byref(ads), ct.byref(device))
            if code:
                raise OSError(f"AdsSyncReadStateReqEx failed: {code}")
            self.ads_state, self.device_state = ads.value, device.value
        except BaseException:
            self.close()
            raise

    def read(self):
        config = self.config
        buffer = (ct.c_ubyte * config["image_bytes"])()
        actual = ct.c_uint32()
        code = self.dll.AdsSyncReadReqEx2(self.port, ct.byref(self.address), config["index_group"],
            config["index_offset"], len(buffer), buffer, ct.byref(actual))
        if code or actual.value != len(buffer):
            raise OSError(f"ADS read failed: code={code}, bytes={actual.value}/{len(buffer)}")
        return bytes(buffer)

    def close(self):
        if self.port:
            port, self.port = self.port, 0
            code = self.dll.AdsPortCloseEx(port)
            if code:
                raise OSError(f"AdsPortCloseEx failed: {code}")


def decode_image(image, config):
    if len(image) != config["image_bytes"]:
        raise ValueError("incomplete ADS process image")
    result = {"input_toggle": (image[config["toggle_byte"]] >> config["toggle_bit"]) & 1}
    for axis, offset in enumerate(config["axis_offsets"], 1):
        status, pos, vel, torque, following, error, mode, sensor = struct.unpack_from("<HiihiHbf", image, offset)
        if not math.isfinite(sensor):
            raise ValueError(f"axis{axis} nonfinite 3C1A torque")
        values = (f"0x{status:04X}", bool(status & 4), bool(status & 8), pos, pos * .0004577637,
                  vel, torque * .1 if config["torque_6077_signedness_verified"] else None,
                  following, f"0x{error:04X}", mode, sensor, torque & 0xFFFF)
        result.update({f"axis{axis}_{name}": value for name, value in zip(AXIS_FIELDS, values)})
    return result


def guard_frame(row, config):
    for axis in (1, 4):
        guard = config["guards"][f"axis{axis}"]
        if (row[f"axis{axis}_fault"] or row[f"axis{axis}_error_code_hex"] != "0x0000"
                or row[f"axis{axis}_operation_enabled"] is not False
                or abs(row[f"axis{axis}_torque_3c1a_nm"]) > guard["max_abs_torque_nm"]):
            raise RuntimeError(f"axis{axis} stationary acquisition guard failed: "
                f"operation_enabled={row[f'axis{axis}_operation_enabled']} "
                f"(expected={guard['expected_operation_enabled']}), "
                f"fault={row[f'axis{axis}_fault']}, error={row[f'axis{axis}_error_code_hex']}, "
                f"torque_3c1a_nm={row[f'axis{axis}_torque_3c1a_nm']}, "
                f"max_abs_torque_nm={guard['max_abs_torque_nm']}")


def validate_task(task):
    if (not isinstance(task, dict) or not isinstance(task.get("task_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", task["task_id"])):
        raise ValueError("simple bounded task_id required")
    if set(task) - {"task_id", "action", "duration_s", "expires_at", "created_at"}:
        raise ValueError("unsupported task fields")
    for name in ("expires_at", "created_at"):
        if name in task:
            instant = datetime.fromisoformat(task[name].replace("Z", "+00:00"))
            if instant.tzinfo is None or (name == "expires_at" and instant <= datetime.now(timezone.utc)):
                raise ValueError("timezone-aware unexpired task metadata required")
    action = task.get("action")
    if action == "probe":
        if task.get("duration_s") not in (None, 10, 30):
            raise ValueError("probe duration cannot request another operation")
    elif action == "capture":
        if type(task.get("duration_s")) is not int or task["duration_s"] not in (10, 30):
            raise ValueError("capture duration must be 10 or 30 seconds")
    else:
        raise ValueError("only probe and capture actions are supported")
    return task


def execute_task(task, config, directory, *, source_factory=AdsReadOnly, clock=time):
    """Execute one bounded read-only task, always close ADS and preserve status."""
    validate_task(task)
    validate_config(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    status = dict(task_id=task["task_id"], action=task["action"], status="STARTING", ok=False,
                  completed_reads=0,
                  motion_commanded=False, ads_write_used=False, source_expected_hz=500,
                  timing_semantics="host receive/query times; actual source frame count and 500 Hz delivery are not established")
    write_json(directory / "segment_status.json", status)
    source, stream = None, None
    result, error = None, None
    rows, toggle_changes, previous_toggle, previous_ns = 0, 0, None, None
    started_ns = clock.perf_counter_ns()
    last_change_ns = started_ns
    try:
        source = source_factory(config)
        status.update(actual_local_ams_net_id=source.local_ams_net_id,
                      ads_state=source.ads_state, device_state=source.device_state)
        if task["action"] == "capture":
            stream = (directory / "leg.csv").open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
        last_frame = None
        query_times = []
        deadline_ns = started_ns + task.get("duration_s", 10) * 1_000_000_000
        reads = 0
        while reads < 20 if task["action"] == "probe" else clock.perf_counter_ns() < deadline_ns:
            start_ns = clock.perf_counter_ns()
            image = source.read()
            end_ns = clock.perf_counter_ns()
            reads += 1
            status.update(completed_reads=reads, last_query_start_ns=start_ns,
                          last_query_end_ns=end_ns, last_raw_image_hex=image.hex())
            row = decode_image(image, config)
            status["last_observed_frame"] = dict(row)
            guard_frame(row, config)
            query_times.append((end_ns-start_ns)/1e6)
            changed = row["input_toggle"] != previous_toggle
            if changed:
                if previous_toggle is not None:
                    toggle_changes += 1
                if stream is not None:
                    row.update(sample_index=rows, host_monotonic_ns=end_ns, query_start_ns=start_ns, query_end_ns=end_ns,
                               host_timestamp_utc=datetime.now(timezone.utc).isoformat(), receive_elapsed_ms=(end_ns-started_ns)/1e6,
                               period_us="" if previous_ns is None else (end_ns-previous_ns)/1000,
                               synchronization="four_axes_same_process_image_snapshot_unverified_device_timing",
                               acquisition_method="ADS_READ_ONLY_SINGLE_INPUT_IMAGE")
                    writer.writerow(row)
                rows += 1
                previous_toggle, previous_ns, last_change_ns = row["input_toggle"], end_ns, end_ns
            last_frame = row
            if end_ns-last_change_ns > config["no_new_sample_timeout_s"] * 1e9:
                raise TimeoutError("input toggle stopped changing")
            clock.sleep(max(config["poll_interval_s"], .005 if task["action"] == "probe" else 0.))
        if toggle_changes == 0:
            raise TimeoutError("no input toggle transitions; capture is blocked")
        end = clock.perf_counter_ns()
        result = dict(reads=reads, rows=rows, toggle_changes=toggle_changes, span_s=(end-started_ns)/1e9,
                      max_read_duration_ms=max(query_times), last_frame=last_frame,
                      actual_local_ams_net_id=source.local_ams_net_id, ads_state=source.ads_state, device_state=source.device_state)
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt):
            error = "KeyboardInterrupt: acquisition cancelled"
    finally:
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                error = (error + "; " if error else "") + f"CSV close failed: {type(exc).__name__}: {exc}"
        if source is not None:
            try:
                source.close()
            except BaseException as exc:
                error = (error + "; " if error else "") + f"ADS close failed: {type(exc).__name__}: {exc}"
        status.update(status="COMPLETE" if error is None else "FAILED", ok=error is None,
                      rows=rows, toggle_changes=toggle_changes, error=error, result=result,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "segment_status.json", status)
    return dict(ok=error is None, result=result, error=error)


def output_files(directory):
    files, total = [], 0
    for name in ("segment_status.json", "leg.csv"):
        path = Path(directory) / name
        if path.exists():
            size = path.stat().st_size
            total += size
            if total > MAX_UPLOAD_BYTES:
                raise ValueError("task output exceeds 10 MB upload limit; files remain local")
            files.append(dict(name=name, content_base64=base64.b64encode(path.read_bytes()).decode("ascii")))
    return files


def load_connection(path):
    connection = read_json(path)
    url = urllib.parse.urlparse(connection.get("server_url", ""))
    if url.scheme != "http" or url.hostname != "192.168.137.1" or url.port != 8765 or url.path not in ("", "/") or url.username or url.password:
        raise ValueError("server_url must be http://192.168.137.1:8765")
    token = connection.get("token")
    if not isinstance(token, str) or not 16 <= len(token) <= 256 or any(c.isspace() for c in token):
        raise ValueError("bounded nonempty session token required")
    expires = datetime.fromisoformat(connection["expires_at"].replace("Z", "+00:00"))
    if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
        raise ValueError("future timezone-aware session expiry required")
    return connection, expires


class Broker:
    def __init__(self, connection):
        self.base = connection["server_url"].rstrip("/")
        self.token = connection["token"]
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, value=None):
        headers = {"X-Lab-Token": self.token}
        body = None
        if value is not None:
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base+path, data=body, headers=headers, method="GET" if body is None else "POST")
        with self.opener.open(request, timeout=10) as response:
            payload = response.read(65537)
            if len(payload) > 65536:
                raise ValueError("broker response is too large")
        return json.loads(payload.decode("utf-8")) if payload else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent,
                        help="directory containing connection.json and leg_capture_config.json")
    args = parser.parse_args(argv)
    directory = args.directory.resolve(strict=True)
    connection, expires = load_connection(directory / "connection.json")
    config_path = directory / "leg_capture_config.json"
    config = validate_config(read_json(config_path))
    if os.name != "nt" or struct.calcsize("P") != 8:
        raise OSError("64-bit Windows Python required")
    broker = Broker(connection)
    hello = dict(hostname=socket.gethostname(), python=sys.version.split()[0], python_bits=struct.calcsize("P")*8,
                 dll_available=Path(config["ads_dll_path"]).is_file(), configured_ams_net_id=config["ams_net_id"],
                 config_path=str(config_path), config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 capabilities=["probe_20_reads", "capture_10_or_30_seconds"], motion_available=False)
    broker.request("/hello", hello)
    end = time.perf_counter() + min(MAX_SESSION_S, (expires-datetime.now(timezone.utc)).total_seconds())
    seen, probe_ok = set(), False
    while time.perf_counter() < end and datetime.now(timezone.utc) < expires:
        task = broker.request("/task")
        if task is None:
            time.sleep(min(1., max(0., end-time.perf_counter())))
            continue
        validate_task(task)
        task_id = task["task_id"]
        if task_id in seen:
            raise RuntimeError("broker repeated a task; no automatic reacquisition")
        seen.add(task_id)
        if time.perf_counter() + (task.get("duration_s", 1) if task["action"] == "capture" else 1) > end:
            raise TimeoutError("task exceeds remaining client session")
        session = directory / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+task_id+"_"+uuid4().hex[:8])
        if task["action"] == "capture" and not probe_ok:
            session.mkdir(parents=True, exist_ok=False)
            result = dict(ok=False, result=None, error="a successful toggle-changing probe is required before capture")
            write_json(session / "segment_status.json", result)
        else:
            result = execute_task(task, config, session)
        result["files"] = output_files(session)
        write_json(session / "upload_result.json", dict(task_id=task_id, ok=result["ok"], uploaded=False))
        broker.request("/result?"+urllib.parse.urlencode({"task_id": task_id}), result)
        write_json(session / "upload_result.json", dict(task_id=task_id, ok=result["ok"], uploaded=True))
        if not result["ok"]:
            return 1
        if task["action"] == "probe":
            probe_ok = True
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Read-only client stopped; no task will be retried.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Read-only client stopped: {type(exc).__name__}: {exc}. Local task files are preserved.", file=sys.stderr)
        raise SystemExit(1)
