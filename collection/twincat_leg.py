"""Read-only ADS process-image capture, isolated from robot control."""
from __future__ import annotations

import csv
import ctypes as ct
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"finite positive {name} required")
    return float(value)


def validate_config(c):
    if c.get("clock_domain") != "LOCAL_HOST_PERF_COUNTER":
        raise ValueError("collector must run on the same Windows host as robot orchestration")
    parts = c["ams_net_id"].split(".")
    if len(parts) != 6 or any(not p.isdecimal() or not 0 <= int(p) <= 255 for p in parts):
        raise ValueError("six-byte AMS Net ID required")
    for key in ("ads_port", "index_group", "index_offset", "image_bytes", "toggle_byte", "toggle_bit"):
        value = c[key]
        if type(value) is not int or value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"invalid {key}")
    if not 0 < c["ads_port"] <= 65535 or not 0 < c["image_bytes"] <= 65536:
        raise ValueError("invalid ADS port/image size")
    if not 0 <= c["toggle_byte"] < c["image_bytes"] or not 0 <= c["toggle_bit"] < 8:
        raise ValueError("toggle outside image")
    if c.get("axis_offsets") != [0, 23, 46, 69] or c["image_bytes"] < 92:
        raise ValueError("this decoder requires the declared four-axis 23-byte PDO layout")
    for key in ("expected_hz", "ads_timeout_ms", "startup_timeout_s", "max_progress_age_s",
                "shutdown_timeout_s", "heartbeat_s", "no_new_sample_timeout_s", "poll_interval_s"):
        positive(c.get(key), key)
    if c["heartbeat_s"] >= c["max_progress_age_s"]:
        raise ValueError("heartbeat must be shorter than progress budget")
    if c["expected_hz"] != 500:
        raise ValueError("current reviewed mapping is a 500 Hz source")
    if c["poll_interval_s"] >= 1 / c["expected_hz"]:
        raise ValueError("poll interval must be shorter than the expected source cycle")
    if c["ads_timeout_ms"] > 0xFFFFFFFF:
        raise ValueError("ADS timeout exceeds API range")
    if not c.get("ads_dll_path"):
        raise ValueError("explicit ADS DLL path required")
    for axis in (1, 4):
        guard = c["guards"][f"axis{axis}"]
        positive(guard.get("max_abs_torque_nm"), f"axis{axis} torque limit")
        if type(guard.get("expected_operation_enabled")) is not bool:
            raise ValueError("reviewed expected leg enable state required")
    return c


class AmsNetId(ct.Structure):
    _pack_ = 1
    _fields_ = [("bytes", ct.c_ubyte * 6)]


class AmsAddr(ct.Structure):
    _pack_ = 1
    _fields_ = [("net_id", AmsNetId), ("port", ct.c_uint16)]


class AdsInputImage:
    """Only AdsSyncReadReqEx2 is exposed; no ADS write operation exists here."""
    def __init__(self, config):
        self.config = validate_config(config)
        if os.name != "nt":
            raise OSError("native TwinCAT ADS capture requires Windows")
        path = Path(config["ads_dll_path"]).resolve(strict=True)
        self.dll = ct.WinDLL(str(path))
        self.dll.AdsPortOpenEx.argtypes = []
        self.dll.AdsPortOpenEx.restype = ct.c_long
        self.dll.AdsPortCloseEx.argtypes = [ct.c_long]
        self.dll.AdsPortCloseEx.restype = ct.c_long
        self.dll.AdsSyncSetTimeoutEx.argtypes = [ct.c_long, ct.c_uint32]
        self.dll.AdsSyncSetTimeoutEx.restype = ct.c_long
        self.dll.AdsSyncReadReqEx2.argtypes = [ct.c_long, ct.POINTER(AmsAddr), ct.c_uint32,
            ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(ct.c_uint32)]
        self.dll.AdsSyncReadReqEx2.restype = ct.c_long
        self.port = self.dll.AdsPortOpenEx()
        if self.port <= 0:
            raise OSError("AdsPortOpenEx failed")
        self.address = AmsAddr(AmsNetId((ct.c_ubyte * 6)(*map(int, config["ams_net_id"].split(".")))), config["ads_port"])
        try:
            status = self.dll.AdsSyncSetTimeoutEx(self.port, math.ceil(config["ads_timeout_ms"]))
            if status:
                raise OSError(f"ADS timeout configuration failed: {status}")
        except BaseException:
            self.close()
            raise

    def read(self):
        c = self.config
        buffer = (ct.c_ubyte * c["image_bytes"])()
        actual = ct.c_uint32()
        status = self.dll.AdsSyncReadReqEx2(self.port, ct.byref(self.address), c["index_group"],
            c["index_offset"], len(buffer), buffer, ct.byref(actual))
        if status or actual.value != len(buffer):
            raise OSError(f"ADS input read failed: code={status}, bytes={actual.value}/{len(buffer)}")
        return bytes(buffer)

    def close(self):
        if getattr(self, "port", 0):
            port, self.port = self.port, 0
            if self.dll.AdsPortCloseEx(port):
                raise OSError("AdsPortCloseEx failed")


AXIS_FIELDS = ("statusword_hex", "operation_enabled", "fault", "position_raw", "position_motor_deg",
               "velocity_raw", "torque_6077_pct_rated", "following_error_raw", "error_code_hex",
               "mode_display", "torque_3c1a_nm", "torque_6077_raw_bits")
FIELDS = ("sample_index", "host_monotonic_ns", "query_start_ns", "query_end_ns", "host_timestamp_utc",
          "receive_elapsed_ms", "period_us", "input_toggle", "synchronization", "acquisition_method") + tuple(
              f"axis{axis}_{field}" for axis in range(1, 5) for field in AXIS_FIELDS)


def decode_image(image, config):
    if len(image) != config["image_bytes"]:
        raise ValueError("incomplete input image")
    result = {"input_toggle": (image[config["toggle_byte"]] >> config["toggle_bit"]) & 1}
    for axis, offset in enumerate(config["axis_offsets"], 1):
        status, pos, vel, torque, following, error, mode, sensor = struct.unpack_from("<HiihiHbf", image, offset)
        if not math.isfinite(sensor):
            raise ValueError(f"axis{axis} nonfinite torque sensor")
        values = (f"0x{status:04X}", bool(status & 4), bool(status & 8), pos, pos * .0004577637,
                  vel, torque * .1 if config.get("torque_6077_signedness_verified", False) else None,
                  following, f"0x{error:04X}", mode, sensor, torque & 0xFFFF)
        result.update({f"axis{axis}_{name}": value for name, value in zip(AXIS_FIELDS, values)})
    return result


def capture(config, output, *, stop_event, notify, source_factory=AdsInputImage):
    """Write complete images after toggle changes; never claim exact frame counts."""
    validate_config(config)
    output = Path(output)
    source = source_factory(config)
    count, previous_toggle, previous_ns = 0, None, None
    origin = time.perf_counter_ns()
    last_sample = last_notify = time.perf_counter()
    try:
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            while not stop_event.is_set():
                start_ns = time.perf_counter_ns()
                image = source.read()
                end_ns = time.perf_counter_ns()
                row = decode_image(image, config)
                for axis in (1, 4):
                    guard = config["guards"][f"axis{axis}"]
                    if (row[f"axis{axis}_fault"] or row[f"axis{axis}_error_code_hex"] != "0x0000"
                            or row[f"axis{axis}_operation_enabled"] != guard["expected_operation_enabled"]
                            or abs(row[f"axis{axis}_torque_3c1a_nm"]) > guard["max_abs_torque_nm"]):
                        raise RuntimeError(f"axis{axis} leg state/torque guard failed")
                now = time.perf_counter()
                if row["input_toggle"] != previous_toggle:
                    row.update(sample_index=count, host_monotonic_ns=end_ns, query_start_ns=start_ns,
                        query_end_ns=end_ns, host_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        receive_elapsed_ms=(end_ns-origin)/1e6,
                        period_us="" if previous_ns is None else (end_ns-previous_ns)/1000,
                        synchronization="four_axes_same_process_image_snapshot_unverified_device_timing",
                        acquisition_method="ADS_READ_ONLY_SINGLE_INPUT_IMAGE")
                    writer.writerow(row)
                    previous_toggle, previous_ns, last_sample = row["input_toggle"], end_ns, now
                    count += 1
                if now - last_sample > config["no_new_sample_timeout_s"]:
                    raise TimeoutError("input toggle stopped changing")
                if count >= 2 and (now-last_notify >= config["heartbeat_s"] or count == 2):
                    stream.flush()
                    notify(dict(state="recording", rows=count, last_sample_ns=previous_ns,
                        latest={key: row[key] for axis in (1, 4) for key in
                            (f"axis{axis}_position_raw", f"axis{axis}_torque_3c1a_nm")}))
                    last_notify = now
                # Windows Event.wait rounded short waits beyond the 2 ms cycle
                # in the host benchmark. Python 3.11+ sleep uses its finer timer.
                time.sleep(config["poll_interval_s"])
            stream.flush()
            os.fsync(stream.fileno())
        if count < 2:
            raise ValueError("capture ended before two fresh samples")
    finally:
        source.close()
    notify(dict(state="completed", rows=count, last_sample_ns=previous_ns))
    return count


class LegCaptureProcess:
    def __init__(self, config_path, output_dir):
        self.config_path = Path(config_path).resolve()
        self.config = validate_config(read_json(config_path))
        self.directory = Path(output_dir)
        self.process = None
        self.status = {}
        self._lock = threading.Lock()
        self._error = None

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stderr = (self.directory / "capture_stderr.log").open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen([sys.executable, "-u", "-m", "collection.twincat_leg",
                "--config", str(self.config_path), "--output", str((self.directory / "leg.csv").resolve())],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, encoding="utf-8",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except BaseException:
            self._stderr.close()
            raise
        def receive():
            try:
                for line in self.process.stdout:
                    record = json.loads(line)
                    with self._lock:
                        self.status = record
            except BaseException as exc:
                with self._lock:
                    self._error = str(exc)
        self._reader = threading.Thread(target=receive, daemon=True)
        self._reader.start()
        deadline = time.perf_counter() + self.config["startup_timeout_s"]
        while time.perf_counter() < deadline:
            if self.health_reason() is None:
                return
            if self.process.poll() is not None:
                break
            time.sleep(.01)
        raise RuntimeError("leg capture failed to become ready: " + str(self.health_reason()))

    def snapshot(self):
        with self._lock:
            return dict(self.status)

    def health_reason(self):
        with self._lock:
            state, error = dict(self.status), self._error
        if error:
            return "leg_capture_protocol_error:" + error
        if self.process is None or self.process.poll() is not None:
            return "leg_capture_process_not_running"
        if state.get("state") != "recording":
            return "leg_capture_not_recording"
        age = (time.perf_counter_ns() - state["last_sample_ns"]) / 1e9
        if age < 0 or age > self.config["max_progress_age_s"]:
            return "leg_capture_stale"
        return None

    def stop(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write("stop\n")
                    self.process.stdin.flush()
                except (OSError, BrokenPipeError):
                    pass
                try:
                    self.process.wait(timeout=self.config["shutdown_timeout_s"])
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
                    raise TimeoutError("ADS worker did not close CSV; partial data excluded")
            self._reader.join(timeout=1)
            with self._lock:
                completed = self.status.get("state") == "completed"
            if self.process.returncode != 0 or not completed:
                raise RuntimeError("ADS capture did not acknowledge a completed, closed CSV")
        finally:
            self._stderr.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = validate_config(read_json(args.config))
    stop = threading.Event()
    def commands():
        # EOF means the controlling process disappeared; finish the read-only capture.
        sys.stdin.readline()
        stop.set()
    threading.Thread(target=commands, daemon=True).start()
    notify = lambda value: print(json.dumps(value), flush=True)
    try:
        capture(config, args.output, stop_event=stop, notify=notify)
    except BaseException as exc:
        notify(dict(state="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
