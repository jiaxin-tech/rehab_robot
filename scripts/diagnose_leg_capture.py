"""Bounded read-only diagnosis of the production leg collector, without a robot.

No probe or robot connection is needed. A new session preserves the exact input
configuration, parent timing, child diagnostics and any partial CSV after failure.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from collection.twincat_leg import FIELDS, LegCaptureProcess, atomic_json, validate_config
from config.bundle import BUNDLE_PATH, ROOT, load_section


SOURCE_FILES = ("scripts/diagnose_leg_capture.py", "collection/twincat_leg.py",
                "collection/leg_capture_diagnostics.py", "collection/leg_progress_publisher.py",
                "config/bundle.py")
MAX_DURATION_S = 30.0
HEALTH_POLL_S = 0.010


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_metadata():
    result = {}
    for key, arguments in (("head", ["rev-parse", "HEAD"]),
                           ("tracked_status", ["status", "--porcelain", "--untracked-files=no"])):
        command = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True,
                                 text=True, encoding="utf-8", timeout=10, check=False)
        if command.returncode:
            raise RuntimeError("cannot preserve Git identity: " + command.stderr.strip())
        result[key] = command.stdout.strip()
    result["tracked_worktree_dirty"] = bool(result["tracked_status"])
    return result


def runtime_metadata():
    info = time.get_clock_info("perf_counter")
    return dict(python_version=sys.version, python_executable=sys.executable,
                python_implementation=platform.python_implementation(), platform=platform.platform(),
                hostname=socket.gethostname(), parent_pid=os.getpid(),
                perf_counter=dict(implementation=info.implementation, monotonic=info.monotonic,
                                  adjustable=info.adjustable, resolution_s=info.resolution),
                sleep_function="time.sleep", requested_health_poll_interval_s=HEALTH_POLL_S)


def summarize_csv(path):
    """Describe accepted rows only; this cannot establish a 500 Hz device rate."""
    if not path.exists():
        return dict(available=False, rows=0, recorded_span_s=None)
    count, first, last, first_ns, previous_ns, maximum_gap_ns = 0, None, None, None, None, 0
    maximum_query_ns = 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("leg CSV header differs from the production collector")
        for row in reader:
            stamp, begin, end = (int(row[key]) for key in ("host_monotonic_ns", "query_start_ns", "query_end_ns"))
            if end != stamp or end < begin or (previous_ns is not None and stamp <= previous_ns):
                raise ValueError("leg CSV timestamps are invalid or nonincreasing")
            if int(row["sample_index"]) != count:
                raise ValueError("leg CSV sample indices are incomplete")
            if first is None:
                first, first_ns = row, stamp
            if previous_ns is not None:
                maximum_gap_ns = max(maximum_gap_ns, stamp - previous_ns)
            maximum_query_ns = max(maximum_query_ns, end - begin)
            previous_ns, last, count = stamp, row, count + 1
    span = (previous_ns - first_ns) / 1e9 if count else None
    return dict(available=True, rows=count, recorded_span_s=span,
                first_host_time_s=first_ns / 1e9 if count else None,
                last_host_time_s=previous_ns / 1e9 if count else None,
                first_utc=first["host_timestamp_utc"] if count else None,
                last_utc=last["host_timestamp_utc"] if count else None,
                maximum_recorded_gap_s=maximum_gap_ns / 1e9 if count > 1 else None,
                maximum_recorded_query_duration_s=maximum_query_ns / 1e9 if count else None,
                observed_accepted_row_rate_hz=(count - 1) / span if count > 1 and span > 0 else None,
                sampling_interpretation="Accepted toggle transitions only; not an exact device frame count or verified 500 Hz acquisition.")


def run(config_path=BUNDLE_PATH, session=None, *, duration_s=10.0, live=False,
        capture_factory=None, clock=time):
    if live is not True:
        raise PermissionError("--live is required for the bounded read-only leg capture")
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or not 0 < duration_s <= MAX_DURATION_S:
        raise ValueError("duration-s must be finite, greater than zero and at most 30 seconds")
    if session is None:
        raise ValueError("a new --session directory is required")
    original = Path(config_path).resolve(strict=True)
    config_bytes = original.read_bytes()
    validate_config(load_section("leg", config_path=original))
    directory = Path(session).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    snapshot_path = directory / "config_snapshot.json"
    with snapshot_path.open("xb") as stream:
        stream.write(config_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    config = validate_config(load_section("leg", config_path=snapshot_path))
    atomic_json(directory / "leg_config_snapshot.json", config)
    started = clock.perf_counter()
    manifest = dict(schema_version=1, action="production_leg_capture_diagnostic", status="STARTING", ok=False,
                    data_classification="DIAGNOSTIC_ONLY", motion_commanded=False, robot_connected=False,
                    ads_write_used=False, clinical_use_authorized=False, training_authorized=False,
                    started_utc=datetime.now(timezone.utc).isoformat(), requested_duration_s=float(duration_s),
                    started_host_time_s=started, config_path=str(original),
                    executed_config_path=str(snapshot_path), config_sha256=sha256_file(snapshot_path),
                    runtime=runtime_metadata(),
                    source_sha256={name: sha256_file(ROOT / name) for name in SOURCE_FILES},
                    csv_complete_acknowledged=False, capture_stop_returned_normally=False, cleanup_error=None,
                    sampling_interpretation="Production mapping and guards unchanged; no exact 500 Hz claim.")
    atomic_json(directory / "manifest.json", manifest)
    capture, monitor_started, monitor_finished, last_poll = None, None, None, None
    maximum_poll_gap = 0.0
    error, cancelled = None, False
    try:
        manifest["git"] = git_metadata()
        factory = LegCaptureProcess if capture_factory is None else capture_factory
        capture = factory(snapshot_path, directory / "leg")
        capture.start()
        monitor_started = clock.perf_counter()
        manifest["capture_ready_host_time_s"] = monitor_started
        deadline = monitor_started + float(duration_s)
        while True:
            now = clock.perf_counter()
            if last_poll is not None:
                maximum_poll_gap = max(maximum_poll_gap, now - last_poll)
            last_poll = now
            reason = capture.health_reason()
            if reason:
                raise RuntimeError(reason)
            if now >= deadline:
                break
            clock.sleep(min(HEALTH_POLL_S, deadline - now))
        monitor_finished = clock.perf_counter()
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        cancelled = isinstance(exc, KeyboardInterrupt)
    finally:
        if monitor_started is not None and monitor_finished is None:
            monitor_finished = clock.perf_counter()
        if capture is not None:
            try:
                capture.stop()
                manifest["capture_stop_returned_normally"] = True
            except (Exception, KeyboardInterrupt) as exc:
                manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            try:
                manifest["child_final_status"] = capture.snapshot()
                manifest["csv_complete_acknowledged"] = (
                    manifest["capture_stop_returned_normally"]
                    and manifest["child_final_status"].get("state") == "completed")
            except Exception as exc:
                manifest["child_status_error"] = f"{type(exc).__name__}: {exc}"
                error = error or manifest["child_status_error"]
    manifest["monitor_duration_s"] = monitor_finished - monitor_started if monitor_started is not None else None
    manifest["maximum_observed_health_poll_gap_s"] = maximum_poll_gap if last_poll is not None else None
    diagnostics_path = directory / "leg/leg_capture_diagnostics.json"
    manifest["child_diagnostics_available"] = diagnostics_path.is_file()
    if diagnostics_path.is_file():
        try:
            diagnostic = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if not isinstance(diagnostic, dict):
                raise ValueError("child diagnostics must be a JSON object")
            manifest["child_diagnostics"] = diagnostic
            if diagnostic.get("completed") is not True or diagnostic.get("error") or diagnostic.get("cleanup_error"):
                raise RuntimeError("child diagnostics do not confirm successful capture and cleanup")
        except Exception as exc:
            manifest["child_diagnostics_error"] = f"{type(exc).__name__}: {exc}"
            if error is None and manifest["cleanup_error"] is None:
                error = manifest["child_diagnostics_error"]
    delivery_path = directory / "leg/progress_delivery_diagnostics.json"
    manifest["progress_delivery_diagnostics_available"] = delivery_path.is_file()
    if delivery_path.is_file():
        try:
            delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
            if not isinstance(delivery, dict):
                raise ValueError("progress delivery diagnostics must be a JSON object")
            manifest["progress_delivery_diagnostics"] = delivery
        except Exception as exc:
            manifest["progress_delivery_diagnostics_error"] = f"{type(exc).__name__}: {exc}"
            error = error or manifest["progress_delivery_diagnostics_error"]
    try:
        summary = summarize_csv(directory / "leg/leg.csv")
        manifest["csv_summary"] = summary
        final = manifest.get("child_final_status", {})
        if not error and not manifest["cleanup_error"]:
            if not manifest["csv_complete_acknowledged"] or final.get("state") != "completed":
                raise RuntimeError("collector did not acknowledge a completed, closed CSV")
            if summary["rows"] < 2 or final.get("rows") != summary["rows"]:
                raise RuntimeError("completed collector row count differs from the retained CSV")
    except Exception as exc:
        manifest["csv_inspection_error"] = f"{type(exc).__name__}: {exc}"
        error = error or manifest["csv_inspection_error"]
    manifest.update(status="CANCELLED" if cancelled else "FAILED" if error or manifest["cleanup_error"] else "CAPTURE_COMPLETE",
                    ok=not cancelled and error is None and manifest["cleanup_error"] is None,
                    error=error, finished_utc=datetime.now(timezone.utc).isoformat(),
                    finished_host_time_s=clock.perf_counter())
    manifest["parent_elapsed_s"] = manifest["finished_host_time_s"] - started
    report = ["# 假腿独立只读采集诊断", "", "状态：" + manifest["status"],
              "机器人连接与运动：无；训练：禁用。", "",
              "请求监测时长：" + str(manifest["requested_duration_s"]) + " 秒；实际监测时长：" + str(manifest["monitor_duration_s"]) + " 秒。",
              "接受的采集帧与实际记录跨度见 manifest.json 的 csv_summary；不代表精确500Hz采集。"]
    if error:
        report.append("失败原因：" + error)
    if manifest["cleanup_error"]:
        report.append("关闭错误：" + manifest["cleanup_error"])
    report.append("子进程诊断：" + ("leg/leg_capture_diagnostics.json" if manifest["child_diagnostics_available"] else "未生成；不能据此判断超时区间内的刷新情况。"))
    report.append("进度传送诊断：" + ("leg/progress_delivery_diagnostics.json" if manifest["progress_delivery_diagnostics_available"] else "未生成。"))
    (directory / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    manifest["output_files_sha256"] = {path.relative_to(directory).as_posix(): sha256_file(path)
                                       for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    atomic_json(directory / "manifest.json", manifest)
    return directory, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--session", type=Path, required=True, help="new directory only")
    parser.add_argument("--duration-s", type=float, default=10.0, help="monitor for >0 and <=30 seconds after startup")
    parser.add_argument("--live", action="store_true", required=True, help="allow this bounded read-only capture")
    args = parser.parse_args(argv)
    try:
        directory, manifest = run(args.config, args.session, duration_s=args.duration_s, live=args.live)
    except (OSError, ValueError, TypeError) as exc:
        print("未启动采集：" + str(exc), file=sys.stderr)
        return 1
    print("记录目录：" + str(directory))
    print("只读采集完成；请回传整个记录目录。" if manifest["ok"] else
          "采集停止；保留原始记录，请回传整个目录：" + str(manifest.get("error") or manifest.get("cleanup_error")))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
