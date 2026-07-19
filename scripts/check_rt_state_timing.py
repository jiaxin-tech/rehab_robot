"""Read-only xCore realtime-state timing diagnostic; never moves the robot."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collection.state import state_age_ms
from config import settings
from scripts.rokae_diagnostic_common import (
    invalid_reason_counts,
    make_robot,
    numeric_summary,
    readonly_connection,
    sequence_drops,
    sleep_until,
    write_report,
)


def measure(robot: Any, *, duration_s: float, poll_hz: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect cached RT frames; dependency injection keeps this fake-testable."""
    period_ns = int(1_000_000_000 / poll_hz)
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
    next_tick_ns = time.perf_counter_ns()
    previous_perf_ns: int | None = None
    previous_state_time_s: float | None = None
    previous_sequence_id: int | None = None
    rows: list[dict[str, Any]] = []
    exceptions = 0
    while time.perf_counter_ns() < deadline_ns:
        started_ns = time.perf_counter_ns()
        try:
            frame = robot.get_state_frame()
            now_monotonic_s = time.monotonic()
            age_ms = state_age_ms(now_monotonic_s, frame.host_monotonic_time_s)
            update_period_ms = None
            if (
                previous_sequence_id is None or frame.sequence_id != previous_sequence_id
            ) and frame.host_monotonic_time_s is not None and previous_state_time_s is not None:
                update_period_ms = (frame.host_monotonic_time_s - previous_state_time_s) * 1000.0
            if frame.sequence_id != previous_sequence_id and frame.host_monotonic_time_s is not None:
                previous_state_time_s = frame.host_monotonic_time_s
            previous_sequence_id = frame.sequence_id
            row = {
                "sample_index": len(rows), "perf_counter_ns": started_ns,
                "sequence_id": frame.sequence_id, "valid": frame.valid,
                "invalid_reason": frame.invalid_reason, "operation_state": frame.operation_state,
                "state_host_time_s": frame.host_monotonic_time_s,
                "state_age_ms": age_ms,
                "rt_update_period_ms": update_period_ms,
                "sample_period_ms": None if previous_perf_ns is None else (started_ns - previous_perf_ns) / 1e6,
            }
            previous_perf_ns = started_ns
        except Exception as exc:
            exceptions += 1
            row = {"sample_index": len(rows), "perf_counter_ns": started_ns, "valid": False,
                   "invalid_reason": f"state_read_error:{type(exc).__name__}:{exc}", "sequence_id": None,
                   "state_age_ms": None, "sample_period_ms": None, "operation_state": None}
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
        if time.perf_counter_ns() - next_tick_ns > period_ns:
            next_tick_ns = time.perf_counter_ns()
    update_summary = numeric_summary(row.get("rt_update_period_ms") for row in rows)
    summary = {
        "diagnostic": "rt_state_timing", "requested_poll_hz": poll_hz, "sample_count": len(rows),
        "valid_count": sum(bool(row["valid"]) for row in rows), "exception_count": exceptions,
        "dropped_frames": sequence_drops([row.get("sequence_id") for row in rows]),
        "state_age_ms": numeric_summary(row.get("state_age_ms") for row in rows),
        "rt_update_period_ms": update_summary,
        "sample_period_ms": numeric_summary(row.get("sample_period_ms") for row in rows),
        "estimated_rt_update_hz": (
            1000.0 / float(update_summary["mean"])
            if update_summary["mean"] else None
        ),
        "unique_state_frames": len({row.get("sequence_id") for row in rows if row.get("sequence_id") is not None}),
        "invalid_reasons": invalid_reason_counts(rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ROKAE RT 状态流频率诊断（只读，不上电、不运动）")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--poll-hz", type=float, default=100.0, help="缓存帧读取频率，不改变 RT 流频率")
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    if args.duration <= 0 or args.poll_hz <= 0:
        parser.error("--duration and --poll-hz must be positive")
    robot = make_robot(args.robot_ip)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"diagnostic": "rt_state_timing", "fatal_error": None}
    try:
        with readonly_connection(robot) as (connected_robot, _):
            rows, summary = measure(connected_robot, duration_s=args.duration, poll_hz=args.poll_hz)
    except Exception as exc:
        summary["fatal_error"] = f"{type(exc).__name__}:{exc}"
    csv_path, json_path = write_report(args.output_dir, "rt_state_timing", rows, summary)
    print(f"RT state: {summary.get('sample_count', 0)} reads, dropped={summary.get('dropped_frames', 0)}, errors={summary.get('exception_count', 0)}")
    print(f"state age ms: {summary.get('state_age_ms')}; RT update period ms: {summary.get('rt_update_period_ms')}")
    print(f"invalid reasons: {summary.get('invalid_reasons', {})}")
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    if summary.get("fatal_error"):
        raise SystemExit(summary["fatal_error"])


if __name__ == "__main__":
    main()
