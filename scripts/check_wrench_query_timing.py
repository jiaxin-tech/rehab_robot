"""Measure direct xCoreSDK getEndTorque query latency without robot motion."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from scripts.rokae_diagnostic_common import make_robot, numeric_summary, readonly_connection, sleep_until, write_report


def measure(robot: Any, *, duration_s: float, target_hz: float, reference_frame: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    period_ns = int(1_000_000_000 / target_hz)
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
    next_tick_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    while time.perf_counter_ns() < deadline_ns:
        scheduled_ns = next_tick_ns
        started_ns = time.perf_counter_ns()
        try:
            raw = robot.get_end_wrench(reference_frame)
            finished_ns = time.perf_counter_ns()
            row = {"sample_index": len(rows), "scheduled_perf_ns": scheduled_ns, "started_perf_ns": started_ns,
                   "finished_perf_ns": finished_ns, "query_duration_ms": (finished_ns - started_ns) / 1e6,
                   "start_lateness_ms": max(0, started_ns - scheduled_ns) / 1e6, "valid": True,
                   "invalid_reason": "", "adapter_query_started_s": raw.get("force_query_started_s"),
                   "adapter_query_finished_s": raw.get("force_query_finished_s"), "raw_force_frame": raw.get("raw_force_frame")}
        except Exception as exc:
            finished_ns = time.perf_counter_ns()
            row = {"sample_index": len(rows), "scheduled_perf_ns": scheduled_ns, "started_perf_ns": started_ns,
                   "finished_perf_ns": finished_ns, "query_duration_ms": (finished_ns - started_ns) / 1e6,
                   "start_lateness_ms": max(0, started_ns - scheduled_ns) / 1e6, "valid": False,
                   "invalid_reason": f"get_end_torque_error:{type(exc).__name__}:{exc}"}
        # A query misses the acquisition deadline when it alone consumes the
        # whole period. Scheduler lateness is retained separately so ordinary
        # host wake-up jitter is not misreported as an SDK-query failure.
        row["deadline_miss"] = row["query_duration_ms"] > period_ns / 1e6
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
        if time.perf_counter_ns() - next_tick_ns > period_ns:
            next_tick_ns = time.perf_counter_ns()
    durations = numeric_summary(row["query_duration_ms"] for row in rows)
    summary = {"diagnostic": "wrench_query_timing", "target_hz": target_hz, "deadline_ms": period_ns / 1e6,
               "sample_count": len(rows), "valid_count": sum(bool(row["valid"]) for row in rows),
               "deadline_miss_count": sum(bool(row["deadline_miss"]) for row in rows),
               "query_duration_ms": durations,
               "stable_50hz_supported": target_hz == 50.0 and sum(bool(row["deadline_miss"]) for row in rows) == 0 and sum(not bool(row["valid"]) for row in rows) == 0}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ROKAE getEndTorque 查询延迟诊断（只读，不上电、不运动）")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    if args.duration <= 0 or args.target_hz <= 0:
        parser.error("--duration and --target-hz must be positive")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"diagnostic": "wrench_query_timing", "fatal_error": None}
    try:
        with readonly_connection(make_robot(args.robot_ip)) as (robot, _):
            rows, summary = measure(robot, duration_s=args.duration, target_hz=args.target_hz, reference_frame=settings.ROBOT_FORCE_RAW_FRAME)
    except Exception as exc:
        summary["fatal_error"] = f"{type(exc).__name__}:{exc}"
    csv_path, json_path = write_report(args.output_dir, "wrench_query_timing", rows, summary)
    print(f"getEndTorque: {summary.get('sample_count', 0)} queries, deadline miss={summary.get('deadline_miss_count', 0)}")
    print(f"query ms (mean/p95/p99/max): {summary.get('query_duration_ms')}")
    print(f"stable 50 Hz: {summary.get('stable_50hz_supported')}")
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    if summary.get("fatal_error"):
        raise SystemExit(summary["fatal_error"])


if __name__ == "__main__":
    main()
