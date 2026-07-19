"""Measure RT-state/internal-wrench alignment using the project snapshot path."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collection.snapshot import read_live_robot_state_sample
from config import settings
from scripts.rokae_diagnostic_common import invalid_reason_counts, make_robot, numeric_summary, readonly_connection, require_confirmed_bias, sleep_until, write_report


def measure(robot: Any, source: Any, *, duration_s: float, sample_hz: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    period_ns = int(1_000_000_000 / sample_hz)
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
    next_tick_ns = time.perf_counter_ns()
    previous = None
    rows: list[dict[str, Any]] = []
    while time.perf_counter_ns() < deadline_ns:
        started_ns = time.perf_counter_ns()
        try:
            sample = read_live_robot_state_sample(robot, source, previous_sample=previous)
            previous = sample
            row = {"sample_index": len(rows), "perf_counter_ns": started_ns, "sequence_id": sample.sequence_id,
                   "valid": sample.valid, "invalid_reason": sample.invalid_reason,
                   "robot_state_age_ms": sample.robot_state_age_ms, "force_sample_age_ms": sample.force_sample_age_ms,
                   "state_internal_skew_ms": sample.state_internal_skew_ms, "force_query_duration_ms": sample.force_query_duration_ms,
                   "force_estimate_valid": sample.force_estimate_valid, "base_transform_kind": sample.base_transform_kind}
        except Exception as exc:
            row = {"sample_index": len(rows), "perf_counter_ns": started_ns, "sequence_id": None, "valid": False,
                   "invalid_reason": f"snapshot_read_error:{type(exc).__name__}:{exc}", "robot_state_age_ms": None,
                   "force_sample_age_ms": None, "state_internal_skew_ms": None, "force_query_duration_ms": None,
                   "force_estimate_valid": False, "base_transform_kind": None}
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
        if time.perf_counter_ns() - next_tick_ns > period_ns:
            next_tick_ns = time.perf_counter_ns()
    summary = {"diagnostic": "snapshot_alignment", "sample_hz": sample_hz, "sample_count": len(rows),
               "valid_count": sum(bool(row["valid"]) for row in rows),
               "robot_state_age_ms": numeric_summary(row["robot_state_age_ms"] for row in rows),
               "force_sample_age_ms": numeric_summary(row["force_sample_age_ms"] for row in rows),
               "state_internal_skew_ms": numeric_summary(row["state_internal_skew_ms"] for row in rows),
               "force_query_duration_ms": numeric_summary(row["force_query_duration_ms"] for row in rows),
               "invalid_reasons": invalid_reason_counts(rows)}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ROKAE RT/wrench 快照对齐诊断（只读，不上电、不运动）")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=float(settings.COLLECT_HZ))
    parser.add_argument("--software-bias", action="store_true", help="显式请求会话软件 bias")
    parser.add_argument("--confirm-unloaded", action="store_true", help="确认已静止、无接触且工具/负载配置正确")
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    if args.duration <= 0 or args.sample_hz <= 0:
        parser.error("--duration and --sample-hz must be positive")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"diagnostic": "snapshot_alignment", "fatal_error": None}
    try:
        with readonly_connection(make_robot(args.robot_ip), use_wrench_stream=True) as (robot, source):
            assert source is not None
            if args.software_bias:
                require_confirmed_bias(source, args.confirm_unloaded)
            rows, summary = measure(robot, source, duration_s=args.duration, sample_hz=args.sample_hz)
            summary["software_bias_used"] = args.software_bias
            summary["base_rotation_config_verified"] = settings.BASE_WRENCH_ROTATION_VERIFIED
    except Exception as exc:
        summary["fatal_error"] = f"{type(exc).__name__}:{exc}"
    csv_path, json_path = write_report(args.output_dir, "snapshot_alignment", rows, summary)
    print(f"snapshot: {summary.get('valid_count', 0)}/{summary.get('sample_count', 0)} valid")
    print(f"age/skew/query ms: {summary.get('robot_state_age_ms')} / {summary.get('state_internal_skew_ms')} / {summary.get('force_query_duration_ms')}")
    print(f"invalid reasons: {summary.get('invalid_reasons', {})}")
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    if summary.get("fatal_error"):
        raise SystemExit(summary["fatal_error"])


if __name__ == "__main__":
    main()
