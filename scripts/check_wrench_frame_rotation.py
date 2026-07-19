"""Manual light-push check for pending world-to-base wrench rotation.

The script records evidence only.  It never changes
``BASE_WRENCH_ROTATION_VERIFIED`` and never sends a motion command.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from scripts.rokae_diagnostic_common import make_robot, readonly_connection, require_confirmed_bias, rotation_push_analysis, sleep_until, write_report


def capture(source: Any, *, duration_s: float, sample_hz: float, phase: str) -> list[dict[str, Any]]:
    period_ns = int(1_000_000_000 / sample_hz)
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
    next_tick_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    while time.perf_counter_ns() < deadline_ns:
        perf_ns = time.perf_counter_ns()
        try:
            frame = source.snapshot()
            row = {"sample_index": len(rows), "phase": phase, "perf_counter_ns": perf_ns, "sequence_id": frame.sequence_id,
                   "valid": frame.valid, "invalid_reason": frame.invalid_reason, "raw_force_frame": frame.raw_force_frame,
                   "raw_force_n": frame.cartesian_force_raw_n, "corrected_force_n": frame.cartesian_force_corrected_n,
                   "base_force_n": frame.cartesian_force_base_n, "base_transform_kind": frame.base_transform_kind}
        except Exception as exc:
            row = {"sample_index": len(rows), "phase": phase, "perf_counter_ns": perf_ns, "valid": False,
                   "invalid_reason": f"wrench_snapshot_error:{type(exc).__name__}:{exc}", "base_force_n": None}
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="手工轻推 world→base wrench 旋转诊断（只读，不运动）")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--direction", choices=("X", "Y", "Z", "x", "y", "z"), required=True, help="预期的正 base 轴")
    parser.add_argument("--baseline-duration", type=float, default=2.0)
    parser.add_argument("--push-duration", type=float, default=3.0)
    parser.add_argument("--sample-hz", type=float, default=float(settings.COLLECT_HZ))
    parser.add_argument("--confirm-unloaded", action="store_true", required=True, help="确认 bias 前无接触、静止、工具/负载正确")
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    if min(args.baseline_duration, args.push_duration, args.sample_hz) <= 0:
        parser.error("durations and --sample-hz must be positive")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"diagnostic": "wrench_frame_rotation", "base_rotation_config_verified": settings.BASE_WRENCH_ROTATION_VERIFIED, "fatal_error": None}
    try:
        with readonly_connection(make_robot(args.robot_ip), use_wrench_stream=True) as (_, source):
            assert source is not None
            require_confirmed_bias(source, args.confirm_unloaded)
            print("Recording unloaded baseline; keep the robot still and untouched...")
            baseline = capture(source, duration_s=args.baseline_duration, sample_hz=args.sample_hz, phase="baseline")
            input(f"Apply one gentle, slow positive base-{args.direction.upper()} push, then press Enter to record. ")
            pushed = capture(source, duration_s=args.push_duration, sample_hz=args.sample_hz, phase="push")
            rows = baseline + pushed
            summary.update(rotation_push_analysis([row.get("base_force_n") for row in baseline], [row.get("base_force_n") for row in pushed], args.direction))
            summary["base_transform_kinds"] = sorted({str(row.get("base_transform_kind")) for row in rows if row.get("base_transform_kind")})
            summary["note"] = "Evidence only: this script does not set BASE_WRENCH_ROTATION_VERIFIED."
    except Exception as exc:
        summary["fatal_error"] = f"{type(exc).__name__}:{exc}"
    csv_path, json_path = write_report(args.output_dir, "wrench_frame_rotation", rows, summary)
    print(f"principal axis/sign: {summary.get('principal_axis')} {summary.get('principal_sign')}; cross-axis ratio={summary.get('cross_axis_ratio')}")
    print(f"expected {args.direction.upper()} positive: {summary.get('expected_axis_positive')}; config flag unchanged={settings.BASE_WRENCH_ROTATION_VERIFIED}")
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    if summary.get("fatal_error"):
        raise SystemExit(summary["fatal_error"])


if __name__ == "__main__":
    main()
