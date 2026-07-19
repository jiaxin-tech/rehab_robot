"""Read-only multi-pose residual-wrench diagnostic for xCoreSDK estimates."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from scripts.rokae_diagnostic_common import (
    make_robot,
    pose_dependence_analysis,
    readonly_connection,
    require_confirmed_bias,
    sleep_until,
    vec_mean,
    write_report,
)


def capture_pose(robot: Any, source: Any, *, pose_index: int, samples: int, sample_hz: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Capture static samples at a manually arranged pose; no motion is issued."""
    period_ns = int(1_000_000_000 / sample_hz)
    next_tick_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    for _ in range(samples):
        perf_ns = time.perf_counter_ns()
        try:
            state = robot.get_state_frame()
            frame = source.snapshot()
            row = {"pose_index": pose_index, "sample_index": len(rows), "perf_counter_ns": perf_ns,
                   "state_sequence_id": state.sequence_id, "wrench_sequence_id": frame.sequence_id,
                   "state_valid": state.valid, "wrench_valid": frame.valid,
                   "invalid_reason": ";".join(part for part in (state.invalid_reason if not state.valid else "", frame.invalid_reason if not frame.valid else "") if part),
                   "tcp_position_m": state.tcp_position_m, "tcp_orientation_rad": state.tcp_orientation_rad,
                   "tcp_linear_velocity_mps": state.tcp_linear_velocity_mps,
                   "raw_force_n": frame.cartesian_force_raw_n, "raw_torque_nm": frame.cartesian_torque_raw_nm,
                   "corrected_force_n": frame.cartesian_force_corrected_n, "corrected_torque_nm": frame.cartesian_torque_corrected_nm,
                   "base_force_n": frame.cartesian_force_base_n, "base_torque_nm": frame.cartesian_torque_base_nm,
                   "base_transform_kind": frame.base_transform_kind}
        except Exception as exc:
            row = {"pose_index": pose_index, "sample_index": len(rows), "perf_counter_ns": perf_ns,
                   "state_valid": False, "wrench_valid": False, "invalid_reason": f"pose_snapshot_error:{type(exc).__name__}:{exc}"}
        rows.append(row)
        next_tick_ns += period_ns
        sleep_until(next_tick_ns, time.perf_counter_ns, time.sleep)
    summary = {"pose_index": pose_index, "sample_count": len(rows),
               "tcp_position_m": next((row.get("tcp_position_m") for row in rows if row.get("tcp_position_m") is not None), None),
               "tcp_orientation_rad": next((row.get("tcp_orientation_rad") for row in rows if row.get("tcp_orientation_rad") is not None), None),
               "raw_force_mean_n": vec_mean(row.get("raw_force_n") for row in rows),
               "raw_torque_mean_nm": vec_mean(row.get("raw_torque_nm") for row in rows),
               "corrected_force_mean_n": vec_mean(row.get("corrected_force_n") for row in rows),
               "corrected_torque_mean_nm": vec_mean(row.get("corrected_torque_nm") for row in rows),
               "base_force_mean_n": vec_mean(row.get("base_force_n") for row in rows),
               "base_torque_mean_nm": vec_mean(row.get("base_torque_nm") for row in rows)}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ROKAE 多姿态静止 wrench 残差诊断（只读，不上电、不运动）")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--poses", type=int, default=3, help="由操作员在 HMI/外部方式安排的静止姿态数量")
    parser.add_argument("--samples-per-pose", type=int, default=100)
    parser.add_argument("--sample-hz", type=float, default=float(settings.COLLECT_HZ))
    parser.add_argument("--force-change-threshold-n", type=float, default=1.0)
    parser.add_argument("--torque-change-threshold-nm", type=float, default=0.2)
    parser.add_argument("--confirm-unloaded", action="store_true", required=True, help="确认 bias 前无接触、静止、工具/负载正确")
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    if args.poses < 2 or args.samples_per_pose <= 0 or args.sample_hz <= 0:
        parser.error("--poses must be at least 2; samples and sample rate must be positive")
    if args.force_change_threshold_n < 0 or args.torque_change_threshold_nm < 0:
        parser.error("pose-dependence thresholds must be non-negative")
    rows: list[dict[str, Any]] = []
    pose_summaries: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"diagnostic": "wrench_pose_dependence", "fatal_error": None}
    try:
        with readonly_connection(make_robot(args.robot_ip), use_wrench_stream=True) as (robot, source):
            assert source is not None
            require_confirmed_bias(source, args.confirm_unloaded)
            for pose_index in range(1, args.poses + 1):
                input(f"Arrange static pose {pose_index}/{args.poses} externally (no command is sent), then press Enter to sample. ")
                pose_rows, pose_summary = capture_pose(robot, source, pose_index=pose_index, samples=args.samples_per_pose, sample_hz=args.sample_hz)
                rows.extend(pose_rows)
                pose_summaries.append(pose_summary)
            summary.update(pose_dependence_analysis(pose_summaries, force_threshold_n=args.force_change_threshold_n, torque_threshold_nm=args.torque_change_threshold_nm))
            summary["per_pose"] = pose_summaries
            summary["base_rotation_config_verified"] = settings.BASE_WRENCH_ROTATION_VERIFIED
            summary["note"] = "The outcome does not determine whether xCoreSDK performs gravity compensation."
    except Exception as exc:
        summary["fatal_error"] = f"{type(exc).__name__}:{exc}"
    csv_path, json_path = write_report(args.output_dir, "wrench_pose_dependence", rows, summary)
    print(f"pose residual force/torque max delta: {summary.get('max_corrected_force_delta_n')} N / {summary.get('max_corrected_torque_delta_nm')} Nm")
    print(f"software-bias pose dependence observed: {summary.get('software_bias_pose_dependence_observed')}")
    print("This does not determine SDK gravity-compensation behavior.")
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    if summary.get("fatal_error"):
        raise SystemExit(summary["fatal_error"])


if __name__ == "__main__":
    main()
