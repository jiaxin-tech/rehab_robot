"""Repeatable, hardware-free quality checks and provisional angle calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from measurement_validation.leg_episode import quality_report, read_leg_episode


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def positive(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def segment_metrics(times: np.ndarray, values: np.ndarray) -> dict:
    valid = np.isfinite(times) & np.isfinite(values)
    t, y = times[valid], values[valid]
    result = {"rows": len(values), "missing": int((~valid).sum()),
              "mean": None, "std": None, "range": None, "slope_per_s": None,
              "end_minus_start": None}
    if len(y):
        edge = max(1, len(y) // 10)
        result.update(mean=float(y.mean()), std=float(y.std(ddof=1)) if len(y) > 1 else None,
                      range=float(np.ptp(y)), end_minus_start=float(y[-edge:].mean() - y[:edge].mean()))
    if len(y) > 1 and np.ptp(t) > 0:
        result["slope_per_s"] = float(np.polyfit(t - t[0], y, 1)[0])
    return result


def stability(times: np.ndarray, counts: np.ndarray, *, max_range: float,
              max_slope: float, min_duration: float, max_gap_s: float = .01) -> dict:
    result = segment_metrics(times, counts)
    reasons = []
    if result["missing"]:
        reasons.append("missing_position_values")
    if len(times) < 3 or times[-1] - times[0] < min_duration:
        reasons.append("segment_too_short")
    result["max_interval_s"] = float(np.max(np.diff(times))) if len(times) > 1 else None
    if result["max_interval_s"] is not None and result["max_interval_s"] > max_gap_s:
        reasons.append("time_gap_exceeded")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        reasons.append("non_increasing_time")
    if result["range"] is None or result["range"] > max_range:
        reasons.append("position_range_exceeded")
    if result["slope_per_s"] is None or abs(result["slope_per_s"]) > max_slope:
        reasons.append("position_drift_exceeded")
    result.update(stable=not reasons, reasons=reasons)
    return result


def inspect_file(path: Path, *, expected_hz: float = 500, gap_factor: float = 2,
                 window_s: float = 1, max_range: float = 50,
                 max_slope: float = 20, torque_drift_nm: float = .2) -> dict:
    episode = read_leg_episode(path)
    frame = episode.source_frame
    t = episode.host_time_s - episode.host_time_s[0]
    dt = np.diff(t)
    period = 1 / expected_hz
    report = quality_report(episode, expected_period_s=period)
    report.update(source_path=str(path.resolve()),
                  thresholds={"expected_hz": expected_hz, "gap_factor": gap_factor,
                              "window_s": window_s, "max_position_range_counts": max_range,
                              "max_position_slope_counts_per_s": max_slope,
                              "torque_drift_flag_nm": torque_drift_nm})
    report["long_intervals"] = [
        {"previous_csv_line": int(i + 2), "csv_line": int(i + 3),
         "start_s": float(t[i]), "end_s": float(t[i + 1]), "interval_ms": float(dt[i] * 1000)}
        for i in np.flatnonzero(dt > period * gap_factor)]
    report["receive_time_duplicates_csv_lines"] = (np.flatnonzero(dt == 0) + 3).tolist()
    # Host timing only estimates scheduling deficits, not device frame loss.
    report["timing_missing_estimate"] = int(np.maximum(np.rint(dt / period) - 1, 0).sum())
    report["missing_estimate_note"] = "Host interval estimate only; not exact device frame loss"
    report["missing_values_by_column"] = {str(c): int(frame[c].isna().sum()) for c in frame
                                           if frame[c].isna().any()}
    report["secondary_clocks"] = {}
    for column in ("host_timestamp_filetime_100ns", "host_timestamp_utc"):
        if column not in frame:
            continue
        if column.endswith("100ns"):
            ticks = [int(x) for x in frame[column]]
        else:
            stamps = pd.to_datetime(frame[column], utc=True, errors="raise", format="ISO8601")
            ticks = [int(x.value) for x in stamps]
        delta = np.asarray([b - a for a, b in zip(ticks, ticks[1:])])
        report["secondary_clocks"][column] = {
            "duplicate_csv_lines": (np.flatnonzero(delta == 0) + 3).tolist(),
            "backward_csv_lines": (np.flatnonzero(delta < 0) + 3).tolist()}
    report["acquisition_metadata"] = {}
    for column in ("acquisition_method", "synchronization", "sync_quality", "configured_cycle_ms",
                   "configured_frequency_hz", "estimated_dropped_cumulative"):
        if column in frame:
            report["acquisition_metadata"][column] = [str(x) for x in frame[column].dropna().unique()]
    report["joint_checks"] = {}
    for name, joint in episode.joints.items():
        entry: dict = {"missing_fields": [key for key in
                       ("position_counts", "torque_nm", "statusword", "error_code", "fault",
                        "operation_enabled", "brake_override") if getattr(joint, key) is None]}
        if joint.torque_nm is not None:
            entry["raw_torque_nm"] = segment_metrics(t, joint.torque_nm)
            change = entry["raw_torque_nm"]["end_minus_start"]
            entry["torque_drift_flag"] = change is not None and abs(change) > torque_drift_nm
            entry["torque_drift_note"] = "Descriptive change; motion/load changes are not distinguished from sensor drift; no tare"
        if joint.position_counts is not None:
            entry["position_counts"] = segment_metrics(t, joint.position_counts)
            entry["whole_segment_stability"] = stability(t, joint.position_counts,
                max_range=max_range, max_slope=max_slope, min_duration=window_s * .8,
                max_gap_s=gap_factor / expected_hz)
            windows = []
            bins = np.floor(t / window_s).astype(int)
            for group in np.unique(bins):
                mask = bins == group
                check = stability(t[mask], joint.position_counts[mask], max_range=max_range,
                                  max_slope=max_slope, min_duration=window_s * .8,
                                  max_gap_s=gap_factor / expected_hz)
                windows.append({"start_s": float(t[mask][0]), "end_s": float(t[mask][-1]), **check})
            entry["position_windows"] = windows
            entry["stability_note"] = "Screening only; stationary counts do not verify the external angle"
        for signal in ("statusword", "error_code", "operation_enabled", "fault"):
            values = getattr(joint, signal)
            if values is not None:
                entry[signal] = {"values": np.unique(values[np.isfinite(values)]).tolist(),
                                 "missing": int((~np.isfinite(values)).sum())}
        report["joint_checks"][name] = entry
    return report
