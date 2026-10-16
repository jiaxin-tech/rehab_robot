"""Repeatable, hardware-free quality checks and provisional angle calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from measurement_validation.leg_episode import quality_report, read_leg_episode


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    report.update(source_path=str(path.resolve()), sha256=digest(path),
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


def calibrate(manifest: dict, data_dir: Path, candidate: dict) -> dict:
    """Fit only declared stable fit points; score independent points before refit."""
    if (not np.isfinite(candidate["counts_per_degree"]) or candidate["counts_per_degree"] <= 0
            or not np.isfinite(candidate["zero_counts"]) or candidate["flexion_sign"] not in (-1, 1)):
        raise ValueError("invalid candidate scale/zero/sign")
    limits = manifest.get("stability", {})
    max_range = float(limits.get("max_range_counts", 50))
    max_slope = float(limits.get("max_slope_counts_per_s", 20))
    min_duration = float(limits.get("min_duration_s", .8))
    max_gap = float(limits.get("max_gap_s", .01))
    if any(not np.isfinite(v) or v <= 0 for v in (max_range, max_slope, min_duration, max_gap)):
        raise ValueError("stability limits must be finite and positive")
    points, ids, selections = [], set(), []
    for spec in manifest["points"]:
        if spec["id"] in ids:
            raise ValueError("duplicate point id")
        ids.add(spec["id"])
        joint = spec["joint"]
        if joint not in ("hip", "knee") or spec["role"] not in ("fit", "validation", "return"):
            raise ValueError("unknown joint or point role")
        path = data_dir / spec["file"]
        sha = digest(path)
        if spec.get("sha256") and sha != spec["sha256"]:
            raise ValueError(f"source hash changed: {path}")
        episode = read_leg_episode(path)
        t = episode.host_time_s - episode.host_time_s[0]
        start, end = float(spec["start_s"]), float(spec["end_s"])
        if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end):
            raise ValueError("segment needs finite 0 <= start_s < end_s")
        for old_path, old_joint, old_start, old_end, old_role in selections:
            if (path.resolve() == old_path and joint == old_joint and start < old_end and end > old_start
                    and spec["role"] != old_role and "fit" in (spec["role"], old_role)):
                raise ValueError("fit and held-out segments must not overlap")
        selections.append((path.resolve(), joint, start, end, spec["role"]))
        mask = (t >= start) & (t < end)
        counts = episode.joints[joint].position_counts
        if counts is None or not mask.any():
            raise ValueError(f"no raw counts in selected segment: {spec['id']}")
        angle = spec.get("angle_deg")
        if angle is not None and not np.isfinite(float(angle)):
            raise ValueError("angle must be finite or null")
        if angle is None and spec["role"] != "return":
            raise ValueError("fit and validation points require an independently measured angle")
        check = stability(t[mask], counts[mask], max_range=max_range,
                          max_slope=max_slope, min_duration=min_duration, max_gap_s=max_gap)
        p = {**spec, "sha256": sha, "segment": check,
             "accepted": bool(check["stable"]), "prediction_deg": None, "prediction_error_deg": None}
        p["prediction_role"] = "in_sample" if spec["role"] == "fit" else "held_out"
        if joint == "hip" and check["mean"] is not None:
            pred = candidate["flexion_sign"] * (check["mean"] - candidate["zero_counts"]) / candidate["counts_per_degree"]
            p["prediction_deg"] = float(pred)
            if angle is not None:
                p["prediction_error_deg"] = float(pred - angle)
        points.append(p)
    fits, pairs = {}, []
    for joint in ("hip", "knee"):
        fit = [p for p in points if p["joint"] == joint and p["role"] == "fit" and p["accepted"]]
        fits[joint] = None
        if len(fit) >= 2 and len({p["angle_deg"] for p in fit}) >= 2:
            angles = np.asarray([p["angle_deg"] for p in fit], dtype=float)
            counts = np.asarray([p["segment"]["mean"] for p in fit])
            slope, zero = np.polyfit(angles, counts, 1)
            if abs(slope) < 1e-12:
                raise ValueError("zero count/angle slope")
            fits[joint] = {"zero_counts": float(zero), "counts_per_degree": float(abs(slope)),
                           "flexion_sign": int(np.sign(slope)), "fit_point_ids": [p["id"] for p in fit]}
            for p in points:
                if p["joint"] == joint and p["segment"]["mean"] is not None:
                    pred = float((p["segment"]["mean"] - zero) / slope)
                    p["refit_prediction_deg"] = pred
                    p["refit_error_deg"] = None if p.get("angle_deg") is None else float(pred - p["angle_deg"])
        groups: dict[str, list] = {}
        for p in points:
            if p["joint"] == joint and p.get("pair_id") and p["accepted"]:
                groups.setdefault(p["pair_id"], []).append(p)
        for pair_id, group in groups.items():
            if len(group) != 2:
                raise ValueError(f"pair {pair_id} needs exactly two accepted points")
            a, b = group
            model = fits[joint] or (candidate if joint == "hip" else None)
            diff = b["segment"]["mean"] - a["segment"]["mean"]
            same_angle = a.get("angle_deg") is not None and a.get("angle_deg") == b.get("angle_deg")
            pairs.append({"joint": joint, "pair_id": pair_id, "point_ids": [a["id"], b["id"]],
                          "difference_counts": float(diff),
                          "difference_deg": None if model is None else float(diff * model["flexion_sign"] / model["counts_per_degree"]),
                          "same_measured_angle": same_angle,
                          "interpretation": "same-angle repeat difference" if same_angle else "return difference only; actual angles differ or are unknown"})
    return {"status": "PROVISIONAL_OFFLINE_ONLY", "candidate": candidate, "points": points,
            "setup": manifest.get("setup", {}),
            "fits": fits, "pairs": pairs, "stability_thresholds": {
                "max_range_counts": max_range, "max_slope_counts_per_s": max_slope,
                "min_duration_s": min_duration, "max_gap_s": max_gap},
            "note": "No motion configuration or safety limits. Two-point residuals are not independent accuracy validation. Missing knee calibration stays null."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    q = commands.add_parser("quality")
    q.add_argument("inputs", type=Path, nargs="+")
    q.add_argument("--output-dir", type=Path, required=True)
    q.add_argument("--expected-hz", type=positive, default=500)
    q.add_argument("--gap-factor", type=positive, default=2)
    q.add_argument("--window-s", type=positive, default=1)
    q.add_argument("--max-range", type=positive, default=50)
    q.add_argument("--max-slope", type=positive, default=20)
    q.add_argument("--torque-drift-nm", type=positive, default=.2)
    c = commands.add_parser("calibrate")
    c.add_argument("manifest", type=Path)
    c.add_argument("--data-dir", type=Path, required=True)
    c.add_argument("--candidate", type=Path, default=Path(__file__).parent / "calibration/hip_candidate.json")
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = calibrate(manifest, args.data_dir,
                           json.loads(args.candidate.read_text(encoding="utf-8")))
        protected = {args.manifest.resolve(), args.candidate.resolve()}
        protected.update((args.data_dir / p["file"]).resolve() for p in manifest["points"])
        if args.output.resolve() in protected:
            parser.error("output must not overwrite manifest, candidate or source data")
        result["manifest_sha256"] = digest(args.manifest)
        result["candidate_sha256"] = digest(args.candidate)
        write_json(args.output, result)
        print(f"Calibration report: {args.output}")
        return 0
    files = sorted({p.resolve() for root in args.inputs for p in
                    (root.glob("*.csv") if root.is_dir() else [root])})
    if not files:
        parser.error("no CSV inputs")
    summary, failed = [], False
    for path in files:
        output = args.output_dir / f"{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}.json"
        try:
            result = inspect_file(path, expected_hz=args.expected_hz, gap_factor=args.gap_factor,
                                  window_s=args.window_s, max_range=args.max_range,
                                  max_slope=args.max_slope, torque_drift_nm=args.torque_drift_nm)
            write_json(output, result)
            summary.append({"file": path.name, "status": "reported", "report": str(output),
                            "rows": result["rows"], "sample_rate_hz": result.get("sample_rate_hz"),
                            "long_intervals": len(result["long_intervals"]),
                            "missing_values": result["missing_values_by_column"],
                            "receive_time_duplicates": len(result["receive_time_duplicates_csv_lines"]),
                            "secondary_clock_duplicates": {k: len(v["duplicate_csv_lines"]) for k, v in result["secondary_clocks"].items()},
                            "joint_screening": {k: {"torque_drift_flag": v.get("torque_drift_flag"),
                                "unstable_position_windows": sum(not w["stable"] for w in v.get("position_windows", []))}
                                for k, v in result["joint_checks"].items()},
                            "usable_for_dynamics": result["usable_for_dynamics"]})
        except (ValueError, OSError, KeyError) as exc:
            failed = True
            summary.append({"file": path.name, "status": "error", "error": str(exc)})
    write_json(args.output_dir / "summary.json", summary)
    lines = ["# Offline acquisition quality", "", "A generated report is not a dynamics approval.", ""]
    for item in summary:
        lines.append(f"- {item['file']}: " + (f"{item['rows']} rows; {item['sample_rate_hz']} Hz; {item['long_intervals']} long intervals" if item["status"] == "reported" else f"ERROR: {item['error']}"))
        if item["status"] == "reported":
            lines.append(f"  Receive duplicates: {item['receive_time_duplicates']}; secondary-clock duplicates: {item['secondary_clock_duplicates']}; missing values: {item['missing_values']}; joint screening: {item['joint_screening']}")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reported {len(summary)} files; errors: {sum(x['status'] == 'error' for x in summary)}; {args.output_dir}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
