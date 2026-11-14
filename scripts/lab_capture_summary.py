"""Inspect completed ADS capture evidence without opening a device.

Recording completion and descriptive quality findings are separate. This module
does not declare dynamic validity, approve motion or change any threshold.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def summarize_capture(directory):
    directory = Path(directory).resolve(strict=True)
    segment = _load(directory / "segment_status.json")
    quality = _load(directory / "quality.json")
    raw = directory / "leg.csv"
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if segment.get("status") != "CAPTURE_COMPLETE" or segment.get("motion_commanded") is not False:
        raise ValueError("capture did not confirm a complete observation-only recording")
    if segment.get("csv_sha256") != digest or quality.get("sha256") != digest:
        raise ValueError("capture/quality evidence does not bind the current raw CSV")
    rows = quality.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 2 or segment.get("rows") != rows:
        raise ValueError("capture/quality row counts are missing or inconsistent")
    findings = quality.get("findings")
    if not isinstance(findings, list) or any(not isinstance(x, str) for x in findings):
        raise ValueError("descriptive quality findings are missing")
    return dict(recording_complete=True, rows=rows, raw_csv=str(raw), raw_csv_sha256=digest,
                sample_rate_hz=quality.get("sample_rate_hz"),
                long_intervals=len(quality.get("long_intervals", [])),
                duplicate_timestamps=len(quality.get("receive_time_duplicates_csv_lines", [])),
                findings=findings, usable_for_dynamics=quality.get("usable_for_dynamics") is True,
                quality_report=str(directory / "quality.json"), motion_approved=False,
                note="Recording completion is separate from measurement quality; raw uncalibrated angle fields and host timing findings require review. This summary does not authorize dynamics or motion.")
