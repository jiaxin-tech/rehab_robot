"""Review saved stationary robot data offline; never authorize robot motion."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid(frame):
    return frame.valid.astype(str).str.lower().eq("true").to_numpy()


def _clock(values):
    time = np.asarray(values, dtype=float)
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("requires at least two finite, strictly increasing timestamps")
    gaps = np.diff(time)
    return dict(span_s=float(time[-1] - time[0]),
                effective_rate_hz=float((len(time) - 1) / (time[-1] - time[0])),
                maximum_gap_ms=float(gaps.max() * 1000),
                gaps_over_40ms=int(np.count_nonzero(gaps > .04)))


def review(session, output_dir):
    session, output = Path(session).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("choose a new output directory; preserve prior reviews")
    paths = [session / "round_status.json", session / "robot/metadata.json"]
    paths += [session / "robot" / name for name in (
        "robot_state.csv", "robot_wrench.csv", "aligned_snapshot.csv",
        "trajectory_command.csv", "wrench_query_events.jsonl")]
    before = {str(path.relative_to(session)): _sha(path) for path in paths}
    status, metadata = (_read(path) for path in paths[:2])
    state, wrench, aligned, commands = (pd.read_csv(path) for path in paths[2:6])
    events = [json.loads(line) for line in paths[6].read_text(encoding="utf-8").splitlines()]
    xyz = state[["tcp_x", "tcp_y", "tcp_z"]].to_numpy(float)
    rpy = state[["tcp_rx", "tcp_ry", "tcp_rz"]].to_numpy(float)
    force = wrench[["fx", "fy", "fz"]].to_numpy(float)
    torque = wrench[["tx", "ty", "tz"]].to_numpy(float)
    if not all(np.isfinite(value).all() for value in (xyz, rpy, force, torque)):
        raise ValueError("nonfinite pose or wrench values; review raw data")
    drift_mm = np.linalg.norm(xyz - xyz[0], axis=1) * 1000
    rotation = Rotation.from_euler("xyz", rpy)
    drift_deg = (rotation[0].inv() * rotation).magnitude() * 180 / np.pi
    force_norm, torque_norm = np.linalg.norm(force, axis=1), np.linalg.norm(torque, axis=1)
    alignment_valid = _valid(aligned)
    good = np.flatnonzero(alignment_valid)
    if not len(good):
        raise ValueError("no valid aligned samples")
    first = int(good[0])
    midpoint = wrench[["query_start_s", "query_end_s"]].mean(axis=1).to_numpy()
    cleanup = [event["outcome"] for event in events if event["type"] == "CLEANUP_OUTCOME"]
    config_path = session.parent / "observation_config.json"
    tool = metadata.get("robot_state_summary", {}).get("robot_metadata", {}).get("sdk_tool_payload", {})
    result = dict(
        schema_version=1, reviewed_utc=datetime.now(timezone.utc).isoformat(),
        session=str(session), hardware_accessed_by_review=False,
        scope="Stationary acquisition and SDK-reported pose/wrench only; no physical path or interaction-force calibration established",
        source_sha256=before, reviewer_sha256=_sha(Path(__file__).resolve()),
        observation_config_sha256=_sha(config_path) if config_path.is_file() else None,
        reported_status=status["status"], motion_commanded=status.get("motion_commanded"),
        command_rows=len(commands),
        state=dict(rows=len(state), valid_rows=int(_valid(state).sum()),
                   clock=_clock(state.host_time_s), mean_xyz_m=xyz.mean(axis=0).tolist(),
                   mean_rpy_rad=rpy.mean(axis=0).tolist(),
                   max_position_change_from_first_mm=float(drift_mm.max()),
                   max_orientation_change_from_first_deg=float(drift_deg.max())),
        wrench=dict(rows=len(wrench), valid_rows=int(_valid(wrench).sum()),
                    frame_types=sorted(wrench.frame_type.unique().tolist()),
                    query_start_clock=_clock(wrench.query_start_s),
                    query_midpoint_clock=_clock(midpoint), publish_clock=_clock(wrench.publish_time_s),
                    force_mean_components_n=force.mean(axis=0).tolist(),
                    force_norm_mean_n=float(force_norm.mean()), force_norm_max_n=float(force_norm.max()),
                    torque_mean_components_nm=torque.mean(axis=0).tolist(),
                    torque_norm_mean_nm=float(torque_norm.mean()), torque_norm_max_nm=float(torque_norm.max())),
        alignment=dict(rows=len(aligned), valid_rows=int(alignment_valid.sum()),
                       startup_invalid_rows=first,
                       time_until_first_valid_s=float(aligned.host_time_s.iloc[first] - aligned.host_time_s.iloc[0]),
                       invalid_rows_after_first_valid=int((~alignment_valid[first:]).sum()),
                       valid_clock=_clock(aligned.host_time_s.to_numpy()[alignment_valid])),
        query_event_counts=dict(Counter(event["type"] for event in events)),
        cleanup_outcomes=cleanup,
        sdk_tool_payload=tool,
        automatic_motion_authorized=False, calibrated_interaction_force_established=False,
        leg_state_or_brake_observed=False,
        interpretation=[
            "Pose changes describe SDK-reported stationary variation, not absolute physical accuracy",
            "Startup invalid samples must not be counted as an interruption after readiness",
            "Raw world wrench is not independently calibrated interaction force; do not tare away this loaded baseline",
            "Zero SDK tool/payload values do not independently identify the active HMI tool, workobject or physical fixture",
            "No knee boundary, release state or coupled position/orientation path was validated",
            "Logger healthy/ready can be false after normal close; assess completion, failure fields and cleanup events",
        ])
    if before != {str(path.relative_to(session)): _sha(path) for path in paths}:
        raise RuntimeError("raw inputs changed during review")
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    origin = float(state.host_time_s.iloc[0])
    axes[0].plot(state.host_time_s - origin, drift_mm * 1000, color="#247e9b")
    axes[0].set_ylabel("SDK position change (um)")
    for index, label in enumerate("XYZ"):
        axes[1].plot(midpoint - origin, force[:, index], label=label, linewidth=1)
        axes[2].plot(midpoint - origin, torque[:, index], label=label, linewidth=1)
    axes[1].set_ylabel("Raw world force (N)")
    axes[2].set_ylabel("Raw world torque (Nm)")
    for axis in axes:
        axis.grid(alpha=.25)
    axes[1].legend(ncol=3, loc="upper right")
    axes[2].legend(ncol=3, loc="upper right")
    axes[2].set_xlabel("Time from first robot state (s), including wrench startup")
    figure.suptitle("Connected stationary observation | no motion commands\nSDK pose variation and raw wrench; physical path unverified", fontsize=11)
    figure_path = output / "stationary_pose_wrench.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    result["figure_sha256"] = _sha(figure_path)
    (output / "review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = review(args.session, args.output_dir)
    print(json.dumps({key: result[key] for key in ("reported_status", "command_rows", "state", "wrench", "alignment")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
