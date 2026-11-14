"""Compare a closed empty-load recording with its frozen TCP plan, offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def review(session, output_dir):
    session, output_dir = Path(session).resolve(), Path(output_dir).resolve()
    config = json.loads((session / "prepared_config.json").read_text(encoding="utf-8"))
    state = json.loads((session / "round_status.json").read_text(encoding="utf-8"))
    metadata = json.loads((session / "robot/metadata.json").read_text(encoding="utf-8"))
    plan = pd.read_csv(session / "tcp_trajectory.csv")
    robot = pd.read_csv(session / "robot/robot_state.csv")
    commands = pd.read_csv(session / "robot/trajectory_command.csv")
    wrench = pd.read_csv(session / "robot/robot_wrench.csv")
    snapshots = pd.read_csv(session / "robot/aligned_snapshot.csv")
    origin = float(metadata["execution_result"]["schedule_origin_host_time_s"])
    times = plan.time_s.to_numpy(float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("plan times must be strictly ordered")
    target = plan[[f"tcp_{axis}_base" for axis in "xyz"]].to_numpy(float)
    received = robot.host_time_s.to_numpy(float) - origin
    actual = robot[[f"tcp_{axis}" for axis in "xyz"]].to_numpy(float)
    valid = robot.valid.astype(str).str.lower().eq("true").to_numpy() & np.isfinite(actual).all(axis=1)
    chosen = valid & (received >= times[0]) & (received <= times[-1])
    t, xyz = received[chosen], actual[chosen]
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("insufficient ordered measured state in the scheduled interval")
    expected = np.column_stack([np.interp(t, times, target[:, axis]) for axis in range(3)])
    error_mm = np.linalg.norm(xyz - expected, axis=1) * 1000
    peak = int(np.argmax(target[:, 2]))
    command_pose = commands[[f"tcp_target_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")]].to_numpy(float)
    planned_pose = plan[[f"tcp_{axis}_base" for axis in "xyz"] + [f"tcp_r{axis}" for axis in "xyz"]].to_numpy(float)
    command_matches = (command_pose.shape == planned_pose.shape
                       and np.allclose(command_pose, planned_pose, atol=1e-12, rtol=0))
    wclock = wrench[["query_start_s", "query_end_s"]].mean(axis=1).to_numpy(float) - origin
    wvalid = wrench.valid.astype(str).str.lower().eq("true").to_numpy()
    wchosen = wvalid & (wclock >= times[0]) & (wclock <= times[-1])
    stamps = np.r_[times[0], wclock[wchosen], times[-1]]
    gaps = np.diff(stamps)
    limit = float(config["motion"]["max_wrench_age_s"])
    large_gaps = [{"start_s": float(stamps[i]), "end_s": float(stamps[i+1]), "gap_ms": float(gap*1000)}
                  for i, gap in enumerate(gaps) if gap > limit]
    snapshot_clock = snapshots.host_time_s.to_numpy(float) - origin
    invalid_motion = ((snapshot_clock >= 0) & (snapshot_clock <= times[-1])
                      & ~snapshots.valid.astype(str).str.lower().eq("true").to_numpy())
    audit = {"passed": False, "error": None}
    try:
        from measurement_validation.robot_execution import audit_robot_execution
        audit["result"] = audit_robot_execution(session, allow_empty=True)
        audit["passed"] = True
    except (ValueError, RuntimeError, OSError, KeyError, AssertionError) as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
    inputs = ("prepared_config.json", "round_status.json", "tcp_trajectory.csv", "dispatch.csv",
              "robot/metadata.json", "robot/robot_state.csv", "robot/robot_wrench.csv",
              "robot/trajectory_command.csv", "robot/aligned_snapshot.csv")
    result = dict(session=str(session), analysis="OFFLINE_RECORDED_PLAN_COMPARISON",
        coordinate_frame="robot Base", position_unit="m", time_alignment="recorded schedule origin; no fitted offset or removed position bias",
        input_sha256={name: hashlib.sha256((session/name).read_bytes()).hexdigest() for name in inputs},
        planned_start_m=target[0].tolist(), planned_peak_m=target[peak].tolist(),
        planned_excursion_m=(target[peak]-target[0]).tolist(), planned_duration_s=float(times[-1]),
        commanded_pose_matches_plan=bool(command_matches), measured_samples_in_cycle=len(t),
        measured_min_m=xyz.min(axis=0).tolist(), measured_max_m=xyz.max(axis=0).tolist(),
        tracking_error_mm={"rms": float(np.sqrt(np.mean(error_mm**2))), "p95": float(np.percentile(error_mm, 95)), "max": float(error_mm.max())},
        last_measured_offset_from_original_anchor_mm=((xyz[-1]-target[0])*1000).tolist(),
        largest_measured_state_interval_ms=float(np.diff(t).max()*1000),
        wrench_gaps_exceeding_limit=large_gaps, wrench_gap_limit_s=limit,
        invalid_aligned_snapshots_during_motion=int(invalid_motion.sum()),
        reported_run_status=state["status"], offline_execution_audit=audit,
        interpretation="Following a frozen plan does not establish that its shape or extent suits rehabilitation. This review neither changes a trajectory nor authorizes replay.",
        hardware_accessed=False, motion_commanded=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(target[:, 1]*100, target[:, 2]*100, "k--", lw=2, label="Frozen plan")
    axes[0, 0].plot(xyz[::8, 1]*100, xyz[::8, 2]*100, lw=1, label="Measured TCP")
    axes[0, 0].scatter(target[[0, peak], 1]*100, target[[0, peak], 2]*100, c=["green", "red"], zorder=4)
    axes[0, 0].set(xlabel="Base Y (cm)", ylabel="Base Z (cm)", title="Spatial path: out and back overlap")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend()
    for axis, label in ((1, "Y"), (2, "Z")):
        axes[0, 1].plot(times, (target[:, axis]-target[0, axis])*100, "--", label=f"{label} plan")
        axes[0, 1].plot(t[::8], (xyz[::8, axis]-target[0, axis])*100, label=f"{label} measured")
    axes[0, 1].set(xlabel="Scheduled time (s)", ylabel="Displacement from anchor (cm)", title="Height and horizontal extent")
    axes[0, 1].legend()
    axes[1, 0].plot(t[::8], error_mm[::8], lw=1)
    axes[1, 0].set(xlabel="Scheduled time (s)", ylabel="Position error (mm)", title="No fitted time or position correction")
    axes[1, 1].plot(stamps[1:], gaps*1000, lw=.7)
    axes[1, 1].axhline(limit*1000, color="red", ls="--", label="Configured gap limit")
    axes[1, 1].set(xlabel="Scheduled time (s)", ylabel="Wrench receive gap (ms)", title="Wrench coverage is a separate check")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.suptitle("Returned empty-load recording: plan, feedback and data coverage")
    fig.savefig(output_dir / "plan_vs_measured.png", dpi=160)
    plt.close(fig)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = review(args.session, args.output_dir)
    print(json.dumps({key: result[key] for key in ("planned_excursion_m", "tracking_error_mm", "offline_execution_audit")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
