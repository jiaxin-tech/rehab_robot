"""Offline review of a saved OEM drag-teaching recording; never authorize replay."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GAP_SCREEN_S = .1
CLOSURE_SCREEN_M = .005
ORIENTATION_SCREEN_DEG = 5.
POSE_FIELDS = ("tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz")
REQUIRED_FIELDS = ("row_index", "row_kind", "host_time_s", "query_start_s", "query_end_s",
                   "sequence_id", "frame_valid", "valid", "invalid_reason", *POSE_FIELDS)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required: " + str(path))
    return value


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _boolean(value, field):
    if str(value).casefold() not in ("true", "false"):
        raise ValueError("explicit true/false required for " + field)
    return str(value).casefold() == "true"


def _rotation_matrices(rpy):
    """Euler xyz / RPY, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    rx, ry, rz = rpy.T
    sx, sy, sz, cx, cy, cz = np.sin(rx), np.sin(ry), np.sin(rz), np.cos(rx), np.cos(ry), np.cos(rz)
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]]).transpose(2, 0, 1)


def _orientation_angles_deg(rpy):
    rotations = _rotation_matrices(rpy)
    relative = rotations[0].T @ rotations
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))


def _verify_inputs(session):
    manifest = _read_json(session / "recording_manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("recording manifest schema_version must be 1")
    if manifest.get("units") != {"position": "base_m", "orientation": "euler_xyz_rad", "joints": "rad"}:
        raise ValueError("recording must explicitly declare base metres and Euler xyz radians")
    hashes = manifest.get("output_files_sha256")
    if not isinstance(hashes, dict) or not {"recorded_tcp.csv", "recording_status.json"} <= hashes.keys():
        raise ValueError("manifest must bind recorded_tcp.csv and recording_status.json")
    verified = {}
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or any(c in name for c in "/\\:"):
            raise ValueError("manifest output must name a local file without a path")
        path = session / name
        if path.resolve().parent != session:
            raise ValueError("manifest output resolves outside the recording directory")
        actual = _sha(path)
        if actual != expected:
            raise ValueError("recording output SHA-256 mismatch: " + name)
        verified[name] = actual
    snapshot = session / "config_snapshot.json"
    snapshot_sha = _sha(snapshot)
    if snapshot_sha != manifest.get("config_snapshot_sha256") or snapshot_sha != manifest.get("config_sha256"):
        raise ValueError("recorded configuration snapshot SHA-256 mismatch")
    status = _read_json(session / "recording_status.json")
    if status.get("status") not in ("PREPARING", "RECORDING", "COMPLETE", "FAILED", "CANCELLED"):
        raise ValueError("unknown recording status")
    with (session / "recorded_tcp.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(REQUIRED_FIELDS) <= set(reader.fieldnames or ()):
            raise ValueError("recorded TCP columns are missing")
        rows = list(reader)
    return manifest, status, rows, verified, snapshot_sha


def _samples(rows):
    selected, dropped = [], []
    all_state_times = []
    seen_sequences = set()
    for number, row in enumerate(rows):
        if int(row["row_index"]) != number:
            raise ValueError("row_index must follow original zero-based CSV order")
        valid, frame_valid = _boolean(row["valid"], "valid"), _boolean(row["frame_valid"], "frame_valid")
        if row["row_kind"] == "state" and row["host_time_s"].strip():
            stamp = float(row["host_time_s"])
            if not np.isfinite(stamp):
                raise ValueError("state receive time must be finite")
            all_state_times.append(stamp)
        if not valid:
            dropped.append(dict(csv_line=number+2, reason=row["invalid_reason"] or row["row_kind"]))
            continue
        if row["row_kind"] != "state" or not frame_valid or row["invalid_reason"].strip():
            raise ValueError("valid row conflicts with state/frame validity or invalid_reason")
        pose = np.array([float(row[field]) for field in POSE_FIELDS])
        times = np.array([float(row[field]) for field in ("host_time_s", "query_start_s", "query_end_s")])
        if not np.isfinite(pose).all() or not np.isfinite(times).all():
            raise ValueError("valid pose and query/receive times must be finite")
        if times[1] > times[2] or times[0] > times[2]:
            raise ValueError("query/receive timing is inconsistent")
        if row["sequence_id"] in seen_sequences or not row["sequence_id"].strip():
            raise ValueError("valid state sequence_id is missing or repeated")
        seen_sequences.add(row["sequence_id"])
        selected.append(dict(index=number, time=float(times[0]), pose=pose))
    if len(all_state_times) > 1 and np.any(np.diff(all_state_times) <= 0):
        raise ValueError("recorded state times are duplicated or out of order; never sort or repair")
    if len(selected) < 2:
        raise ValueError("at least two valid state samples are needed for preview")
    return selected, dropped


def _segments(samples):
    boundaries = [0]
    for index in range(1, len(samples)):
        if (samples[index]["index"] != samples[index-1]["index"]+1
                or samples[index]["time"]-samples[index-1]["time"] > GAP_SCREEN_S):
            boundaries.append(index)
    boundaries.append(len(samples))
    return [slice(a, b) for a, b in zip(boundaries, boundaries[1:])]


def _plot(output, samples, summary):
    time = np.array([sample["time"] for sample in samples])-samples[0]["time"]
    pose = np.array([sample["pose"] for sample in samples])
    xyz, relative = pose[:, :3], pose[:, :3]-pose[0, :3]
    parts, candidate = _segments(samples), summary["max_z_candidate"]["valid_sample_index"]
    banner = ("SYNTHETIC QA | " if summary.get("synthetic_input") else "") + summary["status"] + " | OFFLINE REVIEW ONLY | NOT APPROVED FOR REPLAY"
    paths = []

    def plot_parts(axis, values, x=None):
        for part in parts:
            if x is None:
                axis.plot(*values[part].T, color="#246a9a", linewidth=1.2, marker="." if len(values[part]) == 1 else None)
            else:
                axis.plot(x[part], values[part], color="#246a9a", linewidth=1.2, marker="." if len(values[part]) == 1 else None)

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig = plt.figure(figsize=(12, 5.8), constrained_layout=True)
    span = max(float(np.ptp(xyz, axis=0).max())*1.12, .01)
    for index, (values, label) in enumerate(((xyz, "Absolute base coordinates"), (relative, "Relative to first valid point")), 1):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        plot_parts(axis, values)
        axis.scatter(*values[0], color="green", label="Start", s=35)
        axis.scatter(*values[-1], color="red", label="End", s=25)
        axis.scatter(*values[candidate], color="#b47200", marker="^", label="Max Z candidate (unconfirmed)", s=50)
        center = (values.min(axis=0)+values.max(axis=0))/2
        axis.set(xlim=(center[0]-span/2, center[0]+span/2), ylim=(center[1]-span/2, center[1]+span/2),
                 zlim=(center[2]-span/2, center[2]+span/2), xlabel="Base X (m)", ylabel="Base Y (m)", zlabel="Base Z (m)", title=label)
        axis.set_box_aspect((1, 1, 1))
        axis.legend(fontsize=7, loc="upper left")
    fig.suptitle(banner, fontsize=10)
    save(fig, "trajectory_3d.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, values in enumerate((xyz, relative)):
        for axis, (a, b) in zip(axes[row], ((0, 1), (0, 2), (1, 2))):
            plot_parts(axis, values[:, [a, b]])
            axis.scatter(*values[0, [a, b]], color="green", s=25)
            axis.scatter(*values[-1, [a, b]], color="red", s=20)
            axis.scatter(*values[candidate, [a, b]], color="#b47200", marker="^", s=35)
            axis.set(xlabel=f"Base {'XYZ'[a]} (m)", ylabel=f"Base {'XYZ'[b]} (m)", title="Absolute" if row == 0 else "Relative to start")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=.25)
    fig.suptitle(banner, fontsize=10)
    save(fig, "projections.png")

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for index, label in enumerate("XYZ"):
        for column, values in enumerate((xyz, relative)):
            axis = axes[index, column]
            plot_parts(axis, values[:, index], time)
            axis.axvline(time[candidate], color="#b47200", linestyle=":", linewidth=.9)
            axis.set(ylabel=f"Base {label} (m)", title=("Absolute" if column == 0 else "Relative to first valid point") if index == 0 else "")
            axis.grid(alpha=.25)
    for axis in axes[-1]:
        axis.set_xlabel("Time from first valid state (s)")
    fig.suptitle(banner+"\nGaps and invalid intervals are not connected", fontsize=10)
    save(fig, "xyz_time.png")

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    wrapped = np.degrees(np.arctan2(np.sin(pose[:, 3:]-pose[0, 3:]), np.cos(pose[:, 3:]-pose[0, 3:])))
    for index, label in enumerate(("Roll", "Pitch", "Yaw")):
        plot_parts(axes[index], wrapped[:, index], time)
        axes[index].set_ylabel(label+" delta (deg)")
    plot_parts(axes[3], _orientation_angles_deg(pose[:, 3:]), time)
    axes[3].set(ylabel="Rotation angle (deg)", xlabel="Time from first valid state (s)")
    axes[3].axhline(ORIENTATION_SCREEN_DEG, color="red", linestyle=":", linewidth=.9)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle(banner+"\nOrientation change from first valid state; 5 deg is a review flag", fontsize=10)
    save(fig, "orientation_time.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    plot_parts(axes[0], np.linalg.norm(relative, axis=1)*1000, time)
    axes[0].axhline(CLOSURE_SCREEN_M*1000, color="red", linestyle=":", linewidth=.9)
    axes[0].scatter(time[-1], summary["position_closure_error_m"]*1000, color="red", s=30)
    axes[0].set(ylabel="Distance from start (mm)",
        title=f"End closure: {summary['position_closure_error_m']*1000:.3f} mm; 5 mm is a review flag")
    plot_parts(axes[1], _orientation_angles_deg(pose[:, 3:]), time)
    axes[1].scatter(time[-1], summary["orientation_closure_error_deg"], color="red", s=30)
    axes[1].set(ylabel="Rotation from start (deg)", xlabel="Time from first valid state (s)",
        title=f"End orientation difference: {summary['orientation_closure_error_deg']:.3f} deg")
    for axis in axes:
        axis.axvline(time[candidate], color="#b47200", linestyle=":", linewidth=.9)
        axis.grid(alpha=.25)
    fig.suptitle(banner, fontsize=10)
    save(fig, "closure_review.png")
    return paths


def preview(session, output_dir):
    """Write new review artifacts; never alter, rescale, sort, repair or replay input."""
    session, output = Path(session).expanduser().resolve(strict=True), Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=1, status="FAILED", ok=False, ready_for_replay=False,
        hardware_accessed=False, session=str(session), output_dir=str(output),
        input_modified=False, coordinates_rescaled=False, samples_interpolated=False,
        branch_split_confirmed=False, created_utc=datetime.now(timezone.utc).isoformat(),
        screen_thresholds=dict(gap_s=GAP_SCREEN_S, position_closure_m=CLOSURE_SCREEN_M,
                               orientation_change_deg=ORIENTATION_SCREEN_DEG),
        threshold_interpretation="Offline review flags only; never motion/replay approval limits.")
    try:
        manifest, status, rows, hashes, config_sha = _verify_inputs(session)
        samples, dropped = _samples(rows)
        pose = np.array([sample["pose"] for sample in samples])
        time = np.array([sample["time"] for sample in samples])
        xyz, relative, dt = pose[:, :3], pose[:, :3]-pose[0, :3], np.diff(time)
        angles = _orientation_angles_deg(pose[:, 3:])
        closure = float(np.linalg.norm(relative[-1]))
        partial = status["status"] != "COMPLETE" or status.get("partial") is not False or bool(dropped)
        flags = []
        if partial:
            flags.append("PARTIAL_RECORDING_OR_INVALID_ROWS")
        if np.any(dt > GAP_SCREEN_S):
            flags.append("TIME_GAPS_EXCEED_REVIEW_SCREEN")
        if closure > CLOSURE_SCREEN_M:
            flags.append("END_POSITION_DIFFERS_FROM_START")
        if angles.max() > ORIENTATION_SCREEN_DEG:
            flags.append("LARGE_ORIENTATION_CHANGE")
        if angles[-1] > ORIENTATION_SCREEN_DEG:
            flags.append("END_ORIENTATION_DIFFERS_FROM_START")
        candidate = int(np.argmax(xyz[:, 2]))
        plateau = np.flatnonzero(np.isclose(xyz[:, 2], xyz[candidate, 2], atol=1e-9, rtol=0))
        summary.update(status="PARTIAL" if partial else "PREVIEW_COMPLETE", ok=True,
            synthetic_input=manifest.get("synthetic_input") is True,
            input_hashes_verified=True, recording_status=status["status"], recording_error=status.get("error"),
            recorded_units=manifest["units"], recorded_source_sha256=manifest.get("source_sha256"),
            recorded_source_git=manifest.get("source_git"),
            recording_manifest_sha256=_sha(session / "recording_manifest.json"),
            verified_output_sha256=hashes, config_snapshot_sha256=config_sha,
            total_csv_rows=len(rows), valid_sample_count=len(samples), discarded_rows=dropped,
            first_valid_host_time_s=float(time[0]), valid_span_s=float(time[-1]-time[0]),
            sample_interval_s=dict(min=float(dt.min()), median=float(np.median(dt)), max=float(dt.max())),
            gaps=[dict(previous_csv_line=samples[i]["index"]+2, next_csv_line=samples[i+1]["index"]+2,
                       elapsed_s=float(dt[i])) for i in np.flatnonzero(dt > GAP_SCREEN_S)],
            plotted_contiguous_segments=len(_segments(samples)), start_pose_base_m_rad=pose[0].tolist(),
            end_pose_base_m_rad=pose[-1].tolist(),
            absolute_base_xyz_m={a:dict(min=float(xyz[:, i].min()), max=float(xyz[:, i].max()),
                                       range=float(np.ptp(xyz[:, i]))) for i, a in enumerate("xyz")},
            relative_to_first_valid_xyz_m={a:dict(min=float(relative[:, i].min()), max=float(relative[:, i].max())) for i, a in enumerate("xyz")},
            max_distance_from_start_m=float(np.linalg.norm(relative, axis=1).max()),
            position_closure_error_m=closure, orientation_closure_error_deg=float(angles[-1]),
            maximum_orientation_change_deg=float(angles.max()),
            max_z_candidate=dict(classification="MAX_Z_CANDIDATE_UNCONFIRMED", valid_sample_index=candidate,
                csv_line=samples[candidate]["index"]+2, time_from_first_valid_s=float(time[candidate]-time[0]),
                xyz_base_m=xyz[candidate].tolist(), relative_xyz_m=relative[candidate].tolist(),
                equal_max_z_sample_count=len(plateau),
                interpretation="Maximum height is a candidate only; motion phases and intended turning point require human review."),
            review_flags=flags,
            limitations=["Host receive times do not establish device timing or force synchronization.",
                         "Partial data, closure and orientation flags do not authorize any replay.",
                         "No anatomical geometry, rescaling, phase segmentation or motion commands are generated."])
        paths = _plot(output, samples, summary)
        summary["generated_files_sha256"] = {path.name: _sha(path) for path in paths}
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        summary.update(status="FAILED", ok=False, error=f"{type(exc).__name__}: {exc}")
    _write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory only")
    args = parser.parse_args(argv)
    summary = preview(args.session, args.output_dir)
    print(json.dumps({key: summary.get(key) for key in ("status", "ready_for_replay", "review_flags", "error")}, ensure_ascii=False))
    print(str(args.output_dir.resolve() / "summary.json"))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
