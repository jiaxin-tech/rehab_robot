"""Record state during supervised OEM drag teaching; never replay a trajectory.

Only observation adapter connection/state interfaces are used. Vendor SDK
initialization, connection and disconnection can still affect a controller
session; this is not a promise of zero vendor-side effects.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "row_index", "row_kind", "host_time_s", "query_start_s", "query_end_s",
    "sequence_id", "timestamp_source", "wall_time_iso", "robot_device_time_s",
    "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz",
    "q1", "q2", "q3", "q4", "q5", "q6", "operation_state", "state_age_s",
    "frame_valid", "valid", "invalid_reason", "frame_invalid_reason",
]
POSE = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
SOURCE_FILES = [
    "scripts/record_rehab_trajectory.py", "hardware/rokae_adapter.py",
    "hardware/windows/rokae_xcore.py", "collection/state.py",
    "hardware/safety_events.py", "config/settings.py", "config/bundle.py",
]
OPERATION_STATE_SEMANTICS = (
    "cached adapter operation-state label; not independent verification of OEM drag mode"
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_git():
    def read(*args):
        try:
            result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                    text=True, timeout=5, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": read("rev-parse", "HEAD"),
            "branch": read("branch", "--show-current"),
            "working_tree_status": read("status", "--porcelain")}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, size):
    if value is None:
        return [None] * size
    try:
        values = list(value)
    except TypeError:
        return [None] * size
    return values if len(values) == size else [None] * size


def _adapter(config):
    from hardware.rokae_adapter import RokaeRobotAdapter
    r = config["robot"]
    return RokaeRobotAdapter(
        r["ip"], local_ip=r["local_ip"], robot_class=r["robot_class"],
        collision_source=r["collision_source"],
        collision_event_max_age_s=r.get("collision_event_max_age_s"),
    )


def _announce(message):
    print(message, flush=True)


def _summary(rows):
    states = [r for r in rows if r["row_kind"] == "state"]
    valid = [r for r in states if r["valid"]]
    times = [r["host_time_s"] for r in states if _finite(r["host_time_s"])]
    valid_times = [r["host_time_s"] for r in valid]
    intervals = [b-a for a, b in zip(valid_times, valid_times[1:])]
    def ranges(names):
        return {n: {"min": min(r[n] for r in valid),
                    "max": max(r[n] for r in valid),
                    "range": max(r[n] for r in valid)-min(r[n] for r in valid)}
                for n in names} if valid else {}
    return {
        "rows": len(rows), "state_samples": len(states), "valid_samples": len(valid),
        "invalid_rows": sum(not r["valid"] for r in rows),
        "actual_span_s": max(times)-min(times) if times else 0.0,
        "valid_span_s": valid_times[-1]-valid_times[0] if valid_times else 0.0,
        "interval_s": ({"min": min(intervals), "median": statistics.median(intervals),
                         "max": max(intervals)} if intervals else {}),
        "position_base_m": ranges(POSE[:3]), "orientation_raw_rad": ranges(POSE[3:]),
        "joint_position_rad": ranges([f"q{i}" for i in range(1, 7)]),
        "orientation_range_note": "Raw Euler component ranges; no unwrapping or correction.",
        "usable_for_preview": len(valid) >= 2,
    }


def record(config_path, session, *, duration_s=90.0, live=False,
           adapter_factory=None, clock=time.perf_counter, sleep=time.sleep,
           announce=_announce):
    """Capture raw state into a new directory, retaining failures and cancellation."""
    if live is not True:
        raise PermissionError("recording requires --live; prepare OEM drag teaching manually")
    if not _finite(duration_s) or not 0 < duration_s <= 300:
        raise ValueError("duration_s must be finite and in (0, 300]")
    config_path = Path(config_path).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8-sig"))
    robot = config["robot"]
    for key in ("ip", "local_ip", "robot_class", "collision_source"):
        if not isinstance(robot.get(key), str) or not robot[key].strip():
            raise ValueError("robot." + key + " must be configured")
    max_age = config["motion"]["max_state_age_s"]
    if not _finite(max_age) or max_age <= 0:
        raise ValueError("motion.max_state_age_s must be positive")
    directory = Path(session).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config_snapshot.json").write_bytes(config_bytes)
    state = dict(schema_version=1, status="PREPARING", partial=True,
                 created_at=datetime.now(timezone.utc).isoformat(),
                 requested_duration_s=float(duration_s), max_state_age_s=max_age,
                 ready_for_replay=False, duplicate_polls=0, stale_polls=0,
                 operation_state_semantics=OPERATION_STATE_SEMANTICS,
                 events=[], cleanup=[], error=None,
                 project_commands_sent={"motion": False, "power": False,
                                        "mode": False, "drag": False,
                                        "calibration": False, "wrench": False, "ADS": False})
    manifest = dict(schema_version=1, config_path=str(config_path),
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    config_snapshot_sha256=_sha(directory / "config_snapshot.json"),
                    source_git=_source_git(),
                    source_sha256={name: _sha(ROOT / name) for name in SOURCE_FILES
                                   if (ROOT / name).is_file()},
                    output_files_sha256={}, ready_for_replay=False,
                    units={"position": "base_m", "orientation": "euler_xyz_rad",
                           "joints": "rad"},
                    timestamp_source="host_receive_and_query_times_no_robot_device_clock",
                    operation_state_semantics=OPERATION_STATE_SEMANTICS,
                    operation="OEM drag teaching selected manually; state recording only",
                    vendor_session_side_effects="SDK initialization/connect/disconnect may have controller/session side effects.")
    status_path = directory / "recording_status.json"
    _write_json(status_path, state)
    _write_json(directory / "recording_manifest.json", manifest)
    rows, seen_sequences, seen_times = [], set(), set()
    adapter, capture = None, False
    started = None
    last_receive = None
    last_sequence = None

    def event(kind, query_start, query_end, **values):
        state["events"].append(dict(kind=kind, query_start_s=query_start,
                                    query_end_s=query_end, **values))

    with (directory / "recorded_tcp.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()

        def append(row):
            row["row_index"] = len(rows)
            writer.writerow(row)
            stream.flush()
            rows.append(row)

        def accept(frame, query_start, query_end):
            nonlocal last_receive, last_sequence
            receive, sequence = frame.host_monotonic_time_s, frame.sequence_id
            reason = [frame.invalid_reason] if frame.invalid_reason else []
            age = query_end-receive if _finite(receive) else None
            if age is None or age < 0 or age > max_age:
                reason.append("state_stale_or_host_time_invalid")
                state["stale_polls"] += 1
                event("stale_state", query_start, query_end,
                      sequence_id=sequence, host_time_s=receive if _finite(receive) else None)
            duplicate = ((sequence in seen_sequences) or
                         (_finite(receive) and receive in seen_times))
            if duplicate:
                state["duplicate_polls"] += 1
                event("duplicate_state", query_start, query_end,
                      sequence_id=sequence, host_time_s=receive if _finite(receive) else None,
                      frame_valid=frame.valid, invalid_reason=";".join(reason))
                return
            seen_sequences.add(sequence)
            if type(sequence) is not int or sequence < 0:
                reason.append("state_sequence_invalid")
            elif last_sequence is not None and sequence <= last_sequence:
                reason.append("state_sequence_not_increasing")
            if type(sequence) is int:
                last_sequence = max(last_sequence, sequence) if last_sequence is not None else sequence
            if _finite(receive):
                seen_times.add(receive)
                if last_receive is not None and receive <= last_receive:
                    reason.append("state_host_time_not_increasing")
                last_receive = max(last_receive, receive) if last_receive is not None else receive
            else:
                reason.append("state_host_time_unavailable")
            position = _vector(frame.tcp_position_m, 3)
            orientation = _vector(frame.tcp_orientation_rad, 3)
            joints = _vector(frame.joint_position_rad, 6)
            if not all(_finite(v) for v in position+orientation+joints):
                reason.append("state_pose_or_joints_unavailable_or_nonfinite")
            if frame.valid is not True:
                reason.append("frame_invalid")
            row = dict(row_kind="state", host_time_s=receive,
                       query_start_s=query_start, query_end_s=query_end, sequence_id=sequence,
                       timestamp_source="host_receive_time_no_robot_device_timestamp",
                       wall_time_iso=frame.wall_time_iso, robot_device_time_s=frame.robot_device_time_s,
                       operation_state=frame.operation_state, state_age_s=age,
                       frame_valid=frame.valid, valid=not reason,
                       invalid_reason=";".join(dict.fromkeys(reason)),
                       frame_invalid_reason=frame.invalid_reason)
            row.update(zip(POSE, position+orientation))
            row.update(zip([f"q{i}" for i in range(1, 7)], joints))
            append(row)

        try:
            adapter = (adapter_factory or _adapter)(config)
            adapter.connect()
            adapter.start_state_stream()
            capture = adapter.begin_state_capture() is True
            started = clock()
            deadline = started + duration_s
            state.update(status="RECORDING", started_host_time_s=started,
                         retained_state_capture=capture)
            _write_json(status_path, state)
            announce(f"RECORDING {duration_s:g} seconds; move only using OEM drag teaching; Ctrl+C keeps a partial recording.")
            while clock() < deadline:
                query_start = clock()
                try:
                    frames = adapter.drain_state_frames() if capture else ()
                    if not frames:
                        frames = (adapter.read_state_frame(),)
                    query_end = clock()
                    for frame in frames:
                        accept(frame, query_start, query_end)
                except Exception as exc:
                    query_end = clock()
                    append(dict(row_kind="read_error", query_start_s=query_start,
                                query_end_s=query_end, frame_valid=False, valid=False,
                                invalid_reason=f"{type(exc).__name__}:{exc}"))
                    raise
                sleep(min(.005, max(0., deadline-clock())))
            state.update(status="COMPLETE", partial=False)
        except KeyboardInterrupt:
            state.update(status="CANCELLED", partial=True, error="operator_keyboard_interrupt")
        except Exception as exc:
            state.update(status="FAILED", partial=True, error=f"{type(exc).__name__}:{exc}")
        finally:
            if adapter is not None:
                # No power, mode, drag or motion cleanup: the operator owns the
                # OEM teaching mode. Stop receive-only state and disconnect.
                actions = [("stop_state_stream", adapter.stop_state_stream)]
                if capture:
                    def drain_tail():
                        begin = clock()
                        frames = adapter.drain_state_frames()
                        end = clock()
                        for frame in frames:
                            accept(frame, begin, end)
                    actions.extend([("drain_state_tail", drain_tail),
                                    ("end_state_capture", adapter.end_state_capture)])
                actions.append(("disconnect", adapter.disconnect))
                for name, action in actions:
                    try:
                        action()
                        state["cleanup"].append({"action": name, "ok": True})
                    except BaseException as exc:
                        state["cleanup"].append({"action": name, "ok": False,
                                                 "error": f"{type(exc).__name__}:{exc}"})
                        detail = f"cleanup_failed:{name}:{type(exc).__name__}:{exc}"
                        state.update(status="FAILED", partial=True,
                                     error=state["error"] + ";" + detail if state["error"] else detail)
            state["recording_elapsed_s"] = clock()-started if started is not None else 0.0
    state["summary"] = _summary(rows)
    if state["status"] == "COMPLETE" and not state["summary"]["usable_for_preview"]:
        state.update(status="FAILED", partial=True, error="fewer_than_two_valid_state_samples")
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(status_path, state)
    manifest["output_files_sha256"] = {name: _sha(directory / name)
                                        for name in ("recorded_tcp.csv", "recording_status.json")}
    _write_json(directory / "recording_manifest.json", manifest)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment.json")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=90.)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    state = record(args.config, args.session, duration_s=args.duration_s, live=args.live)
    # Detailed duplicate/query events stay in the status file instead of
    # flooding the operator's terminal after a long recording.
    print(json.dumps({"status": state["status"], "session": str(args.session.resolve()),
                      "partial": state["partial"], "error": state["error"],
                      "summary": state["summary"], "cleanup": state["cleanup"],
                      "ready_for_replay": False},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if state["status"] == "COMPLETE" else 130 if state["status"] == "CANCELLED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
