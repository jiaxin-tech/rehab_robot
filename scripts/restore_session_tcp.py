"""Prepare and execute a supervised return to a previous session's start TCP.

The prepared path begins at a freshly captured live TCP, projects only onto the
already traversed frozen preview, and follows that preview backwards to its
saved anchor.  Execution uses slow NRT MoveL segments so the controller owns
trajectory generation; every segment is bounded, audited, and preceded and
followed by a host wrench check.  Preparation never commands power or motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from collection.twincat_leg import atomic_json
from config.bundle import BUNDLE_PATH, configuration
from control.start_anchor import load_start_anchor
from hardware.rokae_commissioning import (
    COMMISSIONING_CONFIRMATION,
    CommissioningLimits,
    RokaeCommissioningConsole,
)
from scripts.acquire_robot_data import _configured_adapter_factory
from scripts.capture_start_anchor import run_capture
from scripts.lab_session import check_config


RESTORE_CONFIRMATION = "I CONFIRM SUPERVISED TCP RESTORE"
MAX_CAPTURE_TO_PATH_M = 0.010
MAX_SEGMENT_M = 0.004
MAX_START_DRIFT_M = 0.002
MAX_FINAL_POSITION_ERROR_M = 0.002
MAX_ORIENTATION_ERROR_RAD = 0.02
RESTORE_SPEED_RATIO = 10

AdapterFactory = Callable[[str], Any]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _adapter_factory(config: dict[str, Any]) -> AdapterFactory:
    robot = config["robot"]
    return _configured_adapter_factory(
        local_ip=robot["local_ip"],
        robot_class=robot["robot_class"],
        state_interval_ms=None,
        collision_source=robot["collision_source"],
        collision_event_max_age_s=robot["collision_event_max_age_s"],
    )


def _angular_error(left: Sequence[float], right: Sequence[float]) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return float(np.linalg.norm(wrapped))


def _build_reverse_waypoints(
    current_pose: Sequence[float],
    source_preview: pd.DataFrame,
    target_pose: Sequence[float],
    *,
    max_segment_m: float = MAX_SEGMENT_M,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = np.asarray(current_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if current.shape != (6,) or target.shape != (6,) or not (
        np.isfinite(current).all() and np.isfinite(target).all()
    ):
        raise ValueError("current and target TCP poses must contain six finite values")
    required = {"tcp_x_base", "tcp_y_base", "tcp_z_base"}
    missing = required.difference(source_preview.columns)
    if missing or len(source_preview) < 2:
        raise ValueError("source preview does not contain a usable Cartesian path")
    source_xyz = source_preview[
        ["tcp_x_base", "tcp_y_base", "tcp_z_base"]
    ].to_numpy(dtype=float)
    if not np.isfinite(source_xyz).all():
        raise ValueError("source preview contains non-finite positions")
    if float(np.linalg.norm(source_xyz[0] - target[:3])) > 1e-9:
        raise ValueError("source preview start does not match source anchor")
    distances = np.linalg.norm(source_xyz - current[:3], axis=1)
    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])
    if nearest_distance > MAX_CAPTURE_TO_PATH_M:
        raise PermissionError(
            "current TCP is not close enough to the previously traversed path: "
            f"distance_m={nearest_distance:.6f}"
        )
    if nearest_index < 1:
        raise PermissionError("current TCP is already at the source-session start")
    if _angular_error(current[3:], target[3:]) > MAX_ORIENTATION_ERROR_RAD:
        raise PermissionError("current orientation differs too far from source anchor")

    coarse = np.vstack((current[:3], source_xyz[nearest_index::-1]))
    compact = [coarse[0]]
    for point in coarse[1:]:
        if float(np.linalg.norm(point - compact[-1])) > 1e-9:
            compact.append(point)
    dense = [np.asarray(compact[0], dtype=float)]
    for start, end in zip(compact, compact[1:]):
        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(delta))
        pieces = max(1, int(math.ceil(length / float(max_segment_m))))
        for index in range(1, pieces + 1):
            dense.append(np.asarray(start, dtype=float) + delta * (index / pieces))
    xyz = np.vstack(dense)
    segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    orientation = np.repeat(current[None, 3:], len(xyz), axis=0)
    waypoints = pd.DataFrame(
        {
            "waypoint_index": np.arange(len(xyz), dtype=int),
            "tcp_x_base": xyz[:, 0],
            "tcp_y_base": xyz[:, 1],
            "tcp_z_base": xyz[:, 2],
            "tcp_rx": orientation[:, 0],
            "tcp_ry": orientation[:, 1],
            "tcp_rz": orientation[:, 2],
        }
    )
    summary = {
        "nearest_source_preview_index": nearest_index,
        "capture_to_source_path_distance_m": nearest_distance,
        "waypoint_count": len(waypoints),
        "path_length_m": float(segment_lengths.sum()),
        "maximum_segment_m": float(segment_lengths.max(initial=0.0)),
        "straight_line_distance_m": float(np.linalg.norm(current[:3] - target[:3])),
        "orientation_error_to_source_anchor_rad": _angular_error(
            current[3:], target[3:]
        ),
    }
    return waypoints, summary


def _build_direct_waypoints(
    current_pose: Sequence[float],
    target_pose: Sequence[float],
    *,
    max_segment_m: float = MAX_SEGMENT_M,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a straight, bounded return from any reviewed workspace position.

    This removes the requirement to be close to a previous preview path.  It
    does not perform obstacle avoidance, so the straight Cartesian corridor
    must still be physically clear.  The tool orientation must already match
    the reviewed home orientation closely.
    """

    current = np.asarray(current_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if current.shape != (6,) or target.shape != (6,) or not (
        np.isfinite(current).all() and np.isfinite(target).all()
    ):
        raise ValueError("current and target TCP poses must contain six finite values")
    orientation_error = _angular_error(current[3:], target[3:])
    if orientation_error > MAX_ORIENTATION_ERROR_RAD:
        raise PermissionError(
            "current orientation differs too far from the reviewed home orientation"
        )
    delta = target[:3] - current[:3]
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        raise PermissionError("current TCP is already at the reviewed home position")
    pieces = max(1, int(math.ceil(distance / float(max_segment_m))))
    alpha = np.linspace(0.0, 1.0, pieces + 1)
    xyz = current[None, :3] + alpha[:, None] * delta[None, :]
    # Keep the measured orientation fixed during the return.  The final
    # acceptance tolerance is the same reviewed bound used at preparation.
    orientation = np.repeat(current[None, 3:], len(xyz), axis=0)
    waypoints = pd.DataFrame(
        {
            "waypoint_index": np.arange(len(xyz), dtype=int),
            "tcp_x_base": xyz[:, 0],
            "tcp_y_base": xyz[:, 1],
            "tcp_z_base": xyz[:, 2],
            "tcp_rx": orientation[:, 0],
            "tcp_ry": orientation[:, 1],
            "tcp_rz": orientation[:, 2],
        }
    )
    segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return waypoints, {
        "route_mode": "direct_reviewed_workspace",
        "waypoint_count": len(waypoints),
        "path_length_m": float(segment_lengths.sum()),
        "maximum_segment_m": float(segment_lengths.max(initial=0.0)),
        "straight_line_distance_m": distance,
        "orientation_error_to_source_anchor_rad": orientation_error,
    }


def _audit_waypoints(
    waypoints: pd.DataFrame,
    config: dict[str, Any],
    current_pose: Sequence[float],
    target_pose: Sequence[float],
) -> None:
    pose_columns = [
        "tcp_x_base", "tcp_y_base", "tcp_z_base", "tcp_rx", "tcp_ry", "tcp_rz"
    ]
    poses = waypoints[pose_columns].to_numpy(dtype=float)
    if len(poses) < 2 or not np.isfinite(poses).all():
        raise PermissionError("restore waypoints are missing or non-finite")
    if not np.allclose(poses[0], np.asarray(current_pose), rtol=0.0, atol=1e-10):
        raise PermissionError("restore path is not bound to the captured current TCP")
    if not np.allclose(
        poses[-1, :3], np.asarray(target_pose, dtype=float)[:3], rtol=0.0, atol=1e-9
    ):
        raise PermissionError("restore path does not end at the source-session TCP")
    segment_lengths = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    if bool((segment_lengths > MAX_SEGMENT_M + 1e-12).any()):
        raise PermissionError("restore path contains an oversized segment")
    if not np.allclose(poses[:, 3:], poses[0, 3:], rtol=0.0, atol=1e-12):
        raise PermissionError("restore path attempts to rotate the TCP")
    motion = config["motion"]
    lower = np.asarray(motion["workspace_min_base_m"], dtype=float)
    upper = np.asarray(motion["workspace_max_base_m"], dtype=float)
    if not bool(((poses[:, :3] >= lower) & (poses[:, :3] <= upper)).all()):
        raise PermissionError("restore path leaves the configured workspace")


def prepare(
    session: str | Path,
    *,
    source_session: str | Path,
    config_path: str | Path = BUNDLE_PATH,
    direct: bool = False,
    capture_fn=run_capture,
) -> dict[str, Any]:
    """Capture the current TCP and freeze a reverse-path recovery; no motion."""

    directory = Path(session).resolve()
    source = Path(source_session).resolve()
    resolved_config = Path(config_path).resolve()
    source_anchor_path = source / "anchor.json"
    source_preview_path = source / "preview" / "trajectory_preview.csv"
    if not source_anchor_path.is_file():
        raise FileNotFoundError("source session lacks anchor.json")
    if not direct and not source_preview_path.is_file():
        raise FileNotFoundError("source session lacks frozen preview")
    blockers = check_config(resolved_config, empty_load=True)
    if blockers:
        raise ValueError("configuration blocked: " + "; ".join(blockers))
    config = configuration(resolved_config)
    directory.mkdir(parents=True, exist_ok=False)
    status: dict[str, Any] = {
        "status": "PREPARING",
        "motion_commanded": False,
        "source_session": str(source),
        "config_path": str(resolved_config),
    }
    atomic_json(directory / "status.json", status)
    try:
        robot = config["robot"]
        capture_fn(
            robot_ip=robot["ip"],
            output_path=directory / "current_anchor.json",
            adapter_factory=_adapter_factory(config),
            notes=f"TCP restore start for {source.name}",
        )
        current_anchor = load_start_anchor(directory / "current_anchor.json")
        source_anchor = load_start_anchor(source_anchor_path)
        if direct:
            waypoints, path_summary = _build_direct_waypoints(
                current_anchor.tcp_pose_base,
                source_anchor.tcp_pose_base,
            )
        else:
            source_preview = pd.read_csv(source_preview_path)
            waypoints, path_summary = _build_reverse_waypoints(
                current_anchor.tcp_pose_base,
                source_preview,
                source_anchor.tcp_pose_base,
            )
        _audit_waypoints(
            waypoints,
            config,
            current_anchor.tcp_pose_base,
            source_anchor.tcp_pose_base,
        )
        waypoint_path = directory / "waypoints.csv"
        waypoints.to_csv(waypoint_path, index=False)
        manifest = {
            "schema_version": 1,
            "profile": "direct_tcp_home_v1" if direct else "session_tcp_restore_v1",
            "route_mode": "direct" if direct else "reverse_frozen_preview",
            "source_session": str(source),
            "source_anchor_id": source_anchor.anchor_id,
            "current_anchor_id": current_anchor.anchor_id,
            "target_tcp_pose_base_m_rad": list(source_anchor.tcp_pose_base),
            "path_summary": path_summary,
            "config_sha256": _sha256(resolved_config),
            "current_anchor_sha256": _sha256(directory / "current_anchor.json"),
            "source_anchor_sha256": _sha256(source_anchor_path),
            "waypoints_sha256": _sha256(waypoint_path),
        }
        if not direct:
            manifest["source_preview_sha256"] = _sha256(source_preview_path)
        atomic_json(directory / "manifest.json", manifest)
        status.update(status="PREPARED", manifest=manifest)
    except BaseException as exc:
        status.update(status="FAILED", reason=str(exc))
        raise
    finally:
        atomic_json(directory / "status.json", status)
    return status


def _require_wrench(adapter: Any, *, max_force_n: float, max_torque_nm: float) -> None:
    frame = adapter.read_internal_wrench("world")
    if not getattr(frame, "valid", False):
        raise RuntimeError(
            "restore wrench is unavailable: " + str(getattr(frame, "invalid_reason", ""))
        )
    force = np.asarray(frame.cartesian_force_raw_n, dtype=float)
    torque = np.asarray(frame.cartesian_torque_raw_nm, dtype=float)
    if force.shape != (3,) or torque.shape != (3,) or not (
        np.isfinite(force).all() and np.isfinite(torque).all()
    ):
        raise RuntimeError("restore wrench contains invalid values")
    if float(np.linalg.norm(force)) > max_force_n:
        raise RuntimeError("restore force limit exceeded")
    if float(np.linalg.norm(torque)) > max_torque_nm:
        raise RuntimeError("restore torque limit exceeded")


def run(
    session: str | Path,
    *,
    live: bool,
    confirm_unloaded: bool,
    operator_confirmation: str,
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    directory = Path(session).resolve()
    status_path = directory / "status.json"
    status = _load_json(status_path)
    if status.get("status") != "PREPARED":
        raise ValueError("prepare a new restore session before running")
    if live is not True or confirm_unloaded is not True:
        raise PermissionError("restore requires --live and --confirm-unloaded")
    if operator_confirmation != RESTORE_CONFIRMATION:
        raise PermissionError(
            "restore requires exact operator confirmation: " + RESTORE_CONFIRMATION
        )
    manifest = _load_json(directory / "manifest.json")
    config_path = Path(status["config_path"]).resolve()
    source = Path(status["source_session"]).resolve()
    checks = [
        (config_path, manifest["config_sha256"], "configuration"),
        (directory / "current_anchor.json", manifest["current_anchor_sha256"], "current anchor"),
        (source / "anchor.json", manifest["source_anchor_sha256"], "source anchor"),
        (directory / "waypoints.csv", manifest["waypoints_sha256"], "waypoints"),
    ]
    if manifest.get("route_mode") != "direct":
        checks.append(
            (
                source / "preview" / "trajectory_preview.csv",
                manifest["source_preview_sha256"],
                "source preview",
            )
        )
    for path, expected, label in checks:
        if _sha256(path) != expected:
            raise PermissionError(f"prepared {label} changed; prepare a new restore session")

    config = configuration(config_path)
    current_anchor = load_start_anchor(directory / "current_anchor.json")
    source_anchor = load_start_anchor(source / "anchor.json")
    waypoints = pd.read_csv(directory / "waypoints.csv")
    _audit_waypoints(
        waypoints,
        config,
        current_anchor.tcp_pose_base,
        source_anchor.tcp_pose_base,
    )
    robot = config["robot"]
    motion = config["motion"]
    limits = CommissioningLimits(
        max_cartesian_step_m=MAX_SEGMENT_M + MAX_START_DRIFT_M,
        max_speed_ratio=RESTORE_SPEED_RATIO,
        workspace_min_base_m=tuple(motion["workspace_min_base_m"]),
        workspace_max_base_m=tuple(motion["workspace_max_base_m"]),
        max_state_age_s=float(motion["max_state_age_s"]),
        max_network_tolerance_percent=int(motion["rt_network_tolerance_percent"]),
        require_collision_query=False,
        require_joint_soft_limits=True,
    )
    factory = adapter_factory or _adapter_factory(config)
    adapter = factory(robot["ip"])
    console = RokaeCommissioningConsole(
        adapter,
        limits=limits,
        audit_path=directory / "command_audit.jsonl",
    )
    status.update(status="RUNNING", motion_commanded=True)
    atomic_json(status_path, status)
    completed = False
    try:
        console.connect()
        live_start = np.asarray(adapter.read_tcp_pose(), dtype=float)
        prepared_start = np.asarray(current_anchor.tcp_pose_base, dtype=float)
        if float(np.linalg.norm(live_start[:3] - prepared_start[:3])) > MAX_START_DRIFT_M:
            raise PermissionError("robot moved away from the prepared restore start")
        if _angular_error(live_start[3:], prepared_start[3:]) > MAX_ORIENTATION_ERROR_RAD:
            raise PermissionError("robot orientation changed after restore preparation")
        _require_wrench(
            adapter,
            max_force_n=float(motion["max_force_n"]),
            max_torque_nm=float(motion["max_torque_nm"]),
        )
        console.power_on(
            confirmation=COMMISSIONING_CONFIRMATION,
            speed_ratio=RESTORE_SPEED_RATIO,
        )
        for _, row in waypoints.iloc[1:].iterrows():
            target = row[
                ["tcp_x_base", "tcp_y_base", "tcp_z_base", "tcp_rx", "tcp_ry", "tcp_rz"]
            ].to_numpy(dtype=float)
            current = np.asarray(adapter.read_tcp_pose(), dtype=float)
            delta = target[:3] - current[:3]
            if float(np.linalg.norm(delta)) > MAX_SEGMENT_M + MAX_START_DRIFT_M:
                raise RuntimeError("live tracking drift makes the next restore segment unsafe")
            _require_wrench(
                adapter,
                max_force_n=float(motion["max_force_n"]),
                max_torque_nm=float(motion["max_torque_nm"]),
            )
            console.move_cartesian_absolute(
                tuple(float(value) for value in target),
                confirmation=COMMISSIONING_CONFIRMATION,
                position_tolerance_m=MAX_START_DRIFT_M,
                orientation_tolerance_rad=MAX_ORIENTATION_ERROR_RAD,
            )
            _require_wrench(
                adapter,
                max_force_n=float(motion["max_force_n"]),
                max_torque_nm=float(motion["max_torque_nm"]),
            )
        final_pose = np.asarray(adapter.read_tcp_pose(), dtype=float)
        target_pose = np.asarray(source_anchor.tcp_pose_base, dtype=float)
        final_error = float(np.linalg.norm(final_pose[:3] - target_pose[:3]))
        orientation_error = _angular_error(final_pose[3:], target_pose[3:])
        if final_error > MAX_FINAL_POSITION_ERROR_M:
            raise RuntimeError(f"restore final position error is too large: {final_error:.6f} m")
        if orientation_error > MAX_ORIENTATION_ERROR_RAD:
            raise RuntimeError("restore final orientation error is too large")
        completed = True
        status.update(
            status="COMPLETE",
            final_tcp_pose_base_m_rad=final_pose.tolist(),
            final_position_error_m=final_error,
            final_orientation_error_rad=orientation_error,
        )
        status.pop("reason", None)
        return status
    except BaseException as exc:
        status.update(status="FAILED", reason=str(exc))
        try:
            if console.connected:
                console.stop_motion("restore_exception")
        except BaseException as stop_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"additional restore stop failure: {stop_exc}")
        raise
    finally:
        cleanup = console.disconnect()
        status["cleanup"] = cleanup
        if cleanup.get("errors") or cleanup.get("warnings"):
            if completed:
                status.update(status="FAILED", reason="restore cleanup was not confirmed")
        atomic_json(status_path, status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="return slowly along a previous session path to its start TCP"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--session", required=True)
    prepare_parser.add_argument("--source-session", required=True)
    prepare_parser.add_argument("--config", default=str(BUNDLE_PATH))
    prepare_parser.add_argument(
        "--direct",
        action="store_true",
        help=(
            "return along a straight Cartesian corridor from the captured position; "
            "requires a clear obstacle-free workspace"
        ),
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--session", required=True)
    run_parser.add_argument("--live", action="store_true")
    run_parser.add_argument("--confirm-unloaded", action="store_true")
    run_parser.add_argument("--operator-confirmation", default="")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.session,
            source_session=args.source_session,
            config_path=args.config,
            direct=args.direct,
        )
    else:
        result = run(
            args.session,
            live=args.live,
            confirm_unloaded=args.confirm_unloaded,
            operator_confirmation=args.operator_confirmation,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
