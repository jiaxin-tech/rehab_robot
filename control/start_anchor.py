"""Strict, reviewable start anchors for real rehabilitation experiments.

Capturing an anchor is deliberately read-only.  The helper reads one cached
robot state frame (or the adapter's two explicit read methods) and never
connects, enables, powers, stops, or commands the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import uuid


START_ANCHOR_SCHEMA_VERSION = 2
TCP_ORIENTATION_STRATEGY = "fixed"
TCP_ORIENTATION_REPRESENTATION = "euler_xyz_rad"


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _finite_float(value: object, field_name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")
    if non_negative and parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _finite_vector(
    values: object,
    size: int,
    field_name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must contain exactly {size} finite values")
    if len(values) != size:
        raise ValueError(f"{field_name} must contain exactly {size} finite values")
    return tuple(_finite_float(value, field_name) for value in values)


def _optional_nonempty_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be null or a non-empty string")
    return value.strip()


def _strict_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_json_object(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"start anchor JSON not found: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid start anchor JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("start anchor JSON must contain one object")
    return payload


def _atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True)
class FixedTcpOrientation:
    """The captured TCP orientation held unchanged over the first experiment."""

    values_rad: tuple[float, float, float]
    strategy: str = TCP_ORIENTATION_STRATEGY
    representation: str = TCP_ORIENTATION_REPRESENTATION

    def __post_init__(self) -> None:
        if self.strategy != TCP_ORIENTATION_STRATEGY:
            raise ValueError("tcp_orientation.strategy must be 'fixed'")
        if self.representation != TCP_ORIENTATION_REPRESENTATION:
            raise ValueError(
                "tcp_orientation.representation must be 'euler_xyz_rad'"
            )
        values = _finite_vector(self.values_rad, 3, "tcp_orientation.values_rad")
        object.__setattr__(self, "values_rad", values)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "representation": self.representation,
            "values_rad": list(self.values_rad),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "FixedTcpOrientation":
        if not isinstance(payload, Mapping):
            raise ValueError("tcp_orientation must be a JSON object")
        expected = {"strategy", "representation", "values_rad"}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "tcp_orientation fields must be exactly "
                f"{sorted(expected)}; got {sorted(actual)}"
            )
        return cls(
            values_rad=payload["values_rad"],  # type: ignore[arg-type]
            strategy=payload["strategy"],  # type: ignore[arg-type]
            representation=payload["representation"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class StartAnchor:
    """One subject/session-specific, manually reviewable TCP start anchor."""

    capture_host_time_s: float
    tcp_pose_base: tuple[float, float, float, float, float, float]
    tcp_position_base_m: tuple[float, float, float]
    tcp_orientation: FixedTcpOrientation
    robot_joint_positions: tuple[float, float, float, float, float, float]
    trajectory_id: str
    reference_start_q_hip: float
    reference_start_q_knee: float
    anchor_id: str
    robot_model: str | None = None
    robot_serial_number: str | None = None
    controller_version: str | None = None
    tool_name: str | None = None
    workpiece_name: str | None = None
    created_at: str = field(default_factory=_created_at_utc)
    reviewed: bool = False
    notes: str = ""
    schema_version: int = START_ANCHOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != START_ANCHOR_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be the integer {START_ANCHOR_SCHEMA_VERSION}"
            )
        if type(self.reviewed) is not bool:
            raise ValueError("reviewed must be a JSON/Python boolean")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty ISO-8601 string")
        try:
            created = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ValueError("created_at must be a valid ISO-8601 string") from exc
        if created.tzinfo is None:
            raise ValueError("created_at must include an explicit timezone")
        for field_name in ("trajectory_id", "anchor_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "robot_model",
            "robot_serial_number",
            "controller_version",
            "tool_name",
            "workpiece_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonempty_string(getattr(self, field_name), field_name),
            )

        capture_time = _finite_float(
            self.capture_host_time_s,
            "capture_host_time_s",
            non_negative=True,
        )
        pose = _finite_vector(self.tcp_pose_base, 6, "tcp_pose_base")
        position = _finite_vector(
            self.tcp_position_base_m,
            3,
            "tcp_position_base_m",
        )
        joints = _finite_vector(
            self.robot_joint_positions,
            6,
            "robot_joint_positions",
        )
        q_hip = _finite_float(self.reference_start_q_hip, "reference_start_q_hip")
        q_knee = _finite_float(self.reference_start_q_knee, "reference_start_q_knee")
        if not isinstance(self.tcp_orientation, FixedTcpOrientation):
            raise ValueError("tcp_orientation must be a FixedTcpOrientation")
        if pose[:3] != position:
            raise ValueError("tcp_pose_base position does not match tcp_position_base_m")
        if pose[3:] != self.tcp_orientation.values_rad:
            raise ValueError("tcp_pose_base orientation does not match tcp_orientation")

        object.__setattr__(self, "capture_host_time_s", capture_time)
        object.__setattr__(self, "tcp_pose_base", pose)
        object.__setattr__(self, "tcp_position_base_m", position)
        object.__setattr__(self, "robot_joint_positions", joints)
        object.__setattr__(self, "reference_start_q_hip", q_hip)
        object.__setattr__(self, "reference_start_q_knee", q_knee)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capture_host_time_s": self.capture_host_time_s,
            "tcp_pose_base": list(self.tcp_pose_base),
            "tcp_position_base_m": list(self.tcp_position_base_m),
            "tcp_orientation": self.tcp_orientation.to_dict(),
            "robot_joint_positions": list(self.robot_joint_positions),
            "trajectory_id": self.trajectory_id,
            "reference_start_q_hip": self.reference_start_q_hip,
            "reference_start_q_knee": self.reference_start_q_knee,
            "anchor_id": self.anchor_id,
            "robot_model": self.robot_model,
            "robot_serial_number": self.robot_serial_number,
            "controller_version": self.controller_version,
            "tool_name": self.tool_name,
            "workpiece_name": self.workpiece_name,
            "created_at": self.created_at,
            "reviewed": self.reviewed,
            "notes": self.notes,
        }

    def save_json(self, path: str | Path) -> Path:
        """Atomically replace ``path`` with this validated anchor."""

        return _atomic_write_json(path, self.to_dict())

    @classmethod
    def load_json(cls, path: str | Path) -> "StartAnchor":
        payload = _strict_json_object(path)
        expected = {
            "schema_version",
            "capture_host_time_s",
            "tcp_pose_base",
            "tcp_position_base_m",
            "tcp_orientation",
            "robot_joint_positions",
            "trajectory_id",
            "reference_start_q_hip",
            "reference_start_q_knee",
            "anchor_id",
            "robot_model",
            "robot_serial_number",
            "controller_version",
            "tool_name",
            "workpiece_name",
            "created_at",
            "reviewed",
            "notes",
        }
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                f"start anchor fields must be exactly {sorted(expected)}; "
                f"got {sorted(actual)}"
            )
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            capture_host_time_s=payload["capture_host_time_s"],  # type: ignore[arg-type]
            tcp_pose_base=payload["tcp_pose_base"],  # type: ignore[arg-type]
            tcp_position_base_m=payload["tcp_position_base_m"],  # type: ignore[arg-type]
            tcp_orientation=FixedTcpOrientation.from_dict(payload["tcp_orientation"]),
            robot_joint_positions=payload["robot_joint_positions"],  # type: ignore[arg-type]
            trajectory_id=payload["trajectory_id"],  # type: ignore[arg-type]
            reference_start_q_hip=payload["reference_start_q_hip"],  # type: ignore[arg-type]
            reference_start_q_knee=payload["reference_start_q_knee"],  # type: ignore[arg-type]
            anchor_id=payload["anchor_id"],  # type: ignore[arg-type]
            robot_model=payload["robot_model"],  # type: ignore[arg-type]
            robot_serial_number=payload["robot_serial_number"],  # type: ignore[arg-type]
            controller_version=payload["controller_version"],  # type: ignore[arg-type]
            tool_name=payload["tool_name"],  # type: ignore[arg-type]
            workpiece_name=payload["workpiece_name"],  # type: ignore[arg-type]
            created_at=payload["created_at"],  # type: ignore[arg-type]
            reviewed=payload["reviewed"],  # type: ignore[arg-type]
            notes=payload["notes"],  # type: ignore[arg-type]
        )


def _read_anchor_state(adapter: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read pose/joints without invoking any adapter lifecycle or motion method."""

    state_reader = getattr(adapter, "read_state_frame", None)
    if not callable(state_reader):
        state_reader = getattr(adapter, "get_state_frame", None)
    if callable(state_reader):
        frame = state_reader()
        if type(getattr(frame, "valid", None)) is not bool or not frame.valid:
            reason = getattr(frame, "invalid_reason", "robot_state_invalid")
            raise RuntimeError(f"cannot capture start anchor from invalid state: {reason}")
        position = getattr(frame, "tcp_position_m", None)
        orientation = getattr(frame, "tcp_orientation_rad", None)
        joints = getattr(frame, "joint_position_rad", None)
        pose = (*_finite_vector(position, 3, "tcp_position_base_m"),)
        pose += _finite_vector(orientation, 3, "tcp_orientation.values_rad")
        return pose, _finite_vector(joints, 6, "robot_joint_positions")

    pose_reader = getattr(adapter, "read_tcp_pose", None)
    joint_reader = getattr(adapter, "read_joint_positions", None)
    if not callable(pose_reader) or not callable(joint_reader):
        raise TypeError(
            "adapter must expose read_state_frame(), get_state_frame(), or both "
            "read_tcp_pose() and read_joint_positions()"
        )
    pose = _finite_vector(pose_reader(), 6, "tcp_pose_base")
    joints = _finite_vector(joint_reader(), 6, "robot_joint_positions")
    return pose, joints


def _read_anchor_identity(adapter: Any) -> tuple[str | None, str | None, str | None]:
    """Read SDK identity metadata without invoking lifecycle or motion APIs.

    Missing or malformed metadata is retained as ``None`` so the later
    execution gate fails closed.  The read-only adapter method is preferred to
    avoid taking a second state sample merely to obtain metadata.
    """

    metadata_reader = getattr(adapter, "read_robot_metadata", None)
    try:
        if callable(metadata_reader):
            metadata = metadata_reader()
        else:
            summary_reader = getattr(adapter, "get_robot_state_summary", None)
            if not callable(summary_reader):
                return None, None, None
            summary = summary_reader()
            if not isinstance(summary, Mapping):
                return None, None, None
            metadata = summary.get("robot_metadata")
    except Exception:
        return None, None, None
    if not isinstance(metadata, Mapping):
        return None, None, None

    def identity_value(name: str) -> str | None:
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    return (
        identity_value("robot_model"),
        identity_value("robot_serial_number"),
        identity_value("controller_version"),
    )


def capture_start_anchor(
    adapter: Any,
    *,
    trajectory_id: str,
    reference_start_q_hip: float,
    reference_start_q_knee: float,
    anchor_id: str | None = None,
    tool_name: str | None = None,
    workpiece_name: str | None = None,
    notes: str = "",
    clock: Callable[[], float] = time.perf_counter,
) -> StartAnchor:
    """Capture one unreviewed anchor using read methods only.

    Connection and disconnection remain the caller's responsibility.  This
    function intentionally has no option that can enable or move the robot.
    """

    pose, joints = _read_anchor_state(adapter)
    robot_model, robot_serial_number, controller_version = _read_anchor_identity(adapter)
    capture_time = _finite_float(
        clock(),
        "capture_host_time_s",
        non_negative=True,
    )
    resolved_anchor_id = anchor_id or f"anchor_{uuid.uuid4().hex}"
    orientation = FixedTcpOrientation(values_rad=pose[3:])
    return StartAnchor(
        capture_host_time_s=capture_time,
        tcp_pose_base=pose,
        tcp_position_base_m=pose[:3],
        tcp_orientation=orientation,
        robot_joint_positions=joints,
        trajectory_id=trajectory_id,
        reference_start_q_hip=reference_start_q_hip,
        reference_start_q_knee=reference_start_q_knee,
        anchor_id=resolved_anchor_id,
        robot_model=robot_model,
        robot_serial_number=robot_serial_number,
        controller_version=controller_version,
        tool_name=tool_name,
        workpiece_name=workpiece_name,
        reviewed=False,
        notes=notes,
    )


def load_start_anchor(path: str | Path) -> StartAnchor:
    return StartAnchor.load_json(path)


def save_start_anchor(anchor: StartAnchor, path: str | Path) -> Path:
    if not isinstance(anchor, StartAnchor):
        raise TypeError("anchor must be a StartAnchor")
    return anchor.save_json(path)


__all__ = [
    "FixedTcpOrientation",
    "START_ANCHOR_SCHEMA_VERSION",
    "StartAnchor",
    "capture_start_anchor",
    "load_start_anchor",
    "save_start_anchor",
]
