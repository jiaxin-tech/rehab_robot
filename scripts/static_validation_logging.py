"""Default-off sidecar labels for non-human static validation records.

This module never constructs a robot, starts acquisition, changes a command,
or interprets a scientific result.  Callers pass copies of already available
raw wrench/state mappings.  The layer writes a separate label-linked CSV and
does not modify or replace the source raw stream.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, TextIO


SCHEMA_VERSION = "STATIC_VALIDATION_LABEL_SCHEMA_V1"
DEFAULT_ENABLED = False
PHASES = ("PRE", "LOAD", "POST")
PHASE_ORDER = {phase: index for index, phase in enumerate(PHASES)}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/static_validation_logging.json"
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("static_validation_record_schema_v1.json")
LABEL_FILENAME = "static_validation_labels.csv"
METADATA_FILENAME = "static_validation_logging_metadata.json"

CSV_FIELDS = (
    "schema_version",
    "record_id",
    "cell_id",
    "session_id",
    "protocol_sha256",
    "pose_id",
    "direction_id",
    "load_level_id",
    "repeat_id",
    "phase",
    "phase_sample_index",
    "raw_measurement_source",
    "raw_measurement_id",
    "query_start_s",
    "query_end_s",
    "query_midpoint_s",
    "query_latency_ms",
    "fx_raw_n",
    "fy_raw_n",
    "fz_raw_n",
    "raw_force_frame",
    "state_host_time_s",
    "state_sequence_id",
    "robot_state_valid",
    "robot_state_invalid_reason",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_rx_rad",
    "tcp_ry_rad",
    "tcp_rz_rad",
    "q1_rad",
    "q2_rad",
    "q3_rad",
    "q4_rad",
    "q5_rad",
    "q6_rad",
    "robot_operation_state",
    "robot_model",
    "robot_serial_number",
    "controller_version",
    "active_tool_name",
    "active_workobject_name",
    "active_hmi_tool_workobject_verified",
    "tcp_translation_m_json",
    "tcp_rpy_rad_json",
    "payload_mass_kg",
    "payload_cog_m_json",
    "payload_inertia_kg_m2_json",
    "sdk_available_tool_names_json",
    "sdk_available_workobject_names_json",
    "valid",
    "status",
    "invalid_reason",
)


class StaticValidationLoggingError(RuntimeError):
    """Base error for fail-closed sidecar logging."""


class StaticValidationLoggingDisabled(StaticValidationLoggingError):
    """Raised when a write is attempted without explicit enablement."""


def _require_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_sha256(value: Any, field: str) -> str:
    text = _require_nonempty(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _mapping(value: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
        if not isinstance(result, dict):
            raise TypeError("dataclass conversion did not produce a mapping")
        return result
    raise TypeError("record inputs must be mappings, dataclass instances, or None")


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_vector(value: Any, length: int) -> tuple[float | None, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return (None,) * length
    try:
        items = list(value)
    except TypeError:
        return (None,) * length
    return tuple(_finite_float(items[index]) if index < len(items) else None for index in range(length))


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return None


def _digest_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_static_validation_logging_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the explicit default-off identity without mutating global settings."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StaticValidationLoggingError("static validation logging config must be an object")
    if payload.get("schema_version") != 1:
        raise StaticValidationLoggingError("unsupported static validation logging config schema")
    if not isinstance(payload.get("enabled"), bool):
        raise StaticValidationLoggingError("static validation logging enabled must be boolean")
    return payload


@dataclass(frozen=True)
class StaticValidationCell:
    """Immutable identity for one protocol pose/direction/load/repeat cell."""

    session_id: str
    protocol_sha256: str
    pose_id: str
    direction_id: str
    load_level_id: str
    repeat_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_nonempty(self.session_id, "session_id"))
        object.__setattr__(self, "protocol_sha256", _validate_sha256(self.protocol_sha256, "protocol_sha256"))
        object.__setattr__(self, "pose_id", _require_nonempty(self.pose_id, "pose_id"))
        object.__setattr__(self, "direction_id", _require_nonempty(self.direction_id, "direction_id"))
        object.__setattr__(self, "load_level_id", _require_nonempty(self.load_level_id, "load_level_id"))
        if isinstance(self.repeat_id, bool) or not isinstance(self.repeat_id, int) or self.repeat_id < 1:
            raise ValueError("repeat_id must be an integer >= 1")

    @property
    def cell_id(self) -> str:
        return _digest_payload({
            "session_id": self.session_id,
            "protocol_sha256": self.protocol_sha256,
            "pose_id": self.pose_id,
            "direction_id": self.direction_id,
            "load_level_id": self.load_level_id,
            "repeat_id": self.repeat_id,
        })


def build_static_validation_record(
    *,
    cell: StaticValidationCell,
    phase: str,
    phase_sample_index: int,
    raw_measurement_source: str,
    raw_measurement_id: str,
    wrench_sample: Mapping[str, Any] | Any | None,
    robot_state: Mapping[str, Any] | Any | None,
    robot_metadata: Mapping[str, Any] | Any | None,
) -> dict[str, Any]:
    """Copy already acquired values into one label-linked, fail-closed row."""
    phase_value = _require_nonempty(phase, "phase").upper()
    if phase_value not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if isinstance(phase_sample_index, bool) or not isinstance(phase_sample_index, int) or phase_sample_index < 0:
        raise ValueError("phase_sample_index must be an integer >= 0")
    source_value = _require_nonempty(raw_measurement_source, "raw_measurement_source")
    measurement_value = _require_nonempty(raw_measurement_id, "raw_measurement_id")

    wrench = _mapping(wrench_sample)
    state = _mapping(robot_state)
    metadata = _mapping(robot_metadata)
    errors: list[str] = []

    query_start = _finite_float(wrench.get("force_query_started_s", wrench.get("query_start_s")))
    query_end = _finite_float(wrench.get("force_query_finished_s", wrench.get("query_end_s")))
    supplied_midpoint = _finite_float(wrench.get("host_monotonic_time_s", wrench.get("query_midpoint_s")))
    midpoint: float | None = None
    latency_ms: float | None = None
    if query_start is None or query_end is None:
        errors.append("missing_query_timestamp")
    elif query_end < query_start:
        errors.append("query_timestamp_order_invalid")
    else:
        midpoint = (query_start + query_end) / 2.0
        latency_ms = (query_end - query_start) * 1000.0
        if supplied_midpoint is not None:
            tolerance = max(1e-12, abs(query_end - query_start) * 1e-9)
            if abs(supplied_midpoint - midpoint) > tolerance:
                errors.append("query_midpoint_inconsistent")
    force = _finite_vector(wrench.get("cartesian_force_raw_n", wrench.get("force")), 3)
    if any(value is None for value in force):
        errors.append("raw_force_missing_or_nonfinite")
    source_valid = wrench.get("valid")
    if source_valid is False:
        errors.append(str(wrench.get("invalid_reason") or "wrench_source_invalid"))

    tcp = _finite_vector(state.get("tcp_position_m"), 3) + _finite_vector(state.get("tcp_orientation_rad"), 3)
    joints = _finite_vector(state.get("joint_position_rad"), 6)
    state_valid = state.get("valid")
    if not state:
        errors.append("robot_state_missing")
    elif state_valid is not True:
        errors.append(str(state.get("invalid_reason") or "robot_state_invalid_or_unverified"))
    if all(value is None for value in tcp) and all(value is None for value in joints):
        errors.append("robot_pose_and_joint_state_missing")

    sdk_tool_payload = _mapping(metadata.get("sdk_tool_payload"))
    active_verified = sdk_tool_payload.get(
        "active_hmi_tool_workobject_verified",
        metadata.get("active_hmi_tool_workobject_verified"),
    )
    active_tool_name = metadata.get("active_tool_name")
    active_workobject_name = metadata.get("active_workobject_name")
    if not metadata:
        errors.append("robot_metadata_missing")
    if active_verified is not True:
        errors.append("active_tool_workobject_not_verified")

    unique_errors = list(dict.fromkeys(error for error in errors if error))
    valid = not unique_errors
    record_identity = {
        "cell_id": cell.cell_id,
        "phase": phase_value,
        "phase_sample_index": phase_sample_index,
        "raw_measurement_source": source_value,
        "raw_measurement_id": measurement_value,
    }
    row: dict[str, Any] = {field: None for field in CSV_FIELDS}
    row.update({
        "schema_version": SCHEMA_VERSION,
        "record_id": _digest_payload(record_identity),
        "cell_id": cell.cell_id,
        "session_id": cell.session_id,
        "protocol_sha256": cell.protocol_sha256,
        "pose_id": cell.pose_id,
        "direction_id": cell.direction_id,
        "load_level_id": cell.load_level_id,
        "repeat_id": cell.repeat_id,
        "phase": phase_value,
        "phase_sample_index": phase_sample_index,
        "raw_measurement_source": source_value,
        "raw_measurement_id": measurement_value,
        "query_start_s": query_start,
        "query_end_s": query_end,
        "query_midpoint_s": midpoint,
        "query_latency_ms": latency_ms,
        "fx_raw_n": force[0],
        "fy_raw_n": force[1],
        "fz_raw_n": force[2],
        "raw_force_frame": wrench.get("raw_force_frame", wrench.get("reference_frame")),
        "state_host_time_s": _finite_float(state.get("host_monotonic_time_s", state.get("sample_time_s"))),
        "state_sequence_id": state.get("sequence_id"),
        "robot_state_valid": state_valid if isinstance(state_valid, bool) else None,
        "robot_state_invalid_reason": state.get("invalid_reason") or None,
        "tcp_x_m": tcp[0], "tcp_y_m": tcp[1], "tcp_z_m": tcp[2],
        "tcp_rx_rad": tcp[3], "tcp_ry_rad": tcp[4], "tcp_rz_rad": tcp[5],
        "q1_rad": joints[0], "q2_rad": joints[1], "q3_rad": joints[2],
        "q4_rad": joints[3], "q5_rad": joints[4], "q6_rad": joints[5],
        "robot_operation_state": state.get("operation_state", state.get("robot_operation_state")),
        "robot_model": metadata.get("robot_model"),
        "robot_serial_number": metadata.get("robot_serial_number"),
        "controller_version": metadata.get("controller_version"),
        "active_tool_name": active_tool_name,
        "active_workobject_name": active_workobject_name,
        "active_hmi_tool_workobject_verified": active_verified if isinstance(active_verified, bool) else None,
        "tcp_translation_m_json": _json_or_none(sdk_tool_payload.get("toolset_end_translation_m")),
        "tcp_rpy_rad_json": _json_or_none(sdk_tool_payload.get("toolset_end_rpy_rad")),
        "payload_mass_kg": _finite_float(sdk_tool_payload.get("toolset_load_mass_kg")),
        "payload_cog_m_json": _json_or_none(sdk_tool_payload.get("toolset_load_cog_m")),
        "payload_inertia_kg_m2_json": _json_or_none(sdk_tool_payload.get("toolset_load_inertia_kg_m2")),
        "sdk_available_tool_names_json": _json_or_none(sdk_tool_payload.get("sdk_available_tool_names")),
        "sdk_available_workobject_names_json": _json_or_none(sdk_tool_payload.get("sdk_available_workobject_names")),
        "valid": valid,
        "status": "VALID" if valid else "INVALID",
        "invalid_reason": ";".join(unique_errors),
    })
    return row


class StaticValidationLabelLogger:
    """Separate CSV sidecar; disabled unless explicitly enabled by the caller."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        session_id: str,
        protocol_sha256: str,
        raw_measurement_source: str,
        enabled: bool = DEFAULT_ENABLED,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.session_id = _require_nonempty(session_id, "session_id")
        self.protocol_sha256 = _validate_sha256(protocol_sha256, "protocol_sha256")
        self.raw_measurement_source = _require_nonempty(raw_measurement_source, "raw_measurement_source")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self.enabled = enabled
        self._run_metadata = dict(run_metadata or {})
        self._stream: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._seen_record_ids: set[str] = set()
        self._seen_raw_measurement_ids: set[str] = set()
        self._phase_state: dict[str, int] = {}
        self._phase_sample_indices: set[tuple[str, str, int]] = set()
        self._row_count = 0
        self._valid_count = 0
        self._invalid_count = 0
        self._closed = False

    @property
    def active(self) -> bool:
        return self.enabled and self._stream is not None and not self._closed

    @property
    def label_path(self) -> Path:
        return self.output_directory / LABEL_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.output_directory / METADATA_FILENAME

    def start(self) -> "StaticValidationLabelLogger":
        if not self.enabled:
            raise StaticValidationLoggingDisabled("static validation logging is default-off")
        if self._stream is not None or self._closed:
            raise StaticValidationLoggingError("logger cannot be started twice")
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise StaticValidationLoggingError("output directory must be absent or empty; raw/old files are never overwritten")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._stream = self.label_path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=list(CSV_FIELDS), extrasaction="raise")
        self._writer.writeheader()
        self._flush()
        self._write_metadata(status="recording")
        return self

    def _flush(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def _write_metadata(self, *, status: str) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "default_enabled": DEFAULT_ENABLED,
            "enabled_for_this_run": self.enabled,
            "status": status,
            "session_id": self.session_id,
            "protocol_sha256": self.protocol_sha256,
            "raw_measurement_source": self.raw_measurement_source,
            "label_file": LABEL_FILENAME,
            "label_file_sha256": _sha256_file(self.label_path) if self.label_path.is_file() else None,
            "record_count": self._row_count,
            "valid_record_count": self._valid_count,
            "invalid_record_count": self._invalid_count,
            "raw_wrench_overwritten": False,
            "robot_connected_by_logger": False,
            "robot_action_count": 0,
            "control_or_safety_modified": False,
            "run_metadata": self._run_metadata,
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.metadata_path)

    def append(
        self,
        *,
        cell: StaticValidationCell,
        phase: str,
        phase_sample_index: int,
        raw_measurement_id: str,
        wrench_sample: Mapping[str, Any] | Any | None,
        robot_state: Mapping[str, Any] | Any | None,
        robot_metadata: Mapping[str, Any] | Any | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise StaticValidationLoggingDisabled("static validation logging is default-off")
        if not self.active or self._writer is None:
            raise StaticValidationLoggingError("logger is not active")
        if cell.session_id != self.session_id or cell.protocol_sha256 != self.protocol_sha256:
            raise StaticValidationLoggingError("cell session/protocol identity does not match logger")
        row = build_static_validation_record(
            cell=cell,
            phase=phase,
            phase_sample_index=phase_sample_index,
            raw_measurement_source=self.raw_measurement_source,
            raw_measurement_id=raw_measurement_id,
            wrench_sample=wrench_sample,
            robot_state=robot_state,
            robot_metadata=robot_metadata,
        )
        phase_index = PHASE_ORDER[row["phase"]]
        previous_phase = self._phase_state.get(row["cell_id"])
        if previous_phase is None and phase_index != 0:
            raise StaticValidationLoggingError("first phase for every cell must be PRE")
        if previous_phase is not None and (phase_index < previous_phase or phase_index > previous_phase + 1):
            raise StaticValidationLoggingError("phase order must progress PRE -> LOAD -> POST without reversal or skip")
        phase_sample_key = (row["cell_id"], row["phase"], row["phase_sample_index"])
        if phase_sample_key in self._phase_sample_indices:
            raise StaticValidationLoggingError("duplicate phase_sample_index within cell/phase")
        if row["record_id"] in self._seen_record_ids:
            raise StaticValidationLoggingError("duplicate static validation record identity")
        if row["raw_measurement_id"] in self._seen_raw_measurement_ids:
            raise StaticValidationLoggingError("raw measurement is already linked to another label record")

        self._writer.writerow(row)
        self._flush()
        self._phase_state[row["cell_id"]] = phase_index
        self._phase_sample_indices.add(phase_sample_key)
        self._seen_record_ids.add(row["record_id"])
        self._seen_raw_measurement_ids.add(row["raw_measurement_id"])
        self._row_count += 1
        if row["valid"]:
            self._valid_count += 1
        else:
            self._invalid_count += 1
        self._write_metadata(status="recording")
        return dict(row)

    def close(self) -> None:
        if self._closed:
            return
        if self._stream is not None:
            self._flush()
            self._stream.close()
            self._stream = None
        if self.enabled and self.output_directory.exists():
            self._write_metadata(status="closed")
        self._closed = True

    def __enter__(self) -> "StaticValidationLabelLogger":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False
