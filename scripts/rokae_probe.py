"""Connection/state/wrench health probe for the project ROKAE adapter.

This module deliberately imports the native-adapter implementation only when
the CLI is executed.  Importing :mod:`scripts.rokae_probe` is therefore safe on
development machines without xCoreSDK.  Project code in this probe does not
explicitly send motion targets, power-on commands, or mode-switch commands.
Adapter construction, vendor SDK initialization, connect, and disconnect may
still have controller/session side effects defined by the vendor SDK.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
from numbers import Real
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


AdapterFactory = Callable[[str], Any]
_MISSING = object()

PROJECT_ACTION_STATEMENT = (
    "Project code did not explicitly send a motion target, power-on command, "
    "or mode-switch command."
)
VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE = (
    "Adapter construction, vendor SDK initialization, connect, and disconnect "
    "may have controller/session side effects."
)


class ProbeSemanticError(ValueError):
    """Raised when a completed adapter read is not semantically usable."""


def _json_safe(value: Any) -> Any:
    """Convert normalized adapter results into strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _error_payload(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _run_check(
    method: Callable[[], Any],
    *,
    validator: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    try:
        value = method()
    except Exception as exc:
        finished_ns = time.perf_counter_ns()
        return {
            "ok": False,
            "transport_ok": False,
            "semantic_valid": None,
            "duration_ms": (finished_ns - started_ns) / 1_000_000.0,
            "error": _error_payload(exc),
        }
    finished_ns = time.perf_counter_ns()
    serialized = _json_safe(value)
    if validator is not None:
        try:
            validator(value)
        except Exception as exc:
            return {
                "ok": False,
                "transport_ok": True,
                "semantic_valid": False,
                "duration_ms": (finished_ns - started_ns) / 1_000_000.0,
                "value": serialized,
                "error": _error_payload(exc),
            }
    return {
        "ok": True,
        "transport_ok": True,
        "semantic_valid": True if validator is not None else None,
        "duration_ms": (finished_ns - started_ns) / 1_000_000.0,
        "value": serialized,
        "error": None,
    }


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return _MISSING


def _finite_vector(value: Any, *, size: int, field_name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ProbeSemanticError(
            f"{field_name} must contain exactly {size} finite numbers"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ProbeSemanticError(
            f"{field_name} must contain exactly {size} finite numbers"
        ) from exc
    if len(items) != size or any(
        isinstance(item, bool)
        or not isinstance(item, Real)
        or not math.isfinite(float(item))
        for item in items
    ):
        raise ProbeSemanticError(
            f"{field_name} must contain exactly {size} finite numbers"
        )
    return tuple(float(item) for item in items)


def _finite_scalar(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProbeSemanticError(f"{field_name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProbeSemanticError(f"{field_name} must be finite")
    return parsed


def _validate_state_summary(summary: Any) -> None:
    if not isinstance(summary, Mapping):
        raise ProbeSemanticError("robot_state_summary must be a mapping")
    if summary.get("state_valid") is not True:
        raise ProbeSemanticError("robot_state_summary.state_valid must be true")
    thread_alive = summary.get(
        "state_stream_thread_alive",
        summary.get(
            "state_thread_alive",
            summary.get("thread_alive", _MISSING),
        ),
    )
    if thread_alive is not True:
        raise ProbeSemanticError(
            "robot_state_summary.state_stream_thread_alive must be true"
        )


def _validate_tcp_pose(pose: Any) -> None:
    values = pose
    if isinstance(pose, Mapping):
        values = _field(pose, "tcp_pose_base_m_rad")
        if values is _MISSING:
            position = _field(pose, "position_base_m", "tcp_position_m")
            orientation = _field(pose, "orientation_rad", "tcp_orientation_rad")
            if position is _MISSING or orientation is _MISSING:
                raise ProbeSemanticError(
                    "tcp_pose must provide a 6D pose or position plus orientation"
                )
            values = (
                *_finite_vector(position, size=3, field_name="tcp_position"),
                *_finite_vector(orientation, size=3, field_name="tcp_orientation"),
            )
    _finite_vector(values, size=6, field_name="tcp_pose")


def _validate_joint_positions(joints: Any) -> None:
    values = joints
    if isinstance(joints, Mapping):
        values = _field(joints, "joint_position_rad", "joint_positions")
        if values is _MISSING:
            raise ProbeSemanticError("joint_positions field is missing")
    _finite_vector(values, size=6, field_name="joint_positions")


def _validate_internal_wrench(wrench: Any) -> None:
    if _field(wrench, "valid") is not True:
        raise ProbeSemanticError("internal_wrench.valid must be true")
    query_start = _field(
        wrench,
        "host_query_start_s",
        "force_query_started_s",
        "query_start_s",
    )
    query_end = _field(
        wrench,
        "host_query_end_s",
        "force_query_finished_s",
        "query_end_s",
    )
    if query_start is _MISSING or query_end is _MISSING:
        raise ProbeSemanticError("internal_wrench query timing is missing")
    started_s = _finite_scalar(query_start, field_name="internal_wrench.query_start_s")
    finished_s = _finite_scalar(query_end, field_name="internal_wrench.query_end_s")
    if finished_s < started_s:
        raise ProbeSemanticError(
            "internal_wrench query_end_s must not precede query_start_s"
        )
    midpoint = _field(wrench, "host_monotonic_time_s")
    if midpoint is not _MISSING:
        midpoint_s = _finite_scalar(
            midpoint,
            field_name="internal_wrench.host_monotonic_time_s",
        )
        if not started_s <= midpoint_s <= finished_s:
            raise ProbeSemanticError(
                "internal_wrench host_monotonic_time_s must lie within query bounds"
            )
    duration_ms = _field(wrench, "query_duration_ms")
    if duration_ms is not _MISSING:
        parsed_duration_ms = _finite_scalar(
            duration_ms,
            field_name="internal_wrench.query_duration_ms",
        )
        if parsed_duration_ms < 0.0:
            raise ProbeSemanticError(
                "internal_wrench.query_duration_ms must be non-negative"
            )


def _validate_true(value: Any, *, field_name: str) -> None:
    if value is not True:
        raise ProbeSemanticError(f"{field_name} must be true")


def _connection_state(adapter: Any) -> bool:
    state = getattr(adapter, "is_connected")
    return bool(state() if callable(state) else state)


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "transport_ok": False,
        "semantic_valid": None,
        "skipped": True,
        "duration_ms": None,
        "error": {"type": "ProbeSkipped", "message": reason},
    }


def probe_adapter(adapter: Any, *, robot_ip: str | None = None) -> dict[str, Any]:
    """Exercise the observation/session subset of ``RokaeRobotAdapter``.

    Expected methods are ``connect``, ``disconnect``, ``is_connected``,
    ``start_state_stream``, ``stop_state_stream``, ``read_tcp_pose``,
    ``read_joint_positions``, ``read_internal_wrench``, and
    ``get_robot_state_summary``.  Cleanup runs after every attempted connection,
    including partially failed connection or stream-start operations.  Project
    code here sends no explicit target, power-on command, or mode switch; vendor
    initialization/connect/disconnect side effects remain possible.
    """
    probe_started_ns = time.perf_counter_ns()
    checks: dict[str, dict[str, Any]] = {}
    cleanup: dict[str, dict[str, Any]] = {}
    connect_attempted = False
    stream_start_attempted = False

    try:
        connect_attempted = True
        checks["connect"] = _run_check(adapter.connect)
        if checks["connect"]["ok"]:
            checks["is_connected"] = _run_check(lambda: _connection_state(adapter))
        else:
            checks["is_connected"] = _skipped("connect failed")

        connected = bool(
            checks["connect"]["ok"]
            and checks["is_connected"]["ok"]
            and checks["is_connected"].get("value") is True
        )
        if not connected:
            for name in (
                "start_state_stream",
                "robot_state_summary",
                "tcp_pose",
                "joint_positions",
                "internal_wrench",
            ):
                checks[name] = _skipped("adapter connection was not confirmed")
        else:
            stream_start_attempted = True
            checks["start_state_stream"] = _run_check(adapter.start_state_stream)
            checks["robot_state_summary"] = _run_check(
                adapter.get_robot_state_summary,
                validator=_validate_state_summary,
            )
            checks["tcp_pose"] = _run_check(
                adapter.read_tcp_pose,
                validator=_validate_tcp_pose,
            )
            checks["joint_positions"] = _run_check(
                adapter.read_joint_positions,
                validator=_validate_joint_positions,
            )
            checks["internal_wrench"] = _run_check(
                adapter.read_internal_wrench,
                validator=_validate_internal_wrench,
            )
    finally:
        if stream_start_attempted:
            cleanup["stop_state_stream"] = _run_check(adapter.stop_state_stream)
        else:
            cleanup["stop_state_stream"] = _skipped("state stream was not started")

        if connect_attempted:
            cleanup["disconnect"] = _run_check(adapter.disconnect)
            if cleanup["disconnect"]["ok"]:
                cleanup["is_disconnected"] = _run_check(
                    lambda: not _connection_state(adapter),
                    validator=lambda value: _validate_true(
                        value,
                        field_name="is_disconnected",
                    ),
                )
            else:
                cleanup["is_disconnected"] = _skipped("disconnect failed")
        else:
            cleanup["disconnect"] = _skipped("connect was not attempted")
            cleanup["is_disconnected"] = _skipped("connect was not attempted")

    required_checks = (
        "connect",
        "is_connected",
        "start_state_stream",
        "robot_state_summary",
        "tcp_pose",
        "joint_positions",
        "internal_wrench",
    )
    required_cleanup = ("stop_state_stream", "disconnect", "is_disconnected")
    success = all(checks[name]["ok"] for name in required_checks) and all(
        cleanup[name]["ok"] for name in required_cleanup
    )
    probe_finished_ns = time.perf_counter_ns()
    return {
        "schema_version": 1,
        "probe": "rokae_connection_state_wrench_health",
        "project_action_statement": PROJECT_ACTION_STATEMENT,
        "vendor_session_side_effects_possible": True,
        "vendor_session_side_effect_disclosure": (
            VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE
        ),
        "robot_ip": robot_ip,
        "success": success,
        "duration_ms": (probe_finished_ns - probe_started_ns) / 1_000_000.0,
        "checks": checks,
        "cleanup": cleanup,
    }


def _default_adapter_factory(robot_ip: str) -> Any:
    """Load the real adapter lazily so ordinary imports remain SDK-independent."""
    try:
        from hardware.rokae_adapter import RokaeRobotAdapter
    except ImportError as exc:
        raise RuntimeError(
            "hardware.rokae_adapter.RokaeRobotAdapter is unavailable; install or "
            "finish the project hardware adapter before running the real probe"
        ) from exc
    return RokaeRobotAdapter(robot_ip)


def run_probe(robot_ip: str, *, adapter_factory: AdapterFactory | None = None) -> dict[str, Any]:
    factory = adapter_factory or _default_adapter_factory
    try:
        adapter = factory(robot_ip)
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": "rokae_connection_state_wrench_health",
            "project_action_statement": PROJECT_ACTION_STATEMENT,
            "vendor_session_side_effects_possible": True,
            "vendor_session_side_effect_disclosure": (
                VENDOR_SESSION_SIDE_EFFECT_DISCLOSURE
            ),
            "robot_ip": robot_ip,
            "success": False,
            "duration_ms": 0.0,
            "checks": {
                "adapter_factory": {
                    "ok": False,
                    "transport_ok": False,
                    "semantic_valid": None,
                    "duration_ms": None,
                    "error": _error_payload(exc),
                }
            },
            "cleanup": {},
        }
    return probe_adapter(adapter, robot_ip=robot_ip)


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: AdapterFactory | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "ROKAE connection/state/wrench health probe; project code sends no "
            "explicit motion target, power-on command, or mode switch"
        )
    )
    parser.add_argument("--ip", required=True, help="ROKAE controller IP address")
    args = parser.parse_args(argv)
    result = run_probe(args.ip, adapter_factory=adapter_factory)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
