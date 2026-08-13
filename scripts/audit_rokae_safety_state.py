"""Strictly read-only ROKAE safety/state API audit.

Only methods documented by the bundled xCoreSDK 0.7.0 stubs as query/getter
operations are called.  No event watcher, recovery, reset, mode, power,
collision-configuration, register, or motion method is used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

from hardware.windows.rokae_xcore import RokaeRobot


PLANNED_APIS = (
    "BaseRobot.robotInfo(ec)",
    "BaseRobot.operationState(ec)",
    "BaseRobot.powerState(ec)",
    "BaseRobot.operateMode(ec)",
    "BaseRobot.queryEventInfo(Event.safety, ec)",
    "BaseRobot.queryControllerLog(10, {warning, error}, ec, 0)",
)


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return {
            "enum_type": type(value).__name__,
            "name": str(value.name),
            "value": int(value.value),
            "string": str(value),
        }
    fields = {}
    for name in ("id", "timestamp", "content", "repair", "type", "version", "joint_num"):
        if hasattr(value, name):
            fields[name] = _safe(getattr(value, name))
    return {"python_type": type(value).__name__, **fields} if fields else repr(value)


def _error_code(message: str) -> int | None:
    match = re.search(r"\((\d+)\)", message)
    return int(match.group(1)) if match else None


def _query(
    name: str,
    method: Callable[..., Any],
    *args: Any,
    confidence: str,
    notes: str,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    ec: dict[str, Any] = {}
    try:
        value = method(*args, ec)
        finished_ns = time.perf_counter_ns()
        code = int(ec.get("ec", 0))
        if code:
            raise RuntimeError(f"xCoreSDK {name} failed ({code}): {ec.get('message', '')}")
        return {
            "api": name,
            "read_only_confidence": confidence,
            "idle_callable": True,
            "supported": True,
            "success": True,
            "latency_ms": (finished_ns - started_ns) / 1e6,
            "argument_python_types": [type(arg).__name__ for arg in args] + ["dict"],
            "return_type": type(value).__name__,
            "value": _safe(value),
            "ec": _safe(ec),
            "exception": None,
            "notes": notes,
        }
    except Exception as exc:
        finished_ns = time.perf_counter_ns()
        message = str(exc)
        return {
            "api": name,
            "read_only_confidence": confidence,
            "idle_callable": False,
            "supported": False,
            "success": False,
            "latency_ms": (finished_ns - started_ns) / 1e6,
            "argument_python_types": [type(arg).__name__ for arg in args] + ["dict"],
            "return_type": None,
            "value": None,
            "ec": _safe(ec),
            "exception": {
                "type": type(exc).__name__,
                "message": message,
                "sdk_error_code": _error_code(message),
            },
            "notes": notes,
        }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ROKAE safety-state read-only audit",
        "",
        f"- Timestamp UTC: `{payload['timestamp_utc']}`",
        f"- Robot: `{payload['identity'].get('model')}` / controller `{payload['identity'].get('controller')}`",
        f"- SDK: `{payload['identity'].get('sdk')}`",
        f"- Robot IP / local RT IP: `{payload['robot_ip']}` / `{payload['local_ip']}`",
        f"- Result: **{payload['result']}**",
        "",
        "No power, mode, recovery, reset, watcher-registration, collision configuration, register write, or motion API was called.",
        "",
        "| Safety information | SDK API | Read-only evidence | IDLE callable | Result | Error |",
        "|---|---|---|---:|---|---|",
    ]
    labels = {
        "robotInfo": "robot/controller identity",
        "operationState": "operation state",
        "powerState": "power / E-stop / safety-door state",
        "operateMode": "manual/automatic mode",
        "queryEventInfo(safety)": "collision event state",
        "queryControllerLog": "recent warning/error evidence",
    }
    for row in payload["queries"]:
        error = row["exception"]["message"] if row["exception"] else ""
        result = json.dumps(row["value"], ensure_ascii=False) if row["success"] else "unsupported/error"
        lines.append(
            f"| {labels.get(row['api'], row['api'])} | `{row['api']}` | "
            f"{row['read_only_confidence']} | {'yes' if row['idle_callable'] else 'no'} | "
            f"{result} | {error} |"
        )
    lines += [
        "",
        "## Collision query finding",
        "",
        "The bundled stub declares `queryEventInfo(eventType: Event, ec: dict) -> dict`. "
        "The tested arguments were exactly `Event.safety` (Python `Event`, numeric value 1) "
        "and a Python `dict`; the documented result key is `EventInfoKey.Safety.Collided == 'collided'`.",
        "",
        payload["collision_finding"],
        "",
        "## Pre-motion observation",
        "",
        payload["pre_motion_observation"],
        "",
    ]
    return "\n".join(lines)


def run(robot_ip: str, local_ip: str, output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    robot = RokaeRobot(robot_ip, local_ip=local_ip)
    connected = False
    disconnected = False
    queries: list[dict[str, Any]] = []
    identity: dict[str, Any] = {}
    cleanup_error: str | None = None
    print("STRICT READ-ONLY planned SDK APIs:")
    for api in PLANNED_APIS:
        print(f"  - {api}")
    print("No mode change, power, recovery, reset, watcher, collision configuration, or motion API will be called.")
    try:
        robot.connect()
        connected = True
        native = robot._robot
        sdk = robot._sdk
        info = robot._robot_info
        identity = {
            "sdk": robot._sdk_version,
            "controller": str(info.version),
            "model": str(info.type),
            "serial": str(info.id),
        }
        queries.append(_query("robotInfo", native.robotInfo, confidence="high: bundled stub says query", notes="Basic controller identity."))
        operation = _query("operationState", native.operationState, confidence="high: bundled stub says query current running state", notes="Must be IDLE for this audit.")
        queries.append(operation)
        operation_name = ((operation.get("value") or {}).get("name") if isinstance(operation.get("value"), dict) else None)
        if operation_name != "idle":
            raise RuntimeError(f"strict read-only audit requires IDLE; observed {operation_name!r}")
        queries.append(_query("powerState", native.powerState, confidence="high: bundled stub documents power/E-stop/safety-door getter", notes="Enum includes on/off/estop/gstop/unknown."))
        queries.append(_query("operateMode", native.operateMode, confidence="high: bundled stub says query current operation mode", notes="Observation only; setOperateMode was not called."))
        queries.append(_query("queryEventInfo(safety)", native.queryEventInfo, sdk.Event.safety, confidence="high: bundled stub explicitly documents active event query", notes=f"Event.safety type={type(sdk.Event.safety).__name__}, value={int(sdk.Event.safety)}; expected key={sdk.EventInfoKey.Safety.Collided!r}."))
        levels = {sdk.LogInfoLevel.warning, sdk.LogInfoLevel.error}
        queries.append(_query("queryControllerLog", native.queryControllerLog, 10, levels, confidence="high: bundled stub and vendor example explicitly document log query", notes="Historical warning/error evidence, not a synchronized current safety latch."))
    finally:
        if connected:
            try:
                robot.disconnect()
                disconnected = not robot.is_connected
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}:{exc}"

    by_name = {row["api"]: row for row in queries}
    power_ok = by_name.get("powerState", {}).get("success") is True
    operation_ok = by_name.get("operationState", {}).get("success") is True
    collision_ok = by_name.get("queryEventInfo(safety)", {}).get("success") is True
    result = "PASS" if power_ok and operation_ok and collision_ok and disconnected else "BLOCKED WITH EVIDENCE"
    collision_finding = (
        "Collision state was returned successfully."
        if collision_ok
        else "The documented call form still returns SDK error 259 on this SDK/controller/model. "
        "No alternate enum/string/integer form was guessed and no watcher was registered."
    )
    pre_motion = (
        "Use `powerState` for E-stop/safety-door/power observation, `operationState` for IDLE, "
        "and `queryEventInfo(Event.safety)` for collision."
        if collision_ok
        else "`powerState` is the verified read-only E-stop/safety-door signal and `operationState` is the verified IDLE signal. "
        "No verified current collision/protective-stop getter is available, so they are insufficient as the complete pre-motion gate."
    )
    payload = {
        "schema_version": 1,
        "diagnostic": "rokae_safety_state_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "robot_ip": robot_ip,
        "local_ip": local_ip,
        "identity": identity,
        "planned_apis": list(PLANNED_APIS),
        "forbidden_api_called": False,
        "queries": queries,
        "cleanup": {"disconnect_attempted": connected, "disconnected_confirmed": disconnected, "error": cleanup_error},
        "collision_finding": collision_finding,
        "pre_motion_observation": pre_motion,
        "result": result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"safety_state_audit_{stamp}.json"
    md_path = output_dir / f"safety_state_audit_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return json_path, md_path, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly read-only ROKAE safety-state API audit")
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    json_path, md_path, payload = run(args.robot_ip, args.local_ip, Path(args.output_dir))
    print(json.dumps({"result": payload["result"], "identity": payload["identity"], "cleanup": payload["cleanup"]}, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
