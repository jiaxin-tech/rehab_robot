"""Observe a second read-only SDK session while RT mode is prepared.

The script never starts a realtime loop or sends a target.  It always attempts
to return to NRT mode, power down, and disconnect in ``finally``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from hardware.rokae_adapter import RokaeRobotAdapter
from hardware.windows.rokae_xcore import _load_sdk


def _call(robot, name):
    ec = {}
    value = getattr(robot, name)(ec)
    code = int(ec.get("ec", 0))
    if code:
        raise RuntimeError(f"{name} failed ({code}): {ec.get('message')}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    adapter = RokaeRobotAdapter(args.robot_ip, local_ip=args.local_ip)
    native = adapter.native_robot
    second = None
    report = {
        "diagnostic": "rt_second_session_no_motion",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "motion_loop_started": False,
        "target_sent": False,
        "cleanup": [],
    }
    try:
        adapter.connect()
        native.rt_network_tolerance_percent = 20
        native.rt_filter_hz = 50.0
        native.enable_realtime(load=0.0)
        report["parent_prepared"] = native._realtime_prepared_state()

        sdk = _load_sdk()
        second = getattr(sdk, native.robot_class)()
        second.connectToRobot(args.robot_ip, args.local_ip)
        operation = _call(second, "operationState")
        power = _call(second, "powerState")
        mode = _call(second, "operateMode")
        report["second_session"] = {
            "operation_state": operation.name,
            "operation_state_value": int(operation.value),
            "power_state": power.name,
            "operate_mode": mode.name,
        }
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}:{exc}"
    finally:
        if second is not None:
            try:
                ec = {}
                second.disconnectFromRobot(ec)
                if int(ec.get("ec", 0)):
                    raise RuntimeError(str(ec))
                report["cleanup"].append("second_disconnected")
            except BaseException as exc:
                report["cleanup"].append(
                    f"second_disconnect_error:{type(exc).__name__}:{exc}"
                )
        for label, action in (
            ("parent_rt_stopped", lambda: native.stop_realtime(switch_to_nrt=True)),
            ("parent_powered_off", native.disable),
            ("parent_disconnected", adapter.disconnect),
        ):
            try:
                action()
                report["cleanup"].append(label)
            except BaseException as exc:
                report["cleanup"].append(f"{label}_error:{type(exc).__name__}:{exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
