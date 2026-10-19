"""Observe the vendor safety callback without active safety queries or motion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from hardware.safety_events import validate_collision_source


def diagnose(*, robot_ip, local_ip, output, duration_s, max_age_s,
             robot_factory=None, clock=time.perf_counter, sleep=time.sleep):
    validate_collision_source("events", max_age_s)
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not robot_ip or not local_ip:
        raise ValueError("explicit robot_ip and local_ip required")
    output = Path(output)
    # Reserve the evidence path before any connection; never overwrite a log.
    with output.open("x", encoding="utf-8") as stream:
        report = dict(schema_version=1, diagnostic="COLLISION_EVENT_OBSERVATION_ONLY",
            started_at=datetime.now(timezone.utc).isoformat(), robot_ip=robot_ip,
            local_ip=local_ip, collision_source="events", max_age_s=max_age_s,
            budget_status="DIAGNOSTIC_INPUT_NOT_EXECUTION_APPROVAL",
            duration_s=duration_s, snapshots=[], errors=[], cleanup={},
            motion_requested=False, execution_authorized=False,
            snapshot_semantics="Sampled callback cache, not an exhaustive raw event log")
        robot = None
        attempted = False
        try:
            if robot_factory is None:
                from hardware.windows.rokae_xcore import RokaeRobot
                robot_factory = RokaeRobot
            robot = robot_factory(robot_ip, local_ip=local_ip, collision_source="events",
                                  collision_event_max_age_s=max_age_s)
            attempted = True
            robot.connect()
            info = getattr(robot, "_robot_info", None)
            report["identity"] = dict(sdk=getattr(robot, "_sdk_version", None),
                model=str(info.type) if info is not None else None,
                controller=str(info.version) if info is not None else None,
                serial=str(info.id) if info is not None else None)
            deadline = clock() + duration_s
            while True:
                status = robot.get_safety_event_status()
                if not isinstance(status, dict):
                    raise RuntimeError("event status unavailable")
                report["snapshots"].append(dict(host_time_s=clock(), **status))
                remaining = deadline - clock()
                if remaining <= 0:
                    break
                sleep(min(.02, remaining))
        except Exception as exc:
            report["errors"].append(f"{type(exc).__name__}:{exc}")
        finally:
            if attempted:
                try:
                    robot.disconnect()
                    report["cleanup"]["disconnected_confirmed"] = not robot.is_connected
                    report["cleanup"]["status_after_disconnect"] = robot.get_safety_event_status()
                except Exception as exc:
                    report["cleanup"]["disconnected_confirmed"] = False
                    report["errors"].append(f"cleanup:{type(exc).__name__}:{exc}")
            snapshots = report["snapshots"]
            last = snapshots[-1] if snapshots else {}
            report["assessment"] = dict(
                observed_event_count=max((s.get("sequence", 0) for s in snapshots), default=0),
                observed_clear=any(s.get("valid") is True and s.get("collision_state") is False for s in snapshots),
                observed_collision=any(s.get("collision_state") is True or s.get("collision_latched") is True for s in snapshots),
                final_status_valid=last.get("valid") is True,
                unresolved_reasons=sorted({s["invalid_reason"] for s in snapshots if s.get("invalid_reason")}),
                execution_authorized=False)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--max-age-s", type=float, required=True)
    args = parser.parse_args(argv)
    report = diagnose(robot_ip=args.robot_ip, local_ip=args.local_ip, output=args.output,
                      duration_s=args.duration_s, max_age_s=args.max_age_s)
    print(json.dumps(dict(output=str(args.output), assessment=report["assessment"],
                          errors=report["errors"]), ensure_ascii=False))
    assessment = report["assessment"]
    return 0 if (assessment["final_status_valid"] and not assessment["observed_collision"]
                 and not report["errors"] and report["cleanup"].get("disconnected_confirmed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
