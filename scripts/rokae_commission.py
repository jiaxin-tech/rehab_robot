"""Supervised ROKAE commissioning CLI: connect, inspect, power, jog, stop.

Examples
--------
Read-only connectivity self-test against the controller, then disconnect::

    python -m scripts.rokae_commission --ip 192.168.50.103 self-test

Status only (connects, prints state, then disconnects)::

    python -m scripts.rokae_commission --ip 192.168.50.103 status

Supervised power-on, small joint jog, then power-off::

    python -m scripts.rokae_commission --ip 192.168.50.103 --confirm \\
        power-on --speed-ratio 5
    python -m scripts.rokae_commission --ip 192.168.50.103 --confirm \\
        jog-joint --joint 1 --delta-deg 1.0
    python -m scripts.rokae_commission --ip 192.168.50.103 --confirm power-off

Interactive operator console::

    python -m scripts.rokae_commission --ip 192.168.50.103 console

Every power/motion action requires the exact confirmation text
``I CONFIRM SUPERVISED COMMISSIONING MOTION``.  A physical emergency stop and a
trained operator must remain available.  This tool is not a safety-rated stop.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from typing import Any, Callable, Sequence

from hardware.network_probe import DEFAULT_PROBE_PORTS, probe_controller
from hardware.rokae_commissioning import (
    COMMISSIONING_CONFIRMATION,
    CommissioningError,
    CommissioningLimits,
    RokaeCommissioningConsole,
    check_confirmation,
)

AdapterFactory = Callable[[str], Any]

JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _default_adapter_factory(robot_ip: str, *, local_ip: str = ""):
    from hardware.rokae_adapter import RokaeRobotAdapter

    return RokaeRobotAdapter(robot_ip, local_ip=local_ip)


def _build_limits(args: argparse.Namespace) -> CommissioningLimits:
    workspace_min = None
    workspace_max = None
    if args.workspace_min is not None and args.workspace_max is not None:
        workspace_min = tuple(float(v) for v in args.workspace_min)
        workspace_max = tuple(float(v) for v in args.workspace_max)
    return CommissioningLimits(
        max_joint_step_rad=math.radians(float(args.max_joint_step_deg)),
        max_cartesian_step_m=float(args.max_cartesian_step_mm) / 1000.0,
        max_speed_ratio=int(args.max_speed_ratio),
        joint_soft_limit_margin_rad=math.radians(float(args.soft_limit_margin_deg)),
        workspace_min_base_m=workspace_min,
        workspace_max_base_m=workspace_max,
        max_state_age_s=float(args.max_state_age_s),
        max_network_tolerance_percent=int(args.max_network_tolerance_percent),
        require_collision_query=not args.allow_missing_collision_query,
        require_joint_soft_limits=not args.allow_missing_soft_limits,
    )


def _open_console(
    args: argparse.Namespace,
    *,
    adapter_factory: AdapterFactory | None,
) -> RokaeCommissioningConsole:
    factory = adapter_factory or (
        lambda ip: _default_adapter_factory(ip, local_ip=getattr(args, "local_ip", "") or "")
    )
    adapter = factory(args.ip)
    console = RokaeCommissioningConsole(
        adapter,
        limits=_build_limits(args),
        audit_path=args.audit_log,
    )
    console.connect()
    return console


def _run_single(
    args: argparse.Namespace,
    *,
    adapter_factory: AdapterFactory | None,
) -> dict[str, Any]:
    command = args.command
    if command == "network":
        result = probe_controller(
            args.ip,
            ports=tuple(args.probe_port or DEFAULT_PROBE_PORTS),
            timeout_s=float(args.probe_timeout_s),
        )
        return result.to_dict()

    if command == "self-test":
        factory = adapter_factory or (
            lambda ip: _default_adapter_factory(
                ip, local_ip=getattr(args, "local_ip", "") or ""
            )
        )
        adapter = factory(args.ip)
        console = RokaeCommissioningConsole(
            adapter,
            limits=_build_limits(args),
            audit_path=args.audit_log,
        )
        result = console.self_test()
        return result

    console = _open_console(args, adapter_factory=adapter_factory)
    try:
        if command == "status":
            return {"status": console.status()}
        if command == "power-on":
            return console.power_on(
                confirmation=args.confirm,
                speed_ratio=args.speed_ratio,
            )
        if command == "power-off":
            return console.power_off(confirmation=args.confirm)
        if command == "reset":
            return console.reset(confirmation=args.confirm)
        if command == "prepare-realtime":
            return console.prepare_realtime(
                confirmation=args.confirm,
                network_tolerance_percent=args.network_tolerance_percent,
            )
        if command == "end-realtime":
            return console.end_realtime(confirmation=args.confirm)
        if command == "stop":
            return console.stop_motion("cli_stop")
        if command == "jog-joint":
            return console.jog_joint(
                args.joint,
                math.radians(args.delta_deg),
                confirmation=args.confirm,
            )
        if command == "jog-cartesian":
            delta = tuple(v / 1000.0 for v in args.delta_mm)
            return console.jog_cartesian(delta, confirmation=args.confirm)
        if command == "hold":
            return console.hold_pose(confirmation=args.confirm, seconds=args.seconds)
        raise ValueError(f"unsupported command: {command}")
    finally:
        shutdown = console.disconnect()
        if shutdown["errors"]:
            print(
                "cleanup errors: " + "; ".join(shutdown["errors"]),
                file=sys.stderr,
            )


def _console_help() -> str:
    return (
        "commands: status | network | self-test | power-on | power-off | reset | stop |\n"
        "          prepare-realtime | end-realtime |\n"
        "          jog-joint <1-6> <deg> | jog-cartesian <dx_mm> <dy_mm> <dz_mm> |\n"
        "          hold [seconds] | help | quit\n"
        f"power/motion require the exact text: {COMMISSIONING_CONFIRMATION}"
    )


def _interactive_console(
    args: argparse.Namespace,
    *,
    adapter_factory: AdapterFactory | None,
) -> int:
    console = _open_console(args, adapter_factory=adapter_factory)
    print(_console_help(), flush=True)
    try:
        while True:
            try:
                raw = input("rokae> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            parts = shlex.split(raw)
            if not parts:
                continue
            verb = parts[0].lower()
            try:
                if verb in ("quit", "exit"):
                    break
                if verb == "help":
                    print(_console_help())
                elif verb == "status":
                    print(json.dumps(_json_safe(console.status()), ensure_ascii=False, indent=2))
                elif verb == "self-test":
                    print(json.dumps(_json_safe(console.self_test()), ensure_ascii=False, indent=2))
                elif verb in ("power-on", "on"):
                    print(json.dumps(_json_safe(
                        console.power_on(confirmation=console_confirmation(args))
                    ), ensure_ascii=False, indent=2))
                elif verb in ("power-off", "off"):
                    print(json.dumps(_json_safe(
                        console.power_off(confirmation=console_confirmation(args))
                    ), ensure_ascii=False, indent=2))
                elif verb == "reset":
                    print(json.dumps(_json_safe(
                        console.reset(confirmation=console_confirmation(args))
                    ), ensure_ascii=False, indent=2))
                elif verb in ("prepare-realtime", "realtime-on"):
                    print(json.dumps(_json_safe(
                        console.prepare_realtime(confirmation=console_confirmation(args))
                    ), ensure_ascii=False, indent=2))
                elif verb in ("end-realtime", "realtime-off"):
                    print(json.dumps(_json_safe(
                        console.end_realtime(confirmation=console_confirmation(args))
                    ), ensure_ascii=False, indent=2))
                elif verb == "network":
                    print(json.dumps(_json_safe(
                        probe_controller(args.ip).to_dict()
                    ), ensure_ascii=False, indent=2))
                elif verb == "stop":
                    print(json.dumps(_json_safe(console.stop_motion("console_stop")), ensure_ascii=False, indent=2))
                elif verb == "jog-joint":
                    if len(parts) != 3:
                        raise CommissioningError("usage: jog-joint <1-6> <deg>")
                    joint = int(parts[1]) - 1
                    print(json.dumps(_json_safe(console.jog_joint(
                        joint,
                        math.radians(float(parts[2])),
                        confirmation=console_confirmation(args),
                    )), ensure_ascii=False, indent=2))
                elif verb == "jog-cartesian":
                    if len(parts) != 4:
                        raise CommissioningError("usage: jog-cartesian <dx_mm> <dy_mm> <dz_mm>")
                    delta = tuple(float(value) / 1000.0 for value in parts[1:4])
                    print(json.dumps(_json_safe(console.jog_cartesian(
                        delta,
                        confirmation=console_confirmation(args),
                    )), ensure_ascii=False, indent=2))
                elif verb == "hold":
                    seconds = float(parts[1]) if len(parts) == 2 else 1.0
                    print(json.dumps(_json_safe(console.hold_pose(
                        confirmation=console_confirmation(args),
                        seconds=seconds,
                    )), ensure_ascii=False, indent=2))
                else:
                    print(_console_help())
            except Exception as exc:
                print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        shutdown = console.disconnect()
        if shutdown["errors"]:
            print("cleanup errors: " + "; ".join(shutdown["errors"]), file=sys.stderr)
    return 0


def console_confirmation(args: argparse.Namespace) -> str | None:
    """Interactive console asks once when the CLI flag was not supplied."""
    if args.confirm == COMMISSIONING_CONFIRMATION:
        return args.confirm
    supplied = input(f"type exactly '{COMMISSIONING_CONFIRMATION}' to proceed: ").strip()
    return supplied or None


def _shared_option_specs():
    """(flags, kwargs) for options accepted before or after the subcommand."""
    return (
        (("--local-ip",), dict(
            help=(
                "this Windows PC's address on the robot subnet; required only for "
                "realtime preparation (xCoreSDK uses a second network channel)"
            ),
        )),
        (("--confirm",), dict(
            help=f"exact confirmation text for power/motion: {COMMISSIONING_CONFIRMATION}",
        )),
        (("--audit-log",), dict(help="optional JSONL path for the command audit trail")),
        (("--max-joint-step-deg",), dict(type=float, default=3.0)),
        (("--max-cartesian-step-mm",), dict(type=float, default=10.0)),
        (("--max-speed-ratio",), dict(type=int, default=10)),
        (("--soft-limit-margin-deg",), dict(type=float, default=3.0)),
        (("--max-state-age-s",), dict(type=float, default=0.5)),
        (("--workspace-min",), dict(type=float, nargs=3, metavar=("X", "Y", "Z"))),
        (("--workspace-max",), dict(type=float, nargs=3, metavar=("X", "Y", "Z"))),
        (("--allow-missing-collision-query",), dict(action="store_true", default=False)),
        (("--allow-missing-soft-limits",), dict(action="store_true", default=False)),
        (("--max-network-tolerance-percent",), dict(
            type=int,
            default=20,
            help="upper bound accepted for realtime network tolerance",
        )),
        (("--probe-port",), dict(
            type=int,
            action="append",
            help="TCP port to probe for the network command (repeatable)",
        )),
        (("--probe-timeout-s",), dict(type=float, default=0.5)),
    )


def _add_shared_options(target: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    """Add shared options; subcommand copies use SUPPRESS so they never clobber.

    Both ``--confirm X power-on`` and ``power-on --confirm X`` must work, and a
    value supplied before the subcommand must survive when the subcommand copy
    was not supplied.
    """
    for flags, kwargs in _shared_option_specs():
        options = dict(kwargs)
        if suppress_defaults:
            options["default"] = argparse.SUPPRESS
        target.add_argument(*flags, **options)


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    _add_shared_options(shared, suppress_defaults=False)
    sub_shared = argparse.ArgumentParser(add_help=False)
    _add_shared_options(sub_shared, suppress_defaults=True)
    parser = argparse.ArgumentParser(
        parents=[shared],
        description=(
            "Supervised ROKAE commissioning: SDK-free network check, read-only "
            "connectivity self-test, status, power, realtime preparation for the "
            "frozen experiment, bounded jog, and operator stop. Real motion needs "
            "a trained operator at the physical emergency stop."
        ),
    )
    parser.add_argument("--ip", required=True, help="ROKAE controller IP address")
    sub = parser.add_subparsers(dest="command", required=True)

    def child(name: str, **kwargs):
        return sub.add_parser(name, parents=[sub_shared], **kwargs)

    child("network", help="SDK-free ping/TCP reachability check (no robot commands)")
    child("self-test", help="read-only connectivity self-test with cleanup")
    child("status", help="print one robot status snapshot")
    off = child("power-on", help="automatic mode + servo power + default speed")
    off.add_argument("--speed-ratio", type=int)
    child("power-off", help="stop and power down the servos")
    child("reset", help="clear servo alarm and reset motion")
    realtime = child(
        "prepare-realtime",
        help="select realtime command mode and power on without starting motion",
    )
    realtime.add_argument("--network-tolerance-percent", type=int)
    child(
        "end-realtime",
        help="stop realtime, return to non-realtime mode, and power down",
    )
    child("stop", help="operator stop (no confirmation required)")
    joint = child("jog-joint", help="relative jog of one joint")
    joint.add_argument("--joint", type=int, required=True, help="1-based joint index 1..6")
    joint.add_argument("--delta-deg", type=float, required=True)
    cart = child("jog-cartesian", help="relative base-frame Cartesian jog")
    cart.add_argument("--delta-mm", type=float, nargs=3, required=True, metavar=("DX", "DY", "DZ"))
    hold = child("hold", help="command the current pose and hold it")
    hold.add_argument("--seconds", type=float, default=1.0)
    child("console", help="interactive operator console")
    return parser




def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: AdapterFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    # Refuse a missing/typoed confirmation before any adapter or socket exists.
    action = {
        "power-on": "power_on",
        "power-off": "power_off",
        "reset": "reset",
        "prepare-realtime": "prepare_realtime",
        "end-realtime": "end_realtime",
        "jog-joint": "jog_joint",
        "jog-cartesian": "jog_cartesian",
        "hold": "hold_pose",
    }.get(args.command)
    if action is not None:
        try:
            check_confirmation(action, args.confirm)
        except CommissioningError as exc:
            print(f"blocked: {exc}", file=sys.stderr)
            return 2
    if args.command == "jog-joint":
        args.joint = int(args.joint) - 1
        if not 0 <= args.joint <= 5:
            raise SystemExit("--joint must be 1..6")
    try:
        if args.command == "console":
            return _interactive_console(args, adapter_factory=adapter_factory)
        result = _run_single(args, adapter_factory=adapter_factory)
    except CommissioningError as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not (isinstance(result, dict) and result.get("success") is False) else 1


if __name__ == "__main__":
    raise SystemExit(main())