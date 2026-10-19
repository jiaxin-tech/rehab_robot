"""Prepare/check the formal-reference temporary commissioning artifact.

This command never connects to the robot, changes power, or sends motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from control.reference_commissioning import (
    build_reference_commissioning_artifact,
    load_reference_commissioning_policy,
)
from scripts.run_robot_leg_round import check_config


def inspect(policy_path: Path, round_config: Path) -> dict:
    policy = load_reference_commissioning_policy(policy_path)
    blockers = check_config(round_config)
    return {
        "mode": policy.metadata()["mode"],
        "policy_valid": True,
        "motion_authorized": False,
        "controller_speed_ratio_percent": policy.controller_speed_ratio_percent,
        "max_tcp_speed_m_s": policy.max_tcp_speed_m_s,
        "remaining_motion_blockers": blockers,
        "note": "check/prepare are offline-only and cannot move or power the robot",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare"))
    parser.add_argument("--policy", type=Path, default=Path("config/reference_commissioning.json"))
    parser.add_argument("--round-config", type=Path, default=Path("config/robot_leg_round.json"))
    parser.add_argument("--session", type=Path)
    args = parser.parse_args(argv)
    status = inspect(args.policy, args.round_config)
    if args.command == "prepare":
        if args.session is None:
            parser.error("prepare requires --session")
        policy = load_reference_commissioning_policy(args.policy)
        status = build_reference_commissioning_artifact(
            policy,
            args.session,
            blockers=status["remaining_motion_blockers"],
        )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if not status.get("remaining_motion_blockers") else 1


if __name__ == "__main__":
    raise SystemExit(main())
