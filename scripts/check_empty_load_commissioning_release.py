"""Offline validation of a reviewed empty-load commissioning release."""
from __future__ import annotations

import argparse
import json

from control.execution_preflight import (
    OPERATOR_CONFIRMATION,
    evaluate_offline_execution_request,
)
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import (
    build_start_anchored_relative_trajectory,
    load_rehab_frame_config,
)
from safety.empty_load_commissioning_release import (
    load_and_validate_empty_load_commissioning_release,
)
from safety.experiment_safety import load_experiment_safety_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--safety-config", required=True)
    parser.add_argument("--frame-config", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--wrench-config", required=True)
    args = parser.parse_args(argv)
    release = load_and_validate_empty_load_commissioning_release(
        args.release,
        reference_path=args.reference,
        safety_config_path=args.safety_config,
        frame_config_path=args.frame_config,
        anchor_path=args.anchor,
        wrench_config_path=args.wrench_config,
    )
    safety = load_experiment_safety_config(args.safety_config)
    frame = load_rehab_frame_config(args.frame_config)
    anchor = load_start_anchor(args.anchor)
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        args.reference,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    result = evaluate_offline_execution_request(
        mode="execute",
        enable_motion=True,
        operator_confirmation=OPERATOR_CONFIRMATION,
        requested_anchor_id=anchor.anchor_id,
        frame=frame,
        anchor=anchor,
        safety=safety,
        trajectory=trajectory,
        audit=audit,
        release_approval=release,
    )
    print(json.dumps(result.as_metadata(), ensure_ascii=False, indent=2))
    return 0 if result.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
