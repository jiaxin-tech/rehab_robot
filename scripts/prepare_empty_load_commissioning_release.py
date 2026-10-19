"""Create one reviewed, hash-bound empty-load commissioning release."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from control.execution_preflight import (
    audit_execution_trajectory_content,
    experiment_safety_digest,
    trajectory_execution_digest,
)
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import (
    build_start_anchored_relative_trajectory,
    load_rehab_frame_config,
)
from safety.empty_load_commissioning_release import (
    CODE_FILES,
    EXPECTED_COLLISION_ERROR_CODE,
    SCOPE,
    load_and_validate_empty_load_commissioning_release,
    sha256_file,
    validate_retimed_reference,
)
from safety.experiment_safety import load_experiment_safety_config
from scripts.acquire_robot_data import load_live_wrench_config


ROOT = Path(__file__).resolve().parents[1]


def bounded_expiry(reviewed_expiry, minutes, issued_at):
    if not math.isfinite(minutes) or not 0 < minutes <= 30:
        raise ValueError("valid-for-minutes must be positive and at most 30")
    expiry = datetime.fromisoformat(reviewed_expiry.replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= issued_at:
        raise PermissionError("reviewed request expiry must be timezone-aware and future")
    return min(expiry, issued_at + timedelta(minutes=minutes))


def _object_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-request", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--safety-config", type=Path, required=True)
    parser.add_argument("--frame-config", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--wrench-config", type=Path, required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--valid-for-minutes", type=float, default=30.0,
                        help="shorten release validity to at most 30 minutes and never extend the reviewed expiry")
    args = parser.parse_args(argv)

    request_path = args.review_request.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("scope") != SCOPE:
        raise PermissionError("review request is not for empty-load commissioning")
    if args.confirmation != request.get("required_confirmation"):
        raise PermissionError("exact reviewed confirmation is required")
    issued_at = datetime.now(timezone.utc)
    expires_at = bounded_expiry(request["expires_at"], args.valid_for_minutes, issued_at)
    validate_retimed_reference(args.reference)
    safety = load_experiment_safety_config(args.safety_config)
    safety.require_execute_allowed()
    frame = load_rehab_frame_config(args.frame_config)
    anchor = load_start_anchor(args.anchor)
    wrench = load_live_wrench_config(args.wrench_config)
    if not frame.reviewed or not anchor.reviewed:
        raise PermissionError("frame and anchor must be reviewed")
    trajectory, audit, _ = build_start_anchored_relative_trajectory(
        args.reference,
        current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame,
    )
    reasons, _, _ = audit_execution_trajectory_content(
        trajectory,
        safety,
        frame=frame,
        anchor=anchor,
        require_official_reference=False,
    )
    if reasons:
        raise PermissionError("commissioning trajectory blocked: " + ";".join(reasons))

    paths = {
        "reference": args.reference.resolve(),
        "safety": args.safety_config.resolve(),
        "frame": args.frame_config.resolve(),
        "anchor": args.anchor.resolve(),
        "wrench": args.wrench_config.resolve(),
    }
    output = args.output.resolve()
    try:
        request_reference = str(request_path.relative_to(output.parent))
    except ValueError:
        request_reference = str(request_path)
    payload = {
        "schema_version": 1,
        "scope": SCOPE,
        "approval_id": "ELC-" + uuid4().hex[:12].upper(),
        "approved": True,
        "reviewer": request["reviewer"],
        "operator": request["operator"],
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "single_cycle": True,
        "training_allowed": False,
        "reference_sha256": sha256_file(paths["reference"]),
        "review_request": {
            "path": request_reference,
            "sha256": sha256_file(request_path),
        },
        "bindings": {name: sha256_file(path) for name, path in paths.items()},
        "object_bindings": {
            "safety": experiment_safety_digest(safety),
            "frame": _object_digest(frame.as_metadata()),
            "anchor": _object_digest(anchor.to_dict()),
            "trajectory": trajectory_execution_digest(trajectory),
        },
        "robot_identity": {
            "robot_model": safety.expected_robot_model,
            "robot_serial_number": safety.expected_robot_serial_number,
            "controller_version": safety.expected_controller_version,
            "sdk_version": wrench["expected_identity"]["sdk_version"],
        },
        "wrench_hz": 50.0,
        "wrench_config_object": wrench,
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "allow_missing_sdk_collision_state_error": EXPECTED_COLLISION_ERROR_CODE,
        "controller_collision_protection_required": True,
        "operator_emergency_stop_required": True,
        "confirmation": args.confirmation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    load_and_validate_empty_load_commissioning_release(
        output,
        reference_path=paths["reference"],
        safety_config_path=paths["safety"],
        frame_config_path=paths["frame"],
        anchor_path=paths["anchor"],
        wrench_config_path=paths["wrench"],
    )
    print(json.dumps({"approved": True, "output": str(output), "sha256": sha256_file(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
