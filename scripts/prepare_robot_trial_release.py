"""Create or approve a hash-bound first-robot-trial release artifact.

The default action writes a draft. ``--approve`` requires the reviewed safety,
frame, anchor, and live-wrench inputs to already pass their local checks and
requires the exact reviewer confirmation string. It never changes the frozen
reference manifest or turns a robot on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety.robot_trial_release import (
    build_robot_trial_release_approval,
    read_json,
    sha256_file,
    write_robot_trial_release,
    validate_robot_trial_release_payload,
    APPROVAL_CONFIRMATION,
    SCOPE, LOADED_SCOPE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--safety-config", type=Path, required=True)
    parser.add_argument("--frame-config", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--wrench-config", type=Path, required=True)
    parser.add_argument("--robot-model", required=True)
    parser.add_argument("--robot-serial-number", required=True)
    parser.add_argument("--controller-version", required=True)
    parser.add_argument("--sdk-version", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--collision-source", choices=("query", "events"), default="events")
    parser.add_argument("--collision-event-max-age-s", type=float, default=None)
    parser.add_argument("--wrench-hz", type=float, required=True)
    parser.add_argument("--scope", choices=(SCOPE, LOADED_SCOPE), default=SCOPE)
    parser.add_argument("--trial-setup", type=Path)
    parser.add_argument("--notes", default="")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--site-review", type=Path, default=None,
                        help="required JSON site review when approving an existing draft")
    parser.add_argument("--input-draft", type=Path, default=None,
                        help="existing draft to approve; omit to create a new draft")
    args = parser.parse_args(argv)

    if args.approve:
        if args.confirmation != APPROVAL_CONFIRMATION:
            raise PermissionError(f"approval requires exact confirmation: {APPROVAL_CONFIRMATION}")
        if args.input_draft is None or args.site_review is None:
            raise ValueError("--approve requires --input-draft and --site-review")
        payload = read_json(args.input_draft)
        if payload.get("approved") is not False:
            raise ValueError("input draft must have approved=false")
        site_review = read_json(args.site_review)
        if site_review.get("scope") != payload.get("scope") or site_review.get("decision") != "approved":
            raise PermissionError("site review scope or decision is invalid")
        if site_review.get("bundle_sha256") != payload.get("bundle_sha256"):
            raise PermissionError("site review bundle_sha256 does not match draft")
        if site_review.get("reviewer") != payload.get("reviewer"):
            raise PermissionError("site review reviewer does not match draft")
        payload["approved"] = True
        payload["site_review"] = {
            "path": str(args.site_review.resolve()),
            "sha256": sha256_file(args.site_review),
        }
        validate_robot_trial_release_payload(
            payload, artifact_path=args.output, reference_path=args.reference,
            safety_config_path=args.safety_config, frame_config_path=args.frame_config,
            anchor_path=args.anchor, wrench_config_path=args.wrench_config)
        if (args.scope != payload["scope"] or args.wrench_hz != payload["wrench_hz"] or args.collision_source != payload["collision_source"]
                or args.collision_event_max_age_s != payload["collision_event_max_age_s"]):
            raise PermissionError("approval acquisition arguments differ from draft")
        identity = dict(robot_model=args.robot_model, robot_serial_number=args.robot_serial_number,
                        controller_version=args.controller_version, sdk_version=args.sdk_version)
        if (identity != payload["robot_identity"] or args.reviewer != payload["reviewer"]
                or args.operator != payload["operator"] or args.expires_at != payload["expires_at"]):
            raise PermissionError("approval identity/reviewer/operator/expiry arguments differ from draft")
        write_robot_trial_release(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    payload = build_robot_trial_release_approval(
        reviewer=args.reviewer,
        operator=args.operator,
        expires_at=args.expires_at,
        robot_identity={
            "robot_model": args.robot_model,
            "robot_serial_number": args.robot_serial_number,
            "controller_version": args.controller_version,
            "sdk_version": args.sdk_version,
        },
        reference_path=args.reference,
        safety_config_path=args.safety_config,
        frame_config_path=args.frame_config,
        anchor_path=args.anchor,
        wrench_config_path=args.wrench_config,
        collision_source=args.collision_source,
        collision_event_max_age_s=args.collision_event_max_age_s,
        wrench_hz=args.wrench_hz,
        scope=args.scope, trial_setup_path=args.trial_setup,
        notes=args.notes,
    )
    write_robot_trial_release(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
