"""Capture one unreviewed StartAnchor without an explicit motion command.

Vendor SDK construction/connect/disconnect may still have controller-session
side effects, so first use remains a supervised hardware-validation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from control.start_anchor import capture_start_anchor, save_start_anchor
from control.start_anchored_relative_trajectory import (
    APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256,
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
)
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)


AdapterFactory = Callable[[str], Any]


def _default_adapter_factory(robot_ip: str):
    from hardware.rokae_adapter import RokaeRobotAdapter

    return RokaeRobotAdapter(robot_ip)


def run_capture(
    *,
    robot_ip: str,
    output_path: str | Path,
    reference: str | Path = DEFAULT_REFERENCE_PATH,
    anchor_id: str | None = None,
    tool_name: str | None = None,
    workpiece_name: str | None = None,
    notes: str = "",
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    reference_frame, reference_metadata = load_closed_reference_trajectory(reference)
    trajectory_ids = reference_frame["trajectory_id"].astype(str).unique().tolist()
    if trajectory_ids != [FIRST_ROBOT_TRIAL_TRAJECTORY_ID]:
        raise ValueError(
            "first anchor capture accepts only "
            f"{FIRST_ROBOT_TRIAL_TRAJECTORY_ID}"
        )
    if (
        reference_metadata.get("sha256")
        != APPROVED_FIRST_ROBOT_TRIAL_REFERENCE_SHA256
    ):
        raise ValueError(
            "first anchor capture requires the pinned reviewed slow-reference SHA-256"
        )
    first = reference_frame.iloc[0]
    adapter = (adapter_factory or _default_adapter_factory)(robot_ip)
    connect_attempted = False
    stream_attempted = False
    try:
        connect_attempted = True
        adapter.connect()
        stream_attempted = True
        adapter.start_state_stream()
        anchor = capture_start_anchor(
            adapter,
            trajectory_id=trajectory_ids[0],
            reference_start_q_hip=float(first["q_hip_rad"]),
            reference_start_q_knee=float(first["q_knee_rad"]),
            anchor_id=anchor_id,
            tool_name=tool_name,
            workpiece_name=workpiece_name,
            notes=notes,
        )
        destination = save_start_anchor(anchor, output_path)
        return {
            "saved": True,
            "path": str(destination.resolve()),
            "anchor": anchor.to_dict(),
            "motion_commanded": False,
            "requires_manual_review": True,
        }
    finally:
        cleanup_errors: list[BaseException] = []
        if stream_attempted:
            try:
                adapter.stop_state_stream()
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
        if connect_attempted:
            try:
                adapter.disconnect()
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
        if cleanup_errors:
            import sys

            active_exception = sys.exc_info()[1]
            detail = ";".join(
                f"{type(item).__name__}:{item}" for item in cleanup_errors
            )
            if active_exception is not None:
                if hasattr(active_exception, "add_note"):
                    active_exception.add_note("anchor capture cleanup failure: " + detail)
            else:
                raise RuntimeError(
                    "anchor capture cleanup failure: " + detail
                ) from cleanup_errors[0]


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: AdapterFactory | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "ROKAE start-anchor observation; project code sends no explicit "
            "motion/power/mode command"
        )
    )
    parser.add_argument("--ip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--anchor-id")
    parser.add_argument(
        "--tool-name",
        help="operator-declared SDK/HMI tool name; capture does not verify it is active",
    )
    parser.add_argument(
        "--workpiece-name",
        help="operator-declared SDK/HMI workpiece name; capture does not verify it is active",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    result = run_capture(
        robot_ip=args.ip,
        output_path=args.output,
        reference=args.reference,
        anchor_id=args.anchor_id,
        tool_name=args.tool_name,
        workpiece_name=args.workpiece_name,
        notes=args.notes,
        adapter_factory=adapter_factory,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
