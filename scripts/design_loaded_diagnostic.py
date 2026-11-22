"""Design and manually review a small passive dummy-leg diagnostic, offline.

This tool never connects to hardware or establishes physical path compatibility.
The review subcommand records an operator's actual manual inspection of the
whole small path, current installation and selected raw TCP force limits.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from config.bundle import ROOT, configuration
from control.continuous_taught_trajectory import (
    TRAJECTORY_ID, audit_continuous_taught_trajectory,
    build_continuous_taught_trajectory,
)
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from measurement_validation.loaded_trial import PROTOCOL, validate_config
from safety.experiment_safety import ExperimentSafetyConfig
from scripts.preview_rehab_trajectory import _plot_preview

DEFAULT_SOURCE = ROOT / "results/manual_lab/20261122_143700/continuous_empty_01"
MAX_REVIEW_ANCHOR_AGE_S = 1800.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _repo_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _settings(config):
    return {key: value for key, value in config["trajectory"].items()
            if key not in ("mode", "fixed_cycle_duration_s")}


def design(config_path, output_dir, *, output_config, source_empty_session=DEFAULT_SOURCE):
    """Create a blocked template and plots from the empty-motion source anchor."""
    output, config_output = Path(output_dir).resolve(), Path(output_config).resolve()
    if output.exists() or config_output.exists():
        raise FileExistsError("choose new output paths; previous designs are preserved")
    source = Path(source_empty_session).resolve()
    candidate = copy.deepcopy(configuration(config_path))
    trajectory = candidate["trajectory"]
    if trajectory.get("mode") != "continuous_taught_tcp":
        raise ValueError("loaded diagnostic design needs the continuous taught source")
    trajectory.update(scale_x=.03, scale_y=.03, scale_z=.03,
                      cycle_duration_s=90., fixed_cycle_duration_s=90.,
                      ramp_duration_s=5., sample_period_s=.02)
    for section in ("motion", "empty_load"):
        candidate[section].update(max_tcp_speed_m_s=.005,
                                  max_tcp_acceleration_m_s2=.01)
    candidate["training"].update(enabled=False, mode="disabled")
    candidate["validation"]["capture_dummy_leg"] = True
    # The force-learning catalog is inapplicable to this diagnostic.
    candidate.pop("force_training", None)
    candidate["loaded_trial"] = dict(
        protocol=PROTOCOL, setup_id=None, connection_type="rigid_fixture",
        tool_description=None, support_description=None, reviewed=False,
        manual_path_checked=False, reviewed_anchor_pose_base=None,
        reviewed_geometry_sha256=trajectory["geometry_sha256"],
        reviewed_scale_xyz=[trajectory[f"scale_{axis}"] for axis in "xyz"],
        review_notes=None, reviewed_at=None, strict_wrench_freshness=True,
        source_empty_session=_repo_path(source),
        source_empty_status_sha256=sha(source / "round_status.json"),
    )
    validate_config(candidate, require_review=False)
    anchor = load_start_anchor(source / "anchor.json")
    frame = load_rehab_frame_config(config_path)
    table, audit, metadata = build_continuous_taught_trajectory(
        _settings(candidate), current_tcp_start_pose=anchor.tcp_pose_base,
        rehab_frame=frame)
    reasons, vmax, amax = audit_continuous_taught_trajectory(
        table, ExperimentSafetyConfig(**candidate["motion"]),
        expected_trajectory_id=TRAJECTORY_ID, frame=frame)
    if reasons:
        raise ValueError("offline diagnostic geometry audit failed: " + ";".join(reasons))
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "trajectory_preview.csv", index=False)
    figures = _plot_preview(table, output)
    write_new_json(config_output, candidate)
    xyz = table[[f"tcp_{axis}_base" for axis in "xyz"]].to_numpy(float)
    deltas = xyz - xyz[0]
    geometry = Path(trajectory["geometry_path"])
    if not geometry.is_absolute():
        geometry = ROOT / geometry
    summary = dict(
        schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        status="BLOCKED_RIGID_CONNECTION_REQUIRES_COUPLED_POSE",
        purpose="passive_dummy_leg_diagnostic_geometry_comparison_only", hardware_accessed=False,
        ready_for_replay=False, formal_force_comparison_ready=False,
        interpretation="Raw robot TCP wrench, not isolated strap tension or human comfort",
        trajectory_metadata=metadata, settings=_settings(candidate),
        duration_s=float(table.time_s.iloc[-1]), command_count=len(table),
        constant_speed_interval_s=[5., 85.], internal_holds_s=0.,
        positive_excursion_base_m={axis: float(deltas[:, i].max()) for i, axis in enumerate("xyz")},
        negative_excursion_base_m={axis: float(deltas[:, i].min()) for i, axis in enumerate("xyz")},
        audited_maximum_speed_m_s=vmax, audited_maximum_acceleration_m_s2=amax,
        trajectory_audit_reasons=list(reasons),
        preview_anchor_use="historical empty-motion anchor for offline plotting only",
        preview_anchor_source=_repo_path(source / "anchor.json"),
        preview_anchor_source_sha256=sha(source / "anchor.json"),
        preview_anchor_pose_base=list(anchor.tcp_pose_base),
        source_empty_force_quality="not approved; source has documented wrench gaps",
        configured_force_caps_use="inherited machine caps, unreviewed for this installation",
        source_config_sha256=sha(config_path), geometry_sha256=sha(geometry),
        source_code_sha256={name: sha(ROOT / name) for name in (
            "scripts/design_loaded_diagnostic.py", "control/continuous_taught_trajectory.py")},
        output_config=_repo_path(config_output), output_config_sha256=sha(config_output),
        output_files_sha256={p.name: sha(p) for p in [output / "trajectory_preview.csv", *figures]},
        blockers=["Reported rigid lower-leg fixture is incompatible with this unverified fixed-orientation TCP proposal",
                  "A coupled position/orientation trajectory requires actual leg geometry and fixture information",
                  "Reducing amplitude does not establish compatibility with a rigid leg connection",
                  "No manual small-path compatibility check has been completed",
                  "Current installation, support, tool and reviewed start anchor are missing",
                  "Raw TCP force and torque limits require installation-specific review",
                  "Source empty-run force coverage failed; no formal load comparison approval"],
    )
    write_new_json(output / "summary.json", summary)
    return summary


def review(config_path, anchor_path, output, *, setup_id, connection_type,
           tool_description, support_description, review_notes,
           manual_path_checked, max_force_n, max_torque_nm, now=None):
    """Record completed physical checks, never infer them from an offline plot."""
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("review requires a new output file")
    if manual_path_checked is not True:
        raise ValueError("review requires --manual-path-checked after the actual complete small-path check")
    candidate = copy.deepcopy(configuration(config_path))
    validate_config(candidate, require_review=False)
    anchor = load_start_anchor(anchor_path)
    if anchor.trajectory_id != TRAJECTORY_ID:
        raise ValueError("capture a current continuous-taught start anchor")
    source = Path(candidate["loaded_trial"]["source_empty_session"])
    if not source.is_absolute():
        source = ROOT / source
    source_anchor_path = source / "anchor.json"
    source_anchor = load_start_anchor(source_anchor_path)
    if anchor.anchor_id == source_anchor.anchor_id or sha(anchor_path) == sha(source_anchor_path):
        raise ValueError("historical empty-load anchor cannot bind a connected-leg manual review")
    stamp = now or datetime.now(timezone.utc)
    captured = datetime.fromisoformat(anchor.created_at)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("review time requires a timezone")
    age = (stamp - captured).total_seconds()
    if age < -5 or age > MAX_REVIEW_ANCHOR_AGE_S:
        raise ValueError("capture the installed start anchor within 30 minutes before review")
    candidate["motion"].update(max_force_n=max_force_n, max_torque_nm=max_torque_nm)
    candidate["loaded_trial"].update(
        setup_id=setup_id, connection_type=connection_type,
        tool_description=tool_description, support_description=support_description,
        reviewed=True, manual_path_checked=True,
        reviewed_anchor_pose_base=list(anchor.tcp_pose_base),
        review_notes=review_notes, reviewed_at=stamp.isoformat(),
    )
    validate_config(candidate, require_review=True)
    write_new_json(destination, candidate)
    return dict(status="MANUAL_REVIEW_RECORDED", output=str(destination),
                hardware_accessed=False, motion_commanded=False,
                setup_id=setup_id, reviewed_anchor_id=anchor.anchor_id,
                next_step="lab_session check, then prepare a new session; review is not a runtime clearance")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("design", help="create a blocked 3%-amplitude 90-second preview")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--source-empty-session", type=Path, default=DEFAULT_SOURCE)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--output-config", type=Path, required=True)
    reviewed = sub.add_parser("review", help="record actual installation and complete small-path manual checks")
    reviewed.add_argument("--config", type=Path, required=True)
    reviewed.add_argument("--anchor", type=Path, required=True,
                          help="newly captured installed start anchor, no older than 30 minutes")
    reviewed.add_argument("--setup-id", required=True)
    reviewed.add_argument("--connection-type", required=True,
                          choices=("rope_or_strap", "passive_joint", "rigid_fixture"))
    reviewed.add_argument("--tool-description", required=True)
    reviewed.add_argument("--support-description", required=True)
    reviewed.add_argument("--review-notes", required=True,
                          help="actual small-path inspection and reason for the selected raw TCP force limits")
    reviewed.add_argument("--manual-path-checked", action="store_true", required=True,
                          help="declare only after manually checking the entire installed small path")
    reviewed.add_argument("--max-force-n", type=float, required=True,
                          help="installation-reviewed raw TCP force threshold, positive and <=100 N")
    reviewed.add_argument("--max-torque-nm", type=float, required=True,
                          help="installation-reviewed raw TCP torque threshold, positive and <=200 Nm")
    reviewed.add_argument("--output", type=Path, required=True, help="new reviewed config; never overwrites")
    args = parser.parse_args(argv)
    if args.action == "design":
        result = design(args.config, args.output_dir, output_config=args.output_config,
                        source_empty_session=args.source_empty_session)
        result = {k: result[k] for k in ("status", "duration_s", "positive_excursion_base_m",
                  "audited_maximum_speed_m_s", "audited_maximum_acceleration_m_s2")}
    else:
        result = review(args.config, args.anchor, args.output,
                        setup_id=args.setup_id, connection_type=args.connection_type,
                        tool_description=args.tool_description, support_description=args.support_description,
                        review_notes=args.review_notes, manual_path_checked=args.manual_path_checked,
                        max_force_n=args.max_force_n, max_torque_nm=args.max_torque_nm)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
