"""Freeze an exact-duration taught-path proposal and report physical conflicts."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from config.bundle import ROOT, configuration
from control.taught_tcp_trajectory import build_taught_tcp_trajectory, _load_geometry, DURATIONS, HOLDS
from control.taught_tcp_timing import review_fixed_cycle
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from scripts.preview_rehab_trajectory import _plot_preview


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def preview(settings_path, config_path, output_dir, *, durations_s=(22., 17., 25.),
            cycle_duration_s=70., output_config=None):
    output = Path(output_dir).resolve()
    if output.exists() or (output_config is not None and Path(output_config).exists()):
        raise FileExistsError("choose new output paths; existing designs are preserved")
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8-sig"))
    geometry, _ = _load_geometry(settings)
    config = configuration(config_path)
    durations = np.asarray(durations_s, dtype=float)
    if durations.shape != (3,) or not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("three finite positive movement durations required")
    settings.update(dict(zip(DURATIONS, durations.tolist())))
    # Validate cycle and all settings before creating any proposal files.
    limits = {k: min(config["motion"][k], config["empty_load"][k])
              for k in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2")}
    review = review_fixed_cycle(settings, cycle_duration_s,
        speed_limit_m_s=limits["max_tcp_speed_m_s"],
        acceleration_limit_m_s2=limits["max_tcp_acceleration_m_s2"],
        margin=config["empty_load"]["retiming_margin"])
    if "fixed_cycle_duration_mismatch" in review["reasons"]:
        raise ValueError("movement plus all holds must equal the requested fixed cycle")
    output.mkdir(parents=True, exist_ok=False)
    old_geometry = Path(settings["geometry_path"])
    if not old_geometry.is_absolute():
        old_geometry = ROOT / old_geometry
    frozen_geometry = output / "geometry.json"
    shutil.copyfile(old_geometry, frozen_geometry)
    settings["geometry_path"] = (frozen_geometry.relative_to(ROOT).as_posix()
        if frozen_geometry.is_relative_to(ROOT) else str(frozen_geometry))
    table, _, metadata = build_taught_tcp_trajectory(settings,
        current_tcp_start_pose=geometry["recorded_anchor_pose"],
        rehab_frame=load_rehab_frame_config(config_path))
    write_json(output / "settings.json", settings)
    table.to_csv(output / "trajectory_preview.csv", index=False)
    figures = _plot_preview(table, output)
    candidate = copy.deepcopy(config)
    candidate["trajectory"] = dict(mode="taught_tcp", fixed_cycle_duration_s=float(cycle_duration_s), **settings)
    candidate["training"].update(enabled=False, mode="disabled")
    candidate["validation"]["capture_dummy_leg"] = False
    if output_config is not None:
        write_json(output_config, candidate)
    summary = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        status="OFFLINE_TIMING_REVIEW_REQUIRED" if not review["allowed"] else "OFFLINE_TIMING_COMPLETE",
        ready_for_replay=False, hardware_accessed=False, execution_review=review,
        shape_changed=False, orientation_strategy="fixed_at_current_anchor", settings=settings,
        phase_boundaries=metadata["phase_boundaries"],
        source_settings_sha256=sha(settings_path), source_config_sha256=sha(config_path),
        geometry_sha256=sha(frozen_geometry),
        physical_limits_changed=False, output_config=str(Path(output_config).resolve()) if output_config else None,
        output_config_sha256=sha(output_config) if output_config else None,
        processing_source_sha256={name:sha(ROOT / name) for name in (
            "scripts/preview_taught_timing.py", "control/taught_tcp_timing.py", "control/taught_tcp_trajectory.py")},
        output_files_sha256={p.name:sha(p) for p in [frozen_geometry, output/"settings.json", output/"trajectory_preview.csv", *figures]})
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--durations-s", nargs=3, type=float, default=(22.,17.,25.))
    parser.add_argument("--cycle-duration-s", type=float, default=70.)
    args = parser.parse_args(argv)
    result = preview(args.settings, args.config, args.output_dir,
        durations_s=args.durations_s, cycle_duration_s=args.cycle_duration_s, output_config=args.output_config)
    print(json.dumps({"status":result["status"], "execution_review":result["execution_review"]}))
    # Successful offline generation is distinct from passing the motion gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
