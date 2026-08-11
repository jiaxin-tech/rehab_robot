"""Command-line entry point for Stage 5B reference-path retiming."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import reference_retiming_data_dir, reference_trajectory_data_dir
from .reference_trajectory_retiming import run_reference_retiming


def _approved_pair(
    parser: argparse.ArgumentParser,
    minimum: float | None,
    maximum: float | None,
    joint: str,
) -> tuple[float, float] | None:
    if minimum is None and maximum is None:
        return None
    if minimum is None or maximum is None:
        parser.error(
            f"--approved-{joint}-min-deg and --approved-{joint}-max-deg "
            "must be supplied together"
        )
    return float(minimum), float(maximum)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retiming of a Stage-5A processed reference path. Source FPS remains "
            "unknown; generated seconds belong only to the prescribed new clock."
        )
    )
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=reference_trajectory_data_dir,
        help="Stage-5A processed output directory (the raw skeleton CSV is not read).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=reference_retiming_data_dir,
    )
    parser.add_argument("--cycle-index", type=int)
    parser.add_argument(
        "--profile",
        choices=("slow", "nominal", "fast"),
        help="Run one profile; omit to generate all three default profiles.",
    )
    parser.add_argument("--flexion-duration", type=float)
    parser.add_argument("--extension-duration", type=float)
    parser.add_argument("--approved-hip-min-deg", type=float)
    parser.add_argument("--approved-hip-max-deg", type=float)
    parser.add_argument("--approved-knee-min-deg", type=float)
    parser.add_argument("--approved-knee-max-deg", type=float)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    approved_hip = _approved_pair(
        parser,
        args.approved_hip_min_deg,
        args.approved_hip_max_deg,
        "hip",
    )
    approved_knee = _approved_pair(
        parser,
        args.approved_knee_min_deg,
        args.approved_knee_max_deg,
        "knee",
    )
    profiles = None if args.profile is None else (args.profile,)
    result = run_reference_retiming(
        processed_directory=args.processed_directory,
        output_directory=args.output_directory,
        cycle_index=args.cycle_index,
        profiles=profiles,
        flexion_duration_s=args.flexion_duration,
        extension_duration_s=args.extension_duration,
        approved_hip_range_deg=approved_hip,
        approved_knee_range_deg=approved_knee,
        generate_plots=not args.no_plots,
    )

    print("Stage 5B reference retiming completed (software-only).")
    print("Source timing status: unknown")
    print("Retimed timing is original: false")
    print(f"Output directory: {Path(args.output_directory).resolve()}")
    print(
        "Original ROM (deg): "
        f"hip={result.rom_audit.original_angle_range_deg['hip']}, "
        f"knee={result.rom_audit.original_angle_range_deg['knee']}"
    )
    print(f"ROM mapping applied: {result.rom_audit.rom_mapping_applied}")
    print(
        "Trajectory requires ROM confirmation: "
        f"{result.rom_audit.trajectory_requires_rom_confirmation}"
    )
    if result.rom_audit.trajectory_requires_rom_confirmation:
        print("Dynamics safely blocked: " + ";".join(result.rom_audit.confirmation_reasons))
    else:
        print("Dynamics evaluated for: baseline, hip_stiff, knee_stiff, heavy_leg")
    print(result.retiming_summary.to_string(index=False))
    if result.skipped_visualizations:
        print("Skipped visualizations:")
        for filename, reason in result.skipped_visualizations.items():
            print(f"  {filename}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

