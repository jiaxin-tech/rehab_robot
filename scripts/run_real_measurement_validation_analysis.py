"""Offline entry point for the three-level real-measurement analysis pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from measurement_validation.analysis import (
    ANALYSIS_ID,
    build_demo_input,
    load_analysis_input,
    run_validation_analysis,
    write_analysis_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze already recorded measurement logs offline. This command does "
            "not connect to or command a robot."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="normalized V1 JSON input")
    source.add_argument(
        "--demo",
        action="store_true",
        help="run deterministic synthetic pipeline demonstration only",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_ns = time.perf_counter_ns()
    payload = build_demo_input() if args.demo else load_analysis_input(args.input)
    result = run_validation_analysis(payload)
    paths = write_analysis_outputs(result, args.output_dir)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        json.dumps(
            {
                ANALYSIS_ID: result[ANALYSIS_ID],
                "evidence_classification": result["evidence_classification"],
                "MEASUREMENT_VALIDITY": result["MEASUREMENT_VALIDITY"],
                "SAME_TRAJECTORY_REPEATABILITY": result[
                    "SAME_TRAJECTORY_REPEATABILITY"
                ],
                "TRAJECTORY_SENSITIVITY": result["TRAJECTORY_SENSITIVITY"],
                "criterion_statuses_before_evidence_classification": result[
                    "criterion_statuses_before_evidence_classification"
                ],
                "FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY": result[
                    "FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY"
                ],
                "robot_action_count": result["robot_action_count"],
                "elapsed_ms": elapsed_ms,
                "outputs": paths,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
