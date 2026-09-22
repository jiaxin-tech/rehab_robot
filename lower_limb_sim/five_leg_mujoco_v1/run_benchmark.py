"""CLI for the frozen five-leg MuJoCo mechanical benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .benchmark import run_full_benchmark
from .plotting import write_required_figures


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "table_1_leg_parameters_and_rom.csv",
        payload["leg_parameters_and_rom"],
    )
    _write_csv(
        output_dir / "table_2_oracle_characterization.csv",
        payload["oracle_characterization"],
    )
    _write_csv(
        output_dir / "table_3_pairwise_rank_correlations.csv",
        payload["pairwise_landscape_analysis"],
    )
    _write_csv(
        output_dir / "table_4_gray_box_prediction_quality.csv",
        payload["gray_box_prediction_quality"],
    )
    _write_csv(output_dir / "full_landscapes.csv", payload["landscape_rows"])
    write_required_figures(payload, output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    started_ns = time.perf_counter_ns()
    payload = run_full_benchmark()
    write_outputs(payload, args.output_dir)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        json.dumps(
            {
                "FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1": payload[
                    "FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1"
                ],
                "READY_FOR_FROZEN_ALGORITHM_COMPARISON_ON_FIVE_LEG_MUJOCO": payload[
                    "READY_FOR_FROZEN_ALGORITHM_COMPARISON_ON_FIVE_LEG_MUJOCO"
                ],
                "personalization_necessity_conclusion": payload[
                    "personalization_necessity_conclusion"
                ],
                "total_candidate_replays": payload["total_candidate_replays"],
                "robot_actions": payload["robot_actions"],
                "elapsed_ms": elapsed_ms,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
