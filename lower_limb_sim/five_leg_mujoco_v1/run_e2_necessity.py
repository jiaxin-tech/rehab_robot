"""CLI for E2 personalization-necessity reevaluation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .e2_necessity import STUDY_ID, run_e2_necessity_study


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results_e2_necessity_v1"


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
    (output_dir / "study_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "e2_full_landscapes.csv", payload["E2_landscape_rows"])
    _write_csv(
        output_dir / "table_1_e2_oracle_characterization.csv",
        payload["E2_oracle_characterization"],
    )
    _write_csv(
        output_dir / "table_2_e2_oracle_transfer_matrix.csv",
        payload["E2_oracle_transfer_matrix"],
    )
    _write_csv(
        output_dir / "table_3_e2_common_candidate_per_leg.csv",
        payload["E2_common_candidate"]["per_leg"],
    )
    _write_csv(
        output_dir / "table_4_e2_branch_driver_summary.csv",
        payload["E2_branch_driver_summary"],
    )
    _write_csv(
        output_dir / "table_5_e2_branch_driver_rows.csv",
        payload["E2_branch_driver_rows"],
    )
    _write_csv(
        output_dir / "table_6_e0_vs_e2_comparison.csv",
        payload["E0_vs_E2_comparison"],
    )
    _write_csv(
        output_dir / "table_7_common_regret_vs_perturbation.csv",
        payload["robustness_aware_necessity"][
            "per_leg_common_regret_vs_variability"
        ],
    )
    _write_csv(
        output_dir / "table_8_e0_e2_pairwise_landscape_similarity.csv",
        payload["E0_E2_pairwise_landscape_similarity"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    started_ns = time.perf_counter_ns()
    payload = run_e2_necessity_study()
    write_outputs(payload, args.output_dir)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        json.dumps(
            {
                STUDY_ID: payload[STUDY_ID],
                "E2_PERSONALIZATION_NECESSITY": payload[
                    "E2_PERSONALIZATION_NECESSITY"
                ],
                "READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON": payload[
                    "READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON"
                ],
                "algorithm_runs": payload["algorithm_runs"],
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
