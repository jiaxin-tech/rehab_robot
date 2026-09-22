"""CLI for the mechanically interpretable endpoint design study."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .endpoint_design import STUDY_ID, run_endpoint_design_study


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results_endpoint_design_v1"


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
    tables = {
        "table_1_endpoint_discriminability.csv": "endpoint_discriminability",
        "table_2_cross_leg_decision_structure.csv": "cross_leg_decision_structure",
        "table_3_pairwise_endpoint_structure.csv": "pairwise_endpoint_structure",
        "table_4_anti_cancellation_summary.csv": "anti_cancellation_summary",
        "table_5_anti_cancellation_examples.csv": "anti_cancellation_representatives",
        "table_6_scalarization_information_loss.csv": "scalarization_information_loss",
        "table_7_robustness_scenarios.csv": "robustness_scenarios",
        "table_8_robustness_summary.csv": "robustness_summary",
        "table_9_peak_persistence.csv": "peak_persistence_summary",
        "table_10_measurement_compatibility.csv": "measurement_compatibility",
        "table_11_endpoint_selection_characterization.csv": "endpoint_selection_characterization",
        "mechanical_endpoint_feature_vector_v1.csv": "feature_rows",
    }
    for filename, key in tables.items():
        _write_csv(output_dir / filename, payload[key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    started_ns = time.perf_counter_ns()
    payload = run_endpoint_design_study()
    write_outputs(payload, args.output_dir)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        json.dumps(
            {
                STUDY_ID: payload[STUDY_ID],
                "MECHANICAL_ENDPOINT_REDESIGN_V1": payload[
                    "MECHANICAL_ENDPOINT_REDESIGN_V1"
                ],
                "RECOMMENDED_PRIMARY_ENDPOINT": payload[
                    "RECOMMENDED_PRIMARY_ENDPOINT"
                ],
                "READY_TO_REEVALUATE_PERSONALIZATION_NECESSITY": payload[
                    "READY_TO_REEVALUATE_PERSONALIZATION_NECESSITY"
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
