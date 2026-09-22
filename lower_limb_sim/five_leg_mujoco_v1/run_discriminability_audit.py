"""CLI for the diagnostic-only V3 trajectory/objective audit."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .discriminability_audit import AUDIT_ID, run_discriminability_audit
from .discriminability_plotting import write_required_figures


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results_discriminability_audit_v1"


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
    write_required_figures(payload, output_dir)
    serializable = {key: value for key, value in payload.items() if key != "_plot_data"}
    (output_dir / "audit_summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tables = {
        "table_1_trajectory_separability.csv": "trajectory_separability",
        "table_2_selected_path_separation.csv": "selected_path_separation",
        "table_3_timing_effects.csv": "timing_effects",
        "table_4_primary_landscape_dynamic_range.csv": "primary_landscape_dynamic_range",
        "table_5_component_sensitivity.csv": "component_sensitivity",
        "table_6_time_local_sensitivity.csv": "time_local_sensitivity",
        "table_7_gradient_shape_by_leg.csv": "gradient_and_shape_by_leg",
        "table_8_cross_leg_normalized_shape.csv": "cross_leg_normalized_shape",
        "table_9_joint_branch_ordering.csv": "joint_branch_ordering",
        "table_10_secondary_metric_oracles.csv": "secondary_metric_oracles",
        "table_11_secondary_metric_interaction.csv": "secondary_metric_cross_leg_interaction",
        "full_diagnostic_response_rows.csv": "response_rows",
    }
    for filename, key in tables.items():
        _write_csv(output_dir / filename, payload[key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    started_ns = time.perf_counter_ns()
    payload = run_discriminability_audit()
    write_outputs(payload, args.output_dir)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    print(
        json.dumps(
            {
                AUDIT_ID: payload[AUDIT_ID],
                "PRIMARY_LIMITATION": payload["PRIMARY_LIMITATION"],
                "NEXT_SCIENTIFIC_DIRECTION": payload["NEXT_SCIENTIFIC_DIRECTION"],
                "diagnostic_candidate_replays": payload["diagnostic_candidate_replays"],
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
