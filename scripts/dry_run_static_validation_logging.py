"""Offline synthetic verification for the default-off static label sidecar.

Running without ``--run`` performs no write.  The explicit dry run uses only
Python dictionaries; it never constructs or connects a robot and never sends
motion, safety, calibration, or load commands.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.static_validation_logging import (  # noqa: E402
    CSV_FIELDS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENABLED,
    DEFAULT_SCHEMA_PATH,
    PHASES,
    StaticValidationCell,
    StaticValidationLabelLogger,
    StaticValidationLoggingDisabled,
    build_static_validation_record,
    load_static_validation_logging_config,
)


DRY_RUN_PROTOCOL_SHA256 = "c88799b838f6304765acb643a706b1a6f1bbe02b1ee4f6c07ed9c486eab2f5c1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthetic_state(sequence_id: int, time_s: float) -> dict[str, Any]:
    return {
        "host_monotonic_time_s": time_s,
        "sequence_id": sequence_id,
        "tcp_position_m": [0.31, 0.02, 0.44],
        "tcp_orientation_rad": [0.0, 0.0, 0.0],
        "joint_position_rad": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "operation_state": "SYNTHETIC_STATIC",
        "valid": True,
        "invalid_reason": "",
    }


def _synthetic_metadata() -> dict[str, Any]:
    return {
        "robot_model": "SYNTHETIC_MODEL",
        "robot_serial_number": "SYNTHETIC_SERIAL",
        "controller_version": "SYNTHETIC_CONTROLLER",
        "active_tool_name": "SYNTHETIC_TOOL",
        "active_workobject_name": "SYNTHETIC_WORKOBJECT",
        "sdk_tool_payload": {
            "active_hmi_tool_workobject_verified": True,
            "toolset_load_mass_kg": 1.0,
            "toolset_load_cog_m": [0.0, 0.0, 0.05],
            "toolset_load_inertia_kg_m2": [0.01, 0.01, 0.01],
            "toolset_end_translation_m": [0.0, 0.0, 0.1],
            "toolset_end_rpy_rad": [0.0, 0.0, 0.0],
            "sdk_available_tool_names": ["SYNTHETIC_TOOL"],
            "sdk_available_workobject_names": ["SYNTHETIC_WORKOBJECT"],
        },
    }


def run_dry_run(output_directory: Path) -> dict[str, Any]:
    config = load_static_validation_logging_config(DEFAULT_CONFIG_PATH)
    if DEFAULT_ENABLED is not False or config["enabled"] is not False:
        raise RuntimeError("default-off identity changed")

    disabled_path = output_directory.parent / f"{output_directory.name}_disabled_probe"
    disabled_logger = StaticValidationLabelLogger(
        disabled_path,
        session_id="STATIC_LABEL_DRY_RUN_DISABLED",
        protocol_sha256=DRY_RUN_PROTOCOL_SHA256,
        raw_measurement_source="synthetic_disabled",
    )
    try:
        disabled_logger.start()
    except StaticValidationLoggingDisabled:
        pass
    else:
        raise RuntimeError("default-off logger unexpectedly started")
    if disabled_path.exists():
        raise RuntimeError("default-off logger created output")

    metadata = _synthetic_metadata()
    raw_input_unchanged = True
    with StaticValidationLabelLogger(
        output_directory,
        session_id="STATIC_LABEL_DRY_RUN_V1",
        protocol_sha256=DRY_RUN_PROTOCOL_SHA256,
        raw_measurement_source="synthetic_mock_wrench_stream",
        enabled=True,
        run_metadata={
            "evidence_role": "OFFLINE_SYNTHETIC_DRY_RUN_ONLY",
            "robot_used": False,
            "motion_used": False,
            "load_applied": False,
            "human_used": False,
        },
    ) as logger:
        sequence = 0
        for repeat_id in (1, 2):
            cell = StaticValidationCell(
                session_id="STATIC_LABEL_DRY_RUN_V1",
                protocol_sha256=DRY_RUN_PROTOCOL_SHA256,
                pose_id="P0_SYNTHETIC_STATIC",
                direction_id="PX_SYNTHETIC",
                load_level_id="L1_SYNTHETIC_LABEL_ONLY",
                repeat_id=repeat_id,
            )
            for phase_index, phase in enumerate(PHASES):
                for phase_sample_index in range(2):
                    query_start = 100.0 + sequence * 0.02
                    query_end = query_start + 0.003
                    force_x = {"PRE": 0.1, "LOAD": 1.0, "POST": 0.12}[phase]
                    wrench = {
                        "force_query_started_s": query_start,
                        "force_query_finished_s": query_end,
                        "host_monotonic_time_s": (query_start + query_end) / 2.0,
                        "cartesian_force_raw_n": [force_x, -0.2, 0.3],
                        "raw_force_frame": "synthetic_world",
                        "valid": True,
                        "invalid_reason": "",
                    }
                    original = deepcopy(wrench)
                    logger.append(
                        cell=cell,
                        phase=phase,
                        phase_sample_index=phase_sample_index,
                        raw_measurement_id=f"synthetic-wrench-{sequence:04d}",
                        wrench_sample=wrench,
                        robot_state=_synthetic_state(sequence, query_end),
                        robot_metadata=metadata,
                    )
                    raw_input_unchanged = raw_input_unchanged and wrench == original
                    sequence += 1

    missing_cell = StaticValidationCell(
        session_id="STATIC_LABEL_DRY_RUN_V1",
        protocol_sha256=DRY_RUN_PROTOCOL_SHA256,
        pose_id="P0_SYNTHETIC_STATIC",
        direction_id="MISSING_DATA_PROBE",
        load_level_id="L1_SYNTHETIC_LABEL_ONLY",
        repeat_id=1,
    )
    missing = build_static_validation_record(
        cell=missing_cell,
        phase="PRE",
        phase_sample_index=0,
        raw_measurement_source="synthetic_missing_probe",
        raw_measurement_id="synthetic-missing-0000",
        wrench_sample={
            "force_query_started_s": 200.0,
            "force_query_finished_s": 200.003,
            "host_monotonic_time_s": 200.0015,
            "cartesian_force_raw_n": [None, None, None],
            "raw_force_frame": "synthetic_world",
        },
        robot_state=None,
        robot_metadata=None,
    )

    label_path = output_directory / "static_validation_labels.csv"
    with label_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    phase_sequences = {
        cell_id: [row["phase"] for row in rows if row["cell_id"] == cell_id]
        for cell_id in {row["cell_id"] for row in rows}
    }
    expected_sequence = ["PRE", "PRE", "LOAD", "LOAD", "POST", "POST"]
    summary = {
        "dry_run_status": "PASS",
        "evidence_role": "OFFLINE_SYNTHETIC_DRY_RUN_ONLY",
        "default_off_verified": not disabled_path.exists(),
        "schema_complete": set(schema["required"]) == set(CSV_FIELDS),
        "row_count": len(rows),
        "cell_count": len(phase_sequences),
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "cell_ids_unique_by_repeat": len(phase_sequences) == 2,
        "phase_sequences_valid": all(sequence == expected_sequence for sequence in phase_sequences.values()),
        "timestamp_order_valid": all(float(row["query_start_s"]) <= float(row["query_midpoint_s"]) <= float(row["query_end_s"]) for row in rows),
        "raw_input_unchanged": raw_input_unchanged,
        "missing_value_fail_closed": missing["valid"] is False and missing["status"] == "INVALID" and missing["fx_raw_n"] is None,
        "missing_invalid_reason": missing["invalid_reason"],
        "robot_constructed_or_connected": False,
        "robot_action_count": 0,
        "physical_load_count": 0,
        "human_data_count": 0,
        "label_csv": str(label_path),
        "label_csv_sha256": _sha256(label_path),
    }
    if not all(
        summary[key]
        for key in (
            "default_off_verified", "schema_complete", "record_ids_unique",
            "cell_ids_unique_by_repeat", "phase_sequences_valid",
            "timestamp_order_valid", "raw_input_unchanged", "missing_value_fail_closed",
        )
    ):
        summary["dry_run_status"] = "FAIL"
    summary_path = output_directory / "dry_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly run the offline synthetic dry run")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if not args.run:
        print("STATIC_VALIDATION_LOGGING_DEFAULT_OFF: no files written; pass --run and --output-dir for synthetic dry run")
        return
    if not args.output_dir:
        parser.error("--output-dir is required with --run")
    summary = run_dry_run(Path(args.output_dir).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["dry_run_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
