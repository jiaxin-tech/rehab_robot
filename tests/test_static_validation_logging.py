"""Offline tests for the default-off static validation label sidecar."""

from __future__ import annotations

import ast
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from scripts.static_validation_logging import (
    CSV_FIELDS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENABLED,
    DEFAULT_SCHEMA_PATH,
    PHASES,
    SCHEMA_VERSION,
    StaticValidationCell,
    StaticValidationLabelLogger,
    StaticValidationLoggingDisabled,
    StaticValidationLoggingError,
    build_static_validation_record,
    load_static_validation_logging_config,
)
from scripts.dry_run_static_validation_logging import run_dry_run


PROTOCOL_SHA = "c88799b838f6304765acb643a706b1a6f1bbe02b1ee4f6c07ed9c486eab2f5c1"
ROOT = Path(__file__).resolve().parents[1]


def cell(*, repeat_id: int = 1, direction_id: str = "+WORLD_X") -> StaticValidationCell:
    return StaticValidationCell(
        session_id="STATIC_TEST_SESSION",
        protocol_sha256=PROTOCOL_SHA,
        pose_id="P0_CURRENT_SAFE_STATIONARY",
        direction_id=direction_id,
        load_level_id="L1_REVIEWED_LOW",
        repeat_id=repeat_id,
    )


def wrench(*, start: float = 10.0, end: float = 10.004) -> dict:
    return {
        "force_query_started_s": start,
        "force_query_finished_s": end,
        "host_monotonic_time_s": (start + end) / 2.0,
        "cartesian_force_raw_n": [1.2, -0.3, 0.4],
        "raw_force_frame": "world",
        "valid": True,
        "invalid_reason": "",
    }


def state(*, sequence_id: int = 7) -> dict:
    return {
        "host_monotonic_time_s": 10.005,
        "sequence_id": sequence_id,
        "tcp_position_m": [0.31, 0.02, 0.44],
        "tcp_orientation_rad": [0.01, 0.02, 0.03],
        "joint_position_rad": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "operation_state": "stationary",
        "valid": True,
        "invalid_reason": "",
    }


def metadata(*, verified: bool = True) -> dict:
    return {
        "robot_model": "mock-model",
        "robot_serial_number": "mock-serial",
        "controller_version": "mock-controller",
        "active_tool_name": "mock-tool",
        "active_workobject_name": "mock-workobject",
        "sdk_tool_payload": {
            "active_hmi_tool_workobject_verified": verified,
            "toolset_load_mass_kg": 1.5,
            "toolset_load_cog_m": [0.0, 0.0, 0.05],
            "toolset_load_inertia_kg_m2": [0.01, 0.02, 0.03],
            "toolset_end_translation_m": [0.0, 0.0, 0.1],
            "toolset_end_rpy_rad": [0.0, 0.0, 0.0],
            "sdk_available_tool_names": ["mock-tool"],
            "sdk_available_workobject_names": ["mock-workobject"],
        },
    }


def build(**overrides) -> dict:
    arguments = {
        "cell": cell(),
        "phase": "PRE",
        "phase_sample_index": 0,
        "raw_measurement_source": "robot_wrench.csv",
        "raw_measurement_id": "raw-wrench-0001",
        "wrench_sample": wrench(),
        "robot_state": state(),
        "robot_metadata": metadata(),
    }
    arguments.update(overrides)
    return build_static_validation_record(**arguments)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def make_logger(path: Path, *, enabled: bool) -> StaticValidationLabelLogger:
    return StaticValidationLabelLogger(
        path,
        session_id="STATIC_TEST_SESSION",
        protocol_sha256=PROTOCOL_SHA,
        raw_measurement_source="robot_wrench.csv",
        enabled=enabled,
        run_metadata={"evidence_role": "OFFLINE_TEST_ONLY"},
    )


def test_default_off_identity_is_redundantly_explicit() -> None:
    config = load_static_validation_logging_config()
    assert DEFAULT_ENABLED is False
    assert config["enabled"] is False
    assert config["schema_version"] == 1
    assert config["record_schema"] == "scripts/static_validation_record_schema_v1.json"
    assert "does not authorize robot connection" in config["notes"]


def test_disabled_logger_creates_no_directory_or_file(tmp_path: Path) -> None:
    output = tmp_path / "disabled"
    logger = make_logger(output, enabled=False)
    assert logger.active is False
    with pytest.raises(StaticValidationLoggingDisabled, match="default-off"):
        logger.start()
    with pytest.raises(StaticValidationLoggingDisabled, match="default-off"):
        logger.append(
            cell=cell(), phase="PRE", phase_sample_index=0,
            raw_measurement_id="raw-1", wrench_sample=wrench(),
            robot_state=state(), robot_metadata=metadata(),
        )
    assert not output.exists()


def test_schema_required_fields_exactly_match_csv_fields() -> None:
    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert set(schema["required"]) == set(CSV_FIELDS)
    assert set(schema["properties"]) == set(CSV_FIELDS)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["phase"]["enum"] == list(PHASES)


def test_schema_contains_every_requested_identity_measurement_and_status_field() -> None:
    required = set(json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))["required"])
    assert {
        "session_id", "protocol_sha256", "pose_id", "direction_id",
        "load_level_id", "repeat_id", "phase", "query_start_s",
        "query_end_s", "query_midpoint_s", "query_latency_ms",
        "fx_raw_n", "fy_raw_n", "fz_raw_n", "valid", "status",
    } <= required
    assert {
        "tcp_x_m", "tcp_y_m", "tcp_z_m", "q1_rad", "q6_rad",
        "robot_operation_state", "active_tool_name", "tcp_translation_m_json",
        "payload_mass_kg", "payload_cog_m_json",
    } <= required


def test_valid_record_copies_labels_timing_wrench_state_and_metadata() -> None:
    row = build()
    assert row["session_id"] == "STATIC_TEST_SESSION"
    assert row["protocol_sha256"] == PROTOCOL_SHA
    assert row["pose_id"] == "P0_CURRENT_SAFE_STATIONARY"
    assert row["direction_id"] == "+WORLD_X"
    assert row["load_level_id"] == "L1_REVIEWED_LOW"
    assert row["repeat_id"] == 1 and row["phase"] == "PRE"
    assert row["query_start_s"] == 10.0
    assert row["query_midpoint_s"] == pytest.approx(10.002)
    assert row["query_end_s"] == 10.004
    assert row["query_latency_ms"] == pytest.approx(4.0)
    assert (row["fx_raw_n"], row["fy_raw_n"], row["fz_raw_n"]) == (1.2, -0.3, 0.4)
    assert row["tcp_x_m"] == 0.31 and row["q6_rad"] == 0.5
    assert row["active_tool_name"] == "mock-tool"
    assert row["payload_mass_kg"] == 1.5
    assert row["valid"] is True and row["status"] == "VALID"


def test_midpoint_and_latency_use_existing_query_bounds_without_new_clock() -> None:
    sample = wrench(start=21.0, end=21.008)
    sample.pop("host_monotonic_time_s")
    row = build(wrench_sample=sample)
    assert row["query_midpoint_s"] == pytest.approx(21.004)
    assert row["query_latency_ms"] == pytest.approx(8.0)
    source = Path(ROOT / "scripts/static_validation_logging.py").read_text(encoding="utf-8")
    assert "perf_counter" not in source
    assert "time.time" not in source


def test_timestamp_order_and_midpoint_consistency_fail_closed() -> None:
    reversed_row = build(wrench_sample=wrench(start=10.1, end=10.0))
    assert reversed_row["valid"] is False
    assert reversed_row["query_midpoint_s"] is None
    assert reversed_row["query_latency_ms"] is None
    assert "query_timestamp_order_invalid" in reversed_row["invalid_reason"]

    inconsistent = wrench()
    inconsistent["host_monotonic_time_s"] = 99.0
    inconsistent_row = build(wrench_sample=inconsistent)
    assert inconsistent_row["valid"] is False
    assert "query_midpoint_inconsistent" in inconsistent_row["invalid_reason"]


def test_phase_labels_accept_only_pre_load_post() -> None:
    assert [build(phase=phase)["phase"] for phase in PHASES] == ["PRE", "LOAD", "POST"]
    with pytest.raises(ValueError, match="phase must be one of"):
        build(phase="BASELINE")
    with pytest.raises(ValueError, match="phase must be one of"):
        build(phase="PRE_LOAD_ZERO")


def test_cell_identity_is_stable_and_unique_across_repeat_or_direction() -> None:
    first = cell()
    same = cell()
    next_repeat = cell(repeat_id=2)
    next_direction = cell(direction_id="-WORLD_X")
    assert first.cell_id == same.cell_id
    assert len({first.cell_id, next_repeat.cell_id, next_direction.cell_id}) == 3
    assert all(len(value) == 64 for value in {first.cell_id, next_repeat.cell_id, next_direction.cell_id})


def test_record_identity_links_one_raw_measurement_to_one_label() -> None:
    row = build(raw_measurement_id="raw-wrench-0042")
    assert row["raw_measurement_source"] == "robot_wrench.csv"
    assert row["raw_measurement_id"] == "raw-wrench-0042"
    assert len(row["record_id"]) == 64
    assert len(row["cell_id"]) == 64
    changed = build(raw_measurement_id="raw-wrench-0043")
    assert row["record_id"] != changed["record_id"]


def test_raw_wrench_and_state_inputs_are_not_mutated() -> None:
    raw_wrench = wrench()
    raw_state = state()
    raw_metadata = metadata()
    originals = deepcopy((raw_wrench, raw_state, raw_metadata))
    build(wrench_sample=raw_wrench, robot_state=raw_state, robot_metadata=raw_metadata)
    assert (raw_wrench, raw_state, raw_metadata) == originals


def test_missing_numeric_data_remains_none_and_record_is_invalid() -> None:
    row = build(
        wrench_sample={
            "force_query_started_s": None,
            "force_query_finished_s": None,
            "cartesian_force_raw_n": [None, float("nan"), None],
            "raw_force_frame": "world",
        },
        robot_state=None,
        robot_metadata=None,
    )
    assert row["query_start_s"] is None
    assert row["query_end_s"] is None
    assert row["query_midpoint_s"] is None
    assert row["query_latency_ms"] is None
    assert row["fx_raw_n"] is None and row["fy_raw_n"] is None and row["fz_raw_n"] is None
    assert row["tcp_x_m"] is None and row["q1_rad"] is None
    assert row["valid"] is False and row["status"] == "INVALID"
    for reason in ("missing_query_timestamp", "raw_force_missing_or_nonfinite", "robot_state_missing", "robot_metadata_missing"):
        assert reason in row["invalid_reason"]


def test_unverified_active_tool_metadata_fails_closed_without_fake_name() -> None:
    robot_metadata = metadata(verified=False)
    robot_metadata["active_tool_name"] = None
    row = build(robot_metadata=robot_metadata)
    assert row["active_tool_name"] is None
    assert row["active_hmi_tool_workobject_verified"] is False
    assert row["valid"] is False
    assert "active_tool_workobject_not_verified" in row["invalid_reason"]


def test_identity_fields_are_required_instead_of_written_as_ambiguous_blanks() -> None:
    with pytest.raises(ValueError, match="session_id"):
        StaticValidationCell("", PROTOCOL_SHA, "P0", "+X", "L1", 1)
    with pytest.raises(ValueError, match="protocol_sha256"):
        StaticValidationCell("S", "not-a-sha", "P0", "+X", "L1", 1)
    with pytest.raises(ValueError, match="repeat_id"):
        StaticValidationCell("S", PROTOCOL_SHA, "P0", "+X", "L1", 0)
    with pytest.raises(ValueError, match="raw_measurement_id"):
        build(raw_measurement_id="")


def test_logger_enforces_pre_load_post_order_per_cell(tmp_path: Path) -> None:
    logger = make_logger(tmp_path / "ordered", enabled=True).start()
    with pytest.raises(StaticValidationLoggingError, match="first phase.*PRE"):
        logger.append(
            cell=cell(), phase="LOAD", phase_sample_index=0,
            raw_measurement_id="bad-load-first", wrench_sample=wrench(),
            robot_state=state(), robot_metadata=metadata(),
        )
    logger.append(
        cell=cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="pre-0", wrench_sample=wrench(),
        robot_state=state(), robot_metadata=metadata(),
    )
    with pytest.raises(StaticValidationLoggingError, match="without reversal or skip"):
        logger.append(
            cell=cell(), phase="POST", phase_sample_index=0,
            raw_measurement_id="post-skips-load", wrench_sample=wrench(),
            robot_state=state(), robot_metadata=metadata(),
        )
    logger.append(
        cell=cell(), phase="LOAD", phase_sample_index=0,
        raw_measurement_id="load-0", wrench_sample=wrench(),
        robot_state=state(), robot_metadata=metadata(),
    )
    with pytest.raises(StaticValidationLoggingError, match="without reversal or skip"):
        logger.append(
            cell=cell(), phase="PRE", phase_sample_index=1,
            raw_measurement_id="pre-after-load", wrench_sample=wrench(),
            robot_state=state(), robot_metadata=metadata(),
        )
    logger.append(
        cell=cell(), phase="POST", phase_sample_index=0,
        raw_measurement_id="post-0", wrench_sample=wrench(),
        robot_state=state(), robot_metadata=metadata(),
    )
    logger.close()
    assert [row["phase"] for row in read_rows(logger.label_path)] == ["PRE", "LOAD", "POST"]


def test_logger_rejects_duplicate_phase_index_record_and_raw_link(tmp_path: Path) -> None:
    logger = make_logger(tmp_path / "duplicates", enabled=True).start()
    logger.append(
        cell=cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="raw-0", wrench_sample=wrench(),
        robot_state=state(), robot_metadata=metadata(),
    )
    with pytest.raises(StaticValidationLoggingError, match="duplicate phase_sample_index"):
        logger.append(
            cell=cell(), phase="PRE", phase_sample_index=0,
            raw_measurement_id="raw-1", wrench_sample=wrench(),
            robot_state=state(), robot_metadata=metadata(),
        )
    with pytest.raises(StaticValidationLoggingError, match="raw measurement is already linked"):
        logger.append(
            cell=cell(), phase="PRE", phase_sample_index=1,
            raw_measurement_id="raw-0", wrench_sample=wrench(),
            robot_state=state(), robot_metadata=metadata(),
        )
    logger.close()


def test_logger_writes_missing_values_as_blank_not_zero(tmp_path: Path) -> None:
    logger = make_logger(tmp_path / "missing", enabled=True).start()
    logger.append(
        cell=cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="missing-0",
        wrench_sample={"force_query_started_s": None, "cartesian_force_raw_n": None},
        robot_state=None,
        robot_metadata=None,
    )
    logger.close()
    row = read_rows(logger.label_path)[0]
    for field in ("query_start_s", "query_end_s", "query_midpoint_s", "query_latency_ms", "fx_raw_n", "fy_raw_n", "fz_raw_n", "tcp_x_m", "q1_rad"):
        assert row[field] == ""
    assert row["valid"] == "False"
    assert row["status"] == "INVALID"
    assert row["invalid_reason"]


def test_logger_metadata_links_sidecar_to_source_and_never_claims_robot_action(tmp_path: Path) -> None:
    logger = make_logger(tmp_path / "metadata", enabled=True).start()
    logger.append(
        cell=cell(), phase="PRE", phase_sample_index=0,
        raw_measurement_id="raw-0", wrench_sample=wrench(),
        robot_state=state(), robot_metadata=metadata(),
    )
    logger.close()
    payload = json.loads(logger.metadata_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "STATIC_TEST_SESSION"
    assert payload["protocol_sha256"] == PROTOCOL_SHA
    assert payload["raw_measurement_source"] == "robot_wrench.csv"
    assert payload["label_file"] == "static_validation_labels.csv"
    assert len(payload["label_file_sha256"]) == 64
    assert payload["raw_wrench_overwritten"] is False
    assert payload["robot_connected_by_logger"] is False
    assert payload["robot_action_count"] == 0
    assert payload["control_or_safety_modified"] is False


def test_output_collision_never_overwrites_existing_raw_or_label_files(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    protected = output / "robot_wrench.csv"
    protected.write_text("do-not-overwrite\n", encoding="utf-8")
    logger = make_logger(output, enabled=True)
    with pytest.raises(StaticValidationLoggingError, match="never overwritten"):
        logger.start()
    assert protected.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_cell_must_match_logger_session_and_protocol(tmp_path: Path) -> None:
    logger = make_logger(tmp_path / "identity", enabled=True).start()
    wrong = StaticValidationCell("OTHER_SESSION", PROTOCOL_SHA, "P0", "+X", "L1", 1)
    with pytest.raises(StaticValidationLoggingError, match="does not match logger"):
        logger.append(
            cell=wrong, phase="PRE", phase_sample_index=0,
            raw_measurement_id="raw-0", wrench_sample=wrench(),
            robot_state=state(), robot_metadata=metadata(),
        )
    logger.close()


def test_dry_run_uses_synthetic_data_and_passes_all_layer_checks(tmp_path: Path) -> None:
    result = run_dry_run(tmp_path / "dry_run")
    assert result["dry_run_status"] == "PASS"
    assert result["evidence_role"] == "OFFLINE_SYNTHETIC_DRY_RUN_ONLY"
    assert result["default_off_verified"] is True
    assert result["schema_complete"] is True
    assert result["row_count"] == 12
    assert result["cell_count"] == 2
    assert result["record_ids_unique"] is True
    assert result["cell_ids_unique_by_repeat"] is True
    assert result["phase_sequences_valid"] is True
    assert result["timestamp_order_valid"] is True
    assert result["raw_input_unchanged"] is True
    assert result["missing_value_fail_closed"] is True
    assert result["robot_constructed_or_connected"] is False
    assert result["robot_action_count"] == 0
    assert result["physical_load_count"] == 0
    assert result["human_data_count"] == 0


def test_dry_run_script_is_default_off_without_explicit_run_flag() -> None:
    source = (ROOT / "scripts/dry_run_static_validation_logging.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--run", action="store_true"' in source
    assert "if not args.run:" in source
    assert "no files written" in source


def test_new_layer_has_no_hardware_control_safety_or_robot_action_imports() -> None:
    path = ROOT / "scripts/static_validation_logging.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not any(name.startswith(("hardware", "control", "safety")) for name in imported)
    assert not {"connect", "enable", "move", "drag", "stop", "calibrate_force_sensors"}.intersection(called)


def test_existing_collection_and_control_paths_do_not_import_new_layer() -> None:
    for relative in (
        "collection/collector.py",
        "collection/episode_logger.py",
        "collection/real_robot_acquisition.py",
        "control/robot_trajectory_executor.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "static_validation_logging" not in text


def test_config_adds_no_force_or_safety_threshold() -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(config) == {"enabled", "notes", "record_schema", "schema_version"}
    assert not any("force" in key.lower() or "threshold" in key.lower() or "limit" in key.lower() for key in config)
