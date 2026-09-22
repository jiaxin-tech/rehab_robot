from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from measurement_validation.analysis import (
    ANALYSIS_ID,
    analyze_same_trajectory_repeatability,
    analyze_static_load,
    analyze_trajectory_sensitivity,
    build_demo_input,
    extract_episode_features,
    run_validation_analysis,
)
from scripts.run_real_measurement_validation_analysis import main


ROOT = Path(__file__).resolve().parents[1]


def _static_cell(repeat_id: int = 1):
    phase_vectors = {
        "PRE": ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        "LOAD": ([5.0, 8.0, 11.0], [1.0, 2.0, 3.0]),
        "POST": ([3.0, 4.0, 5.0], [0.2, 0.2, 0.2]),
    }
    rows = []
    for phase, (force, torque) in phase_vectors.items():
        rows.append(
            {
                "session_id": "SESSION_A",
                "pose_id": "POSE_A",
                "direction_id": "KNOWN_DIRECTION",
                "load_level_id": "LOAD_A",
                "repeat_id": repeat_id,
                "cell_id": f"CELL_{repeat_id}",
                "phase": phase,
                "fx_raw_n": force[0],
                "fy_raw_n": force[1],
                "fz_raw_n": force[2],
                "mx": torque[0],
                "my": torque[1],
                "mz": torque[2],
                "reference_load_vector_n": [3.0, 5.0, 7.0],
                "query_start_s": 1.0,
                "query_end_s": 1.004,
                "state_host_time_s": 1.0,
                "tcp_x_m": 0.3,
                "tcp_y_m": 0.0,
                "tcp_z_m": 0.4,
                "q1_rad": 0.1,
                "valid": True,
                "robot_state_valid": True,
                "invalid_reason": "",
            }
        )
    return rows


def _episode(beta=(0.0, 0.0), repeat=0, *, tool="TOOL_A"):
    samples = []
    for index, phase in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        samples.append(
            {
                "time_s": index * 0.1,
                "phase": phase,
                "branch": "FLEXION" if phase < 0.5 else "EXTENSION",
                "fx": 1.0 + repeat * 0.01,
                "fy": 2.0,
                "fz": 3.0 + beta[0] * 10.0 + phase,
                "mx": 0.1,
                "my": 0.2,
                "mz": 0.3,
                "hip_response": 4.0 + phase,
                "knee_response": 5.0 - phase,
                "joint_measured_torque_1": 0.4 + phase,
                "joint_external_torque_1": 0.3 + phase,
                "valid": True,
                "state_valid": True,
                "invalid_reason": "",
            }
        )
    return {
        "episode_id": f"episode_{beta[0]}_{beta[1]}_{repeat}",
        "entity_id": "DUMMY_A",
        "entity_kind": "DUMMY_LEG",
        "beta_flex": beta[0],
        "beta_extend": beta[1],
        "rom_profile_id": "ROM_A",
        "rom_profile_version": "V1",
        "hardware_setup_id": "SETUP_A",
        "measurement_source": "RECORDED_LOG",
        "robot_model": "R",
        "robot_serial_number": "S",
        "controller_version": "C",
        "active_tool_name": tool,
        "active_workobject_name": "W",
        "tcp_id": "TCP_A",
        "payload_id": "PAYLOAD_A",
        "samples": samples,
    }


def test_pre_load_post_keeps_raw_values_and_calculates_force_and_torque_contrast():
    result = analyze_static_load(_static_cell())
    cell = result["cells"][0]
    assert cell["pre_mean_force_n"] == pytest.approx([1.0, 2.0, 3.0])
    assert cell["load_mean_force_n"] == pytest.approx([5.0, 8.0, 11.0])
    assert cell["post_mean_force_n"] == pytest.approx([3.0, 4.0, 5.0])
    assert cell["baseline_bias_force_n"] == pytest.approx([2.0, 3.0, 4.0])
    assert cell["drift_post_minus_pre_force_n"] == pytest.approx([2.0, 2.0, 2.0])
    assert cell["measured_load_vector_n"] == pytest.approx([3.0, 5.0, 7.0])
    assert cell["load_delta_torque_nm"] == pytest.approx([0.9, 1.9, 2.9])
    assert cell["sign_agreement_by_component"] == {"fx": True, "fy": True, "fz": True}
    assert cell["direction_cosine_agreement"] == pytest.approx(1.0)
    assert cell["raw_phase_samples"]["PRE"][0]["force_n"] == [1.0, 2.0, 3.0]
    assert result["criterion_status"] == "INSUFFICIENT"


def test_missing_and_invalid_samples_remain_missing_and_are_counted():
    rows = _static_cell()
    rows[0]["fx_raw_n"] = None
    rows[0]["invalid_reason"] = "force_missing"
    rows[2]["valid"] = False
    rows[2]["invalid_reason"] = "wrench_invalid"
    result = analyze_static_load(rows)
    cell = result["cells"][0]
    assert cell["raw_phase_samples"]["PRE"][0]["force_n"][0] is None
    assert cell["raw_phase_samples"]["PRE"][0]["valid"] is False
    assert result["usable_record_count"] == 1
    assert result["missing_or_invalid_record_count"] == 2
    assert result["invalid_reason_counts"] == {"force_missing": 1, "wrench_invalid": 1}
    assert cell["complete_pre_load_post"] is False
    assert cell["measured_load_vector_n"] is None


def test_static_repeatability_groups_repeats_without_hardcoded_pass_thresholds():
    rows = _static_cell(1) + _static_cell(2)
    rows[-2]["fz_raw_n"] += 0.2
    result = analyze_static_load(rows)
    repeatability = result["known_load_contrast_repeatability"]
    assert len(repeatability) == 1
    assert repeatability[0]["repeat_count"] == 2
    assert repeatability[0]["component_repeatability_standard_deviation_n"] is not None
    assert result["criteria_source"] == "NOT_SUPPLIED"
    assert result["criterion_checks"] == []


def test_episode_features_preserve_branches_time_local_data_and_provenance():
    raw = _episode()
    result = extract_episode_features(raw)
    assert result["features"]["fx"]["full_cycle_rms"] == pytest.approx(1.0)
    assert result["features"]["fz"]["flexion_rms"] != result["features"]["fz"]["extension_rms"]
    assert result["features"]["fz"]["peak_abs"] == pytest.approx(4.0)
    assert result["time_local_curves"]["fz"] is not None
    assert len(result["time_local_curves"]["fz"]) == 51
    assert result["channel_provenance"]["fx"] == "ROBOT_REPORTED_RAW_CHANNEL"
    assert result["channel_provenance"]["hip_response"] == "MODEL_DERIVED"
    assert result["channel_provenance"]["joint_measured_torque_1"] == (
        "ROBOT_REPORTED_RAW_CHANNEL"
    )
    assert result["channel_provenance"]["joint_external_torque_1"] == "MODEL_DERIVED"
    assert "task_direction_projection" not in result["features"]
    assert result["task_direction_projection"]["available"] is False

    projected = extract_episode_features(
        raw,
        task_direction={
            "vector": [0.0, 1.0, 0.0],
            "frame": "world",
            "source": "reviewed_geometry",
            "geometry_validated": False,
        },
    )
    assert projected["features"]["task_direction_projection"]["full_cycle_rms"] == pytest.approx(2.0)
    assert projected["task_direction_projection"]["decision_eligible"] is False


def test_same_beta_grouping_and_metadata_consistency_are_explicit():
    features = [extract_episode_features(_episode(repeat=index)) for index in range(3)]
    criteria = {
        "minimum_repeats_per_group": 3,
        "minimum_valid_episode_fraction": 1.0,
        "require_metadata_consistency": True,
        "decision_feature_ids": ["fz.full_cycle_rms"],
        "maximum_feature_cv": 0.01,
        "minimum_time_series_correlation": 0.99,
    }
    result = analyze_same_trajectory_repeatability(features, criteria)
    assert result["criterion_status"] == "SUPPORTED"
    assert result["same_condition_group_count"] == 1
    assert result["groups"][0]["repeat_count"] == 3
    assert result["groups"][0]["metadata_consistent"] is True
    assert result["groups"][0]["time_series_similarity"]["fz"]["pair_count"] == 3

    inconsistent = features[:2] + [extract_episode_features(_episode(repeat=2, tool="TOOL_B"))]
    failed = analyze_same_trajectory_repeatability(inconsistent, criteria)
    assert failed["criterion_status"] == "FAILED"
    assert failed["groups"][0]["metadata_consistent"] is False
    assert failed["groups"][0]["inconsistent_setup_metadata_fields"] == [
        "active_tool_name"
    ]


def test_different_beta_effect_uses_within_variation_and_retains_time_local_response():
    demo = build_demo_input()
    features = [extract_episode_features(episode) for episode in demo["episodes"]]
    result = analyze_trajectory_sensitivity(features, demo["criteria"]["level_3"])
    assert result["criterion_status"] == "SUPPORTED"
    assert result["analyzable_context_count"] == 1
    context = result["contexts"][0]
    assert context["beta_condition_count"] == 3
    by_feature = {row["feature_id"]: row for row in context["feature_effects"]}
    assert by_feature["fz.full_cycle_rms"]["pooled_within_trajectory_standard_deviation"] > 0.0
    assert by_feature["fz.full_cycle_rms"]["trajectory_snr"] > 2.0
    assert "fz.flexion_rms" in by_feature
    assert "fz.extension_rms" in by_feature
    local = {row["channel"]: row for row in context["time_local_effects"]}["fz"]
    assert local["maximum_time_local_effect"] > 0.6
    assert 0.2 < local["phase_at_maximum_effect"] < 0.5
    assert local["time_local_snr"] > 2.0


def test_hierarchy_and_evidence_classification_prevent_automatic_personalization():
    demo = build_demo_input()
    result = run_validation_analysis(demo)
    assert result[ANALYSIS_ID] == "IMPLEMENTED_WITH_LIMITATIONS"
    assert result["criterion_statuses_before_evidence_classification"] == {
        "MEASUREMENT_VALIDITY": "SUPPORTED",
        "SAME_TRAJECTORY_REPEATABILITY": "SUPPORTED",
        "TRAJECTORY_SENSITIVITY": "SUPPORTED",
    }
    assert result["MEASUREMENT_VALIDITY"] == "INSUFFICIENT"
    assert result["SAME_TRAJECTORY_REPEATABILITY"] == "INSUFFICIENT"
    assert result["TRAJECTORY_SENSITIVITY"] == "INSUFFICIENT"
    assert result["FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY"] is False
    assert result["personalization_conclusion_emitted"] is False
    assert "PERSONALIZATION_SUPPORTED" not in result
    assert result["robot_connection_attempted"] is False
    assert result["robot_action_count"] == 0

    no_static = {**demo, "evidence_classification": "REAL_MEASUREMENT", "static_records": []}
    blocked = run_validation_analysis(no_static)
    assert blocked["MEASUREMENT_VALIDITY"] == "INSUFFICIENT"
    assert blocked["level_2"]["blocked_by"] == "MEASUREMENT_VALIDITY"
    assert blocked["level_3"]["blocked_by"] == "MEASUREMENT_VALIDITY"


def test_real_classification_can_open_only_the_future_evaluation_gate():
    payload = build_demo_input()
    payload["evidence_classification"] = "REAL_MEASUREMENT"
    result = run_validation_analysis(payload)
    assert result["MEASUREMENT_VALIDITY"] == "SUPPORTED"
    assert result["SAME_TRAJECTORY_REPEATABILITY"] == "SUPPORTED"
    assert result["TRAJECTORY_SENSITIVITY"] == "SUPPORTED"
    assert result["FUTURE_SUBJECT_TRAJECTORY_EVALUATION_READY"] is True
    assert result["personalization_conclusion_emitted"] is False


def test_demo_runner_writes_reports_and_new_code_has_no_robot_runtime_import(tmp_path):
    output = tmp_path / "demo"
    assert main(["--demo", "--output-dir", str(output)]) == 0
    summary = json.loads((output / "validation_summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_classification"] == "OFFLINE_SYNTHETIC_DEMO_ONLY"
    assert summary["not_real_measurement_evidence"] is True
    required = {
        "level_1_static_cells.csv",
        "episode_features.csv",
        "level_2_repeatability_groups.csv",
        "level_3_trajectory_effects.csv",
        "level_3_time_local_effects.csv",
    }
    assert all((output / filename).stat().st_size > 0 for filename in required)

    forbidden_import_roots = {
        "hardware",
        "control",
        "collection",
        "personalization",
        "safety",
    }
    for path in (
        ROOT / "measurement_validation" / "analysis.py",
        ROOT / "scripts" / "run_real_measurement_validation_analysis.py",
    ):
        imported = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip(".").split(".")[0])
        assert not imported & forbidden_import_roots
