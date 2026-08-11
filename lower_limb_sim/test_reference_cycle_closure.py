"""Regression tests for full-joint natural-cycle closure auditing.

The fixture is the already imported Stage-5A software dataset.  These tests do
not import or execute robot hardware, control, motion, or safety code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from lower_limb_sim.reference_cycle_closure import (
    APPROVED_HIP_ROM_DEG,
    APPROVED_KNEE_ROM_DEG,
    CLOSURE_AUDIT_COLUMNS,
    CycleClosureConfig,
    audit_reference_cycle_closure,
    select_best_cycle_candidate,
)


PROCESSED_DIRECTORY = (
    Path(__file__).resolve().parent
    / "data"
    / "reference_trajectories"
    / "processed"
)
PHYSICAL_DERIVATIVE_COLUMNS = (
    "dq_hip_start_rad_s",
    "dq_hip_end_rad_s",
    "dq_knee_start_rad_s",
    "dq_knee_end_rad_s",
    "delta_dq_hip",
    "delta_dq_knee",
    "delta_dq_hip_rad_s",
    "delta_dq_knee_rad_s",
)


@pytest.fixture(scope="module")
def full_angles() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIRECTORY / "reference_full_angles.csv")


@pytest.fixture(scope="module")
def legacy_cycles() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIRECTORY / "detected_cycles.csv")


@pytest.fixture(scope="module")
def audit_result(full_angles: pd.DataFrame, legacy_cycles: pd.DataFrame):
    return audit_reference_cycle_closure(full_angles, legacy_cycles)


def test_full_joint_pca_detects_and_compares_all_natural_cycles(
    audit_result,
) -> None:
    audit = audit_result.closure_audit
    assert tuple(audit.columns) == CLOSURE_AUDIT_COLUMNS
    assert audit["cycle_candidate_id"].tolist() == [0, 1, 2, 3]
    assert audit["peak_frame"].astype(int).tolist() == [5686, 5798, 5895, 5995]
    assert audit_result.metadata["major_peak_frames"] == [5686, 5798, 5895, 5995]
    assert audit_result.metadata["joint_state_valley_frames"] == [
        5731,
        5836,
        5937,
        6040,
    ]
    assert audit_result.metadata["legacy_detected_cycles_used_for_selection"] is False
    np.testing.assert_allclose(
        audit_result.metadata["pca_pc1"],
        [0.6647000783679987, 0.7471103036483813],
        atol=1e-12,
        rtol=0.0,
    )
    assert audit_result.metadata[
        "pca_pc1_explained_variance_ratio"
    ] == pytest.approx(0.9889699555894546, abs=1e-12)
    # Both non-zero PCA loadings and the optimized same-phase boundaries show
    # that neither detection nor closure is a knee-minimum-only rule.
    assert np.count_nonzero(np.abs(audit_result.metadata["pca_pc1"]) > 0.1) == 2
    candidate_two = audit.loc[audit["cycle_candidate_id"].eq(2)].iloc[0]
    assert int(candidate_two["start_frame"]) != int(
        candidate_two["bracket_start_frame"]
    )
    assert int(candidate_two["end_frame"]) != int(
        candidate_two["bracket_end_frame"]
    )


def test_automatic_selection_is_candidate_two_at_the_jointly_optimized_bounds(
    audit_result,
    full_angles: pd.DataFrame,
) -> None:
    audit = audit_result.closure_audit
    assert audit_result.selected_candidate_id == 2
    assert audit["eligible"].tolist() == [False, False, True, True]
    assert audit["selected"].tolist() == [False, False, True, False]
    selected = audit_result.selected_candidate
    assert selected is not None
    assert (
        int(selected["start_frame"]),
        int(selected["peak_frame"]),
        int(selected["end_frame"]),
    ) == (5844, 5895, 5934)
    assert float(selected["closure_score"]) == pytest.approx(
        1.0315252030197288, abs=1e-12
    )

    candidate_three = audit.loc[audit["cycle_candidate_id"].eq(3)].iloc[0]
    assert (
        int(candidate_three["start_frame"]),
        int(candidate_three["peak_frame"]),
        int(candidate_three["end_frame"]),
    ) == (5945, 5995, 6029)
    assert float(selected["closure_score"]) < float(
        candidate_three["closure_score"]
    )

    expected_slice = full_angles.loc[
        full_angles["Frame"].between(5844, 5934, inclusive="both")
    ].copy()
    assert_frame_equal(
        audit_result.selected_measured_cycle,
        expected_slice,
        check_exact=True,
    )


def test_raw_closure_components_are_retained_and_reproduce_the_score(
    audit_result,
) -> None:
    audit = audit_result.closure_audit
    expected_score = np.sqrt(
        audit["delta_q_hip_deg"].to_numpy(float) ** 2
        + audit["delta_q_knee_deg"].to_numpy(float) ** 2
        + (1000.0 * audit["delta_x_pull_m"].to_numpy(float) / 5.0) ** 2
        + (1000.0 * audit["delta_z_pull_m"].to_numpy(float) / 5.0) ** 2
    )
    np.testing.assert_allclose(
        audit["closure_score"], expected_score, atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        audit["pull_closure_error_mm"],
        1000.0
        * np.hypot(
            audit["delta_x_pull_m"].to_numpy(float),
            audit["delta_z_pull_m"].to_numpy(float),
        ),
        atol=1e-12,
        rtol=0.0,
    )
    selected = audit_result.selected_candidate
    assert selected is not None
    assert float(selected["delta_q_hip_deg"]) == pytest.approx(
        -0.45244558113928335, abs=1e-12
    )
    assert float(selected["delta_q_knee_deg"]) == pytest.approx(
        -0.21627638940201876, abs=1e-12
    )
    assert float(selected["pull_closure_error_mm"]) == pytest.approx(
        4.507110393477331, abs=1e-12
    )


def test_complete_projection_and_approved_rom_are_independent_strict_gates(
    audit_result,
    full_angles: pd.DataFrame,
) -> None:
    audit = audit_result.closure_audit
    assert APPROVED_HIP_ROM_DEG == (0.0, 120.0)
    assert APPROVED_KNEE_ROM_DEG == (5.0, 145.0)
    assert audit["cycle_complete"].tolist() == [True, True, True, True]
    assert audit["projection_valid"].tolist() == [False, False, True, True]
    assert audit["rom_valid"].tolist() == [False, True, True, True]
    np.testing.assert_array_equal(
        audit["eligible"].to_numpy(bool),
        audit["cycle_complete"].to_numpy(bool)
        & audit["projection_valid"].to_numpy(bool)
        & audit["rom_valid"].to_numpy(bool),
    )

    one_bad_projection = full_angles.copy(deep=True)
    one_bad_projection.loc[one_bad_projection["Frame"].eq(5900), "angle_valid"] = False
    changed = audit_reference_cycle_closure(one_bad_projection)
    candidate_two = changed.closure_audit.loc[
        changed.closure_audit["cycle_candidate_id"].eq(2)
    ].iloc[0]
    assert bool(candidate_two["cycle_complete"])
    assert bool(candidate_two["rom_valid"])
    assert not bool(candidate_two["projection_valid"])
    assert int(candidate_two["projection_invalid_frame_count"]) == 1
    assert not bool(candidate_two["eligible"])
    assert changed.selected_candidate_id == 3


def test_missing_source_fps_keeps_physical_derivatives_blank_with_reason(
    audit_result,
    full_angles: pd.DataFrame,
) -> None:
    audit = audit_result.closure_audit
    assert audit.loc[:, PHYSICAL_DERIVATIVE_COLUMNS].isna().all().all()
    assert not audit["derivative_valid"].any()
    assert set(audit["derivative_invalid_reason"]) == {"source_fps_not_provided"}
    assert audit_result.metadata["physical_derivatives_available"] is False
    assert (
        audit_result.metadata["physical_derivative_invalid_reason"]
        == "source_fps_not_provided"
    )

    timed = audit_reference_cycle_closure(full_angles, source_fps=100.0)
    assert timed.closure_audit.loc[:, PHYSICAL_DERIVATIVE_COLUMNS].notna().all().all()
    assert timed.closure_audit["derivative_valid"].all()
    assert set(timed.closure_audit["derivative_invalid_reason"]) == {""}


def test_legacy_cycle_table_cannot_change_detection_boundaries_or_selection(
    full_angles: pd.DataFrame,
    legacy_cycles: pd.DataFrame,
) -> None:
    without_legacy = audit_reference_cycle_closure(full_angles)
    malicious_legacy = legacy_cycles.copy(deep=True)
    malicious_legacy["start_frame"] = -10_000
    malicious_legacy["end_frame"] = 99_999
    malicious_legacy["peak_flexion_frame"] += 17
    with_legacy = audit_reference_cycle_closure(full_angles, malicious_legacy)
    columns = [
        "cycle_candidate_id",
        "start_frame",
        "peak_frame",
        "end_frame",
        "closure_score",
        "eligible",
        "selected",
    ]
    assert_frame_equal(
        without_legacy.closure_audit.loc[:, columns],
        with_legacy.closure_audit.loc[:, columns],
        check_exact=True,
    )
    assert without_legacy.selected_candidate_id == with_legacy.selected_candidate_id == 2


def test_selection_helper_is_fail_closed_and_uses_stable_candidate_id_ties() -> None:
    candidates = pd.DataFrame(
        {
            "cycle_candidate_id": [8, 3, 1],
            "closure_score": [0.5, 0.5, 0.1],
            "cycle_complete": [True, True, True],
            "projection_valid": [True, True, True],
            "rom_valid": [True, True, True],
            "eligible": [True, True, False],
        }
    )
    selected = select_best_cycle_candidate(candidates)
    assert selected is not None
    assert int(selected["cycle_candidate_id"]) == 3
    candidates["eligible"] = False
    assert select_best_cycle_candidate(candidates) is None


def test_projection_boolean_parser_and_rom_config_fail_closed(
    full_angles: pd.DataFrame,
) -> None:
    malformed = full_angles.copy(deep=True)
    malformed["angle_valid"] = "true"
    malformed.loc[malformed.index[0], "angle_valid"] = "unknown"
    with pytest.raises(ValueError, match="invalid boolean encodings"):
        audit_reference_cycle_closure(malformed)
    with pytest.raises(ValueError, match="approved_hip_rom_deg"):
        CycleClosureConfig(approved_hip_rom_deg=(120.0, 0.0))
    with pytest.raises(ValueError, match="must remain"):
        CycleClosureConfig(approved_knee_rom_deg=(5.0, 144.0))


def test_module_has_no_robot_hardware_control_or_safety_imports() -> None:
    source_path = Path(__file__).with_name("reference_cycle_closure.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_roots = {
        "control",
        "hardware",
        "rokae",
        "safety",
        "xcore",
        "xCoreSDK",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(forbidden_roots)
