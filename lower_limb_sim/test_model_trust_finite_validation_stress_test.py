from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from .final_model_screened_finite_sequential_validation import (
    freeze_model_screened_shortlist,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .model_trust_finite_validation_stress_test import (
    N_RANDOM_REPEATS,
    PersistedTruthGate,
    deterministic_seed,
    final_regret,
    freeze_random3_candidates,
    geometry_candidate_universe,
    select_model_only,
    select_validated_with_reference,
)
from .run_model_trust_finite_validation_stress_test import (
    DEFAULT_OUTPUT_DIRECTORY,
    FIGURE_FILENAMES,
    FROZEN_V1_MANIFEST_PATH,
    FROZEN_V1_MANIFEST_SHA256,
    REQUIRED_OUTPUTS,
    _case_plan,
)
from . import run_model_trust_finite_validation_stress_test as runner


def _identity_universe() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "trajectory_id": f"T{index:02d}",
                "trajectory_sha256": hashlib.sha256(f"T{index}".encode()).hexdigest(),
                "hip_delta": float(index + 1),
                "knee_delta": -float(index + 1),
                "phase_delta": 0.0,
                "geometrically_admissible": True,
            }
        )
    rows.append(
        {
            "trajectory_id": "REFERENCE",
            "trajectory_sha256": "a" * 64,
            "hip_delta": 0.0,
            "knee_delta": 0.0,
            "phase_delta": 0.0,
            "geometrically_admissible": True,
        }
    )
    return pd.DataFrame(rows)


def _prediction_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trajectory_id": "reference",
                "hip_delta": 0.0,
                "knee_delta": 0.0,
                "phase_delta": 0.0,
                "J_pred": 1.0,
                "domain_coverage": 100.0,
                "model_supported": True,
                "geometrically_admissible": True,
            },
            {
                "trajectory_id": "c1",
                "hip_delta": -1.0,
                "knee_delta": -1.0,
                "phase_delta": 0.0,
                "J_pred": 0.90,
                "domain_coverage": 95.0,
                "model_supported": True,
                "geometrically_admissible": True,
            },
            {
                "trajectory_id": "c2",
                "hip_delta": -2.0,
                "knee_delta": -2.0,
                "phase_delta": 0.0,
                "J_pred": 0.906,
                "domain_coverage": 95.0,
                "model_supported": True,
                "geometrically_admissible": True,
            },
            {
                "trajectory_id": "c3",
                "hip_delta": -3.0,
                "knee_delta": -3.0,
                "phase_delta": 0.0,
                "J_pred": 0.912,
                "domain_coverage": 95.0,
                "model_supported": True,
                "geometrically_admissible": True,
            },
        ]
    )


def test_truth_gate_fails_closed_until_both_manifests_are_persisted() -> None:
    gate = PersistedTruthGate()
    with pytest.raises(PermissionError, match="truth requested before freeze"):
        gate.authorize_truth()
    gate.mark_persisted(protocol_sha256="a" * 64, baseline_manifest_sha256="b" * 64)
    token = gate.authorize_truth()
    assert len(token) == 64
    with pytest.raises(RuntimeError, match="cannot change"):
        gate.mark_persisted(protocol_sha256="a" * 64, baseline_manifest_sha256="b" * 64)


def test_random3_is_reproducible_and_does_not_use_prediction_or_truth() -> None:
    universe = geometry_candidate_universe(_identity_universe())
    first = freeze_random3_candidates(universe, case_id="case", repeats=30)
    second = freeze_random3_candidates(universe, case_id="case", repeats=30)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 90
    assert first.groupby("random_repeat")["trajectory_id"].nunique().eq(3).all()
    assert not first["sampling_uses_J_pred"].astype(bool).any()
    assert not first["sampling_uses_J_truth"].astype(bool).any()
    assert deterministic_seed("case", 0) == deterministic_seed("case", 0)
    with pytest.raises(PermissionError, match="forbidden"):
        freeze_random3_candidates(universe.assign(J_pred=0.5), case_id="leak", repeats=30)
    with pytest.raises(PermissionError, match="forbidden"):
        geometry_candidate_universe(_identity_universe().assign(J_truth=0.5))


def test_reference_is_not_random_candidate_but_is_validation_fallback() -> None:
    universe = geometry_candidate_universe(_identity_universe())
    assert "REFERENCE" not in set(universe["trajectory_id"])
    candidates = pd.DataFrame(
        [
            {"trajectory_id": "bad", "J_truth": 1.1, "hip_delta": 1.0, "knee_delta": 0.0, "phase_delta": 0.0},
            {"trajectory_id": "good", "J_truth": 0.9, "hip_delta": -1.0, "knee_delta": 0.0, "phase_delta": 0.0},
        ]
    )
    assert select_validated_with_reference(candidates)["trajectory_id"] == "good"
    assert select_validated_with_reference(candidates.iloc[[0]])["trajectory_id"] == "REFERENCE"
    assert select_model_only(candidates.iloc[0].to_dict())["trajectory_id"] == "bad"


def test_regret_definition_is_oracle_relative_and_nonnegative() -> None:
    assert np.isclose(final_regret(0.95, 0.90), 0.05)
    assert final_regret(0.90 - 1e-13, 0.90) == 0.0
    with pytest.raises(ValueError, match="below the oracle"):
        final_regret(0.80, 0.90)


def test_top1_and_top3_share_the_same_initial_prediction_and_frozen_core() -> None:
    shortlist = freeze_model_screened_shortlist(_prediction_map(), case_id="synthetic")
    assert shortlist.candidates[0].trajectory_id == "c1"
    source = inspect.getsource(runner)
    assert "frozen_v1._evaluate_case(" in source
    assert "def _evaluate_case(" not in source
    assert "max_candidates=2" in source
    assert "max_candidates=5" not in source


def test_case_plan_contains_all_nine_actual_mismatch_definitions() -> None:
    cases = _case_plan()
    assert len(cases) == 15
    assert cases["scenario_name"].nunique() == 9
    assert not cases["global_scalar_severity_order_used"].astype(bool).any()
    residual = cases.loc[cases["scenario_name"].eq("structured_residual")]
    assert set(residual["mismatch_level"]) == {"SINGLE_DEFINED_LEVEL"}


def test_frozen_scientific_boundaries_and_v1_manifest_are_unchanged() -> None:
    assert hashlib.sha256(FROZEN_V1_MANIFEST_PATH.read_bytes()).hexdigest() == FROZEN_V1_MANIFEST_SHA256
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == 0.005
    assert N_RANDOM_REPEATS == 100


def test_stress_test_sources_have_no_robot_or_optimizer_boundary() -> None:
    directory = Path(__file__).resolve().parent
    sources = "\n".join(
        (directory / name).read_text(encoding="utf-8")
        for name in (
            "model_trust_finite_validation_stress_test.py",
            "run_model_trust_finite_validation_stress_test.py",
        )
    )
    for forbidden in (
        "import hardware",
        "from hardware",
        "import control",
        "from control",
        "import collection",
        "from collection",
        "import safety",
        "from safety",
        "bayes_opt",
        "thompson",
        "connectToRobot",
    ):
        assert forbidden not in sources


def test_formal_artifacts_are_complete_reproducible_and_truth_isolated() -> None:
    output = DEFAULT_OUTPUT_DIRECTORY
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "STRESS_TEST_PROTOCOL.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "FROZEN_BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    assert metadata["STRESS_TEST_PROTOCOL_INTEGRITY"] == "PASS"
    assert metadata["candidate_truth_opened_after_both_manifests_persisted"] is True
    assert metadata["scenario_count"] == 9
    assert metadata["candidate_lattice_size"] == 21025
    assert metadata["random_repeat_count_per_case"] == 100
    assert metadata["random_result_count"] == 1500
    assert metadata["held_out_final_test_read"] is False
    assert metadata["new_prospective_cohort_generated"] is False
    assert metadata["bayesian_optimization_implemented"] is False
    assert metadata["robot_connected"] is False
    assert protocol["candidate_truth_may_open_only_after_both_manifests_persist"] is True
    assert manifest["truth_read_before_manifest_persist"] is False
    assert manifest["K5_skipped_before_truth"] is True
    assert hashlib.sha256((output / "STRESS_TEST_PROTOCOL.json").read_bytes()).hexdigest() == metadata["protocol_sha256"]
    assert hashlib.sha256((output / "FROZEN_BASELINE_MANIFEST.json").read_bytes()).hexdigest() == metadata["frozen_baseline_manifest_sha256"]
    for name in REQUIRED_OUTPUTS:
        assert (output / name).is_file() and (output / name).stat().st_size > 0
    checksum_lines = (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == len(REQUIRED_OUTPUTS)


def test_formal_result_semantics_match_the_registered_baselines() -> None:
    output = DEFAULT_OUTPUT_DIRECTORY
    per_case = pd.read_csv(output / "PER_CASE_RESULTS.csv")
    random = pd.read_csv(output / "RANDOM3_RESULTS.csv")
    budget = pd.read_csv(output / "TRIAL_BUDGET_RESULTS.csv")
    truth_access = pd.read_csv(output / "TRUTH_ACCESS_AUDIT.csv")
    b1 = per_case.loc[per_case["method"].eq("B1_MODEL_ONLY")]
    b3 = per_case.loc[per_case["method"].eq("B3_MODEL_TOP1_VALIDATION")]
    assert b1["reference_fallback_available"].eq(False).all()
    assert b3["reference_fallback_available"].eq(True).all()
    assert set(budget["validation_budget_K"]) == {0, 1, 2, 3, 5}
    assert budget.loc[budget["validation_budget_K"].eq(5), "status"].eq(
        "SKIPPED_REQUIRES_FROZEN_SELECTION_RULE_REDESIGN"
    ).all()
    assert not random["sampling_uses_J_pred"].astype(bool).any()
    assert not random["sampling_uses_J_truth"].astype(bool).any()
    assert truth_access["candidate_frozen_before_access"].astype(bool).all()
    assert not per_case["candidate_truth_used_for_candidate_generation_or_ordering"].astype(bool).any()


def test_formal_figures_are_nonempty_readable_pngs() -> None:
    for relative in FIGURE_FILENAMES:
        path = DEFAULT_OUTPUT_DIRECTORY / relative
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 900
            assert image.height >= 600
            extrema = image.convert("RGB").getextrema()
            assert any(low != high for low, high in extrema)
