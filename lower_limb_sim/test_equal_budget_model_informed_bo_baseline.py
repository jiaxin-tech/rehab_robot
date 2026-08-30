from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from .equal_budget_model_informed_bo_baseline import (
    ALPHA_LOWER, ALPHA_UPPER, BO_VARIANTS, BUDGETS, KERNEL_SPEC,
    PRIMARY_VARIANT, SequentialQueryTruthGate, acquisition_table,
    deterministic_seed, expected_improvement, normalize_alpha,
    run_bo_sequence,
)
from .formal_protocol import ACTIVE_REFERENCE_SHA256, ROM_PROTOCOL_VERSION, THETA_SHANK_DEFINITION
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .run_equal_budget_model_informed_bo_baseline import (
    DEFAULT_OUTPUT_DIRECTORY, FIGURES, OUTPUTS, STRESS_DIRECTORY,
    STRESS_PROTOCOL_SHA256,
)


def _pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"trajectory_id": f"T{i}", "hip_delta": -4.0 + i,
         "knee_delta": -3.0 + .5*i, "phase_delta": -.02 + .005*i,
         "J_pred": 0.96 + .005*i, "domain_coverage": 95.0,
         "model_supported": True, "geometrically_admissible": True}
        for i in range(6)
    ])


def test_alpha_normalization_is_frozen_and_truth_independent() -> None:
    table = pd.DataFrame({"hip_delta": [ALPHA_LOWER[0], ALPHA_UPPER[0]],
                          "knee_delta": [ALPHA_LOWER[1], ALPHA_UPPER[1]],
                          "phase_delta": [ALPHA_LOWER[2], ALPHA_UPPER[2]]})
    assert np.allclose(normalize_alpha(table), [[-1, -1, -1], [1, 1, 1]])


def test_truth_gate_fails_closed_and_enforces_query_order() -> None:
    gate = SequentialQueryTruthGate()
    with pytest.raises(PermissionError, match="config not persisted"):
        gate.authorize("T0")
    gate.mark_persisted("a" * 64)
    token = gate.authorize("T0")
    with pytest.raises(PermissionError):
        gate.authorize("T1")
    gate.complete("T0", token)
    with pytest.raises(PermissionError):
        gate.authorize("T0")


def test_acquisition_rejects_unqueried_truth_column() -> None:
    obs = _pool().iloc[[0]].assign(J_truth=.95, residual=-.01,
                                   truth_was_queried=True)
    with pytest.raises(PermissionError, match="must not contain truth"):
        acquisition_table(_pool().assign(J_truth=.9), obs, seed=1)


def test_residual_gp_uses_only_queried_observations() -> None:
    obs = _pool().iloc[[0]].assign(J_truth=.95, residual=-.01,
                                   truth_was_queried=False)
    with pytest.raises(PermissionError, match="queried truth"):
        acquisition_table(_pool(), obs, seed=1)


def test_first_query_is_c1_and_sequence_is_deterministic() -> None:
    pool = _pool()
    calls = []
    truth = {f"T{i}": .94 + .004*i for i in range(6)}

    def callback(row, index, token):
        calls.append((index, row["trajectory_id"], token))
        return truth[row["trajectory_id"]]

    first, _ = run_bo_sequence(pool, case_id="case", variant=PRIMARY_VARIANT,
                               first_trajectory_id="T0", config_sha256="a"*64,
                               truth_query=callback, max_budget=3)
    calls.clear()
    second, _ = run_bo_sequence(pool, case_id="case", variant=PRIMARY_VARIANT,
                                first_trajectory_id="T0", config_sha256="a"*64,
                                truth_query=callback, max_budget=3)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["trajectory_id"] == "T0"
    assert first["trajectory_id"].nunique() == 3
    assert not first["unqueried_truth_used"].astype(bool).any()


def test_expected_improvement_and_seeds_are_deterministic() -> None:
    observed = expected_improvement(np.array([.9, 1.1]), np.array([.01, .01]),
                                    incumbent=1.0)
    assert observed[0] > observed[1]
    assert deterministic_seed("case", PRIMARY_VARIANT) == deterministic_seed("case", PRIMARY_VARIANT)
    assert len(BO_VARIANTS) == 2 and BUDGETS == (1, 2, 3, 5)
    assert KERNEL_SPEC["kernel"].startswith("ConstantKernel")


def test_frozen_boundaries_and_sources_are_offline_only() -> None:
    assert ACTIVE_REFERENCE_SHA256 == "f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881"
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert THETA_SHANK_DEFINITION == "q_hip - q_knee"
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == .005
    sources = "\n".join((Path(__file__).parent / name).read_text()
                        for name in ("equal_budget_model_informed_bo_baseline.py",
                                     "run_equal_budget_model_informed_bo_baseline.py"))
    for forbidden in ("import hardware", "from hardware", "import control",
                      "from control", "import safety", "from safety",
                      "connectToRobot"):
        assert forbidden not in sources


def test_formal_artifacts_and_truth_integrity() -> None:
    output = DEFAULT_OUTPUT_DIRECTORY
    metadata = json.loads((output / "metadata.json").read_text())
    protocol = json.loads((output / "BO_PROTOCOL.json").read_text())
    config = json.loads((output / "FROZEN_BO_CONFIG.json").read_text())
    assert metadata["BO_PROTOCOL_INTEGRITY"] == "PASS"
    assert metadata["BO_PROTOCOL_SANITY"] == "PASS"
    assert metadata["K1_equals_Model_Top1_all_cases"] is True
    assert metadata["oracle_revealed_after_all_bo_variants"] is True
    assert metadata["unqueried_truth_used"] is False
    assert metadata["prior_conclusions_unchanged"] is True
    assert protocol["preflight"]["stress_protocol_sha256"] == STRESS_PROTOCOL_SHA256
    assert config["all_configuration_frozen_before_truth"] is True
    for name in OUTPUTS:
        assert (output / name).is_file() and (output / name).stat().st_size > 0
    assert len((output / "checksums.sha256").read_text().splitlines()) == len(OUTPUTS)


def test_k1_matches_top1_and_only_queried_candidates_can_win() -> None:
    output = DEFAULT_OUTPUT_DIRECTORY
    bo = pd.read_csv(output / "PER_CASE_BO_RESULTS.csv")
    stress = pd.read_csv(STRESS_DIRECTORY / "PER_CASE_RESULTS.csv")
    k1 = bo.loc[(bo["bo_variant"] == PRIMARY_VARIANT) & (bo["budget"] == 1)].set_index("case_id")
    top1 = stress.loc[stress["method"].eq("B3_MODEL_TOP1_VALIDATION")].set_index("case_id")
    assert np.allclose(k1["final_regret"].sort_index(), top1["final_regret"].sort_index(), atol=1e-11)
    assert not bo["unqueried_candidate_selected"].astype(bool).any()
    log = pd.read_csv(output / "BO_QUERY_LOG.csv")
    assert not log["unqueried_truth_used"].astype(bool).any()
    assert log.groupby(["case_id", "bo_variant"])["query_index"].apply(list).apply(lambda x: x == [1,2,3,4,5]).all()


def test_formal_figures_are_readable() -> None:
    for relative in FIGURES:
        with Image.open(DEFAULT_OUTPUT_DIRECTORY / relative) as image:
            assert image.format == "PNG"
            assert image.width >= 900 and image.height >= 700
            assert any(lo != hi for lo, hi in image.convert("RGB").getextrema())

