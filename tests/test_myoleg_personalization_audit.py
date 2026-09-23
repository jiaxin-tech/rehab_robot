from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pandas as pd
import pytest

from lower_limb_sim.myoleg_benchmark.personalization_audit import (
    _interaction_audit,
    _oracle_table,
    _pool_comparison,
)


def _row(subject: str, candidate: str, index: int, e3: float, feasible: bool,
         beta_flex: float, beta_extend: float, share_shift: float) -> dict:
    return {
        "scope": "DEVELOPMENT",
        "subject_id": subject,
        "family": "BETA_TIMING",
        "candidate_id": candidate,
        "candidate_index": index,
        "beta_flex": beta_flex,
        "beta_extend": beta_extend,
        "share_shift": share_shift,
        "E3": e3,
        "E2": .99 if feasible else 1.02,
        "peak_ratio": 1.0,
        "feasible": feasible,
        "hip_flex_rms_nm": e3,
        "hip_extend_rms_nm": e3,
        "knee_flex_rms_nm": e3,
        "knee_extend_rms_nm": e3,
    }


def test_infeasible_low_e3_is_not_oracle_or_near_oracle() -> None:
    landscape = pd.DataFrame([
        _row("MYOLEG_VP_001", "BETA_TIMING:0", 0, .80, False, -.12, -.12, -.10),
        _row("MYOLEG_VP_001", "BETA_TIMING:1", 1, .90, True, .12, .12, .10),
    ])

    oracle, common = _oracle_table(landscape, .01)

    result = oracle.iloc[0]
    assert result["oracle_candidate_id"] == "BETA_TIMING:1"
    assert result["oracle_status"] == "FEASIBLE_ORACLE"
    assert result["tie_equivalent_count"] == 1
    assert result["near_oracle_count_eps_0.001"] == 1
    assert common.iloc[0]["common_candidate_id"] == "BETA_TIMING:1"


def test_crossed_two_subject_landscape_reports_interaction_and_true_distance() -> None:
    rows = []
    for subject, a, b in (("MYOLEG_VP_001", .90, 1.00), ("MYOLEG_VP_002", 1.00, .90)):
        rows.extend([
            _row(subject, "BETA_TIMING:0", 0, a, True, -.12, -.12, -.10),
            _row(subject, "BETA_TIMING:1", 1, b, True, .12, .12, .10),
        ])
    landscape = pd.DataFrame(rows)
    oracle, _ = _oracle_table(landscape, .01)
    pairwise, summary = _interaction_audit(landscape, oracle)

    assert set(oracle["oracle_candidate_id"]) == {"BETA_TIMING:0", "BETA_TIMING:1"}
    assert summary.iloc[0]["interaction_fraction"] == pytest.approx(1.0)
    assert summary.iloc[0]["median_spearman_rank"] == pytest.approx(-1.0)
    assert pairwise.iloc[0]["oracle_distance"] == pytest.approx(1.0)
    assert not bool(pairwise.iloc[0]["oracle_same"])


def test_pool_prefix_is_causal_and_supplied_scope_cannot_override_subject_id() -> None:
    landscape = pd.DataFrame([
        _row("MYOLEG_VP_001", "BETA_TIMING:0", 0, 1.0, True, 0., 0., 0.),
        _row("MYOLEG_VP_001", "BETA_TIMING:1", 1, .90, True, .12, .12, .10),
        _row("MYOLEG_VP_001", "BETA_TIMING:2", 2, .80, True, -.12, -.12, -.10),
    ])
    oracle, _ = _oracle_table(landscape, .01)
    history = pd.DataFrame([
        {"scope": "DEVELOPMENT", "subject_id": "MYOLEG_VP_001", "family": "BETA_TIMING", "noise_std": 0., "method": "PHYSICS_GREEDY", "seed": 0, "trial": 1, "candidate_id": "BETA_TIMING:0"},
        {"scope": "DEVELOPMENT", "subject_id": "MYOLEG_VP_001", "family": "BETA_TIMING", "noise_std": 0., "method": "PHYSICS_GREEDY", "seed": 0, "trial": 2, "candidate_id": "BETA_TIMING:1"},
        {"scope": "DEVELOPMENT", "subject_id": "MYOLEG_VP_001", "family": "BETA_TIMING", "noise_std": 0., "method": "PHYSICS_GREEDY", "seed": 0, "trial": 3, "candidate_id": "BETA_TIMING:2"},
    ])
    results = pd.DataFrame([
        {"scope": "DEVELOPMENT", "subject_id": "MYOLEG_VP_001", "family": "BETA_TIMING", "noise_std": 0., "method": "PHYSICS_GREEDY", "seed": 0, "budget": 2, "status": "complete", "recommendation_id": "BETA_TIMING:1"},
        {"scope": "DEVELOPMENT", "subject_id": "MYOLEG_VP_001", "family": "BETA_TIMING", "noise_std": 0., "method": "PHYSICS_GREEDY", "seed": 0, "budget": 3, "status": "complete", "recommendation_id": "BETA_TIMING:2"},
    ])
    tmp_path = Path(".cache") / f"myoleg-audit-test-{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True)
    try:
        history.to_csv(tmp_path / "trial_history.csv", index=False)
        results.to_csv(tmp_path / "results.csv", index=False)

        pool, _ = _pool_comparison(landscape, oracle, tmp_path)
        prefix = pool.sort_values("budget").iloc[0]
        assert prefix["executed_count"] == 2
        assert prefix["pool_oracle_id"] == "BETA_TIMING:1"
        assert not bool(prefix["pool_has_full_oracle"])
        assert bool(pool.sort_values("budget").iloc[1]["pool_has_full_oracle"])

        bad = results.copy()
        bad.loc[:, "scope"] = "NATIVE"
        bad.to_csv(tmp_path / "results.csv", index=False)
        with pytest.raises(ValueError, match="EXECUTED_RESULTS_SCOPE_ID_MISMATCH"):
            _pool_comparison(landscape, oracle, tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
