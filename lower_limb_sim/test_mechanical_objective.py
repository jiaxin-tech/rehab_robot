from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from .mechanical_objective import (
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
    compute_torque_metrics,
    evaluate_mechanical_objective,
    rank_feasible_candidates,
)


def _metrics(scale: float = 1.0):
    time = np.linspace(0.0, 2.0, 101)
    hip = scale * (2.0 + np.sin(np.pi * time))
    knee = scale * (3.0 + 0.5 * np.cos(np.pi * time))
    return compute_torque_metrics(time, hip, knee)


def _objective(scale: float = 1.0, *, hip_deviation: float = 0.0, knee_deviation: float = 0.0):
    return evaluate_mechanical_objective(
        trajectory_id=f"candidate_{scale}",
        metrics=_metrics(scale),
        reference_metrics=_metrics(1.0),
        hip_rms_deviation_deg=hip_deviation,
        knee_rms_deviation_deg=knee_deviation,
    )


def _ranking_rows():
    return pd.DataFrame(
        [
            {
                "trajectory_id": "a",
                "trajectory_feasible": True,
                "mechanical_cost_j_rms": 0.900,
                "reference_deviation": 2.0,
                "combined_peak_ratio": 0.9,
                "combined_torque_rate_ratio": 0.9,
            },
            {
                "trajectory_id": "b",
                "trajectory_feasible": True,
                "mechanical_cost_j_rms": 0.904,
                "reference_deviation": 1.0,
                "combined_peak_ratio": 1.0,
                "combined_torque_rate_ratio": 1.0,
            },
            {
                "trajectory_id": "bad",
                "trajectory_feasible": False,
                "mechanical_cost_j_rms": 0.1,
                "reference_deviation": 0.0,
                "combined_peak_ratio": 0.1,
                "combined_torque_rate_ratio": 0.1,
            },
        ]
    )


def test_reference_mechanical_cost_is_exactly_one():
    assert _objective().mechanical_cost_j_rms == pytest.approx(1.0)


def test_joint_rms_ratios_follow_subject_specific_reference_normalization():
    result = _objective(0.8)
    assert result.hip_rms_ratio == pytest.approx(0.8)
    assert result.knee_rms_ratio == pytest.approx(0.8)
    assert result.mechanical_cost_j_rms == pytest.approx(0.8)


def test_subject_normalization_does_not_share_absolute_torque_scale():
    time = np.linspace(0.0, 1.0, 51)
    waveform = 1.0 + 0.2 * np.sin(2.0 * np.pi * time)
    subject_a = compute_torque_metrics(time, waveform, 2.0 * waveform)
    subject_b = compute_torque_metrics(time, 5.0 * waveform, 10.0 * waveform)
    a = evaluate_mechanical_objective(
        trajectory_id="a",
        metrics=subject_a,
        reference_metrics=subject_a,
        hip_rms_deviation_deg=0.0,
        knee_rms_deviation_deg=0.0,
    )
    b = evaluate_mechanical_objective(
        trajectory_id="b",
        metrics=subject_b,
        reference_metrics=subject_b,
        hip_rms_deviation_deg=0.0,
        knee_rms_deviation_deg=0.0,
    )
    assert a.mechanical_cost_j_rms == b.mechanical_cost_j_rms == pytest.approx(1.0)


def test_reference_deviation_is_joint_rms_root_mean_square():
    assert _objective(1.0, hip_deviation=3.0, knee_deviation=4.0).reference_deviation == pytest.approx(
        np.sqrt((3.0**2 + 4.0**2) / 2.0)
    )


def test_torque_metrics_reject_nonmonotone_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_torque_metrics([0.0, 1.0, 0.5], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0])


def test_infeasible_candidate_never_participates_in_ranking():
    ranked = rank_feasible_candidates(_ranking_rows())
    assert "bad" not in set(ranked["trajectory_id"])


def test_equivalence_tolerance_is_frozen_at_half_percent():
    assert OBJECTIVE_EQUIVALENCE_TOLERANCE == pytest.approx(0.005)


def test_equivalent_candidate_prefers_smaller_reference_deviation():
    ranked = rank_feasible_candidates(_ranking_rows())
    assert ranked.iloc[0]["trajectory_id"] == "b"


def test_outside_equivalence_group_primary_cost_wins():
    rows = _ranking_rows().loc[lambda frame: frame.trajectory_id.ne("bad")].copy()
    rows.loc[rows.trajectory_id.eq("b"), "mechanical_cost_j_rms"] = 0.906
    ranked = rank_feasible_candidates(rows)
    assert ranked.iloc[0]["trajectory_id"] == "a"


def test_equivalent_tie_uses_peak_then_torque_rate_then_lexical_id():
    rows = pd.DataFrame(
        [
            {
                "trajectory_id": name,
                "trajectory_feasible": True,
                "mechanical_cost_j_rms": 0.9,
                "reference_deviation": 1.0,
                "combined_peak_ratio": peak,
                "combined_torque_rate_ratio": rate,
            }
            for name, peak, rate in (("z", 1.0, 0.8), ("b", 0.9, 0.9), ("a", 0.9, 0.9))
        ]
    )
    assert rank_feasible_candidates(rows)["trajectory_id"].tolist() == ["a", "b", "z"]


def test_ranking_is_reproducible_under_input_permutation():
    rows = _ranking_rows()
    expected = rank_feasible_candidates(rows)["trajectory_id"].tolist()
    for seed in range(5):
        shuffled = rows.sample(frac=1.0, random_state=seed)
        assert rank_feasible_candidates(shuffled)["trajectory_id"].tolist() == expected


def test_mechanical_module_contains_no_comfort_objective():
    source = inspect.getsource(evaluate_mechanical_objective).lower()
    assert "comfort" not in source
