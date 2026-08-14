from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .dynamic_subject import get_dynamic_subject
from .formal_protocol import (
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
)
from .mechanical_objective import compute_torque_metrics
from .parameter_estimator import baseline_template_from_dynamic_subject, estimate_subject_parameters
from .reference_release import load_frozen_active_reference
from .sequential_personalization import (
    MINIMUM_STEP_HIP_DEG,
    MINIMUM_STEP_KNEE_DEG,
    MINIMUM_STEP_PHASE,
    STOP_MAX_TRIALS,
    STOP_MINIMUM_STEP_WITHOUT_ACCEPTED_IMPROVEMENT,
    STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD,
    STOP_MODEL_UPDATE_FAILED,
    STOP_NO_FEASIBLE_NEIGHBOR,
    STOP_PREDICTED_IMPROVEMENT_BELOW_TOLERANCE,
    CandidateEvaluation,
    SearchAlpha,
    Stage45CVirtualTruthOracle,
    TrustRegionSteps,
    TruthExecution,
    accept_actual_trial,
    boundary_saturation_audit,
    build_coordinate_neighborhood,
    optimizer_metadata,
    propose_next_trial,
    rejection_stop_reason,
    run_subject_personalization,
    shrink_steps,
)


@pytest.fixture(scope="module")
def matched_result():
    return run_subject_personalization("baseline", "matched_linear", max_trials=2)


@pytest.fixture(scope="module")
def mismatch_result():
    return run_subject_personalization("baseline", "combined_mild", max_trials=2)


@pytest.fixture(scope="module")
def knee_result():
    return run_subject_personalization("knee_stiff", "combined_mild", max_trials=2)


def _estimated_parameters(result) -> dict[str, float]:
    row = result.parameter_history.iloc[-1]
    return {
        "mass_scale": float(row["mass_scale_hat"]),
        "k_hip_nm_per_rad": float(row["K_hip_hat"]),
        "k_knee_nm_per_rad": float(row["K_knee_hat"]),
        "b_hip_nm_s_per_rad": float(row["B_hip_hat"]),
        "b_knee_nm_s_per_rad": float(row["B_knee_hat"]),
    }


def test_initial_alpha_is_frozen_reference():
    assert SearchAlpha().neutral
    assert SearchAlpha().key() == (0.0, 0.0, 0.0)


def test_coordinate_neighborhood_contains_at_most_seven_points():
    assert len(build_coordinate_neighborhood(SearchAlpha(), TrustRegionSteps())) == 7


def test_coordinate_neighborhood_changes_exactly_one_parameter():
    points = build_coordinate_neighborhood(SearchAlpha(), TrustRegionSteps())
    assert {point.key() for point in points} == {
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 0.01),
        (0.0, 0.0, -0.01),
    }


def test_coordinate_neighborhood_does_not_clip_at_search_bound():
    points = build_coordinate_neighborhood(SearchAlpha(-5.0, 0.0, 0.0), TrustRegionSteps())
    assert SearchAlpha(-6.0, 0.0, 0.0) in points


def test_rejection_shrinks_all_steps_by_half():
    reduced = shrink_steps(TrustRegionSteps())
    assert reduced == TrustRegionSteps(0.5, 0.5, 0.005)


def test_steps_never_shrink_below_frozen_minimum():
    reduced = shrink_steps(TrustRegionSteps(0.25, 0.25, 0.0025))
    assert reduced == TrustRegionSteps(
        MINIMUM_STEP_HIP_DEG, MINIMUM_STEP_KNEE_DEG, MINIMUM_STEP_PHASE
    )
    assert reduced.at_minimum


@pytest.mark.parametrize(
    ("actual", "best", "expected"),
    ((0.9949, 1.0, True), (0.995, 1.0, False), (1.01, 1.0, False)),
)
def test_acceptance_requires_strict_improvement_beyond_tolerance(actual, best, expected):
    assert accept_actual_trial(actual, best) is expected


@pytest.mark.parametrize(
    ("alpha", "parameter", "direction"),
    (
        (SearchAlpha(-5.0, 0.0, 0.0), "hip_amplitude_delta_deg", "lower"),
        (SearchAlpha(2.0, 0.0, 0.0), "hip_amplitude_delta_deg", "upper"),
        (SearchAlpha(0.0, -5.0, 0.0), "knee_amplitude_delta_deg", "lower"),
        (SearchAlpha(0.0, 2.0, 0.0), "knee_amplitude_delta_deg", "upper"),
        (SearchAlpha(0.0, 0.0, -0.03), "knee_phase_shift", "lower"),
        (SearchAlpha(0.0, 0.0, 0.03), "knee_phase_shift", "upper"),
    ),
)
def test_boundary_saturation_reports_parameter_and_direction(alpha, parameter, direction):
    audit = boundary_saturation_audit(alpha)
    assert audit["OBJECTIVE_BOUNDARY_SATURATION"] is True
    assert audit["boundary_parameter"] == parameter
    assert audit["boundary_direction"] == direction
    assert audit["search_bounds_expanded"] is False


def test_neutral_point_is_not_boundary_saturated():
    assert boundary_saturation_audit(SearchAlpha())["OBJECTIVE_BOUNDARY_SATURATION"] is False


def test_generator_propagates_frozen_parent_sha_and_fixed_duration():
    generated = generate_personalized_trajectory()
    assert generated.metadata["parent_reference_sha256"] == ACTIVE_REFERENCE_SHA256
    assert generated.metadata["total_duration_s"] == pytest.approx(24.0)
    assert generated.metadata["duration_optimization_enabled"] is False


def test_generated_candidate_preserves_theta_shank_difference():
    trajectory = generate_personalized_trajectory(None, 0.0, -1.0, 0.0).trajectory
    np.testing.assert_allclose(
        trajectory["theta_shank_rad"],
        trajectory["q_hip_rad"] - trajectory["q_knee_rad"],
        atol=1e-14,
        rtol=0.0,
    )


def test_generated_candidate_uses_formal_rom_v2():
    trajectory = generate_personalized_trajectory(None, 0.0, -1.0, 0.0).trajectory
    assert ROM_PROTOCOL_VERSION == "ROM_PROTOCOL_V2"
    assert FORMAL_HIP_ROM_DEG == (0.0, 120.0)
    assert FORMAL_KNEE_ROM_DEG == (5.0, 145.0)
    assert np.rad2deg(trajectory["q_knee_rad"]).max() <= 145.0


def test_proposal_marks_exactly_one_candidate_as_executed(matched_result):
    assert int(matched_result.candidate_audit["proposed_for_execution"].sum()) == 1


def test_infeasible_real_neighbors_are_not_proposed(matched_result):
    infeasible = matched_result.candidate_audit.loc[
        ~matched_result.candidate_audit["trajectory_feasible"].astype(bool)
    ]
    assert not infeasible.empty
    assert not infeasible["proposed_for_execution"].astype(bool).any()


def test_matched_truth_prediction_is_numerically_close_to_actual(matched_result):
    executed = matched_result.history.loc[matched_result.history["trial_id"].gt(0)]
    assert float(executed["prediction_error"].abs().max()) < 1e-10


def test_mismatch_truth_retains_auditable_prediction_error(mismatch_result):
    executed = mismatch_result.history.loc[mismatch_result.history["trial_id"].gt(0)]
    assert not executed.empty
    assert float(executed["prediction_error"].abs().max()) > 1e-6


def test_missing_model_reliability_threshold_stops_fail_closed(mismatch_result):
    assert mismatch_result.summary["stop_reason"] == STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD


def test_predicted_small_improvement_falls_back_to_reference(knee_result):
    assert knee_result.summary["stop_reason"] == STOP_PREDICTED_IMPROVEMENT_BELOW_TOLERANCE
    assert knee_result.final_alpha == SearchAlpha()
    assert knee_result.summary["final_actual_J"] == pytest.approx(1.0)


def test_train_and_heldout_roles_are_isolated(mismatch_result):
    audit = mismatch_result.data_leakage_audit
    assert audit["initial_identification_role"] == "train"
    assert audit["heldout_rows_used_for_proposal"] == 0
    assert audit["heldout_rows_used_for_parameter_fitting"] == 0
    assert audit["heldout_rows_used_for_ranking"] == 0
    assert audit["heldout_rows_used_for_stopping"] == 0
    assert audit["truth_calls_unchanged_during_every_proposal"] is True


def test_heldout_is_final_only_and_never_fit(mismatch_result):
    heldout = mismatch_result.heldout_generalization
    assert set(heldout["data_role"]) == {"held_out_final_test_only"}
    assert not heldout["used_for_parameter_fitting"].astype(bool).any()
    assert not heldout["used_for_proposal"].astype(bool).any()


class _WorseAfterReferenceOracle:
    def __init__(self, subject_id: str, scenario_name: str) -> None:
        self._delegate = Stage45CVirtualTruthOracle(subject_id, scenario_name)
        self.scenario_name = scenario_name

    @property
    def truth_calls(self) -> int:
        return self._delegate.truth_calls

    def simulate(self, trajectory: pd.DataFrame) -> TruthExecution:
        result = self._delegate.simulate(trajectory)
        if self.truth_calls == 1:
            return result
        hip = 1.5 * result.actual_hip_torque_nm
        knee = 1.5 * result.actual_knee_torque_nm
        return TruthExecution(
            estimator_observations=result.estimator_observations,
            actual_hip_torque_nm=hip,
            actual_knee_torque_nm=knee,
            actual_metrics=compute_torque_metrics(trajectory["time_s"], hip, knee),
            observation_valid=True,
            invalid_reason="",
        )


@pytest.fixture(scope="module")
def rejected_result():
    return run_subject_personalization(
        "baseline",
        "matched_linear",
        max_trials=2,
        model_reliability_threshold=10.0,
        truth_oracle_factory=_WorseAfterReferenceOracle,
    )


def test_prediction_good_truth_bad_trial_is_rejected(rejected_result):
    row = rejected_result.history.iloc[-1]
    assert row["predicted_improvement"] > 0.005
    assert row["actual_improvement"] < 0.0
    assert bool(row["accepted"]) is False


def test_rejected_trial_does_not_update_verified_best(rejected_result):
    assert rejected_result.final_alpha == SearchAlpha()
    assert rejected_result.summary["final_actual_J"] == pytest.approx(1.0)
    row = rejected_result.history.iloc[-1]
    assert row["best_alpha_after_hip"] == pytest.approx(0.0)
    assert row["best_alpha_after_knee"] == pytest.approx(0.0)
    assert row["best_alpha_after_phase"] == pytest.approx(0.0)
    assert row["best_accepted_trajectory_id_after"].endswith("k+0.000_p+0.00000")


def test_rejected_executed_data_still_enters_reidentification(rejected_result):
    assert int(rejected_result.parameter_history.iloc[-1]["adaptation_trial_count"]) == 2
    assert bool(rejected_result.parameter_history.iloc[-1]["model_update_success"]) is True


def test_rejection_records_step_shrink_policy(rejected_result):
    # The executed row records the step used for its proposal; the pure shrink
    # rule itself is checked independently above.
    assert rejected_result.summary["stop_reason"] == STOP_MAX_TRIALS
    assert rejected_result.history.iloc[-1]["rejection_reason"] == "actual_not_better_than_verified_best"
    assert rejected_result.history.iloc[-1]["step_hip_after"] == pytest.approx(0.5)
    assert rejected_result.history.iloc[-1]["step_knee_after"] == pytest.approx(0.5)
    assert rejected_result.history.iloc[-1]["step_phase_after"] == pytest.approx(0.005)


def test_rejection_at_minimum_step_has_explicit_stop_reason():
    assert rejection_stop_reason(
        TrustRegionSteps(
            MINIMUM_STEP_HIP_DEG,
            MINIMUM_STEP_KNEE_DEG,
            MINIMUM_STEP_PHASE,
        )
    ) == STOP_MINIMUM_STEP_WITHOUT_ACCEPTED_IMPROVEMENT


def test_model_update_failure_is_explicit_and_falls_back():
    calls = 0

    def estimator(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("synthetic estimator failure")
        return estimate_subject_parameters(*args, **kwargs)

    result = run_subject_personalization(
        "baseline", "matched_linear", max_trials=2, estimator=estimator
    )
    assert result.summary["stop_reason"] == STOP_MODEL_UPDATE_FAILED
    assert result.final_alpha == SearchAlpha()
    assert bool(result.history.iloc[-1]["model_update_success"]) is False


def test_all_infeasible_neighbors_stop_without_execution(matched_result):
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    parameters = _estimated_parameters(matched_result)

    def generator(**kwargs):
        generated = generate_personalized_trajectory(**kwargs)
        alpha = SearchAlpha(
            kwargs["hip_amplitude_delta_deg"],
            kwargs["knee_amplitude_delta_deg"],
            kwargs["knee_phase_shift"],
        )
        if alpha.neutral:
            return generated
        constraints = replace(
            generated.constraints,
            trajectory_feasible=False,
            invalid_reason="synthetic_all_neighbors_infeasible",
        )
        return replace(generated, constraints=constraints)

    proposal = propose_next_trial(
        current=SearchAlpha(),
        steps=TrustRegionSteps(),
        estimated_parameters=parameters,
        template=template,
        generator=generator,
    )
    assert proposal.candidate is None
    assert proposal.stop_reason == STOP_NO_FEASIBLE_NEIGHBOR


def test_domain_coverage_failure_is_never_proposed(matched_result):
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))
    parameters = _estimated_parameters(matched_result)

    def generator(**kwargs):
        generated = generate_personalized_trajectory(**kwargs)
        if kwargs["knee_amplitude_delta_deg"] == -1.0:
            constraints = replace(
                generated.constraints,
                domain_coverage_valid=False,
                trajectory_feasible=False,
                invalid_reason="domain_coverage_insufficient",
            )
            return replace(generated, constraints=constraints)
        return generated

    proposal = propose_next_trial(
        current=SearchAlpha(),
        steps=TrustRegionSteps(),
        estimated_parameters=parameters,
        template=template,
        generator=generator,
    )
    failed = proposal.candidate_audit.loc[
        proposal.candidate_audit["invalid_reason"].str.contains(
            "identification_domain_insufficient", regex=False
        )
        & np.isclose(proposal.candidate_audit["alpha_knee"], -1.0)
        & np.isclose(proposal.candidate_audit["alpha_hip"], 0.0)
    ]
    assert len(failed) == 1
    assert proposal.candidate is None or proposal.candidate.row["trajectory_id"] not in set(
        failed["trajectory_id"]
    )


def test_all_predicted_neighbors_worse_stops_before_execution(monkeypatch):
    import lower_limb_sim.sequential_personalization as module

    rows = [
        CandidateEvaluation(
            SearchAlpha(),
            None,
            {
                "trajectory_id": "current",
                "trajectory_feasible": True,
                "mechanical_cost_j_rms": 1.0,
                "reference_deviation": 0.0,
                "combined_peak_ratio": 1.0,
                "combined_torque_rate_ratio": 1.0,
                "alpha_hip": 0.0,
                "alpha_knee": 0.0,
                "alpha_phase": 0.0,
            },
        ),
        CandidateEvaluation(
            SearchAlpha(-1.0, 0.0, 0.0),
            None,
            {
                "trajectory_id": "worse",
                "trajectory_feasible": True,
                "mechanical_cost_j_rms": 1.01,
                "reference_deviation": 1.0,
                "combined_peak_ratio": 1.01,
                "combined_torque_rate_ratio": 1.01,
                "alpha_hip": -1.0,
                "alpha_knee": 0.0,
                "alpha_phase": 0.0,
            },
        ),
    ]
    metrics = compute_torque_metrics(
        [0.0, 0.5, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]
    )
    monkeypatch.setattr(module, "evaluate_candidate_neighborhood", lambda **kwargs: (rows, metrics))
    proposal = module.propose_next_trial(
        current=SearchAlpha(),
        steps=TrustRegionSteps(),
        estimated_parameters={},
        template=None,
    )
    assert proposal.candidate is None
    assert proposal.stop_reason == STOP_PREDICTED_IMPROVEMENT_BELOW_TOLERANCE


def test_parent_sha_error_propagates_fail_closed(matched_result):
    template = baseline_template_from_dynamic_subject(get_dynamic_subject("baseline"))

    def bad_generator(**kwargs):
        raise RuntimeError("REFERENCE_HASH_MISMATCH")

    with pytest.raises(RuntimeError, match="REFERENCE_HASH_MISMATCH"):
        propose_next_trial(
            current=SearchAlpha(),
            steps=TrustRegionSteps(),
            estimated_parameters=_estimated_parameters(matched_result),
            template=template,
            generator=bad_generator,
        )


def test_legacy_symmetric_parent_is_rejected():
    parent = load_frozen_active_reference()
    legacy = replace(
        parent,
        manifest={**parent.manifest, "reference_id": "reference_closed_symmetric"},
    )
    with pytest.raises(PermissionError, match="frozen active asymmetric"):
        generate_personalized_trajectory(legacy)


def test_max_trial_limit_is_explicit_offline_design_parameter():
    result = run_subject_personalization(
        "baseline",
        "matched_linear",
        max_trials=2,
        model_reliability_threshold=10.0,
    )
    assert len(result.history) == 2
    assert result.summary["stop_reason"] == STOP_MAX_TRIALS


def test_subject_histories_are_independent(matched_result, knee_result):
    assert matched_result.history is not knee_result.history
    assert set(matched_result.history["subject_id"]) == {"baseline"}
    assert set(knee_result.history["subject_id"]) == {"knee_stiff"}


def test_optimizer_metadata_records_no_guessed_safety_or_reliability_threshold():
    metadata = optimizer_metadata()
    assert metadata["model_reliability_threshold"] is None
    assert metadata["equivalence_tolerance_is_robot_safety_threshold"] is False
    assert metadata["robot_motion_authorized"] is False


def test_sequential_module_imports_no_hardware_or_safety_package():
    import lower_limb_sim.sequential_personalization as module

    tree = ast.parse(inspect.getsource(module))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith(("hardware", "safety")) for name in imported)


def test_formal_artifact_directory_contains_required_outputs():
    output = Path(__file__).resolve().parent / "formal_artifacts" / "sequential_personalization"
    required = {
        "sequential_personalization_history.csv",
        "sequential_personalization_summary.csv",
        "mechanical_cost_vs_iteration.png",
        "alpha_evolution_vs_iteration.png",
        "reference_vs_final_personalized.png",
        "predicted_vs_actual_improvement.png",
        "subject_specific_final_parameters.png",
        "DATA_LEAKAGE_AUDIT.md",
    }
    assert required.issubset({path.name for path in output.iterdir()})
