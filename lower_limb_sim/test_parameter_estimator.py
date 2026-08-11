"""第四阶段五参数估计与可辨识性的离线测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import (
    L1,
    L2,
    identification_lower_bounds,
    identification_upper_bounds,
)
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.identifiability_analysis import (
    analyze_identifiability,
    compare_excitation_sets,
)
from lower_limb_sim.identification_dataset import (
    build_identification_dataset,
    split_identification_dataset,
)
from lower_limb_sim.parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
    compute_torque_metrics,
    estimate_subject_parameters,
    predict_joint_torque,
)


@pytest.fixture(scope="module")
def clean_fits():
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    results = {}
    for subject_id in ("baseline", "hip_stiff", "knee_stiff", "heavy_leg"):
        dataset = build_identification_dataset(
            get_dynamic_subject(subject_id),
            "clean",
            sampling_frequency_hz=20.0,
        )
        splits = split_identification_dataset(dataset)
        estimate = estimate_subject_parameters(splits["train"], template, L1, L2)
        results[subject_id] = (dataset, splits, estimate)
    return template, results


def _truth(subject_id: str) -> dict[str, float]:
    return {
        "mass_scale": 1.3 if subject_id == "heavy_leg" else 1.0,
        "k_hip_nm_per_rad": 30.0 if subject_id == "hip_stiff" else 15.0,
        "k_knee_nm_per_rad": 30.0 if subject_id == "knee_stiff" else 12.0,
        "b_hip_nm_s_per_rad": 2.0,
        "b_knee_nm_s_per_rad": 1.5,
    }


@pytest.mark.parametrize(
    "subject_id",
    ("baseline", "hip_stiff", "knee_stiff", "heavy_leg"),
)
def test_clean_subject_parameters_are_recovered(
    clean_fits,
    subject_id: str,
) -> None:
    template, results = clean_fits
    _, splits, estimate = results[subject_id]
    assert estimate.optimizer_success
    truth = _truth(subject_id)
    for parameter, true_value in truth.items():
        relative_error = (
            abs(estimate.estimated_parameters[parameter] - true_value)
            / true_value
        )
        limit = 0.05 if parameter.startswith("b_") else 0.02
        assert relative_error < limit
    test_metrics = compute_torque_metrics(
        splits["test"],
        template,
        estimate.estimated_parameters,
        L1,
        L2,
    )
    assert test_metrics["torque_rmse_combined_nm"] < 1e-6


def test_stiff_and_heavy_variants_are_distinguished(clean_fits) -> None:
    _, results = clean_fits
    assert (
        results["hip_stiff"][2].estimated_parameters["k_hip_nm_per_rad"]
        > results["baseline"][2].estimated_parameters["k_hip_nm_per_rad"]
    )
    assert (
        results["knee_stiff"][2].estimated_parameters["k_knee_nm_per_rad"]
        > results["baseline"][2].estimated_parameters["k_knee_nm_per_rad"]
    )
    assert (
        results["heavy_leg"][2].estimated_parameters["mass_scale"]
        > results["baseline"][2].estimated_parameters["mass_scale"]
    )


@pytest.mark.parametrize("subject_id", ("baseline", "hip_stiff", "knee_stiff", "heavy_leg"))
def test_estimates_are_bounded_and_non_negative(clean_fits, subject_id: str) -> None:
    _, results = clean_fits
    parameters = results[subject_id][2].estimated_parameters
    for name in PARAMETER_NAMES:
        assert identification_lower_bounds[name] <= parameters[name]
        assert parameters[name] <= identification_upper_bounds[name]
        assert parameters[name] >= 0.0


def test_test_prediction_does_not_depend_on_subject_id_or_saved_measured_torque(
    clean_fits,
) -> None:
    template, results = clean_fits
    _, splits, estimate = results["baseline"]
    test = splits["test"].copy()
    reference = predict_joint_torque(
        test,
        template,
        estimate.estimated_parameters,
        L1,
    )
    test["subject_id"] = "decoy_subject_that_has_no_truth_lookup"
    test["tau_measured_hip_nm"] = 1e12
    test["tau_measured_knee_nm"] = -1e12
    observed = predict_joint_torque(
        test,
        template,
        estimate.estimated_parameters,
        L1,
    )
    assert np.allclose(reference[0], observed[0], atol=0.0, rtol=0.0)
    assert np.allclose(reference[1], observed[1], atol=0.0, rtol=0.0)


def _fit_custom_subject(subject):
    baseline = get_dynamic_subject("baseline")
    template = baseline_template_from_dynamic_subject(baseline)
    dataset = build_identification_dataset(
        subject,
        "clean",
        sampling_frequency_hz=20.0,
    )
    return estimate_subject_parameters(
        split_identification_dataset(dataset)["train"],
        template,
        L1,
        L2,
    )


def test_changing_only_damping_does_not_change_estimated_stiffness() -> None:
    baseline = get_dynamic_subject("baseline")
    changed = replace(
        baseline,
        subject_id="damping_only_test",
        b_hip_nm_s_per_rad=4.0,
        b_knee_nm_s_per_rad=3.0,
    )
    estimate = _fit_custom_subject(changed)
    assert abs(estimate.estimated_parameters["k_hip_nm_per_rad"] - 15.0) < 1e-6
    assert abs(estimate.estimated_parameters["k_knee_nm_per_rad"] - 12.0) < 1e-6


def test_changing_only_stiffness_does_not_change_estimated_damping() -> None:
    baseline = get_dynamic_subject("baseline")
    changed = replace(
        baseline,
        subject_id="stiffness_only_test",
        k_hip_nm_per_rad=25.0,
        k_knee_nm_per_rad=22.0,
    )
    estimate = _fit_custom_subject(changed)
    assert abs(estimate.estimated_parameters["b_hip_nm_s_per_rad"] - 2.0) < 1e-6
    assert abs(estimate.estimated_parameters["b_knee_nm_s_per_rad"] - 1.5) < 1e-6


def _analysis_with_common_scales(
    dataframe: pd.DataFrame,
    template,
    parameters,
    common,
    name: str,
):
    scales = (
        common.torque_scales_nm["hip"],
        common.torque_scales_nm["knee"],
    )
    return analyze_identifiability(
        dataframe,
        template,
        parameters,
        L1,
        L2,
        analysis_set=name,
        torque_scales_nm=scales,
    )


def test_removing_fast_trajectories_does_not_improve_mass_or_damping_information(
    clean_fits,
) -> None:
    template, results = clean_fits
    dataset, _, estimate = results["baseline"]
    complete = analyze_identifiability(
        dataset,
        template,
        estimate.estimated_parameters,
        L1,
        L2,
        analysis_set="complete",
    )
    reduced = _analysis_with_common_scales(
        dataset.loc[~dataset["speed_profile"].eq("fast")],
        template,
        estimate.estimated_parameters,
        complete,
        "without_fast",
    )
    for parameter in (
        "mass_scale",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    ):
        assert (
            reduced.information_diagonal[parameter]
            <= complete.information_diagonal[parameter] + 1e-9
        )
    assert reduced.singular_values[-1] <= complete.singular_values[-1] + 1e-9


def test_removing_dominant_family_reduces_target_joint_information(
    clean_fits,
) -> None:
    template, results = clean_fits
    dataset, _, estimate = results["baseline"]
    complete = analyze_identifiability(
        dataset,
        template,
        estimate.estimated_parameters,
        L1,
        L2,
        analysis_set="complete",
    )
    without_hip = _analysis_with_common_scales(
        dataset.loc[~dataset["trajectory_family"].eq("hip_dominant")],
        template,
        estimate.estimated_parameters,
        complete,
        "without_hip",
    )
    without_knee = _analysis_with_common_scales(
        dataset.loc[~dataset["trajectory_family"].eq("knee_dominant")],
        template,
        estimate.estimated_parameters,
        complete,
        "without_knee",
    )
    assert (
        without_hip.information_diagonal["k_hip_nm_per_rad"]
        <= complete.information_diagonal["k_hip_nm_per_rad"] + 1e-9
    )
    assert (
        without_knee.information_diagonal["k_knee_nm_per_rad"]
        <= complete.information_diagonal["k_knee_nm_per_rad"] + 1e-9
    )


def test_complete_excitation_is_full_rank_and_better_than_single_trajectory(
    clean_fits,
) -> None:
    template, results = clean_fits
    dataset, _, estimate = results["baseline"]
    comparison = compare_excitation_sets(
        dataset,
        template,
        estimate.estimated_parameters,
        L1,
        L2,
    )
    single = comparison["A_coupled_nominal"]
    complete = comparison["C_all_families_all_speeds"]
    assert complete.numerical_rank == len(PARAMETER_NAMES)
    assert np.isfinite(complete.singular_values).all()
    assert complete.singular_values[-1] > single.singular_values[-1]
    assert complete.condition_number < single.condition_number
    single_correlation = np.asarray(single.parameter_correlation)
    complete_correlation = np.asarray(complete.parameter_correlation)
    upper = np.triu_indices(len(PARAMETER_NAMES), 1)
    assert np.max(np.abs(complete_correlation[upper])) < np.max(
        np.abs(single_correlation[upper])
    )
