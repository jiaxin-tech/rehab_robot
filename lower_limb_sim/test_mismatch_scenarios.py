"""阶段 4.5C 模型失配场景定义测试。"""

from __future__ import annotations

import inspect

import numpy as np

import lower_limb_sim.parameter_estimator as parameter_estimator
from lower_limb_sim.config import L1
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.mismatch_dynamics import mismatch_inverse_dynamics
from lower_limb_sim.mismatch_scenarios import (
    MISMATCH_SCENARIOS,
    build_mismatch_subject,
    get_mismatch_scenario,
)
from lower_limb_sim.mismatch_subject import MISMATCH_PARAMETER_NAMES


EXPECTED_SCENARIOS = {
    "matched_linear",
    "nonlinear_stiffness_mild",
    "nonlinear_stiffness_strong",
    "hip_knee_coupling_mild",
    "hip_knee_coupling_strong",
    "nonlinear_damping_mild",
    "structured_residual",
    "combined_mild",
    "combined_strong",
}


def test_all_nine_scenario_names_are_unique() -> None:
    assert len(MISMATCH_SCENARIOS) == 9
    assert len(set(MISMATCH_SCENARIOS)) == len(MISMATCH_SCENARIOS)
    assert set(MISMATCH_SCENARIOS) == EXPECTED_SCENARIOS


def test_matched_linear_has_every_additional_parameter_zero() -> None:
    scenario = get_mismatch_scenario("matched_linear")

    assert set(scenario.generator_parameters) == set(MISMATCH_PARAMETER_NAMES)
    assert all(value == 0.0 for value in scenario.generator_parameters.values())
    assert scenario.model_mismatch_terms == ()


def test_mild_scenarios_are_weaker_than_corresponding_strong_scenarios() -> None:
    stiffness_mild = get_mismatch_scenario("nonlinear_stiffness_mild")
    stiffness_strong = get_mismatch_scenario("nonlinear_stiffness_strong")
    for name in ("k3_hip_nm_per_rad3", "k3_knee_nm_per_rad3"):
        assert (
            stiffness_mild.generator_parameters[name]
            < stiffness_strong.generator_parameters[name]
        )

    coupling_mild = get_mismatch_scenario("hip_knee_coupling_mild")
    coupling_strong = get_mismatch_scenario("hip_knee_coupling_strong")
    for name in ("k_coupling_nm_per_rad", "k_coupling_asymmetry"):
        assert (
            coupling_mild.generator_parameters[name]
            < coupling_strong.generator_parameters[name]
        )

    combined_mild = get_mismatch_scenario("combined_mild")
    combined_strong = get_mismatch_scenario("combined_strong")
    active = set(combined_mild.model_mismatch_terms)
    assert active == set(combined_strong.model_mismatch_terms)
    for name, mild_value in combined_mild.generator_parameters.items():
        if mild_value > 0.0:
            assert mild_value < combined_strong.generator_parameters[name]


def test_combined_scenarios_contain_every_required_term() -> None:
    required_terms = {
        "nonlinear_stiffness",
        "hip_knee_coupling",
        "nonlinear_damping",
        "structured_residual",
    }
    required_nonzero_parameters = {
        "k3_hip_nm_per_rad3",
        "k3_knee_nm_per_rad3",
        "k_coupling_nm_per_rad",
        "k_coupling_asymmetry",
        "b2_hip_nm_s2_per_rad2",
        "b2_knee_nm_s2_per_rad2",
        "residual_torque_scale_nm",
        "residual_torque_frequency",
    }
    for name in ("combined_mild", "combined_strong"):
        scenario = get_mismatch_scenario(name)
        assert set(scenario.model_mismatch_terms) == required_terms
        assert all(
            scenario.generator_parameters[parameter] > 0.0
            for parameter in required_nonzero_parameters
        )


def test_scenario_seed_and_generated_torque_are_reproducible() -> None:
    scenario_first = get_mismatch_scenario("structured_residual")
    scenario_second = get_mismatch_scenario("structured_residual")
    base = get_dynamic_subject("baseline")
    subject_first = scenario_first.create_subject(base)
    subject_second = scenario_second.create_subject(base)
    state = (0.8, 1.1, 0.3, -0.4, 0.2, -0.1)
    result_first = mismatch_inverse_dynamics(
        *state,
        subject_first,
        L1,
        residual_random_seed=scenario_first.random_seed,
    )
    result_second = mismatch_inverse_dynamics(
        *state,
        subject_second,
        L1,
        residual_random_seed=scenario_second.random_seed,
    )

    assert scenario_first.as_metadata_dict() == scenario_second.as_metadata_dict()
    assert subject_first == subject_second
    assert result_first == result_second


def test_every_scenario_has_complete_metadata() -> None:
    expected_metadata = {
        "scenario_name",
        "generator_parameters",
        "estimator_model_description",
        "random_seed",
        "model_mismatch_terms",
    }
    for name in MISMATCH_SCENARIOS:
        scenario = get_mismatch_scenario(name)
        metadata = scenario.as_metadata_dict()
        assert set(metadata) == expected_metadata
        assert metadata["scenario_name"] == name
        assert set(metadata["generator_parameters"]) == set(MISMATCH_PARAMETER_NAMES)
        assert "five_parameter_linear_gray_box" in metadata[
            "estimator_model_description"
        ]
        assert isinstance(metadata["random_seed"], int)
        assert isinstance(metadata["model_mismatch_terms"], list)


def test_all_scenarios_remain_finite_over_configured_motion_range() -> None:
    base = get_dynamic_subject("baseline")
    q_hip = np.deg2rad(np.linspace(0.0, 120.0, 61))
    q_knee = np.deg2rad(np.linspace(5.0, 130.0, 61))
    dq_hip = np.linspace(-2.0, 2.0, 61)
    dq_knee = np.linspace(2.0, -2.0, 61)
    ddq_hip = np.linspace(-4.0, 4.0, 61)
    ddq_knee = np.linspace(4.0, -4.0, 61)

    for name in MISMATCH_SCENARIOS:
        scenario = get_mismatch_scenario(name)
        subject = build_mismatch_subject(base, name)
        result = mismatch_inverse_dynamics(
            q_hip,
            q_knee,
            dq_hip,
            dq_knee,
            ddq_hip,
            ddq_knee,
            subject,
            L1,
            residual_random_seed=scenario.random_seed,
        )
        for value in vars(result).values():
            assert np.isfinite(np.asarray(value)).all(), name
        assert np.max(np.abs(result.tau_total_hip_nm)) < 500.0, name
        assert np.max(np.abs(result.tau_total_knee_nm)) < 500.0, name


def test_generator_truth_is_not_part_of_five_parameter_estimator_interface() -> None:
    source = inspect.getsource(parameter_estimator)
    signature = inspect.signature(parameter_estimator.estimate_subject_parameters)
    expected_parameters = {
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    }

    assert set(parameter_estimator.PARAMETER_NAMES) == expected_parameters
    assert set(MISMATCH_PARAMETER_NAMES).isdisjoint(parameter_estimator.PARAMETER_NAMES)
    assert "mismatch_subject" not in source
    assert "mismatch_scenarios" not in source
    assert "generator_parameters" not in signature.parameters
    for parameter in MISMATCH_PARAMETER_NAMES:
        assert parameter not in signature.parameters


def test_building_scenario_subject_does_not_modify_base_subject() -> None:
    base = get_dynamic_subject("baseline")
    before = base.as_metadata_dict()
    generated = build_mismatch_subject(base, "combined_strong")

    assert base.as_metadata_dict() == before
    assert generated.base_dynamic_subject() == base
    assert generated.mismatch_parameters_dict() == dict(
        get_mismatch_scenario("combined_strong").generator_parameters,
    )

