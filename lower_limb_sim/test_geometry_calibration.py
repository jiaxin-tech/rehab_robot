"""Stage 4.5D truth/assumption separation and scenario tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from lower_limb_sim.geometry_calibration import (
    AssumedGeometry,
    TrueGeometry,
    build_assumed_geometry,
    calibration_error_from_geometries,
    create_true_geometry,
)
from lower_limb_sim.geometry_error_scenarios import (
    BASE_GEOMETRY_ERROR_SCENARIOS,
    GEOMETRY_ERROR_SCENARIOS,
    GEOMETRY_ERROR_SCENARIO_DEFINITIONS,
    GEOMETRY_SENSITIVITY_VARIANTS,
    INDEPENDENT_JOINT_MEASUREMENT,
    OBSERVATION_MODES,
    TCP_INVERSE_KINEMATICS,
    get_geometry_error_scenario,
)
from lower_limb_sim.virtual_subject import BASELINE_SUBJECT


@pytest.fixture
def true_geometry() -> TrueGeometry:
    return create_true_geometry(BASELINE_SUBJECT)


def test_true_and_assumed_geometry_are_strictly_separate_value_objects(
    true_geometry: TrueGeometry,
) -> None:
    assumed = build_assumed_geometry(true_geometry, L1_error_m=0.01)

    assert isinstance(true_geometry, TrueGeometry)
    assert isinstance(assumed, AssumedGeometry)
    assert true_geometry is not assumed
    assert {field.name for field in fields(TrueGeometry)}.isdisjoint(
        {field.name for field in fields(AssumedGeometry)}
    )
    assert set(true_geometry.as_metadata_dict()) == {
        "L1_true_m",
        "L2_true_m",
        "hip_center_x_true_m",
        "hip_center_z_true_m",
        "q0_hip_true_rad",
        "q0_knee_true_rad",
    }
    assert set(assumed.as_metadata_dict()) == {
        "L1_assumed_m",
        "L2_assumed_m",
        "hip_center_x_assumed_m",
        "hip_center_z_assumed_m",
        "q0_hip_assumed_rad",
        "q0_knee_assumed_rad",
    }
    with pytest.raises(FrozenInstanceError):
        assumed.L1_assumed_m = 9.0  # type: ignore[misc]


def test_assumed_geometry_cannot_access_truth_or_retain_truth_reference(
    true_geometry: TrueGeometry,
) -> None:
    assumed = build_assumed_geometry(true_geometry)

    forbidden_attributes = (
        "true_geometry",
        "L1_true_m",
        "L2_true_m",
        "hip_center_x_true_m",
        "hip_center_z_true_m",
        "q0_hip_true_rad",
        "q0_knee_true_rad",
    )
    for name in forbidden_attributes:
        assert not hasattr(assumed, name)
        with pytest.raises(AttributeError):
            getattr(assumed, name)
    # slots prevents a hidden, mutable instance dictionary from being attached.
    assert not hasattr(assumed, "__dict__")


@pytest.mark.parametrize("link", ["L1", "L2"])
@pytest.mark.parametrize("magnitude_cm", [1, 2])
def test_positive_and_negative_L1_L2_errors_are_applied_additively(
    true_geometry: TrueGeometry,
    link: str,
    magnitude_cm: int,
) -> None:
    positive = get_geometry_error_scenario(
        f"{link}_error_{magnitude_cm}cm_positive"
    ).create_assumed_geometry(true_geometry)
    negative = get_geometry_error_scenario(
        f"{link}_error_{magnitude_cm}cm_negative"
    ).create_assumed_geometry(true_geometry)
    true_value = getattr(true_geometry, f"{link}_true_m")

    assert getattr(positive, f"{link}_assumed_m") == pytest.approx(
        true_value + magnitude_cm / 100.0
    )
    assert getattr(negative, f"{link}_assumed_m") == pytest.approx(
        true_value - magnitude_cm / 100.0
    )


def test_three_centimetre_L2_sign_variants_are_present(
    true_geometry: TrueGeometry,
) -> None:
    plus = get_geometry_error_scenario("L2_error_3cm_positive")
    minus = get_geometry_error_scenario("L2_error_3cm_negative")
    plus_geometry = plus.create_assumed_geometry(true_geometry)
    minus_geometry = minus.create_assumed_geometry(true_geometry)

    assert plus_geometry.L2_assumed_m - true_geometry.L2_true_m == pytest.approx(
        0.03
    )
    assert minus_geometry.L2_assumed_m - true_geometry.L2_true_m == pytest.approx(
        -0.03
    )


@pytest.mark.parametrize("axis", ["x", "z"])
@pytest.mark.parametrize("direction, expected", [("positive", 0.01), ("negative", -0.01)])
def test_hip_center_offset_uses_world_coordinate_direction(
    true_geometry: TrueGeometry,
    axis: str,
    direction: str,
    expected: float,
) -> None:
    scenario = get_geometry_error_scenario(
        f"hip_center_{axis}_error_1cm_{direction}"
    )
    assumed = scenario.create_assumed_geometry(true_geometry)

    assert getattr(assumed, f"hip_center_{axis}_assumed_m") == pytest.approx(
        getattr(true_geometry, f"hip_center_{axis}_true_m") + expected
    )
    other_axis = "z" if axis == "x" else "x"
    assert getattr(assumed, f"hip_center_{other_axis}_assumed_m") == pytest.approx(
        getattr(true_geometry, f"hip_center_{other_axis}_true_m")
    )


@pytest.mark.parametrize("error_deg", [3, 5])
def test_q0_error_has_correct_radian_unit_and_positive_sign(
    true_geometry: TrueGeometry,
    error_deg: int,
) -> None:
    scenario = get_geometry_error_scenario(f"q0_error_{error_deg}deg")
    assumed = scenario.create_assumed_geometry(true_geometry)
    errors = calibration_error_from_geometries(true_geometry, assumed)

    expected = np.deg2rad(error_deg)
    assert errors["q0_hip_error_rad"] == pytest.approx(expected)
    assert errors["q0_knee_error_rad"] == pytest.approx(expected)
    assert np.rad2deg(errors["q0_hip_error_rad"]) == pytest.approx(error_deg)
    assert np.rad2deg(errors["q0_knee_error_rad"]) == pytest.approx(error_deg)


def test_fixed_seed_reproduces_position_and_angle_noise() -> None:
    tcp = get_geometry_error_scenario("tcp_position_noise_medium")
    angle = get_geometry_error_scenario("independent_angle_noise_medium")

    tcp_first = tcp.sample_tcp_position_noise(128)
    tcp_second = tcp.sample_tcp_position_noise(128)
    angle_first = angle.sample_independent_angle_noise(128)
    angle_second = angle.sample_independent_angle_noise(128)

    np.testing.assert_array_equal(tcp_first[0], tcp_second[0])
    np.testing.assert_array_equal(tcp_first[1], tcp_second[1])
    np.testing.assert_array_equal(angle_first[0], angle_second[0])
    np.testing.assert_array_equal(angle_first[1], angle_second[1])
    assert not np.array_equal(
        tcp_first[0],
        tcp.with_random_seed(tcp.random_seed + 1).sample_tcp_position_noise(128)[0],
    )


def test_exact_nineteen_base_scenarios_and_direction_variants_are_registered() -> None:
    expected_base = {
        "matched_geometry",
        "L1_error_1cm",
        "L1_error_2cm",
        "L2_error_1cm",
        "L2_error_2cm",
        "L2_error_3cm",
        "hip_center_x_error_1cm",
        "hip_center_z_error_1cm",
        "hip_center_combined_error_2cm",
        "q0_error_3deg",
        "q0_error_5deg",
        "tcp_position_noise_low",
        "tcp_position_noise_medium",
        "tcp_position_noise_high",
        "independent_angle_noise_low",
        "independent_angle_noise_medium",
        "independent_angle_noise_high",
        "combined_geometry_mild",
        "combined_geometry_strong",
    }
    assert len(BASE_GEOMETRY_ERROR_SCENARIOS) == 19
    assert set(BASE_GEOMETRY_ERROR_SCENARIOS) == expected_base
    assert set(BASE_GEOMETRY_ERROR_SCENARIOS).issubset(GEOMETRY_ERROR_SCENARIOS)
    assert set(GEOMETRY_SENSITIVITY_VARIANTS).issubset(GEOMETRY_ERROR_SCENARIOS)
    assert set(GEOMETRY_ERROR_SCENARIOS) == set(
        GEOMETRY_ERROR_SCENARIO_DEFINITIONS
    )


def test_mode_applicability_and_static_L2_scope_are_explicit() -> None:
    for scenario in GEOMETRY_ERROR_SCENARIO_DEFINITIONS.values():
        assert scenario.static_calibration_only
        assert not scenario.l2_time_varying
        assert set(scenario.applicable_observation_modes).issubset(
            OBSERVATION_MODES
        )

    tcp = get_geometry_error_scenario("tcp_position_noise_low")
    independent = get_geometry_error_scenario("independent_angle_noise_low")
    assert tcp.applicable_observation_modes == (TCP_INVERSE_KINEMATICS,)
    assert independent.applicable_observation_modes == (
        INDEPENDENT_JOINT_MEASUREMENT,
    )


def test_invalid_geometry_and_dynamic_slip_configuration_are_rejected(
    true_geometry: TrueGeometry,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_assumed_geometry(true_geometry, L2_error_m=-true_geometry.L2_true_m)

    base = get_geometry_error_scenario("matched_geometry")
    with pytest.raises(ValueError, match="dynamic L2/strap slip"):
        type(base)(
            scenario_name="forbidden_dynamic_slip",
            scenario_category="forbidden",
            l2_time_varying=True,
        )
