"""Regression tests for the measured asymmetric periodic reference."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_CLOSED_REFERENCE,
    MEASURED_ASYMMETRIC_NOMINAL_ID,
    MEASURED_ASYMMETRIC_SLOW_ID,
    MEASURED_RAW_REFERENCE,
    build_reference_measured_raw,
    fit_measured_asymmetric_periodic_reference,
    retime_measured_asymmetric_periodic_reference,
)


DATA_DIRECTORY = (
    Path(__file__).resolve().parent
    / "data"
    / "reference_trajectories"
    / "processed"
)


def _full_angles() -> pd.DataFrame:
    return pd.read_csv(DATA_DIRECTORY / "reference_full_angles.csv")


def _raw() -> pd.DataFrame:
    return build_reference_measured_raw(
        _full_angles(),
        start_frame=5844,
        peak_frame=5895,
        end_frame=5934,
    )


def _model():
    return fit_measured_asymmetric_periodic_reference(_raw())


def test_reference_measured_raw_preserves_every_stage5a_source_value():
    full = _full_angles()
    expected = full.loc[full["Frame"].between(5844, 5934)].reset_index(drop=True)
    raw = _raw()
    pd.testing.assert_frame_equal(
        raw.loc[:, expected.columns].reset_index(drop=True),
        expected,
        check_dtype=True,
        check_exact=True,
    )
    assert raw["reference_version"].eq(MEASURED_RAW_REFERENCE).all()
    assert not raw["source_values_modified"].astype(bool).any()
    assert raw["extension_source_is_measured"].astype(bool).all()
    assert not raw["measured_extension_is_reversed_flexion"].astype(bool).any()


def test_measured_extension_is_not_reversed_flexion():
    raw = _raw()
    flexion = raw.loc[raw["cycle_phase"].eq("flexion")]
    extension = raw.loc[raw["cycle_phase"].eq("extension")]
    common = np.linspace(0.0, 1.0, 1001)
    flexion_phase = flexion["segment_phase"].to_numpy(dtype=float)
    extension_phase = extension["segment_phase"].to_numpy(dtype=float)
    for column in ("q_hip_measured_rad", "q_knee_measured_rad"):
        flexion_values = PchipInterpolator(
            flexion_phase, flexion[column].to_numpy(dtype=float)
        )(common)
        # Reverse only for this identity test; the production path never uses it.
        extension_values = PchipInterpolator(
            extension_phase, extension[column].to_numpy(dtype=float)
        )(common)[::-1]
        assert not np.allclose(
            flexion_values, extension_values, atol=np.deg2rad(0.1), rtol=0.0
        )


def test_periodic_closure_is_small_and_preserves_measured_asymmetry():
    model = _model()
    assert model.fit_accepted, model.rejection_reasons
    deviation = model.deviation_audit
    assert np.isclose(deviation.natural_delta_q_hip_deg, -0.45244558113928124)
    assert np.isclose(deviation.natural_delta_q_knee_deg, -0.21627638940202093)
    assert np.isclose(deviation.natural_pull_closure_error_mm, 4.507110393477331)
    assert not deviation.natural_closure_below_numerical_tolerance
    assert deviation.hip_max_deviation_deg <= 0.5
    assert deviation.knee_max_deviation_deg <= 0.5
    assert deviation.pull_point_max_deviation_mm <= 2.5
    asymmetry = model.asymmetry_audit
    assert asymmetry.hip_flexion_extension_asymmetry_rmse_deg > 10.0
    assert asymmetry.knee_flexion_extension_asymmetry_rmse_deg > 10.0
    assert asymmetry.pull_path_asymmetry_rmse_mm > 100.0
    assert asymmetry.asymmetry_preserved
    assert min(
        asymmetry.hip_asymmetry_retention_ratio,
        asymmetry.knee_asymmetry_retention_ratio,
        asymmetry.pull_asymmetry_retention_ratio,
    ) > 0.98


def test_periodic_spline_has_zero_c2_warning_counts():
    continuity = _model().continuity_audit
    assert continuity.spline_degree == 3
    assert continuity.continuity_order == 2
    assert continuity.position_continuity_warning_count == 0
    assert continuity.velocity_continuity_warning_count == 0
    assert continuity.acceleration_continuity_warning_count == 0
    assert continuity.passed


def test_retimed_slow_and_nominal_are_asymmetric_closed_and_c2():
    model = _model()
    for profile, duration, trajectory_id in (
        ("slow", 24.0, MEASURED_ASYMMETRIC_SLOW_ID),
        ("nominal", 12.0, MEASURED_ASYMMETRIC_NOMINAL_ID),
    ):
        trajectory = retime_measured_asymmetric_periodic_reference(
            model,
            profile=profile,
            total_duration_s=duration,
        )
        assert len(trajectory) == 401
        assert np.isclose(trajectory["time_s"].iloc[-1], duration)
        assert trajectory["trajectory_id"].eq(trajectory_id).all()
        assert trajectory["reference_version"].eq(
            MEASURED_ASYMMETRIC_CLOSED_REFERENCE
        ).all()
        assert trajectory["extension_source_is_measured"].astype(bool).all()
        assert not trajectory[
            "measured_extension_is_reversed_flexion"
        ].astype(bool).any()
        np.testing.assert_allclose(
            trajectory[["q_hip_rad", "q_knee_rad"]].iloc[0],
            trajectory[["q_hip_rad", "q_knee_rad"]].iloc[-1],
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            trajectory[["x_pull_m", "z_pull_m"]].iloc[0],
            trajectory[["x_pull_m", "z_pull_m"]].iloc[-1],
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            trajectory["theta_shank_rad"],
            trajectory["q_hip_rad"] - trajectory["q_knee_rad"],
            atol=1e-14,
            rtol=0.0,
        )
        assert np.rad2deg(trajectory["q_hip_rad"]).between(0.0, 120.0).all()
        assert np.rad2deg(trajectory["q_knee_rad"]).between(5.0, 145.0).all()
        assert trajectory["trajectory_sample_valid"].astype(bool).all()
        assert trajectory["formal_execution_allowed"].astype(bool).all()
        assert np.allclose(
            trajectory.iloc[[0, -1]][
                [
                    "dq_hip_rad_s",
                    "dq_knee_rad_s",
                    "ddq_hip_rad_s2",
                    "ddq_knee_rad_s2",
                ]
            ],
            0.0,
            atol=1e-12,
            rtol=0.0,
        )


def test_start_anchored_relative_pull_displacement_closes():
    trajectory = retime_measured_asymmetric_periodic_reference(
        _model(), profile="slow", total_duration_s=24.0
    )
    pull = trajectory[["x_pull_m", "z_pull_m"]].to_numpy(dtype=float)
    delta = pull - pull[0]
    np.testing.assert_allclose(delta[0], 0.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(delta[-1], 0.0, atol=1e-12, rtol=0.0)


def test_measured_reference_modules_have_no_robot_or_safety_imports():
    module_paths = (
        Path(__file__).with_name("reference_measured_asymmetric.py"),
        Path(__file__).with_name("visualize_reference_measured_asymmetric.py"),
    )
    forbidden = ("hardware", "safety", "xCoreSDK_python")
    for path in module_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imported
            for prefix in forbidden
        )
