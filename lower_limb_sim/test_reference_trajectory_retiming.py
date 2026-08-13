"""Stage 5B reference-path retiming and software-boundary tests."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from lower_limb_sim.config import (
    L1,
    L2,
    hip_range_deg,
    knee_range_deg,
    reference_retiming_durations_s,
    reference_trajectory_data_dir,
)
from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.reference_trajectory_retiming import (
    ApprovedRom,
    SUBJECT_IDS,
    apply_approved_rom_mapping,
    build_reference_phase_path,
    load_processed_reference_cycle,
    retime_reference_path,
    run_reference_retiming,
)


SAMPLES_PER_SEGMENT = 101
KINEMATIC_COLUMNS = (
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
    "x_knee_m",
    "z_knee_m",
    "x_pull_m",
    "z_pull_m",
)


@pytest.fixture(scope="module")
def source_cycle_and_metadata() -> tuple[pd.DataFrame, dict[str, object]]:
    assert reference_trajectory_data_dir.is_dir()
    return load_processed_reference_cycle(reference_trajectory_data_dir)


@pytest.fixture(scope="module")
def raw_phase_path(
    source_cycle_and_metadata: tuple[pd.DataFrame, dict[str, object]],
) -> pd.DataFrame:
    source_cycle, _ = source_cycle_and_metadata
    return build_reference_phase_path(
        source_cycle,
        samples_per_segment=SAMPLES_PER_SEGMENT,
    )


@pytest.fixture(scope="module")
def approved_phase_path(
    raw_phase_path: pd.DataFrame,
) -> pd.DataFrame:
    mapped, audit = apply_approved_rom_mapping(
        raw_phase_path,
        approved_rom=ApprovedRom(knee_deg=(5.0, 130.0)),
    )
    assert audit.dynamics_allowed
    return mapped


@pytest.fixture(scope="module")
def default_result():
    return run_reference_retiming(
        processed_directory=reference_trajectory_data_dir,
        profiles=("slow", "nominal", "fast"),
        samples_per_segment=SAMPLES_PER_SEGMENT,
        save_outputs=False,
        generate_plots=False,
    )


@pytest.fixture(scope="module")
def approved_result():
    return run_reference_retiming(
        processed_directory=reference_trajectory_data_dir,
        profiles=("slow", "nominal", "fast"),
        approved_knee_range_deg=(5.0, 130.0),
        samples_per_segment=SAMPLES_PER_SEGMENT,
        save_outputs=False,
        generate_plots=False,
    )


def test_phase_path_is_available_without_source_fps(
    source_cycle_and_metadata: tuple[pd.DataFrame, dict[str, object]],
    raw_phase_path: pd.DataFrame,
) -> None:
    source_cycle, source_metadata = source_cycle_and_metadata
    assert source_metadata["fps"] is None
    assert source_cycle["time_s"].isna().all()
    assert "time_s" not in raw_phase_path
    assert raw_phase_path["source_timing_status"].eq("unknown").all()
    assert raw_phase_path["global_phase"].iloc[0] == pytest.approx(0.0)
    assert raw_phase_path["global_phase"].iloc[-1] == pytest.approx(1.0)
    assert np.all(np.diff(raw_phase_path["global_phase"]) >= 0.0)
    for _, segment in raw_phase_path.groupby("cycle_phase", sort=False):
        assert segment["segment_phase"].iloc[0] == pytest.approx(0.0)
        assert segment["segment_phase"].iloc[-1] == pytest.approx(1.0)
        assert np.all(np.diff(segment["segment_phase"]) > 0.0)


def test_metadata_separates_unknown_source_timing_from_retimed_clock(
    default_result,
) -> None:
    metadata = default_result.metadata
    assert metadata["source_fps"] is None
    assert metadata["source_timing_status"] == "unknown"
    assert metadata["retimed_trajectory"] is True
    assert metadata["retimed_timing_is_original"] is False
    assert "not the original" in metadata["retimed_timing_warning"]
    assert metadata["phase_parameterization"] == (
        "joint_space_geometric_arc_length_per_segment"
    )
    assert metadata["minimum_jerk_controls"] == (
        "path_phase_not_joint_endpoint_line"
    )
    for trajectory in default_result.retimed_by_profile.values():
        assert trajectory["source_timing_status"].eq("unknown").all()
        assert trajectory["retimed_trajectory"].all()
        assert not trajectory["retimed_timing_is_original"].any()


def test_minimum_jerk_clock_follows_curved_reference_not_endpoint_line(
    approved_result,
) -> None:
    trajectory = approved_result.retimed_by_profile["nominal"]
    maximum_deviation = 0.0
    for _, segment in trajectory.groupby("cycle_phase", sort=False):
        q = segment[["q_hip_rad", "q_knee_rad"]].to_numpy(dtype=float)
        path_phase = segment["segment_phase"].to_numpy(dtype=float)
        endpoint_line = q[0] + path_phase[:, None] * (q[-1] - q[0])
        maximum_deviation = max(
            maximum_deviation,
            float(np.max(np.linalg.norm(q - endpoint_line, axis=1))),
        )
    # The provided path bends by several degrees in joint space.  A direct
    # minimum-jerk interpolation between only its endpoints would fail this.
    assert maximum_deviation > np.deg2rad(2.0)


def test_minimum_jerk_branch_endpoints_have_zero_velocity_and_acceleration(
    approved_result,
) -> None:
    trajectory = approved_result.retimed_by_profile["nominal"]
    transition = np.flatnonzero(
        trajectory["cycle_phase"].to_numpy()[1:]
        != trajectory["cycle_phase"].to_numpy()[:-1]
    )
    assert transition.tolist() == [SAMPLES_PER_SEGMENT - 1]
    audited_rows = trajectory.iloc[[0, int(transition[0]), -1]]
    endpoint_state = audited_rows[
        [
            "dq_hip_rad_s",
            "dq_knee_rad_s",
            "ddq_hip_rad_s2",
            "ddq_knee_rad_s2",
        ]
    ].to_numpy(dtype=float)
    np.testing.assert_allclose(endpoint_state, 0.0, atol=1e-12, rtol=0.0)
    assert np.all(np.diff(trajectory["time_s"]) > 0.0)


def test_three_profiles_have_identical_spatial_path(
    approved_result,
) -> None:
    retimed = approved_result.retimed_by_profile
    assert set(retimed) == {"slow", "nominal", "fast"}
    reference = retimed["nominal"]
    spatial_columns = (
        "segment_phase",
        "global_phase",
        "q_hip_rad",
        "q_knee_rad",
        "q_hip_raw_rad",
        "q_knee_raw_rad",
        "q_hip_smoothed_rad",
        "q_knee_smoothed_rad",
        "x_knee_m",
        "z_knee_m",
        "x_pull_m",
        "z_pull_m",
        "x_ankle_observed_m",
        "z_ankle_observed_m",
    )
    for profile in ("slow", "fast"):
        assert len(retimed[profile]) == len(reference)
        np.testing.assert_allclose(
            retimed[profile].loc[:, spatial_columns],
            reference.loc[:, spatial_columns],
            atol=1e-12,
            rtol=0.0,
        )


def test_duration_controls_velocity_and_acceleration_scaling(
    approved_result,
) -> None:
    retimed = approved_result.retimed_by_profile
    slow = retimed["slow"]
    nominal = retimed["nominal"]
    fast = retimed["fast"]
    assert slow["time_s"].iloc[-1] == pytest.approx(24.0)
    assert nominal["time_s"].iloc[-1] == pytest.approx(12.0)
    assert fast["time_s"].iloc[-1] == pytest.approx(6.0)
    for joint in ("hip", "knee"):
        np.testing.assert_allclose(
            nominal[f"dq_{joint}_rad_s"],
            2.0 * slow[f"dq_{joint}_rad_s"],
            atol=2e-12,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            fast[f"dq_{joint}_rad_s"],
            4.0 * slow[f"dq_{joint}_rad_s"],
            atol=4e-12,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            nominal[f"ddq_{joint}_rad_s2"],
            4.0 * slow[f"ddq_{joint}_rad_s2"],
            atol=5e-11,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            fast[f"ddq_{joint}_rad_s2"],
            16.0 * slow[f"ddq_{joint}_rad_s2"],
            atol=2e-10,
            rtol=2e-12,
        )
    assert reference_retiming_durations_s["slow"]["flexion"] == 12.0
    assert reference_retiming_durations_s["nominal"]["flexion"] == 6.0
    assert reference_retiming_durations_s["fast"]["flexion"] == 3.0


def test_chain_rule_uses_plus_sign_and_matches_finite_difference(
    approved_result,
) -> None:
    trajectory = approved_result.retimed_by_profile["nominal"]
    time_s = trajectory["time_s"].to_numpy(dtype=float)
    phase_rate = trajectory[
        "minimum_jerk_phase_rate_s_inv"
    ].to_numpy(dtype=float)
    phase_acceleration = trajectory[
        "minimum_jerk_phase_acceleration_s_inv2"
    ].to_numpy(dtype=float)
    transition = int(
        np.flatnonzero(
            trajectory["cycle_phase"].to_numpy()[1:]
            != trajectory["cycle_phase"].to_numpy()[:-1]
        )[0]
        + 1
    )
    finite_difference_mask = np.ones(len(trajectory), dtype=bool)
    finite_difference_mask[:4] = False
    finite_difference_mask[-4:] = False
    finite_difference_mask[transition - 4 : transition + 4] = False

    for joint in ("hip", "knee"):
        q_s = trajectory[f"dq_{joint}_ds_rad"].to_numpy(dtype=float)
        q_ss = trajectory[f"d2q_{joint}_ds2_rad"].to_numpy(dtype=float)
        reported = trajectory[f"ddq_{joint}_rad_s2"].to_numpy(dtype=float)
        plus_formula = q_ss * phase_rate**2 + q_s * phase_acceleration
        wrong_minus_formula = q_ss * phase_rate**2 - q_s * phase_acceleration
        np.testing.assert_allclose(reported, plus_formula, atol=1e-13, rtol=0.0)

        finite_difference = np.gradient(
            trajectory[f"dq_{joint}_rad_s"].to_numpy(dtype=float),
            time_s,
            edge_order=2,
        )
        plus_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        finite_difference[finite_difference_mask]
                        - plus_formula[finite_difference_mask]
                    )
                    ** 2
                )
            )
        )
        minus_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        finite_difference[finite_difference_mask]
                        - wrong_minus_formula[finite_difference_mask]
                    )
                    ** 2
                )
            )
        )
        assert plus_rmse < 0.5 * minus_rmse


def test_subtractive_shank_angle_and_existing_forward_kinematics_are_preserved(
    approved_result,
) -> None:
    for trajectory in approved_result.retimed_by_profile.values():
        q_hip = trajectory["q_hip_rad"].to_numpy(dtype=float)
        q_knee = trajectory["q_knee_rad"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            trajectory["theta_shank_rad"],
            q_hip - q_knee,
            atol=1e-14,
            rtol=0.0,
        )
        expected = forward_kinematics(q_hip, q_knee, L1, L2)
        for column, values in zip(
            ("x_knee_m", "z_knee_m", "x_pull_m", "z_pull_m"),
            expected,
        ):
            np.testing.assert_allclose(
                trajectory[column], values, atol=1e-13, rtol=0.0
            )
    assert tuple(float(value) for value in hip_range_deg) == (0.0, 120.0)
    assert tuple(float(value) for value in knee_range_deg) == (5.0, 145.0)


def test_default_143_degree_knee_path_is_valid_under_formal_v2_rom(
    default_result,
) -> None:
    audit = default_result.rom_audit
    assert 142.0 < audit.original_angle_range_deg["knee"][1] < 144.0
    assert not audit.trajectory_requires_rom_confirmation
    assert audit.dynamics_allowed
    assert not audit.rom_mapping_applied
    assert audit.confirmation_reasons == ()
    assert default_result.dynamics_by_profile_subject
    assert default_result.metadata["dynamics_evaluated"]
    assert default_result.metadata["dynamics_allowed"]
    assert default_result.subject_comparison["dynamics_evaluated"].all()
    for trajectory in default_result.retimed_by_profile.values():
        assert np.rad2deg(trajectory["q_knee_rad"].max()) > 142.0
        assert trajectory["joint_limit_valid"].all()
        assert trajectory["dynamics_allowed"].all()


def test_explicit_rom_mapping_is_affine_not_clipping_and_preserves_originals(
    raw_phase_path: pd.DataFrame,
) -> None:
    original = raw_phase_path.copy(deep=True)
    # Historical 5--130 mapping regression only; the active protocol is V2.
    mapped, audit = apply_approved_rom_mapping(
        raw_phase_path,
        approved_rom=ApprovedRom(knee_deg=(5.0, 130.0)),
    )
    pd.testing.assert_frame_equal(raw_phase_path, original, check_exact=True)
    np.testing.assert_array_equal(
        mapped["q_knee_smoothed_rad"], original["q_knee_smoothed_rad"]
    )
    np.testing.assert_array_equal(
        mapped["q_hip_reference_rad"], original["q_hip_smoothed_rad"]
    )

    knee_original = original["q_knee_smoothed_rad"].to_numpy(dtype=float)
    normalized = (knee_original - knee_original.min()) / (
        knee_original.max() - knee_original.min()
    )
    expected = np.deg2rad(5.0) + normalized * np.deg2rad(125.0)
    np.testing.assert_allclose(
        mapped["q_knee_reference_rad"], expected, atol=1e-14, rtol=0.0
    )
    assert np.rad2deg(mapped["q_knee_reference_rad"].min()) == pytest.approx(5.0)
    assert np.rad2deg(mapped["q_knee_reference_rad"].max()) == pytest.approx(130.0)
    assert mapped["q_knee_reference_rad"].nunique() > 0.8 * len(mapped)
    assert audit.rom_mapping_applied_by_joint == {"hip": False, "knee": True}
    assert "approved_min" in audit.mapping_formula["knee"]
    assert "clip" not in audit.mapping_formula["knee"].lower()
    assert audit.dynamics_allowed


def test_observed_ankle_remains_separate_from_model_pull_point(
    approved_phase_path: pd.DataFrame,
) -> None:
    reference = retime_reference_path(
        approved_phase_path,
        profile="nominal",
        flexion_duration_s=6.0,
        extension_duration_s=6.0,
        samples_per_segment=SAMPLES_PER_SEGMENT,
    )
    ankle = reference[
        ["x_ankle_observed_m", "z_ankle_observed_m"]
    ].to_numpy(dtype=float)
    pull = reference[["x_pull_m", "z_pull_m"]].to_numpy(dtype=float)
    assert np.max(np.linalg.norm(ankle - pull, axis=1)) > 0.01
    assert not reference["observed_ankle_is_pull_point"].any()

    tampered_path = approved_phase_path.copy(deep=True)
    tampered_path["x_ankle_observed_m"] += 1.0
    tampered_path["z_ankle_observed_m"] -= 1.0
    tampered = retime_reference_path(
        tampered_path,
        profile="nominal",
        flexion_duration_s=6.0,
        extension_duration_s=6.0,
        samples_per_segment=SAMPLES_PER_SEGMENT,
    )
    np.testing.assert_allclose(
        tampered[["q_hip_rad", "q_knee_rad", "x_pull_m", "z_pull_m"]],
        reference[["q_hip_rad", "q_knee_rad", "x_pull_m", "z_pull_m"]],
        atol=0.0,
        rtol=0.0,
    )
    assert not np.allclose(
        tampered[["x_ankle_observed_m", "z_ankle_observed_m"]], ankle
    )


def test_approved_rom_runs_four_subjects_on_exactly_the_same_geometry(
    approved_result,
) -> None:
    assert set(approved_result.dynamics_by_profile_subject) == {
        "slow",
        "nominal",
        "fast",
    }
    for profile, subject_tables in (
        approved_result.dynamics_by_profile_subject.items()
    ):
        assert set(subject_tables) == set(SUBJECT_IDS)
        reference = approved_result.retimed_by_profile[profile]
        for subject_id, dynamics in subject_tables.items():
            assert dynamics["subject_id"].eq(subject_id).all()
            np.testing.assert_allclose(
                dynamics.loc[:, KINEMATIC_COLUMNS],
                reference.loc[:, KINEMATIC_COLUMNS],
                atol=0.0,
                rtol=0.0,
            )
            assert dynamics["dynamic_sample_valid"].any()
            assert dynamics["clinical_validation_status"].eq(
                "not_clinically_validated"
            ).all()
        baseline = subject_tables["baseline"]
        hip_stiff = subject_tables["hip_stiff"]
        knee_stiff = subject_tables["knee_stiff"]
        assert not np.allclose(
            baseline["tau_total_hip_nm"], hip_stiff["tau_total_hip_nm"]
        )
        assert not np.allclose(
            baseline["tau_total_knee_nm"], knee_stiff["tau_total_knee_nm"]
        )


def test_public_parameter_validation_is_fail_closed(
    raw_phase_path: pd.DataFrame,
    approved_phase_path: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="minimum must be below maximum"):
        ApprovedRom(knee_deg=(130.0, 5.0))
    with pytest.raises(ValueError, match="within configured ROM"):
        ApprovedRom(knee_deg=(0.0, 150.0))
    with pytest.raises(ValueError, match="finite and positive"):
        retime_reference_path(
            approved_phase_path,
            profile="bad",
            flexion_duration_s=0.0,
            extension_duration_s=1.0,
        )
    with pytest.raises(ValueError, match="at least 3"):
        build_reference_phase_path(raw_phase_path, samples_per_segment=2)
    with pytest.raises(ValueError, match="unknown retiming profiles"):
        run_reference_retiming(
            profiles=("not_a_profile",),
            save_outputs=False,
            generate_plots=False,
        )


def test_stage5b_has_no_hardware_sdk_or_runtime_stack_import() -> None:
    import lower_limb_sim.reference_trajectory_retiming as module

    source_path = Path(inspect.getsourcefile(module) or "")
    assert source_path.is_file()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_roots = {
        "hardware",
        "control",
        "collection",
        "safety",
        "rokae",
        "xCoreSDK",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    assert imported_roots.isdisjoint(forbidden_roots)

    # A dirty hardware tree is expected during the separately scoped real-robot
    # refactor.  The invariant for Stage 5B is import isolation, asserted above.
