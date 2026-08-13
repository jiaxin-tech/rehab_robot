"""Stage 5C reference execution and candidate-screening regression tests.

These tests intentionally use a small phase grid.  They exercise the same
software-only pipeline as the command-line runner while keeping the full test
suite practical.  No robot hardware, collection, control, safety, or SDK
module is imported here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import lower_limb_sim.reference_local_excitation as local_excitation_module
from lower_limb_sim.config import (
    hip_range_deg,
    knee_range_deg,
    reference_trajectory_data_dir,
)
from lower_limb_sim.geometry_error_metrics import (
    ESTIMATED_DOMAIN_STATE_COLUMNS,
    StateDomainBounds,
)
from lower_limb_sim.reference_execution_trajectory import (
    CLOSED_REFERENCE,
    MEASURED_REFERENCE,
    HipRomApproval,
    KneeRomApproval,
    apply_execution_rom_policy,
    build_execution_reference_versions,
    closure_metrics,
)
from lower_limb_sim.reference_local_excitation import (
    LOCAL_TRAJECTORY_SPLIT,
    build_local_phase_paths,
    fit_local_identification_domain,
    fit_local_subject_parameters,
)
from lower_limb_sim.reference_trajectory_retiming import (
    build_reference_phase_path,
    load_processed_reference_cycle,
)
from lower_limb_sim.run_reference_candidate_evaluation import (
    CANDIDATE_SPECIFICATIONS,
    evaluate_candidate_trajectories,
    run_reference_candidate_evaluation,
)


SAMPLES_PER_SEGMENT = 31
CONFIGURED_KNEE_ROM_AT_IMPORT = tuple(float(value) for value in knee_range_deg)
EXPLICIT_HIP_APPROVAL = (0.0, 120.0)
EXPLICIT_TEST_APPROVAL = (5.0, 145.0)
STAGE5C_SOURCE_FILES = (
    "reference_execution_trajectory.py",
    "reference_local_excitation.py",
    "run_reference_candidate_evaluation.py",
    "visualize_reference_candidates.py",
)


@pytest.fixture(scope="module")
def source_cycle() -> pd.DataFrame:
    cycle, metadata = load_processed_reference_cycle(reference_trajectory_data_dir)
    assert metadata["fps"] is None
    return cycle


@pytest.fixture(scope="module")
def raw_execution_versions(source_cycle: pd.DataFrame) -> pd.DataFrame:
    return build_execution_reference_versions(
        source_cycle, samples_per_segment=SAMPLES_PER_SEGMENT
    )


@pytest.fixture(scope="module")
def unapproved_result():
    return run_reference_candidate_evaluation(
        processed_directory=reference_trajectory_data_dir,
        approved_knee_range_deg=None,
        samples_per_segment=SAMPLES_PER_SEGMENT,
        save_outputs=False,
        generate_plots=False,
    )


@pytest.fixture(scope="module")
def approved_result():
    # This explicit run-local approval matches the one formal protocol.
    return run_reference_candidate_evaluation(
        processed_directory=reference_trajectory_data_dir,
        approved_hip_range_deg=EXPLICIT_HIP_APPROVAL,
        approved_knee_range_deg=EXPLICIT_TEST_APPROVAL,
        rom_approval_source="pytest_explicit_run_local_approval",
        samples_per_segment=SAMPLES_PER_SEGMENT,
        save_outputs=False,
        generate_plots=False,
    )


def _version(dataframe: pd.DataFrame, name: str) -> pd.DataFrame:
    return dataframe.loc[dataframe["reference_version"].eq(name)].reset_index(
        drop=True
    )


def _estimated_parameters(result) -> dict[str, dict[str, float]]:
    return {
        str(subject_id): {
            str(row.parameter): float(row.estimated_value)
            for row in group.itertuples(index=False)
        }
        for subject_id, group in result.local_parameter_estimates.groupby(
            "subject_id", sort=False
        )
    }


def test_missing_knee_approval_blocks_all_formal_stage5c_work(
    unapproved_result,
) -> None:
    assert not unapproved_result.rom_audit.formal_execution_allowed
    assert unapproved_result.rom_audit.trajectory_requires_rom_confirmation
    assert "approved_knee_rom_missing" in unapproved_result.rom_audit.block_reasons
    assert unapproved_result.local_identification_dataset.empty
    assert unapproved_result.local_parameter_estimates.empty
    assert unapproved_result.local_domain_coverage.empty
    assert unapproved_result.candidate_metrics.empty
    assert unapproved_result.candidate_feasibility.empty
    assert unapproved_result.candidate_trajectories == {}
    assert unapproved_result.metadata["formal_outputs_generated"] is False


def test_stage5c_runner_rejects_nonformal_or_mapped_rom() -> None:
    with pytest.raises(ValueError, match="exactly match ROM_PROTOCOL_V2"):
        run_reference_candidate_evaluation(
            processed_directory=reference_trajectory_data_dir,
            approved_knee_range_deg=(5.0, 130.0),
            save_outputs=False,
            generate_plots=False,
        )
    with pytest.raises(ValueError, match="forbids ROM amplitude mapping"):
        run_reference_candidate_evaluation(
            processed_directory=reference_trajectory_data_dir,
            apply_smooth_rom_mapping=True,
            save_outputs=False,
            generate_plots=False,
        )


def test_run_local_approval_does_not_modify_global_knee_or_hip_limits(
    approved_result,
) -> None:
    assert tuple(float(value) for value in knee_range_deg) == (5.0, 145.0)
    assert tuple(float(value) for value in knee_range_deg) == CONFIGURED_KNEE_ROM_AT_IMPORT
    assert tuple(float(value) for value in hip_range_deg) == (0.0, 120.0)
    assert approved_result.rom_audit.approved_hip_range_deg == EXPLICIT_HIP_APPROVAL
    assert approved_result.rom_audit.approved_knee_range_deg == EXPLICIT_TEST_APPROVAL
    assert approved_result.rom_audit.configured_knee_range_deg == (5.0, 145.0)
    assert approved_result.metadata["configured_knee_range_deg"] == [
        5.0,
        145.0,
    ]
    assert approved_result.metadata["approved_hip_rom_deg"] == [0.0, 120.0]
    assert approved_result.metadata["approved_knee_rom_deg"] == [5.0, 145.0]
    assert approved_result.metadata["rom_approval_status"] == "approved"
    assert approved_result.metadata["rom_mapping_applied"] is False
    assert approved_result.metadata["reference_path_preserved"] is True
    assert approved_result.metadata["reference_max_knee_flexion_deg"] == pytest.approx(
        143.04834840975474
    )


def test_explicit_hip_approval_cannot_expand_the_configured_range() -> None:
    with pytest.raises(ValueError, match="cannot expand"):
        HipRomApproval(0.0, 121.0)


def test_approved_145_knee_rom_releases_reference_without_mapping(
    approved_result,
) -> None:
    assert approved_result.rom_audit.formal_execution_allowed
    assert not approved_result.rom_audit.trajectory_requires_rom_confirmation
    assert approved_result.rom_audit.block_reasons == ()
    assert approved_result.rom_audit.original_knee_range_deg[1] < 145.0
    assert approved_result.rom_audit.rom_mapping_applied is False
    closed = _version(approved_result.execution_versions, CLOSED_REFERENCE)
    np.testing.assert_array_equal(
        closed["q_hip_reference_rad"], closed["q_hip_original_rad"]
    )
    np.testing.assert_array_equal(
        closed["q_knee_reference_rad"], closed["q_knee_original_rad"]
    )


def test_measured_asymmetric_version_preserves_the_full_measured_phase_path(
    source_cycle: pd.DataFrame,
    raw_execution_versions: pd.DataFrame,
) -> None:
    expected = build_reference_phase_path(
        source_cycle, samples_per_segment=SAMPLES_PER_SEGMENT
    ).reset_index(drop=True)
    measured = _version(raw_execution_versions, MEASURED_REFERENCE)
    assert_frame_equal(
        measured.loc[:, expected.columns],
        expected,
        check_dtype=False,
        check_exact=True,
    )
    np.testing.assert_array_equal(
        measured["q_hip_reference_rad"], measured["q_hip_smoothed_rad"]
    )
    np.testing.assert_array_equal(
        measured["q_knee_reference_rad"], measured["q_knee_smoothed_rad"]
    )
    assert not measured["repeatable_loop"].any()
    assert measured["extension_source_is_measured"].all()


def test_closed_symmetric_version_closes_joint_angles_and_pull_point(
    raw_execution_versions: pd.DataFrame,
    approved_result,
) -> None:
    closed = _version(raw_execution_versions, CLOSED_REFERENCE)
    errors = closure_metrics(closed)
    assert errors["q_hip_closure_error_deg"] == pytest.approx(0.0, abs=1e-12)
    assert errors["q_knee_closure_error_deg"] == pytest.approx(0.0, abs=1e-12)
    assert errors["pull_point_closure_error_m"] == pytest.approx(0.0, abs=1e-14)
    assert closed["repeatable_loop"].all()

    c0 = approved_result.candidate_trajectories["C0"]
    np.testing.assert_allclose(
        c0.loc[[c0.index[0], c0.index[-1]], ["q_hip_rad", "q_knee_rad"]],
        np.repeat(
            c0.loc[[c0.index[0]], ["q_hip_rad", "q_knee_rad"]].to_numpy(),
            2,
            axis=0,
        ),
        atol=1e-12,
        rtol=0.0,
    )
    audited = c0.iloc[[0, SAMPLES_PER_SEGMENT - 1, -1]]
    np.testing.assert_allclose(
        audited[
            [
                "dq_hip_rad_s",
                "dq_knee_rad_s",
                "ddq_hip_rad_s2",
                "ddq_knee_rad_s2",
            ]
        ],
        0.0,
        atol=1e-11,
        rtol=0.0,
    )


def test_closed_extension_is_exact_time_reverse_of_measured_flexion(
    raw_execution_versions: pd.DataFrame,
) -> None:
    closed = _version(raw_execution_versions, CLOSED_REFERENCE)
    flexion = closed.loc[closed["cycle_phase"].eq("flexion")].reset_index(drop=True)
    extension = closed.loc[closed["cycle_phase"].eq("extension")].reset_index(
        drop=True
    )
    for column in (
        "source_frame",
        "q_hip_raw_rad",
        "q_knee_raw_rad",
        "q_hip_original_rad",
        "q_knee_original_rad",
        "x_ankle_observed_m",
        "z_ankle_observed_m",
    ):
        np.testing.assert_array_equal(
            extension[column].to_numpy(), flexion[column].to_numpy()[::-1]
        )
    assert not extension["extension_source_is_measured"].any()
    assert extension["reference_provenance"].eq(
        "synthetic_time_reverse_of_measured_flexion"
    ).all()


def test_all_local_adjustments_are_zero_at_cycle_endpoints(
    approved_result,
) -> None:
    paths = build_local_phase_paths(approved_result.execution_versions)
    base = paths["reference_slow"]
    for trajectory_id, path in paths.items():
        for column in ("q_hip_reference_rad", "q_knee_reference_rad"):
            delta = path[column].to_numpy(float) - base[column].to_numpy(float)
            assert delta[0] == pytest.approx(0.0, abs=1e-14), trajectory_id
            assert delta[-1] == pytest.approx(0.0, abs=1e-14), trajectory_id
        assert path["pointwise_angle_clipping_applied"].eq(False).all()


def test_authorized_rom_mapping_is_one_whole_path_affine_map_not_clip(
    raw_execution_versions: pd.DataFrame,
) -> None:
    # Historical 5--130 protocol regression only; never an active loader input.
    mapped, audit = apply_execution_rom_policy(
        raw_execution_versions,
        approved_knee_rom=KneeRomApproval(5.0, 130.0),
        apply_smooth_rom_mapping=True,
    )
    original_closed = _version(raw_execution_versions, CLOSED_REFERENCE)
    mapped_closed = _version(mapped, CLOSED_REFERENCE)
    original = original_closed["q_knee_original_rad"].to_numpy(float)
    normalized = (original - original.min()) / (original.max() - original.min())
    expected = np.deg2rad(5.0) + normalized * np.deg2rad(125.0)
    np.testing.assert_allclose(
        mapped_closed["q_knee_reference_rad"], expected, atol=2e-14, rtol=0.0
    )
    assert audit.rom_mapping_applied
    assert audit.mapping_formula is not None
    assert "original_max - original_min" in audit.mapping_formula

    # The measured record is immutable even when the execution copy is mapped.
    np.testing.assert_array_equal(
        _version(mapped, MEASURED_REFERENCE)["q_knee_reference_rad"],
        _version(raw_execution_versions, MEASURED_REFERENCE)["q_knee_reference_rad"],
    )

    package_dir = Path(__file__).resolve().parent
    for filename in (
        "reference_execution_trajectory.py",
        "reference_local_excitation.py",
    ):
        tree = ast.parse((package_dir / filename).read_text(encoding="utf-8"))
        clip_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "clip")
                or (isinstance(node.func, ast.Name) and node.func.id == "clip")
            )
        ]
        assert clip_calls == [], f"pointwise clip found in {filename}"


def test_knee_phase_perturbation_preserves_knee_peak_and_endpoints(
    approved_result,
) -> None:
    paths = build_local_phase_paths(approved_result.execution_versions)
    base = paths["reference_slow"]["q_knee_reference_rad"].to_numpy(float)
    for trajectory_id in (
        "knee_phase_advance_3pct",
        "knee_phase_delay_3pct",
    ):
        changed = paths[trajectory_id]["q_knee_reference_rad"].to_numpy(float)
        assert changed[0] == pytest.approx(base[0], abs=1e-14)
        assert changed[-1] == pytest.approx(base[-1], abs=1e-14)
        assert np.max(changed) == pytest.approx(np.max(base), abs=1e-13)
        assert np.max(np.abs(changed - base)) > np.deg2rad(0.05)


def test_test_trajectory_never_enters_parameter_fit(
    approved_result,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = approved_result.local_identification_dataset.copy(deep=True)
    sentinel = 987654.321
    test_mask = dataset["dataset_split"].eq("test") & dataset["subject_id"].eq(
        "baseline"
    )
    dataset.loc[test_mask, ["fx_observed_n", "fz_observed_n"]] = sentinel
    captured: list[pd.DataFrame] = []
    real_estimator = local_excitation_module.estimate_subject_parameters

    def spy_estimator(training_dataframe, *args, **kwargs):
        captured.append(training_dataframe.copy(deep=True))
        return real_estimator(training_dataframe, *args, **kwargs)

    monkeypatch.setattr(
        local_excitation_module, "estimate_subject_parameters", spy_estimator
    )
    fit_local_subject_parameters(dataset, subject_ids=("baseline",))

    assert len(captured) == 1
    estimator_input = captured[0]
    assert not estimator_input["fx_observed_n"].eq(sentinel).any()
    assert not estimator_input["fz_observed_n"].eq(sentinel).any()
    expected_train_count = int(
        (
            dataset["subject_id"].eq("baseline")
            & dataset["dataset_split"].eq("train")
        ).sum()
    )
    assert len(estimator_input) == expected_train_count
    assert approved_result.metadata["test_used_for_parameter_fit"] is False
    assert LOCAL_TRAJECTORY_SPLIT["knee_phase_delay_3pct"] == "test"


def test_identification_domain_uses_training_estimated_states_only(
    approved_result,
) -> None:
    dataset = approved_result.local_identification_dataset.copy(deep=True)
    reference_bounds = fit_local_identification_domain(dataset)
    nontrain = ~dataset["dataset_split"].eq("train")
    observed_state_columns = (
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    dataset.loc[nontrain, list(observed_state_columns)] = 1.0e6
    # A truth-looking field must not be consulted by the runtime-domain fit.
    dataset["q_hip_true_rad"] = -1.0e9
    poisoned_bounds = fit_local_identification_domain(dataset)
    assert poisoned_bounds == reference_bounds
    assert poisoned_bounds.columns == ESTIMATED_DOMAIN_STATE_COLUMNS
    assert all("true" not in column for column in poisoned_bounds.columns)
    assert approved_result.metadata["test_used_for_domain_fit"] is False


def test_fast_profile_is_excluded_from_main_candidate_set_and_ranking(
    approved_result,
) -> None:
    assert set(CANDIDATE_SPECIFICATIONS) == {f"C{index}" for index in range(9)}
    assert set(approved_result.candidate_trajectories) == set(CANDIDATE_SPECIFICATIONS)
    for trajectory in approved_result.candidate_trajectories.values():
        assert trajectory["main_candidate_profile"].eq("slow").all()
        assert not trajectory["software_stress_test"].any()
        assert trajectory["ranking_eligible_profile"].all()
    assert approved_result.candidate_metrics[
        "fast_excluded_from_main_ranking"
    ].all()
    assert approved_result.metadata["fast_profile"][
        "included_in_candidate_ranking"
    ] is False


def test_subtractive_shank_angle_convention_holds_everywhere(
    approved_result,
) -> None:
    tables = [
        approved_result.execution_versions,
        approved_result.local_identification_dataset,
        *approved_result.candidate_trajectories.values(),
    ]
    for dataframe in tables:
        np.testing.assert_allclose(
            dataframe["theta_shank_rad"],
            dataframe["q_hip_rad"] - dataframe["q_knee_rad"],
            atol=1e-14,
            rtol=0.0,
        )
    assert approved_result.metadata["model_angle_definition"] == (
        "theta_shank = q_hip - q_knee"
    )


def test_candidates_above_rom_or_below_domain_coverage_are_explicitly_rejected(
    approved_result,
) -> None:
    trajectories = {
        candidate_id: trajectory.copy(deep=True)
        for candidate_id, trajectory in approved_result.candidate_trajectories.items()
    }
    violating = trajectories["C1"]
    middle = violating.index[len(violating) // 2]
    violating.loc[middle, "q_knee_rad"] = np.deg2rad(151.0)
    violating.loc[middle, "theta_shank_rad"] = (
        violating.loc[middle, "q_hip_rad"] - violating.loc[middle, "q_knee_rad"]
    )
    narrow_domain = StateDomainBounds(
        columns=ESTIMATED_DOMAIN_STATE_COLUMNS,
        lower=(0.0,) * len(ESTIMATED_DOMAIN_STATE_COLUMNS),
        upper=(0.0,) * len(ESTIMATED_DOMAIN_STATE_COLUMNS),
        valid_training_samples=1,
    )
    evaluation = evaluate_candidate_trajectories(
        trajectories,
        _estimated_parameters(approved_result),
        narrow_domain,
        approved_knee_rom=KneeRomApproval(*EXPLICIT_TEST_APPROVAL),
    )
    c0 = evaluation.feasibility.set_index("candidate_id").loc["C0"]
    c1 = evaluation.feasibility.set_index("candidate_id").loc["C1"]
    assert not bool(c0["candidate_feasible"])
    assert "identification_domain_insufficient" in c0["infeasible_reasons"]
    assert not bool(c1["candidate_feasible"])
    assert "rom_violation" in c1["infeasible_reasons"]
    assert "identification_domain_insufficient" in c1["infeasible_reasons"]
    assert evaluation.pareto.empty or not evaluation.pareto["candidate_id"].isin(
        ["C0", "C1"]
    ).any()


def test_candidate_force_is_only_a_software_relative_audit_metric(
    approved_result,
) -> None:
    comparison = approved_result.candidate_subject_comparison
    assert comparison["force_is_software_relative_metric_only"].all()
    assert not comparison["force_is_real_robot_safety_threshold"].any()
    assert comparison["simulation_status"].eq("software_only").all()
    assert approved_result.metadata["force_metrics_are_software_relative_only"] is True
    assert (
        approved_result.metadata["force_metrics_are_real_robot_safety_thresholds"]
        is False
    )


def test_candidate_paths_remain_closed_within_approved_rom_and_never_raise_maxima(
    approved_result,
) -> None:
    c0 = approved_result.candidate_trajectories["C0"]
    reference_max = c0[["q_hip_rad", "q_knee_rad"]].max()
    for candidate_id, trajectory in approved_result.candidate_trajectories.items():
        errors = closure_metrics(trajectory)
        assert abs(errors["q_hip_closure_error_deg"]) <= 1e-9, candidate_id
        assert abs(errors["q_knee_closure_error_deg"]) <= 1e-9, candidate_id
        assert errors["pull_point_closure_error_m"] <= 1e-12, candidate_id
        assert trajectory["q_hip_rad"].min() >= np.deg2rad(hip_range_deg[0]) - 1e-12
        assert trajectory["q_hip_rad"].max() <= np.deg2rad(hip_range_deg[1]) + 1e-12
        assert trajectory["q_knee_rad"].min() >= np.deg2rad(5.0) - 1e-12
        assert trajectory["q_knee_rad"].max() <= np.deg2rad(145.0) + 1e-12
        assert trajectory["q_hip_rad"].max() <= reference_max["q_hip_rad"] + 1e-12
        assert trajectory["q_knee_rad"].max() <= reference_max["q_knee_rad"] + 1e-12


def test_stage5c_has_no_real_robot_hardware_control_collection_or_sdk_imports() -> None:
    package_dir = Path(__file__).resolve().parent
    forbidden_roots = {
        "hardware",
        "collection",
        "control",
        "safety",
        "sdk",
        "xcoresdk",
        "rokae",
    }
    for filename in STAGE5C_SOURCE_FILES:
        path = package_dir / filename
        assert path.is_file(), filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        for imported in imported_modules:
            components = {component.lower() for component in imported.split(".")}
            assert components.isdisjoint(forbidden_roots), (
                f"forbidden real-robot dependency {imported!r} imported by {filename}"
            )
