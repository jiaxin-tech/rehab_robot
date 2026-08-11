"""第四阶段辨识数据、观测来源和噪声时序的离线测试。"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lower_limb_sim.config import (
    L1,
    L2,
    identification_dataset_split,
)
from lower_limb_sim.dynamic_subject import get_dynamic_subject
from lower_limb_sim.identification_dataset import (
    build_identification_dataset,
    split_identification_dataset,
)
from lower_limb_sim.noise_models import apply_noise_scenario
from lower_limb_sim.observation_model import joint_torque_from_endpoint_force
from lower_limb_sim.parameter_estimator import (
    baseline_template_from_dynamic_subject,
    estimate_subject_parameters,
)
import lower_limb_sim.parameter_estimator as parameter_estimator_module


@pytest.fixture(scope="module")
def clean_dataset() -> pd.DataFrame:
    return build_identification_dataset(
        get_dynamic_subject("baseline"),
        "clean",
        sampling_frequency_hz=20.0,
    )


def test_dataset_split_has_no_trajectory_overlap(
    clean_dataset: pd.DataFrame,
) -> None:
    splits = split_identification_dataset(clean_dataset)
    keys = {
        name: set(
            dataframe[["trajectory_family", "speed_profile"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for name, dataframe in splits.items()
    }
    assert keys["train"].isdisjoint(keys["validation"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["validation"].isdisjoint(keys["test"])
    assert set.union(*keys.values()) == set(identification_dataset_split)


def test_test_trajectories_never_enter_training(
    clean_dataset: pd.DataFrame,
) -> None:
    training = split_identification_dataset(clean_dataset)["train"]
    assert training["dataset_split"].eq("train").all()
    assert set(
        training[["trajectory_family", "speed_profile"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == {
        key
        for key, split in identification_dataset_split.items()
        if split == "train"
    }


def test_identifier_input_contains_no_true_parameter_or_total_torque_fields(
    clean_dataset: pd.DataFrame,
) -> None:
    forbidden_exact = {
        "mass_scale",
        "k_hip_nm_per_rad",
        "k_knee_nm_per_rad",
        "b_hip_nm_s_per_rad",
        "b_knee_nm_s_per_rad",
    }
    assert forbidden_exact.isdisjoint(clean_dataset.columns)
    assert not any(column.startswith("true_") for column in clean_dataset)
    assert not any(column.startswith("tau_total") for column in clean_dataset)
    source = inspect.getsource(parameter_estimator_module)
    assert "get_dynamic_subject" not in source
    assert "DYNAMIC_SUBJECTS" not in source


def test_tau_measured_is_recomputed_from_jacobian_transpose_force(
    clean_dataset: pd.DataFrame,
) -> None:
    tau_hip, tau_knee = joint_torque_from_endpoint_force(
        clean_dataset["q_hip_rad"],
        clean_dataset["q_knee_rad"],
        clean_dataset["fx_observed_n"],
        clean_dataset["fz_observed_n"],
        L1,
        L2,
    )
    assert np.allclose(tau_hip, clean_dataset["tau_measured_hip_nm"], atol=1e-12)
    assert np.allclose(tau_knee, clean_dataset["tau_measured_knee_nm"], atol=1e-12)
    assert (
        clean_dataset["torque_reconstruction_consistency_error_nm"].max()
        < 1e-9
    )


def test_estimator_reconstructs_force_torque_instead_of_trusting_saved_tau(
    clean_dataset: pd.DataFrame,
) -> None:
    training = split_identification_dataset(clean_dataset)["train"]
    tampered = training.copy()
    tampered["tau_measured_hip_nm"] = 1e8
    tampered["tau_measured_knee_nm"] = -1e8
    template = baseline_template_from_dynamic_subject(
        get_dynamic_subject("baseline")
    )
    reference = estimate_subject_parameters(training, template, L1, L2)
    observed = estimate_subject_parameters(tampered, template, L1, L2)
    assert np.allclose(
        list(reference.estimated_parameters.values()),
        list(observed.estimated_parameters.values()),
        atol=1e-10,
    )


def test_each_trajectory_time_is_strictly_increasing(
    clean_dataset: pd.DataFrame,
) -> None:
    for _, trajectory in clean_dataset.groupby(
        ["trajectory_family", "speed_profile"]
    ):
        assert (np.diff(trajectory["time_s"]) > 0.0).all()


def test_identification_paths_are_above_bed_reachable_and_include_120deg_hip(
    clean_dataset: pd.DataFrame,
) -> None:
    assert clean_dataset["x_pull_m"].min() >= 0.0
    assert clean_dataset["z_pull_m"].min() >= 0.0
    assert clean_dataset["z_knee_m"].min() >= 0.0
    assert (
        clean_dataset["trajectory_id"]
        .eq("identification_excitation_trajectory")
        .all()
    )
    hip_path = clean_dataset.loc[
        clean_dataset["trajectory_family"].eq("hip_dominant")
    ]
    assert np.isclose(hip_path["q_hip_rad"].max(), np.deg2rad(120.0))


def test_force_mapping_invalid_sample_never_enters_fit(
    clean_dataset: pd.DataFrame,
) -> None:
    training = split_identification_dataset(clean_dataset)["train"]
    poisoned = training.copy()
    index = poisoned.index[100]
    poisoned.loc[index, "force_mapping_valid"] = False
    poisoned.loc[index, "fx_observed_n"] = 1e9
    poisoned.loc[index, "fz_observed_n"] = -1e9
    template = baseline_template_from_dynamic_subject(
        get_dynamic_subject("baseline")
    )
    reference = estimate_subject_parameters(
        training.drop(index=index),
        template,
        L1,
        L2,
    )
    observed = estimate_subject_parameters(poisoned, template, L1, L2)
    assert np.allclose(
        list(reference.estimated_parameters.values()),
        list(observed.estimated_parameters.values()),
        atol=1e-10,
    )


def test_dropout_and_freeze_are_explicitly_marked_invalid(
    clean_dataset: pd.DataFrame,
) -> None:
    dropout = apply_noise_scenario(
        clean_dataset,
        "random_dropout_5pct",
        random_seed=123,
    ).dataframe
    dropped = dropout["invalid_reason"].str.contains("wrench_dropout")
    assert dropped.any()
    assert (~dropout.loc[dropped, "sample_valid"]).all()
    assert dropout.loc[dropped, ["fx_observed_n", "fz_observed_n"]].isna().all(
        axis=None
    )

    frozen = apply_noise_scenario(
        clean_dataset,
        "stale_freeze",
        random_seed=123,
    ).dataframe
    stale = frozen["wrench_is_stale"].astype(bool)
    assert stale.any()
    assert (~frozen.loc[stale, "sample_valid"]).all()
    assert frozen.loc[stale, "invalid_reason"].str.contains(
        "stale_wrench_freeze"
    ).all()


def test_delay_is_causal_and_initial_history_is_not_filled(
    clean_dataset: pd.DataFrame,
) -> None:
    delayed = apply_noise_scenario(
        clean_dataset,
        "timing_delay_32ms",
        random_seed=42,
    ).dataframe
    for keys, group in delayed.groupby(["trajectory_family", "speed_profile"]):
        first = group.iloc[0]
        assert not first["sample_valid"]
        assert np.isnan(first["fx_observed_n"])
        valid = group.loc[group["sample_valid"].astype(bool)]
        assert np.allclose(valid["wrench_delay_s"], 0.032, atol=1e-12)


def test_all_clean_valid_samples_are_finite(
    clean_dataset: pd.DataFrame,
) -> None:
    valid = clean_dataset["sample_valid"].astype(bool)
    numeric = clean_dataset.loc[valid].select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_noise_is_reproducible_for_fixed_seed(
    clean_dataset: pd.DataFrame,
) -> None:
    first = apply_noise_scenario(
        clean_dataset,
        "combined_realistic",
        random_seed=2026,
    ).dataframe
    second = apply_noise_scenario(
        clean_dataset,
        "combined_realistic",
        random_seed=2026,
    ).dataframe
    assert_frame_equal(first, second)


def test_advanced_angle_noise_derivatives_do_not_read_ground_truth_derivatives(
    clean_dataset: pd.DataFrame,
) -> None:
    poisoned = clean_dataset.copy()
    derivative_columns = [
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    ]
    poisoned.loc[:, derivative_columns] = 1e6
    first = apply_noise_scenario(
        clean_dataset,
        "advanced_angle_noise",
        random_seed=77,
    ).dataframe
    second = apply_noise_scenario(
        poisoned,
        "advanced_angle_noise",
        random_seed=77,
    ).dataframe
    assert np.allclose(
        first[derivative_columns],
        second[derivative_columns],
        atol=0.0,
        rtol=0.0,
    )
