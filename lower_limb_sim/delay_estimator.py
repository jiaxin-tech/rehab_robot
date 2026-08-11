"""仅用 train 拟合、validation 选点的固定 wrench 延迟网格搜索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    delay_search_common_margin_s,
    delay_search_minimum_coverage_ratio,
    delay_search_values_ms,
    identification_initial_guess,
    identification_loss,
    identification_lower_bounds,
    identification_upper_bounds,
    max_alignment_interpolation_gap_s,
)
from .parameter_estimator import (
    PARAMETER_NAMES,
    BaselineSubjectTemplate,
    compute_torque_metrics,
    estimate_subject_parameters,
    measured_joint_torque,
    valid_observations,
)
from .timestamp_alignment import align_wrench_to_state_timestamps

DELAY_RMSE_TIE_TOLERANCE_NM = 1e-12

# 显式观测白名单。wrench_age、旧 wrench_delay、noise_scenario、subject_id、
# true/tau_total 字段、invalid_reason 和 DataFrame attrs 均不进入自动搜索。
DELAY_ESTIMATOR_INPUT_COLUMNS = (
    "trajectory_id",
    "trajectory_family",
    "speed_profile",
    "phase",
    "time_s",
    "trajectory_sample_index",
    "dataset_split",
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
    "fx_observed_n",
    "fz_observed_n",
    "sample_valid",
    "force_mapping_valid",
    "wrench_is_stale",
    "state_timestamp_s",
    "wrench_timestamp_s",
)


@dataclass(frozen=True)
class DelayEstimationResult:
    estimated_delay_s: float
    estimated_parameters: dict[str, float]
    optimizer_success: bool
    optimizer_message: str
    search_curve: pd.DataFrame
    delay_selection_splits: tuple[str, str]
    test_used_for_delay_selection: bool
    alignment_mode: str
    common_training_samples: int
    common_validation_samples: int
    estimator_input_columns: tuple[str, ...]
    search_boundary_hit: bool
    search_warning: str | None

    @property
    def estimated_delay_ms(self) -> float:
        return 1000.0 * self.estimated_delay_s


def sanitize_delay_estimation_input(
    dataframe: pd.DataFrame,
    expected_split: str,
) -> pd.DataFrame:
    """投影为无真值/场景/审计元数据的观测表，并清空 attrs。"""

    if expected_split not in {"train", "validation"}:
        raise ValueError("delay selection accepts only train or validation.")
    missing = set(DELAY_ESTIMATOR_INPUT_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(f"delay estimator input is missing: {sorted(missing)}")
    observed_splits = set(dataframe["dataset_split"].astype(str))
    if observed_splits != {expected_split}:
        raise ValueError(
            f"expected only {expected_split!r} rows, got {sorted(observed_splits)}"
        )
    sanitized = dataframe.loc[:, DELAY_ESTIMATOR_INPUT_COLUMNS].copy(deep=True)
    sanitized["invalid_reason"] = ""
    sanitized.attrs.clear()
    return sanitized.reset_index(drop=True)


def _candidate_grid(
    candidate_delays_s: Sequence[float] | None,
) -> np.ndarray:
    if candidate_delays_s is None:
        candidates = np.asarray(delay_search_values_ms, dtype=float) / 1000.0
    else:
        candidates = np.asarray(candidate_delays_s, dtype=float)
    if (
        candidates.ndim != 1
        or not len(candidates)
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("candidate_delays_s must be a non-empty finite vector.")
    candidates = np.unique(np.round(candidates, decimals=12))
    if np.any(np.diff(candidates) <= 0.0):
        raise ValueError("candidate delays must be strictly increasing.")
    return candidates


def _align_candidate(
    dataframe: pd.DataFrame,
    candidate_s: float,
    L1: float,
    L2: float,
    max_gap_s: float,
    common_margin_s: float,
) -> pd.DataFrame:
    return align_wrench_to_state_timestamps(
        dataframe,
        candidate_s,
        mode="offline_only",
        max_interpolation_gap_s=max_gap_s,
        evaluation_margin_s=common_margin_s,
        L1_m=L1,
        L2_m=L2,
    ).dataframe


def _common_candidate_support(
    dataframe: pd.DataFrame,
    candidates: np.ndarray,
    L1: float,
    L2: float,
    max_gap_s: float,
    common_margin_s: float,
) -> np.ndarray:
    common = np.ones(len(dataframe), dtype=bool)
    for candidate in candidates:
        aligned = _align_candidate(
            dataframe,
            float(candidate),
            L1,
            L2,
            max_gap_s,
            common_margin_s,
        )
        common &= aligned["sample_valid"].astype(bool).to_numpy()
    if not common.any():
        raise ValueError("delay candidates have no common valid support.")
    return common


def _apply_common_support(
    dataframe: pd.DataFrame,
    common_support: np.ndarray,
) -> pd.DataFrame:
    result = dataframe.copy()
    result["sample_valid"] = (
        result["sample_valid"].astype(bool).to_numpy() & common_support
    )
    outside = ~common_support
    if outside.any():
        result.loc[outside, "alignment_invalid_reason"] = (
            "candidate_common_support_excluded"
        )
        result.loc[outside, "invalid_reason"] = (
            "candidate_common_support_excluded"
        )
    return result


def _nominal_margin_support_count(
    dataframe: pd.DataFrame,
    common_margin_s: float,
) -> int:
    support = np.zeros(len(dataframe), dtype=bool)
    for _, group in dataframe.groupby(
        ["trajectory_family", "speed_profile"],
        sort=False,
    ):
        indices = group.index.to_numpy(dtype=int)
        time_s = group["state_timestamp_s"].to_numpy(dtype=float)
        support[indices] = (
            (time_s >= time_s[0] + common_margin_s - 1e-12)
            & (time_s <= time_s[-1] - common_margin_s + 1e-12)
            & group["force_mapping_valid"].astype(bool).to_numpy()
        )
    count = int(support.sum())
    if count == 0:
        raise ValueError("common margin removes every candidate sample.")
    return count


def _fixed_training_torque_scales(
    aligned_training: pd.DataFrame,
    L1: float,
    L2: float,
) -> tuple[float, float]:
    valid = valid_observations(aligned_training)
    measured_hip, measured_knee = measured_joint_torque(valid, L1, L2)
    return (
        max(float(np.std(measured_hip)), 1.0),
        max(float(np.std(measured_knee)), 1.0),
    )


def estimate_wrench_delay(
    training_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    L1: float,
    L2: float,
    *,
    candidate_delays_s: Sequence[float] | None = None,
    max_interpolation_gap_s: float = max_alignment_interpolation_gap_s,
    common_margin_s: float = delay_search_common_margin_s,
    minimum_coverage_ratio: float = delay_search_minimum_coverage_ratio,
    loss: str = identification_loss,
) -> DelayEstimationResult:
    """搜索延迟；函数签名刻意不接受 test dataframe 或 true delay。"""

    if (
        not np.isfinite(minimum_coverage_ratio)
        or minimum_coverage_ratio <= 0.0
        or minimum_coverage_ratio > 1.0
    ):
        raise ValueError("minimum_coverage_ratio must lie in (0, 1].")
    training = sanitize_delay_estimation_input(training_dataframe, "train")
    validation = sanitize_delay_estimation_input(
        validation_dataframe,
        "validation",
    )
    candidates = _candidate_grid(candidate_delays_s)
    common_training = _common_candidate_support(
        training,
        candidates,
        L1,
        L2,
        max_interpolation_gap_s,
        common_margin_s,
    )
    common_validation = _common_candidate_support(
        validation,
        candidates,
        L1,
        L2,
        max_interpolation_gap_s,
        common_margin_s,
    )
    common_training_count = int(common_training.sum())
    common_validation_count = int(common_validation.sum())
    nominal_training_count = _nominal_margin_support_count(
        training,
        common_margin_s,
    )
    nominal_validation_count = _nominal_margin_support_count(
        validation,
        common_margin_s,
    )
    common_training_coverage = common_training_count / nominal_training_count
    common_validation_coverage = (
        common_validation_count / nominal_validation_count
    )
    if (
        common_training_coverage < minimum_coverage_ratio
        or common_validation_coverage < minimum_coverage_ratio
    ):
        raise ValueError(
            "candidate common support is below minimum_coverage_ratio."
        )

    zero_reference = _apply_common_support(
        _align_candidate(
            training,
            0.0,
            L1,
            L2,
            max_interpolation_gap_s,
            common_margin_s,
        ),
        common_training,
    )
    fixed_scales = _fixed_training_torque_scales(zero_reference, L1, L2)
    rows: list[dict[str, object]] = []
    estimates: dict[float, object] = {}
    for candidate in candidates:
        candidate_float = float(candidate)
        aligned_training = _apply_common_support(
            _align_candidate(
                training,
                candidate_float,
                L1,
                L2,
                max_interpolation_gap_s,
                common_margin_s,
            ),
            common_training,
        )
        aligned_validation = _apply_common_support(
            _align_candidate(
                validation,
                candidate_float,
                L1,
                L2,
                max_interpolation_gap_s,
                common_margin_s,
            ),
            common_validation,
        )
        train_valid = int(aligned_training["sample_valid"].sum())
        validation_valid = int(aligned_validation["sample_valid"].sum())
        train_coverage = train_valid / nominal_training_count
        validation_coverage = validation_valid / nominal_validation_count
        row: dict[str, object] = {
            "candidate_delay_ms": 1000.0 * candidate_float,
            "train_valid_samples": train_valid,
            "validation_valid_samples": validation_valid,
            "valid_train_samples": train_valid,
            "valid_validation_samples": validation_valid,
            "valid_sample_count": train_valid + validation_valid,
            "train_coverage_ratio": train_coverage,
            "validation_coverage_ratio": validation_coverage,
            "optimizer_success": False,
            "optimizer_cost": np.nan,
            "train_rmse_nm": np.inf,
            "validation_rmse_nm": np.inf,
            "validation_torque_rmse_hip_nm": np.inf,
            "validation_torque_rmse_knee_nm": np.inf,
            "validation_torque_rmse_combined_nm": np.inf,
            "failure_reason": "",
        }
        if (
            train_coverage < minimum_coverage_ratio
            or validation_coverage < minimum_coverage_ratio
        ):
            row["failure_reason"] = "insufficient_common_support_coverage"
            rows.append(row)
            continue
        try:
            estimate = estimate_subject_parameters(
                aligned_training,
                baseline_subject_template,
                L1,
                L2,
                initial_guess=identification_initial_guess,
                bounds=(
                    identification_lower_bounds,
                    identification_upper_bounds,
                ),
                loss=loss,
                fixed_torque_scales_nm=fixed_scales,
            )
            metrics = compute_torque_metrics(
                aligned_validation,
                baseline_subject_template,
                estimate.estimated_parameters,
                L1,
                L2,
            )
            training_metrics = compute_torque_metrics(
                aligned_training,
                baseline_subject_template,
                estimate.estimated_parameters,
                L1,
                L2,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row.update(
            {
                "optimizer_success": estimate.optimizer_success,
                "optimizer_cost": estimate.cost,
                "train_rmse_nm": training_metrics[
                    "torque_rmse_combined_nm"
                ],
                "validation_rmse_nm": metrics[
                    "torque_rmse_combined_nm"
                ],
                "validation_torque_rmse_hip_nm": metrics[
                    "torque_rmse_hip_nm"
                ],
                "validation_torque_rmse_knee_nm": metrics[
                    "torque_rmse_knee_nm"
                ],
                "validation_torque_rmse_combined_nm": metrics[
                    "torque_rmse_combined_nm"
                ],
                **{
                    f"train_estimate_{name}": value
                    for name, value in estimate.estimated_parameters.items()
                },
            }
        )
        if not estimate.optimizer_success:
            row["failure_reason"] = estimate.optimizer_message
        estimates[candidate_float] = estimate
        rows.append(row)

    curve = pd.DataFrame(rows).sort_values("candidate_delay_ms").reset_index(
        drop=True
    )
    eligible = curve.loc[
        curve["optimizer_success"].astype(bool)
        & np.isfinite(curve["validation_torque_rmse_combined_nm"])
    ].copy()
    if eligible.empty:
        raise RuntimeError("all delay candidates failed.")
    minimum_validation_rmse = float(
        eligible["validation_torque_rmse_combined_nm"].min()
    )
    tied = eligible.loc[
        eligible["validation_torque_rmse_combined_nm"]
        <= minimum_validation_rmse + DELAY_RMSE_TIE_TOLERANCE_NM
    ].copy()
    tied["absolute_candidate_delay_ms"] = tied["candidate_delay_ms"].abs()
    # Explicit deterministic tie-break: smallest absolute delay, then the
    # signed delay. ``mergesort`` keeps the behavior stable across runs.
    best = tied.sort_values(
        ["absolute_candidate_delay_ms", "candidate_delay_ms"],
        kind="mergesort",
    ).iloc[0]
    selected_s = float(best["candidate_delay_ms"]) / 1000.0
    curve["selected"] = np.isclose(
        curve["candidate_delay_ms"],
        best["candidate_delay_ms"],
        atol=1e-12,
        rtol=0.0,
    )
    selected_estimate = estimates[selected_s]
    boundary_hit = bool(
        np.isclose(selected_s, float(candidates[0]), atol=1e-12, rtol=0.0)
        or np.isclose(
            selected_s,
            float(candidates[-1]),
            atol=1e-12,
            rtol=0.0,
        )
    )
    search_warning = (
        "selected_search_boundary_possible_out_of_search_range"
        if boundary_hit
        else None
    )
    return DelayEstimationResult(
        estimated_delay_s=selected_s,
        estimated_parameters=dict(selected_estimate.estimated_parameters),
        optimizer_success=bool(selected_estimate.optimizer_success),
        optimizer_message=str(selected_estimate.optimizer_message),
        search_curve=curve,
        delay_selection_splits=("train", "validation"),
        test_used_for_delay_selection=False,
        alignment_mode="offline_only",
        common_training_samples=common_training_count,
        common_validation_samples=common_validation_count,
        estimator_input_columns=DELAY_ESTIMATOR_INPUT_COLUMNS,
        search_boundary_hit=boundary_hit,
        search_warning=search_warning,
    )
