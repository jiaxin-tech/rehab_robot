"""五参数数值灵敏度、信息量、奇异值和相关性分析。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    identification_lower_bounds,
    identification_parameter_scales,
    identification_upper_bounds,
)
from .parameter_estimator import (
    PARAMETER_NAMES,
    BaselineSubjectTemplate,
    _parameter_vector,
    measured_joint_torque,
    predict_joint_torque,
    valid_observations,
)


@dataclass(frozen=True)
class IdentifiabilityResult:
    analysis_set: str
    valid_samples: int
    parameter_count: int
    numerical_rank: int
    singular_values: list[float]
    condition_number: float
    parameter_correlation: list[list[float]]
    information_diagonal: dict[str, float]
    highly_correlated_pairs: list[dict[str, float | str]]
    torque_scales_nm: dict[str, float]

    def as_serializable_dict(self) -> dict[str, object]:
        return asdict(self)


def _common_torque_scales(
    dataframe: pd.DataFrame,
    L1: float,
    L2: float,
) -> tuple[float, float]:
    data = valid_observations(dataframe)
    hip, knee = measured_joint_torque(data, L1, L2)
    return max(float(np.std(hip)), 1.0), max(float(np.std(knee)), 1.0)


def numerical_sensitivity_matrix(
    dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
    L1: float,
    L2: float,
    *,
    torque_scales_nm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, tuple[float, float]]:
    """中央差分计算无量纲残差灵敏度矩阵。

    行分别是全部髋残差、全部膝残差；列按 PARAMETER_NAMES 排列。列乘固定
    参数尺度、行除固定关节力矩尺度，使奇异值比较不受单位选择支配。
    """

    data = valid_observations(dataframe)
    vector = _parameter_vector(parameters, "parameters")
    lower = _parameter_vector(identification_lower_bounds, "lower bounds")
    upper = _parameter_vector(identification_upper_bounds, "upper bounds")
    parameter_scales = _parameter_vector(
        identification_parameter_scales,
        "parameter scales",
    )
    if torque_scales_nm is None:
        torque_scales_nm = _common_torque_scales(data, L1, L2)
    hip_scale, knee_scale = torque_scales_nm

    base_hip, base_knee = predict_joint_torque(
        data,
        baseline_subject_template,
        vector,
        L1,
    )
    columns: list[np.ndarray] = []
    for index, scale in enumerate(parameter_scales):
        step = max(abs(vector[index]) * 1e-6, scale * 1e-6, 1e-8)
        can_lower = vector[index] - step >= lower[index]
        can_upper = vector[index] + step <= upper[index]
        if can_lower and can_upper:
            minus = vector.copy()
            plus = vector.copy()
            minus[index] -= step
            plus[index] += step
            minus_hip, minus_knee = predict_joint_torque(
                data,
                baseline_subject_template,
                minus,
                L1,
            )
            plus_hip, plus_knee = predict_joint_torque(
                data,
                baseline_subject_template,
                plus,
                L1,
            )
            derivative_hip = (plus_hip - minus_hip) / (2.0 * step)
            derivative_knee = (plus_knee - minus_knee) / (2.0 * step)
        elif can_upper:
            plus = vector.copy()
            plus[index] += step
            plus_hip, plus_knee = predict_joint_torque(
                data,
                baseline_subject_template,
                plus,
                L1,
            )
            derivative_hip = (plus_hip - base_hip) / step
            derivative_knee = (plus_knee - base_knee) / step
        elif can_lower:
            minus = vector.copy()
            minus[index] -= step
            minus_hip, minus_knee = predict_joint_torque(
                data,
                baseline_subject_template,
                minus,
                L1,
            )
            derivative_hip = (base_hip - minus_hip) / step
            derivative_knee = (base_knee - minus_knee) / step
        else:
            raise ValueError(f"cannot perturb parameter {PARAMETER_NAMES[index]}.")

        # residual = measured - predicted；负号不影响信息矩阵/奇异值。
        columns.append(
            np.concatenate(
                (
                    -derivative_hip / hip_scale,
                    -derivative_knee / knee_scale,
                )
            )
            * scale
        )
    return np.column_stack(columns), data, torque_scales_nm


def analyze_identifiability(
    dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
    L1: float,
    L2: float,
    *,
    analysis_set: str = "custom",
    torque_scales_nm: tuple[float, float] | None = None,
    high_correlation_threshold: float = 0.9,
) -> IdentifiabilityResult:
    sensitivity, data, scales = numerical_sensitivity_matrix(
        dataframe,
        baseline_subject_template,
        parameters,
        L1,
        L2,
        torque_scales_nm=torque_scales_nm,
    )
    singular_values = np.linalg.svd(sensitivity, compute_uv=False)
    tolerance = (
        np.finfo(float).eps
        * max(sensitivity.shape)
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > tolerance
        else float("inf")
    )
    information = sensitivity.T @ sensitivity
    covariance_shape = np.linalg.pinv(information)
    diagonal = np.sqrt(np.maximum(np.diag(covariance_shape), 0.0))
    denominator = np.outer(diagonal, diagonal)
    correlation = np.divide(
        covariance_shape,
        denominator,
        out=np.zeros_like(covariance_shape),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    highly_correlated: list[dict[str, float | str]] = []
    for first in range(len(PARAMETER_NAMES)):
        for second in range(first + 1, len(PARAMETER_NAMES)):
            value = float(correlation[first, second])
            if abs(value) >= high_correlation_threshold:
                highly_correlated.append(
                    {
                        "parameter_1": PARAMETER_NAMES[first],
                        "parameter_2": PARAMETER_NAMES[second],
                        "correlation": value,
                    }
                )

    return IdentifiabilityResult(
        analysis_set=analysis_set,
        valid_samples=len(data),
        parameter_count=len(PARAMETER_NAMES),
        numerical_rank=rank,
        singular_values=[float(value) for value in singular_values],
        condition_number=condition,
        parameter_correlation=correlation.tolist(),
        information_diagonal={
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, np.diag(information))
        },
        highly_correlated_pairs=highly_correlated,
        torque_scales_nm={"hip": scales[0], "knee": scales[1]},
    )


def compare_excitation_sets(
    complete_dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
    L1: float,
    L2: float,
) -> dict[str, IdentifiabilityResult]:
    """比较 A 单轨迹、B coupled 三速度、C 全部九条及几何筛选。"""

    valid_complete = valid_observations(complete_dataframe)
    common_scales = _common_torque_scales(valid_complete, L1, L2)
    subsets = {
        "A_coupled_nominal": complete_dataframe.loc[
            complete_dataframe["trajectory_family"].eq("coupled")
            & complete_dataframe["speed_profile"].eq("nominal")
        ],
        "B_coupled_all_speeds": complete_dataframe.loc[
            complete_dataframe["trajectory_family"].eq("coupled")
        ],
        "C_all_families_all_speeds": complete_dataframe,
    }
    if "jacobian_condition_number" in valid_complete:
        limit = float(
            valid_complete["jacobian_condition_number"].quantile(0.95)
        )
        subsets["C_without_extreme_geometry"] = complete_dataframe.loc[
            complete_dataframe["jacobian_condition_number"] <= limit
        ]

    return {
        name: analyze_identifiability(
            subset,
            baseline_subject_template,
            parameters,
            L1,
            L2,
            analysis_set=name,
            torque_scales_nm=common_scales,
        )
        for name, subset in subsets.items()
    }


def force_amplitude_sensitivity_analysis(
    dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
    L1: float,
    L2: float,
) -> pd.DataFrame:
    """按观测力四分位报告样本灵敏度，不按力幅值删除数据。"""

    sensitivity, data, _ = numerical_sensitivity_matrix(
        dataframe,
        baseline_subject_template,
        parameters,
        L1,
        L2,
    )
    sample_count = len(data)
    per_sample_norm = np.sqrt(
        np.sum(sensitivity[:sample_count] ** 2, axis=1)
        + np.sum(sensitivity[sample_count:] ** 2, axis=1)
    )
    force = data["force_magnitude_observed_n"].to_numpy(dtype=float)
    bins = pd.qcut(
        force,
        q=4,
        labels=("Q1_low", "Q2", "Q3", "Q4_high"),
        duplicates="drop",
    )
    analysis = pd.DataFrame(
        {
            "force_amplitude_bin": bins,
            "force_magnitude_observed_n": force,
            "scaled_sensitivity_norm": per_sample_norm,
        }
    )
    return (
        analysis.groupby("force_amplitude_bin", observed=True)
        .agg(
            sample_count=("scaled_sensitivity_norm", "size"),
            force_min_n=("force_magnitude_observed_n", "min"),
            force_max_n=("force_magnitude_observed_n", "max"),
            force_mean_n=("force_magnitude_observed_n", "mean"),
            mean_scaled_sensitivity_norm=("scaled_sensitivity_norm", "mean"),
        )
        .reset_index()
    )


def save_identifiability_outputs(
    results: Mapping[str, IdentifiabilityResult],
    output_dir: str | Path,
    force_amplitude_analysis: pd.DataFrame | None = None,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    singular_rows = []
    for name, result in results.items():
        maximum_correlation = 0.0
        correlation = np.asarray(result.parameter_correlation)
        if correlation.shape == (len(PARAMETER_NAMES), len(PARAMETER_NAMES)):
            upper = np.abs(correlation[np.triu_indices(len(PARAMETER_NAMES), 1)])
            maximum_correlation = float(np.max(upper))
        summary_rows.append(
            {
                "analysis_set": name,
                "valid_samples": result.valid_samples,
                "numerical_rank": result.numerical_rank,
                "condition_number": result.condition_number,
                "largest_singular_value": result.singular_values[0],
                "smallest_singular_value": result.singular_values[-1],
                "maximum_absolute_parameter_correlation": maximum_correlation,
                "highly_correlated_pair_count": len(
                    result.highly_correlated_pairs
                ),
            }
        )
        singular_rows.extend(
            {
                "analysis_set": name,
                "singular_value_index": index + 1,
                "singular_value": value,
            }
            for index, value in enumerate(result.singular_values)
        )

    summary_path = destination / "identifiability_summary.csv"
    singular_path = destination / "sensitivity_singular_values.csv"
    correlation_path = destination / "parameter_correlation_matrix.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(singular_rows).to_csv(singular_path, index=False)

    complete_key = (
        "C_all_families_all_speeds"
        if "C_all_families_all_speeds" in results
        else next(iter(results))
    )
    pd.DataFrame(
        results[complete_key].parameter_correlation,
        index=PARAMETER_NAMES,
        columns=PARAMETER_NAMES,
    ).rename_axis("parameter").to_csv(correlation_path)
    paths = {
        "summary": summary_path,
        "singular_values": singular_path,
        "correlation": correlation_path,
    }
    if force_amplitude_analysis is not None:
        force_path = destination / "force_amplitude_sensitivity.csv"
        force_amplitude_analysis.to_csv(force_path, index=False)
        paths["force_amplitude"] = force_path
    return paths
