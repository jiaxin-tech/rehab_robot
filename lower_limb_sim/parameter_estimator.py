"""泄漏隔离的五参数虚拟受试者动力学辨识器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import (
    identification_initial_guess,
    identification_loss,
    identification_lower_bounds,
    identification_parameter_names,
    identification_parameter_scales,
    identification_upper_bounds,
)
from .dynamic_subject import DynamicVirtualSubject
from .full_dynamics import inverse_dynamics
from .observation_model import joint_torque_from_endpoint_force

PARAMETER_NAMES = identification_parameter_names
REQUIRED_OBSERVATION_COLUMNS = (
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
    "fx_observed_n",
    "fz_observed_n",
    "sample_valid",
)


@dataclass(frozen=True)
class BaselineSubjectTemplate:
    """辨识器可读取的已知人体比例模板。

    特意不包含真实刚度、真实阻尼、真实 ``mass_scale`` 或 subject lookup。
    """

    mass_thigh_kg: float
    mass_shank_kg: float
    com_thigh_m: float
    com_shank_m: float
    inertia_thigh_kg_m2: float
    inertia_shank_kg_m2: float
    q0_hip_rad: float
    q0_knee_rad: float
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        positive = (
            self.mass_thigh_kg,
            self.mass_shank_kg,
            self.com_thigh_m,
            self.com_shank_m,
            self.inertia_thigh_kg_m2,
            self.inertia_shank_kg_m2,
            self.gravity_m_s2,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("baseline anthropometric values must be positive.")
        if not np.isfinite(self.q0_hip_rad) or not np.isfinite(self.q0_knee_rad):
            raise ValueError("baseline neutral angles must be finite.")


@dataclass(frozen=True)
class ParameterEstimationResult:
    """有界最小二乘的结果、残差统计和局部不确定度。"""

    estimated_parameters: dict[str, float]
    optimizer_success: bool
    optimizer_message: str
    cost: float
    number_of_function_evaluations: int
    residual_statistics: dict[str, float]
    jacobian_singular_values: list[float]
    parameter_covariance: list[list[float]]
    parameter_standard_errors: dict[str, float]
    torque_scales_nm: dict[str, float]
    valid_training_samples: int
    loss: str

    def as_serializable_dict(self) -> dict[str, object]:
        return asdict(self)


def baseline_template_from_dynamic_subject(
    subject: DynamicVirtualSubject,
) -> BaselineSubjectTemplate:
    """显式投影为不含刚度、阻尼和 ID 的 baseline 模板。"""

    return BaselineSubjectTemplate(
        mass_thigh_kg=subject.mass_thigh_kg,
        mass_shank_kg=subject.mass_shank_kg,
        com_thigh_m=subject.com_thigh_m,
        com_shank_m=subject.com_shank_m,
        inertia_thigh_kg_m2=subject.inertia_thigh_kg_m2,
        inertia_shank_kg_m2=subject.inertia_shank_kg_m2,
        q0_hip_rad=subject.q0_hip_rad,
        q0_knee_rad=subject.q0_knee_rad,
        gravity_m_s2=subject.gravity_m_s2,
    )


def _parameter_vector(
    values: Mapping[str, float] | Sequence[float],
    name: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        try:
            result = np.array([values[key] for key in PARAMETER_NAMES], dtype=float)
        except KeyError as exc:
            raise ValueError(f"{name} is missing {exc.args[0]!r}.") from exc
    else:
        result = np.asarray(values, dtype=float)
    if result.shape != (len(PARAMETER_NAMES),) or not np.isfinite(result).all():
        raise ValueError(
            f"{name} must contain {len(PARAMETER_NAMES)} finite values."
        )
    return result


def candidate_subject_from_parameters(
    baseline_subject_template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
) -> DynamicVirtualSubject:
    """纯函数式建立候选模型，不修改 baseline 或虚拟真值对象。"""

    vector = _parameter_vector(parameters, "parameters")
    mass_scale, k_hip, k_knee, b_hip, b_knee = vector
    return DynamicVirtualSubject(
        subject_id="candidate_identification_model",
        mass_thigh_kg=mass_scale * baseline_subject_template.mass_thigh_kg,
        mass_shank_kg=mass_scale * baseline_subject_template.mass_shank_kg,
        com_thigh_m=baseline_subject_template.com_thigh_m,
        com_shank_m=baseline_subject_template.com_shank_m,
        inertia_thigh_kg_m2=(
            mass_scale * baseline_subject_template.inertia_thigh_kg_m2
        ),
        inertia_shank_kg_m2=(
            mass_scale * baseline_subject_template.inertia_shank_kg_m2
        ),
        b_hip_nm_s_per_rad=b_hip,
        b_knee_nm_s_per_rad=b_knee,
        k_hip_nm_per_rad=k_hip,
        k_knee_nm_per_rad=k_knee,
        q0_hip_rad=baseline_subject_template.q0_hip_rad,
        q0_knee_rad=baseline_subject_template.q0_knee_rad,
        gravity_m_s2=baseline_subject_template.gravity_m_s2,
    )


def _reject_leakage_columns(dataframe: pd.DataFrame) -> None:
    forbidden = [
        column
        for column in dataframe.columns
        if column.startswith("true_")
        or column.startswith("ground_truth")
        or column.startswith("tau_total")
    ]
    if forbidden:
        raise ValueError(
            "identification dataframe contains forbidden ground-truth columns: "
            f"{sorted(forbidden)}"
        )


def valid_observations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """返回仅含有效、有限观测的副本。"""

    _reject_leakage_columns(dataframe)
    missing = set(REQUIRED_OBSERVATION_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"identification dataframe is missing columns: {sorted(missing)}"
        )
    # ``copy=True`` keeps this mask writable even when pandas Copy-on-Write is
    # enabled by the surrounding test/application process.
    valid = dataframe["sample_valid"].astype(bool).to_numpy(copy=True)
    if "force_mapping_valid" in dataframe:
        valid &= dataframe["force_mapping_valid"].astype(bool).to_numpy()
    if "wrench_is_stale" in dataframe:
        valid &= ~dataframe["wrench_is_stale"].astype(bool).to_numpy()
    finite_columns = REQUIRED_OBSERVATION_COLUMNS[:-1]
    finite = np.isfinite(
        dataframe.loc[:, finite_columns].to_numpy(dtype=float)
    ).all(axis=1)
    selected = dataframe.loc[valid & finite].copy()
    if selected.empty:
        raise ValueError("identification dataframe has no valid finite samples.")
    return selected


def measured_joint_torque(
    dataframe: pd.DataFrame,
    L1: float,
    L2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """只从当前 q 和观测力重建力矩，不读取保存的 tau 字段。"""

    return tuple(
        np.asarray(value, dtype=float)
        for value in joint_torque_from_endpoint_force(
            dataframe["q_hip_rad"].to_numpy(dtype=float),
            dataframe["q_knee_rad"].to_numpy(dtype=float),
            dataframe["fx_observed_n"].to_numpy(dtype=float),
            dataframe["fz_observed_n"].to_numpy(dtype=float),
            L1,
            L2,
        )
    )


def predict_joint_torque(
    dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
    L1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """用候选五参数重新运行现有完整动力学。"""

    candidate = candidate_subject_from_parameters(
        baseline_subject_template,
        parameters,
    )
    dynamics = inverse_dynamics(
        dataframe["q_hip_rad"].to_numpy(dtype=float),
        dataframe["q_knee_rad"].to_numpy(dtype=float),
        dataframe["dq_hip_rad_s"].to_numpy(dtype=float),
        dataframe["dq_knee_rad_s"].to_numpy(dtype=float),
        dataframe["ddq_hip_rad_s2"].to_numpy(dtype=float),
        dataframe["ddq_knee_rad_s2"].to_numpy(dtype=float),
        candidate,
        L1,
    )
    return (
        np.asarray(dynamics.tau_total_hip_nm, dtype=float),
        np.asarray(dynamics.tau_total_knee_nm, dtype=float),
    )


def _torque_scales(
    measured_hip: np.ndarray,
    measured_knee: np.ndarray,
) -> tuple[float, float]:
    # 1 N·m 下限避免极小信号导致不合理放大；分别缩放防止某一关节主导。
    return (
        max(float(np.std(measured_hip)), 1.0),
        max(float(np.std(measured_knee)), 1.0),
    )


def estimate_subject_parameters(
    training_dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    L1: float,
    L2: float,
    initial_guess: Mapping[str, float] | Sequence[float] = (
        identification_initial_guess
    ),
    bounds: tuple[
        Mapping[str, float] | Sequence[float],
        Mapping[str, float] | Sequence[float],
    ] = (identification_lower_bounds, identification_upper_bounds),
    *,
    loss: str = identification_loss,
    fixed_torque_scales_nm: tuple[float, float] | None = None,
) -> ParameterEstimationResult:
    """以有界、量纲缩放的 ``scipy.optimize.least_squares`` 辨识五参数。"""

    data = valid_observations(training_dataframe)
    measured_hip, measured_knee = measured_joint_torque(data, L1, L2)
    if fixed_torque_scales_nm is None:
        hip_scale, knee_scale = _torque_scales(measured_hip, measured_knee)
    else:
        scales = np.asarray(fixed_torque_scales_nm, dtype=float)
        if (
            scales.shape != (2,)
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
        ):
            raise ValueError(
                "fixed_torque_scales_nm must contain two finite positive values."
            )
        hip_scale, knee_scale = float(scales[0]), float(scales[1])
    initial = _parameter_vector(initial_guess, "initial_guess")
    lower = _parameter_vector(bounds[0], "lower bounds")
    upper = _parameter_vector(bounds[1], "upper bounds")
    if np.any(lower >= upper):
        raise ValueError("every lower parameter bound must be below its upper bound.")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial_guess must lie within parameter bounds.")
    x_scale = _parameter_vector(
        identification_parameter_scales,
        "identification_parameter_scales",
    )

    def normalized_residual(vector: np.ndarray) -> np.ndarray:
        predicted_hip, predicted_knee = predict_joint_torque(
            data,
            baseline_subject_template,
            vector,
            L1,
        )
        return np.concatenate(
            (
                (measured_hip - predicted_hip) / hip_scale,
                (measured_knee - predicted_knee) / knee_scale,
            )
        )

    optimized = least_squares(
        normalized_residual,
        initial,
        bounds=(lower, upper),
        x_scale=x_scale,
        loss=loss,
        f_scale=1.0,
        max_nfev=500,
    )
    estimated = {
        key: float(value) for key, value in zip(PARAMETER_NAMES, optimized.x)
    }
    predicted_hip, predicted_knee = predict_joint_torque(
        data,
        baseline_subject_template,
        optimized.x,
        L1,
    )
    residual_hip = measured_hip - predicted_hip
    residual_knee = measured_knee - predicted_knee
    combined = np.concatenate((residual_hip, residual_knee))
    residual_statistics = {
        "torque_rmse_hip_nm": float(np.sqrt(np.mean(residual_hip**2))),
        "torque_rmse_knee_nm": float(np.sqrt(np.mean(residual_knee**2))),
        "torque_rmse_combined_nm": float(np.sqrt(np.mean(combined**2))),
        "torque_mae_hip_nm": float(np.mean(np.abs(residual_hip))),
        "torque_mae_knee_nm": float(np.mean(np.abs(residual_knee))),
        "residual_mean_hip_nm": float(np.mean(residual_hip)),
        "residual_mean_knee_nm": float(np.mean(residual_knee)),
    }

    jacobian = np.asarray(optimized.jac, dtype=float)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    degrees_of_freedom = max(jacobian.shape[0] - jacobian.shape[1], 1)
    raw_normalized_residual = normalized_residual(optimized.x)
    variance = float(raw_normalized_residual @ raw_normalized_residual)
    variance /= degrees_of_freedom
    covariance = variance * np.linalg.pinv(jacobian.T @ jacobian)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return ParameterEstimationResult(
        estimated_parameters=estimated,
        optimizer_success=bool(optimized.success),
        optimizer_message=str(optimized.message),
        cost=float(optimized.cost),
        number_of_function_evaluations=int(optimized.nfev),
        residual_statistics=residual_statistics,
        jacobian_singular_values=[float(value) for value in singular_values],
        parameter_covariance=covariance.tolist(),
        parameter_standard_errors={
            key: float(value)
            for key, value in zip(PARAMETER_NAMES, standard_errors)
        },
        torque_scales_nm={"hip": hip_scale, "knee": knee_scale},
        valid_training_samples=len(data),
        loss=loss,
    )


def _r2(measured: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.sum((measured - np.mean(measured)) ** 2))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(1.0 - np.sum(residual**2) / denominator)


def _vaf(measured: np.ndarray, residual: np.ndarray) -> float:
    variance = float(np.var(measured))
    if variance <= np.finfo(float).eps:
        return float("nan")
    return float(100.0 * (1.0 - np.var(residual) / variance))


def compute_torque_metrics(
    dataframe: pd.DataFrame,
    baseline_subject_template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
    L1: float,
    L2: float,
) -> dict[str, float | int]:
    """计算一个 split 上的 RMSE、MAE、R² 和 VAF。"""

    data = valid_observations(dataframe)
    measured_hip, measured_knee = measured_joint_torque(data, L1, L2)
    predicted_hip, predicted_knee = predict_joint_torque(
        data,
        baseline_subject_template,
        parameters,
        L1,
    )
    residual_hip = measured_hip - predicted_hip
    residual_knee = measured_knee - predicted_knee
    combined = np.concatenate((residual_hip, residual_knee))
    return {
        "valid_samples": len(data),
        "torque_rmse_hip_nm": float(np.sqrt(np.mean(residual_hip**2))),
        "torque_rmse_knee_nm": float(np.sqrt(np.mean(residual_knee**2))),
        "torque_rmse_combined_nm": float(np.sqrt(np.mean(combined**2))),
        "torque_mae_hip_nm": float(np.mean(np.abs(residual_hip))),
        "torque_mae_knee_nm": float(np.mean(np.abs(residual_knee))),
        "R2_hip": _r2(measured_hip, residual_hip),
        "R2_knee": _r2(measured_knee, residual_knee),
        "VAF_hip_percent": _vaf(measured_hip, residual_hip),
        "VAF_knee_percent": _vaf(measured_knee, residual_knee),
    }
