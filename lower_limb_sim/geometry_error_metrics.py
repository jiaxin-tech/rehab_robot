"""Stage 4.5D geometry-observation metrics and domain audits.

All functions that need ground truth live in this evaluation-only module.  The
runtime domain classifier is intentionally separated and rejects columns whose
names contain ``true`` so it can be reused without leaking simulator state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import model_mismatch_nrmse_epsilon_nm
from .parameter_estimator import (
    PARAMETER_NAMES,
    BaselineSubjectTemplate,
    predict_joint_torque,
    valid_observations,
)


EVALUATION_GROUP_COLUMNS = (
    "subject_id",
    "scenario_name",
    "observation_mode",
    "dataset_split",
)

ESTIMATED_DOMAIN_STATE_COLUMNS = (
    "q_hip_est_rad",
    "q_knee_est_rad",
    "dq_hip_est_rad_s",
    "dq_knee_est_rad_s",
    "ddq_hip_est_rad_s2",
    "ddq_knee_est_rad_s2",
)

TRUE_DOMAIN_STATE_COLUMNS = (
    "q_hip_true_rad",
    "q_knee_true_rad",
    "dq_hip_true_rad_s",
    "dq_knee_true_rad_s",
    "ddq_hip_true_rad_s2",
    "ddq_knee_true_rad_s2",
)


@dataclass(frozen=True)
class StateDomainBounds:
    """Axis-aligned training coverage in a named six-dimensional state."""

    columns: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    valid_training_samples: int

    def as_serializable_dict(self) -> dict[str, object]:
        return asdict(self)


def _group_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in EVALUATION_GROUP_COLUMNS if column in dataframe]


def _iter_groups(dataframe: pd.DataFrame):
    columns = _group_columns(dataframe)
    if not columns:
        yield {}, dataframe
        return
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, group in dataframe.groupby(grouper, sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        yield dict(zip(columns, values)), group


def _finite_rmse(error: np.ndarray) -> float:
    finite = np.asarray(error, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan


def _finite_peak(error: np.ndarray) -> float:
    finite = np.abs(np.asarray(error, dtype=float))
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else np.nan


def _valid_state_mask(dataframe: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(dataframe), dtype=bool)
    for column in (
        "ik_valid",
        "joint_continuity_valid",
        "derivative_valid",
        "state_estimation_valid",
    ):
        if column in dataframe:
            mask &= dataframe[column].fillna(False).astype(bool).to_numpy()
    finite_columns = [
        *TRUE_DOMAIN_STATE_COLUMNS,
        *ESTIMATED_DOMAIN_STATE_COLUMNS,
    ]
    finite_columns = [column for column in finite_columns if column in dataframe]
    if finite_columns:
        mask &= np.isfinite(
            dataframe.loc[:, finite_columns].to_numpy(dtype=float)
        ).all(axis=1)
    return mask


def compute_kinematic_reconstruction_metrics(
    reconstructed_state: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all requested q/dq/ddq metrics per mode and fixed split."""

    required = set(TRUE_DOMAIN_STATE_COLUMNS) | set(ESTIMATED_DOMAIN_STATE_COLUMNS)
    missing = required.difference(reconstructed_state.columns)
    if missing:
        raise ValueError(f"reconstructed_state missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for identity, group in _iter_groups(reconstructed_state):
        valid = _valid_state_mask(group)
        selected = group.loc[valid]
        q_hip_error_deg = np.rad2deg(
            selected["q_hip_est_rad"].to_numpy(dtype=float)
            - selected["q_hip_true_rad"].to_numpy(dtype=float)
        )
        q_knee_error_deg = np.rad2deg(
            selected["q_knee_est_rad"].to_numpy(dtype=float)
            - selected["q_knee_true_rad"].to_numpy(dtype=float)
        )
        errors = {
            name: (
                selected[estimated].to_numpy(dtype=float)
                - selected[truth].to_numpy(dtype=float)
            )
            for name, estimated, truth in (
                (
                    "dq_hip",
                    "dq_hip_est_rad_s",
                    "dq_hip_true_rad_s",
                ),
                (
                    "dq_knee",
                    "dq_knee_est_rad_s",
                    "dq_knee_true_rad_s",
                ),
                (
                    "ddq_hip",
                    "ddq_hip_est_rad_s2",
                    "ddq_hip_true_rad_s2",
                ),
                (
                    "ddq_knee",
                    "ddq_knee_est_rad_s2",
                    "ddq_knee_true_rad_s2",
                ),
            )
        }
        ik_valid = (
            group["ik_valid"].fillna(False).astype(bool).to_numpy()
            if "ik_valid" in group
            else np.ones(len(group), dtype=bool)
        )
        continuity = (
            group["joint_continuity_valid"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
            if "joint_continuity_valid" in group
            else np.ones(len(group), dtype=bool)
        )
        rows.append(
            {
                **identity,
                "total_samples": int(len(group)),
                "valid_state_samples": int(valid.sum()),
                "q_hip_rmse_deg": _finite_rmse(q_hip_error_deg),
                "q_knee_rmse_deg": _finite_rmse(q_knee_error_deg),
                "q_hip_peak_error_deg": _finite_peak(q_hip_error_deg),
                "q_knee_peak_error_deg": _finite_peak(q_knee_error_deg),
                "dq_hip_rmse_rad_s": _finite_rmse(errors["dq_hip"]),
                "dq_knee_rmse_rad_s": _finite_rmse(errors["dq_knee"]),
                "ddq_hip_rmse_rad_s2": _finite_rmse(errors["ddq_hip"]),
                "ddq_knee_rmse_rad_s2": _finite_rmse(errors["ddq_knee"]),
                "invalid_ik_rate": float(1.0 - np.mean(ik_valid)),
                "joint_discontinuity_count": int(np.sum(~continuity & ik_valid)),
            }
        )
    return pd.DataFrame(rows)


def compute_observation_metrics(
    reconstructed_state: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Jacobian and reconstructed torque observation errors."""

    required = {
        "jacobian_frobenius_error",
        "jacobian_condition_error",
        "tau_measured_true_hip_nm",
        "tau_measured_true_knee_nm",
        "tau_measured_est_hip_nm",
        "tau_measured_est_knee_nm",
    }
    missing = required.difference(reconstructed_state.columns)
    if missing:
        raise ValueError(f"observation table missing columns: {sorted(missing)}")
    rows = []
    for identity, group in _iter_groups(reconstructed_state):
        valid = _valid_state_mask(group)
        selected = group.loc[valid]
        hip_error = (
            selected["tau_measured_est_hip_nm"].to_numpy(dtype=float)
            - selected["tau_measured_true_hip_nm"].to_numpy(dtype=float)
        )
        knee_error = (
            selected["tau_measured_est_knee_nm"].to_numpy(dtype=float)
            - selected["tau_measured_true_knee_nm"].to_numpy(dtype=float)
        )
        rows.append(
            {
                **identity,
                "valid_observation_samples": int(len(selected)),
                "jacobian_frobenius_error": _finite_rmse(
                    selected["jacobian_frobenius_error"].to_numpy(dtype=float)
                ),
                "jacobian_condition_error": _finite_rmse(
                    selected["jacobian_condition_error"].to_numpy(dtype=float)
                ),
                "tau_observation_hip_rmse_nm": _finite_rmse(hip_error),
                "tau_observation_knee_rmse_nm": _finite_rmse(knee_error),
                "tau_observation_combined_rmse_nm": _finite_rmse(
                    np.concatenate((hip_error, knee_error))
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_parameter_error_table(
    subject_id: str,
    scenario_name: str,
    estimates_by_mode: Mapping[str, Mapping[str, float]],
    true_parameters: Mapping[str, float],
) -> pd.DataFrame:
    """Report five-parameter errors; truth is used only at final evaluation."""

    rows = []
    error_column = {
        "mass_scale": "mass_scale_error_percent",
        "k_hip_nm_per_rad": "k_hip_error_percent",
        "k_knee_nm_per_rad": "k_knee_error_percent",
        "b_hip_nm_s_per_rad": "b_hip_error_percent",
        "b_knee_nm_s_per_rad": "b_knee_error_percent",
    }
    for mode, estimate in estimates_by_mode.items():
        summary: dict[str, object] = {
            "subject_id": subject_id,
            "scenario_name": scenario_name,
            "observation_mode": mode,
        }
        maximum_error = 0.0
        for parameter in PARAMETER_NAMES:
            truth = float(true_parameters[parameter])
            value = float(estimate[parameter])
            relative = (
                100.0 * abs(value - truth) / abs(truth)
                if truth != 0.0
                else np.nan
            )
            summary[f"{parameter}_true"] = truth
            summary[f"{parameter}_estimated"] = value
            summary[error_column[parameter]] = relative
            if np.isfinite(relative):
                maximum_error = max(maximum_error, relative)
        summary["maximum_parameter_error_percent"] = maximum_error
        rows.append(summary)
    return pd.DataFrame(rows)


def fit_state_domain_bounds(
    training_dataframe: pd.DataFrame,
    columns: Sequence[str] = ESTIMATED_DOMAIN_STATE_COLUMNS,
) -> StateDomainBounds:
    """Fit runtime coverage using estimated state only.

    Column names containing ``true`` are rejected to make accidental simulator
    leakage visible during testing and code review.
    """

    names = tuple(str(column) for column in columns)
    if len(names) != 6 or any("true" in column.lower() for column in names):
        raise ValueError("runtime domain bounds require six non-true state columns.")
    missing = set(names).difference(training_dataframe.columns)
    if missing:
        raise ValueError(f"training dataframe missing domain columns: {sorted(missing)}")
    values = training_dataframe.loc[:, names].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    if "state_estimation_valid" in training_dataframe:
        finite &= training_dataframe["state_estimation_valid"].astype(bool).to_numpy()
    if not finite.any():
        raise ValueError("no valid estimated training states for domain bounds.")
    selected = values[finite]
    return StateDomainBounds(
        columns=names,
        lower=tuple(np.min(selected, axis=0).astype(float)),
        upper=tuple(np.max(selected, axis=0).astype(float)),
        valid_training_samples=int(finite.sum()),
    )


def classify_state_domain(
    dataframe: pd.DataFrame,
    bounds: StateDomainBounds,
) -> np.ndarray:
    """Classify domain membership without accessing true state columns."""

    if any("true" in column.lower() for column in bounds.columns):
        raise ValueError("runtime domain classifier cannot use true state columns.")
    values = dataframe.loc[:, bounds.columns].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    lower = np.asarray(bounds.lower, dtype=float)
    upper = np.asarray(bounds.upper, dtype=float)
    inside = finite & np.all((values >= lower) & (values <= upper), axis=1)
    if "state_estimation_valid" in dataframe:
        inside &= dataframe["state_estimation_valid"].astype(bool).to_numpy()
    return inside


def _truth_domain_membership(dataframe: pd.DataFrame) -> np.ndarray:
    """Evaluation-only true membership; never called by runtime classifier."""

    training = dataframe.loc[dataframe["dataset_split"].eq("train")]
    values = training.loc[:, TRUE_DOMAIN_STATE_COLUMNS].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    lower = np.min(values[finite], axis=0)
    upper = np.max(values[finite], axis=0)
    all_values = dataframe.loc[:, TRUE_DOMAIN_STATE_COLUMNS].to_numpy(dtype=float)
    return np.isfinite(all_values).all(axis=1) & np.all(
        (all_values >= lower) & (all_values <= upper),
        axis=1,
    )


def attach_and_evaluate_domain_membership(
    reconstructed_state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, StateDomainBounds]]:
    """Attach estimated/true membership and report false accept/reject rates."""

    frames = []
    metric_rows = []
    bounds_by_mode: dict[str, StateDomainBounds] = {}
    identity_columns = [
        column
        for column in ("subject_id", "scenario_name", "observation_mode")
        if column in reconstructed_state
    ]
    grouper: str | list[str] = (
        identity_columns[0] if len(identity_columns) == 1 else identity_columns
    )
    groups = (
        reconstructed_state.groupby(grouper, sort=False, dropna=False)
        if identity_columns
        else [((), reconstructed_state)]
    )
    for key, raw_group in groups:
        group = raw_group.copy()
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(identity_columns, key_values))
        training = group.loc[group["dataset_split"].eq("train")]
        bounds = fit_state_domain_bounds(training)
        mode_key = "/".join(str(identity.get(name, "")) for name in identity_columns)
        bounds_by_mode[mode_key] = bounds
        group["domain_membership_estimated"] = classify_state_domain(group, bounds)
        group["domain_membership_true"] = _truth_domain_membership(group)
        for split, split_group in group.groupby("dataset_split", sort=False):
            estimated = split_group["domain_membership_estimated"].astype(bool).to_numpy()
            truth = split_group["domain_membership_true"].astype(bool).to_numpy()
            false_accept = estimated & ~truth
            false_reject = ~estimated & truth
            true_outside = max(int(np.sum(~truth)), 1)
            true_inside = max(int(np.sum(truth)), 1)
            metric_rows.append(
                {
                    **identity,
                    "dataset_split": split,
                    "samples": int(len(split_group)),
                    "domain_false_accept_count": int(false_accept.sum()),
                    "domain_false_reject_count": int(false_reject.sum()),
                    "domain_false_accept_rate": float(
                        false_accept.sum() / true_outside
                    ),
                    "domain_false_reject_rate": float(
                        false_reject.sum() / true_inside
                    ),
                }
            )
        frames.append(group)
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(metric_rows),
        bounds_by_mode,
    )


def q0_stiffness_correlation_analysis(
    training_dataframe: pd.DataFrame,
    baseline_template: BaselineSubjectTemplate,
    estimated_parameters: Mapping[str, float],
    L1_assumed_m: float,
    *,
    parameter_step_fraction: float = 1e-4,
    q0_step_rad: float = 1e-4,
) -> dict[str, float | int | bool]:
    """Numerically audit K-q0 correlation without changing the main estimator."""

    data = valid_observations(training_dataframe)
    base = dict(estimated_parameters)
    columns: list[np.ndarray] = []
    specifications = (
        ("k_hip_nm_per_rad", "parameter"),
        ("k_knee_nm_per_rad", "parameter"),
        ("q0_hip_rad", "template"),
        ("q0_knee_rad", "template"),
    )
    for name, kind in specifications:
        if kind == "parameter":
            step = max(abs(float(base[name])) * parameter_step_fraction, 1e-5)
            plus, minus = dict(base), dict(base)
            plus[name] += step
            minus[name] -= step
            plus_tau = predict_joint_torque(data, baseline_template, plus, L1_assumed_m)
            minus_tau = predict_joint_torque(data, baseline_template, minus, L1_assumed_m)
        else:
            step = q0_step_rad
            plus_template = replace(
                baseline_template,
                **{name: float(getattr(baseline_template, name)) + step},
            )
            minus_template = replace(
                baseline_template,
                **{name: float(getattr(baseline_template, name)) - step},
            )
            plus_tau = predict_joint_torque(data, plus_template, base, L1_assumed_m)
            minus_tau = predict_joint_torque(data, minus_template, base, L1_assumed_m)
        derivative = np.concatenate(
            (
                (np.asarray(plus_tau[0]) - np.asarray(minus_tau[0])) / (2 * step),
                (np.asarray(plus_tau[1]) - np.asarray(minus_tau[1])) / (2 * step),
            )
        )
        columns.append(derivative)
    sensitivity = np.column_stack(columns)
    information_inverse = np.linalg.pinv(sensitivity.T @ sensitivity)
    standard = np.sqrt(np.maximum(np.diag(information_inverse), 0.0))
    denominator = np.outer(standard, standard)
    correlation = np.divide(
        information_inverse,
        denominator,
        out=np.full_like(information_inverse, np.nan),
        where=denominator > np.finfo(float).eps,
    )
    singular_values = np.linalg.svd(sensitivity, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > np.finfo(float).eps
        else np.inf
    )
    return {
        "k_hip_q0_hip_correlation": float(correlation[0, 2]),
        "k_knee_q0_knee_correlation": float(correlation[1, 3]),
        "maximum_absolute_k_q0_correlation": float(
            np.nanmax(np.abs((correlation[0, 2], correlation[1, 3])))
        ),
        "sensitivity_condition_number": condition,
        "sensitivity_rank": int(np.linalg.matrix_rank(sensitivity)),
        "main_estimator_parameter_count": len(PARAMETER_NAMES),
        "q0_included_in_main_estimator": False,
    }


SENSITIVITY_METRICS = (
    "joint_angle_rmse_deg",
    "tau_observation_combined_rmse_nm",
    "interpolation_combined_nrmse_percent",
    "maximum_parameter_error_percent",
    "invalid_ik_rate",
)


def build_geometry_sensitivity_ranking(
    experiment_summary: pd.DataFrame,
    *,
    metrics: Iterable[str] = SENSITIVITY_METRICS,
) -> pd.DataFrame:
    """Aggregate signs/seeds and rank each geometry-error source."""

    required = {"error_source", "random_seed", *metrics}
    missing = required.difference(experiment_summary.columns)
    if missing:
        raise ValueError(f"experiment summary missing: {sorted(missing)}")
    rows = []
    group_columns = ["error_source"]
    if "observation_mode" in experiment_summary.columns:
        group_columns.append("observation_mode")
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    for key, group in experiment_summary.groupby(grouper, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size:
                row = {
                    **identity,
                    "ranking_metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95.0)),
                    "worst_case": float(np.max(values)),
                    "run_count": int(values.size),
                }
            else:
                row = {
                    **identity,
                    "ranking_metric": metric,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "p95": np.nan,
                    "worst_case": np.nan,
                    "run_count": 0,
                }
            rows.append(row)
    result = pd.DataFrame(rows)
    result["rank"] = result.groupby("ranking_metric")["mean"].rank(
        method="dense",
        ascending=False,
        na_option="bottom",
    )
    return result.sort_values(
        ["ranking_metric", "rank", *group_columns]
    ).reset_index(drop=True)


def combined_nrmse_percent(
    true_hip: np.ndarray,
    true_knee: np.ndarray,
    predicted_hip: np.ndarray,
    predicted_knee: np.ndarray,
) -> float:
    """Small helper using the stage-4.5C fixed NRMSE convention."""

    true = np.concatenate((true_hip, true_knee)).astype(float)
    residual = true - np.concatenate((predicted_hip, predicted_knee)).astype(float)
    rmse = float(np.sqrt(np.mean(residual**2)))
    return float(
        100.0
        * rmse
        / (float(np.max(true) - np.min(true)) + model_mismatch_nrmse_epsilon_nm)
    )


__all__ = [
    "ESTIMATED_DOMAIN_STATE_COLUMNS",
    "SENSITIVITY_METRICS",
    "StateDomainBounds",
    "attach_and_evaluate_domain_membership",
    "build_geometry_sensitivity_ranking",
    "classify_state_domain",
    "combined_nrmse_percent",
    "compute_kinematic_reconstruction_metrics",
    "compute_observation_metrics",
    "compute_parameter_error_table",
    "fit_state_domain_bounds",
    "q0_stiffness_correlation_analysis",
]
