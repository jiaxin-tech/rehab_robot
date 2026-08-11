"""Metrics and residual diagnostics for the stage 4.5C model-mismatch study.

This module consumes only motion states and already-computed true/predicted
torques (plus optional endpoint-force predictions).  It deliberately has no
dependency on the complex subject or mismatch-scenario definitions, so model
generation parameters cannot enter the evaluation path accidentally.

The fixed normalised error convention is::

    NRMSE = RMSE / (max(true torque) - min(true torque) + epsilon) * 100

When the true range is smaller than ``minimum_nrmse_range_nm``, NRMSE is
reported as ``NaN`` and ``nrmse_unreliable_small_range`` is set instead of
emitting a misleadingly large percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    model_mismatch_diagnostic_correlation_threshold,
    model_mismatch_nrmse_epsilon_nm,
    model_mismatch_nrmse_minimum_range_nm,
)
from .force_mapping import endpoint_force_from_joint_torque


PREDICTION_MODELS = ("generic", "identified")
RESIDUAL_FEATURE_COLUMNS = (
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
)

_TRUE_TORQUE_ALIASES = {
    "hip": (
        "tau_true_hip_nm",
        "tau_complex_true_hip_nm",
        "tau_measured_hip_nm",
    ),
    "knee": (
        "tau_true_knee_nm",
        "tau_complex_true_knee_nm",
        "tau_measured_knee_nm",
    ),
}
_PREDICTED_TORQUE_ALIASES = {
    "generic": {
        "hip": (
            "tau_generic_hip_nm",
            "tau_generic_baseline_hip_nm",
            "tau_predicted_generic_hip_nm",
            "tau_generic_baseline_model_hip_nm",
        ),
        "knee": (
            "tau_generic_knee_nm",
            "tau_generic_baseline_knee_nm",
            "tau_predicted_generic_knee_nm",
            "tau_generic_baseline_model_knee_nm",
        ),
    },
    "identified": {
        "hip": (
            "tau_identified_hip_nm",
            "tau_identified_equivalent_hip_nm",
            "tau_predicted_identified_hip_nm",
            "tau_identified_equivalent_model_hip_nm",
            "tau_predicted_hip_nm",
        ),
        "knee": (
            "tau_identified_knee_nm",
            "tau_identified_equivalent_knee_nm",
            "tau_predicted_identified_knee_nm",
            "tau_identified_equivalent_model_knee_nm",
            "tau_predicted_knee_nm",
        ),
    },
}
_TRUE_FORCE_ALIASES = {
    "x": ("fx_true_n", "fx_complex_true_n", "fx_observed_n"),
    "z": ("fz_true_n", "fz_complex_true_n", "fz_observed_n"),
}
_PREDICTED_FORCE_ALIASES = {
    "generic": {
        "x": (
            "fx_generic_n",
            "fx_generic_baseline_n",
            "fx_predicted_generic_n",
        ),
        "z": (
            "fz_generic_n",
            "fz_generic_baseline_n",
            "fz_predicted_generic_n",
        ),
    },
    "identified": {
        "x": (
            "fx_identified_n",
            "fx_identified_equivalent_n",
            "fx_predicted_identified_n",
            "fx_predicted_n",
        ),
        "z": (
            "fz_identified_n",
            "fz_identified_equivalent_n",
            "fz_predicted_identified_n",
            "fz_predicted_n",
        ),
    },
}


@dataclass(frozen=True)
class MismatchMetricBundle:
    """All tabular outputs needed by one model-mismatch experiment."""

    trajectory_metrics: pd.DataFrame
    generic_vs_identified: pd.DataFrame
    residual_feature_correlations: pd.DataFrame
    residual_diagnostics: pd.DataFrame


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    *,
    description: str,
    required: bool = True,
) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    if required:
        raise ValueError(
            f"prediction dataframe is missing {description}; expected one of "
            f"{list(candidates)}"
        )
    return None


def _validate_prediction_model(prediction_model: str) -> None:
    if prediction_model not in PREDICTION_MODELS:
        raise ValueError(
            f"prediction_model must be one of {PREDICTION_MODELS}, got "
            f"{prediction_model!r}."
        )


def _torque_columns(
    dataframe: pd.DataFrame,
    prediction_model: str,
) -> dict[str, str]:
    _validate_prediction_model(prediction_model)
    return {
        "true_hip": _first_existing_column(
            dataframe,
            _TRUE_TORQUE_ALIASES["hip"],
            description="true hip torque",
        ),
        "true_knee": _first_existing_column(
            dataframe,
            _TRUE_TORQUE_ALIASES["knee"],
            description="true knee torque",
        ),
        "predicted_hip": _first_existing_column(
            dataframe,
            _PREDICTED_TORQUE_ALIASES[prediction_model]["hip"],
            description=f"{prediction_model} hip torque",
        ),
        "predicted_knee": _first_existing_column(
            dataframe,
            _PREDICTED_TORQUE_ALIASES[prediction_model]["knee"],
            description=f"{prediction_model} knee torque",
        ),
    }


def resolve_mismatch_prediction_columns(
    dataframe: pd.DataFrame,
    prediction_model: str,
) -> dict[str, str]:
    """Return the concrete torque columns selected for one predictor.

    The canonical names are ``tau_true_*``, ``tau_generic_*`` and
    ``tau_identified_*``.  The aliases keep the metric layer compatible with
    older stage-4 prediction tables without changing any dynamics formula.
    """

    return _torque_columns(dataframe, prediction_model)


def _trajectory_group_columns(dataframe: pd.DataFrame) -> list[str]:
    # ``trajectory_id`` may intentionally be a common software-validation tag,
    # so family/name/speed remain part of the identity when available.
    candidates = (
        "subject_id",
        "scenario_name",
        "mismatch_scenario",
        "dataset_split",
        "split",
        "trajectory_id",
        "trajectory_family",
        "trajectory_name",
        "speed_profile",
    )
    columns: list[str] = []
    for candidate in candidates:
        if candidate in dataframe.columns and candidate not in columns:
            columns.append(candidate)
    return columns


def _diagnostic_group_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "subject_id",
            "scenario_name",
            "mismatch_scenario",
        )
        if column in dataframe.columns
    ]


def _iter_groups(
    dataframe: pd.DataFrame,
    group_columns: Sequence[str],
) -> Iterable[tuple[dict[str, object], pd.DataFrame]]:
    if not group_columns:
        yield {}, dataframe
        return
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in dataframe.groupby(grouper, sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        yield dict(zip(group_columns, key_tuple)), group


def _boolean_column(
    dataframe: pd.DataFrame,
    column: str,
    default: bool = True,
) -> np.ndarray:
    if column not in dataframe.columns:
        return np.full(len(dataframe), default, dtype=bool)
    values = dataframe[column]
    return values.notna().to_numpy() & values.fillna(False).astype(bool).to_numpy()


def _valid_prediction_mask(
    dataframe: pd.DataFrame,
    prediction_model: str,
    torque_columns: dict[str, str],
) -> np.ndarray:
    valid = _boolean_column(dataframe, "sample_valid")
    valid &= _boolean_column(
        dataframe,
        f"{prediction_model}_prediction_valid",
    )
    finite_columns = list(torque_columns.values())
    if "q_hip_rad" in dataframe and "q_knee_rad" in dataframe:
        finite_columns.extend(("q_hip_rad", "q_knee_rad"))
    finite = np.isfinite(
        dataframe.loc[:, finite_columns].to_numpy(dtype=float)
    ).all(axis=1)
    return valid & finite


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 2:
        return np.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def _safe_vaf_percent(true: np.ndarray, predicted: np.ndarray) -> float:
    finite = np.isfinite(true) & np.isfinite(predicted)
    true = np.asarray(true[finite], dtype=float)
    predicted = np.asarray(predicted[finite], dtype=float)
    if true.size < 2:
        return np.nan
    true_variance = float(np.var(true))
    if true_variance <= np.finfo(float).eps:
        return np.nan
    residual_variance = float(np.var(true - predicted))
    return float(100.0 * (1.0 - residual_variance / true_variance))


def _rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual)))) if residual.size else np.nan


def _mae(residual: np.ndarray) -> float:
    return float(np.mean(np.abs(residual))) if residual.size else np.nan


def _nrmse_percent(
    rmse_nm: float,
    true: np.ndarray,
    epsilon_nm: float,
    minimum_range_nm: float,
) -> tuple[float, float, bool]:
    if true.size == 0 or not np.isfinite(rmse_nm):
        return np.nan, np.nan, True
    true_range = float(np.max(true) - np.min(true))
    unreliable = bool(
        not np.isfinite(true_range) or true_range < minimum_range_nm
    )
    if unreliable:
        return np.nan, true_range, True
    return float(100.0 * rmse_nm / (true_range + epsilon_nm)), true_range, False


def _peak_error_percent(
    true: np.ndarray,
    predicted: np.ndarray,
    epsilon: float,
) -> tuple[float, bool]:
    if true.size == 0:
        return np.nan, True
    true_peak = float(np.max(np.abs(true)))
    predicted_peak = float(np.max(np.abs(predicted)))
    unreliable = bool(true_peak <= epsilon or not np.isfinite(true_peak))
    if unreliable:
        return np.nan, True
    return float(100.0 * abs(predicted_peak - true_peak) / (true_peak + epsilon)), False


def _strongest_signed_correlation(
    residuals: Sequence[np.ndarray],
    features: Sequence[np.ndarray],
) -> float:
    correlations = [
        _safe_pearson(residual, feature)
        for residual in residuals
        for feature in features
    ]
    finite = [value for value in correlations if np.isfinite(value)]
    if not finite:
        return np.nan
    return float(max(finite, key=abs))


def _provided_or_mapped_force(
    dataframe: pd.DataFrame,
    prediction_model: str,
    torque_columns: dict[str, str],
    link_1_m: float,
    link_2_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_fx_column = _first_existing_column(
        dataframe,
        _TRUE_FORCE_ALIASES["x"],
        description="true endpoint Fx",
        required=False,
    )
    true_fz_column = _first_existing_column(
        dataframe,
        _TRUE_FORCE_ALIASES["z"],
        description="true endpoint Fz",
        required=False,
    )
    predicted_fx_column = _first_existing_column(
        dataframe,
        _PREDICTED_FORCE_ALIASES[prediction_model]["x"],
        description=f"{prediction_model} endpoint Fx",
        required=False,
    )
    predicted_fz_column = _first_existing_column(
        dataframe,
        _PREDICTED_FORCE_ALIASES[prediction_model]["z"],
        description=f"{prediction_model} endpoint Fz",
        required=False,
    )

    q_hip = dataframe["q_hip_rad"].to_numpy(dtype=float)
    q_knee = dataframe["q_knee_rad"].to_numpy(dtype=float)
    if true_fx_column is not None and true_fz_column is not None:
        true_fx = dataframe[true_fx_column].to_numpy(dtype=float)
        true_fz = dataframe[true_fz_column].to_numpy(dtype=float)
        true_valid = np.isfinite(true_fx) & np.isfinite(true_fz)
    else:
        true_force = endpoint_force_from_joint_torque(
            q_hip,
            q_knee,
            dataframe[torque_columns["true_hip"]].to_numpy(dtype=float),
            dataframe[torque_columns["true_knee"]].to_numpy(dtype=float),
            link_1_m,
            link_2_m,
        )
        true_fx = np.asarray(true_force.fx_robot_on_leg_n, dtype=float)
        true_fz = np.asarray(true_force.fz_robot_on_leg_n, dtype=float)
        true_valid = np.asarray(true_force.force_mapping_valid, dtype=bool)

    if predicted_fx_column is not None and predicted_fz_column is not None:
        predicted_fx = dataframe[predicted_fx_column].to_numpy(dtype=float)
        predicted_fz = dataframe[predicted_fz_column].to_numpy(dtype=float)
        predicted_valid = np.isfinite(predicted_fx) & np.isfinite(predicted_fz)
    else:
        predicted_force = endpoint_force_from_joint_torque(
            q_hip,
            q_knee,
            dataframe[torque_columns["predicted_hip"]].to_numpy(dtype=float),
            dataframe[torque_columns["predicted_knee"]].to_numpy(dtype=float),
            link_1_m,
            link_2_m,
        )
        predicted_fx = np.asarray(predicted_force.fx_robot_on_leg_n, dtype=float)
        predicted_fz = np.asarray(predicted_force.fz_robot_on_leg_n, dtype=float)
        predicted_valid = np.asarray(
            predicted_force.force_mapping_valid,
            dtype=bool,
        )

    finite = (
        np.isfinite(true_fx)
        & np.isfinite(true_fz)
        & np.isfinite(predicted_fx)
        & np.isfinite(predicted_fz)
    )
    return true_fx, true_fz, predicted_fx, predicted_fz, (
        true_valid & predicted_valid & finite
    )


def _empty_metric_values() -> dict[str, object]:
    names = (
        "hip_torque_rmse_nm",
        "knee_torque_rmse_nm",
        "combined_torque_rmse_nm",
        "hip_nrmse_percent",
        "knee_nrmse_percent",
        "combined_nrmse_percent",
        "hip_mae_nm",
        "knee_mae_nm",
        "hip_peak_error_percent",
        "knee_peak_error_percent",
        "combined_peak_error_percent",
        "hip_correlation",
        "knee_correlation",
        "hip_vaf",
        "knee_vaf",
        "endpoint_force_rmse_n",
        "endpoint_force_peak_error_percent",
        "residual_mean_hip_nm",
        "residual_mean_knee_nm",
        "residual_angle_correlation",
        "residual_velocity_correlation",
        "hip_true_torque_range_nm",
        "knee_true_torque_range_nm",
        "combined_true_torque_range_nm",
    )
    result: dict[str, object] = {name: np.nan for name in names}
    result.update(
        {
            "nrmse_unreliable_small_range": True,
            "hip_nrmse_unreliable_small_range": True,
            "knee_nrmse_unreliable_small_range": True,
            "combined_nrmse_unreliable_small_range": True,
            "peak_error_unreliable_small_true_peak": True,
            "valid_torque_sample_count": 0,
            "valid_endpoint_force_sample_count": 0,
            "metric_valid": False,
            "metric_invalid_reason": "no_valid_finite_samples",
        }
    )
    return result


def compute_trajectory_metrics(
    predictions: pd.DataFrame,
    prediction_model: str = "identified",
    *,
    link_1_m: float = L1,
    link_2_m: float = L2,
    nrmse_epsilon_nm: float = model_mismatch_nrmse_epsilon_nm,
    minimum_nrmse_range_nm: float = model_mismatch_nrmse_minimum_range_nm,
) -> pd.DataFrame:
    """Compute all requested metrics independently for every trajectory.

    ``sample_valid`` and ``<model>_prediction_valid`` are honoured when those
    columns exist.  Invalid groups remain in the returned table with explicit
    flags, so difficult strong-mismatch or extrapolation trajectories cannot be
    hidden by filtering.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                *_trajectory_group_columns(predictions),
                "prediction_model",
                *_empty_metric_values().keys(),
            ]
        )
    if not np.isfinite(nrmse_epsilon_nm) or nrmse_epsilon_nm <= 0.0:
        raise ValueError("nrmse_epsilon_nm must be finite and positive.")
    if (
        not np.isfinite(minimum_nrmse_range_nm)
        or minimum_nrmse_range_nm < 0.0
    ):
        raise ValueError(
            "minimum_nrmse_range_nm must be finite and non-negative."
        )
    if not np.isfinite(link_1_m) or not np.isfinite(link_2_m):
        raise ValueError("link lengths must be finite.")
    if "q_hip_rad" not in predictions or "q_knee_rad" not in predictions:
        raise ValueError(
            "prediction dataframe must contain q_hip_rad and q_knee_rad for "
            "endpoint-force evaluation."
        )

    torque_columns = _torque_columns(predictions, prediction_model)
    group_columns = _trajectory_group_columns(predictions)
    rows: list[dict[str, object]] = []
    for identity, raw_group in _iter_groups(predictions, group_columns):
        row: dict[str, object] = {
            **identity,
            "prediction_model": prediction_model,
            "total_sample_count": int(len(raw_group)),
        }
        valid_mask = _valid_prediction_mask(
            raw_group,
            prediction_model,
            torque_columns,
        )
        group = raw_group.loc[valid_mask]
        if group.empty:
            row.update(_empty_metric_values())
            rows.append(row)
            continue

        true_hip = group[torque_columns["true_hip"]].to_numpy(dtype=float)
        true_knee = group[torque_columns["true_knee"]].to_numpy(dtype=float)
        predicted_hip = group[torque_columns["predicted_hip"]].to_numpy(
            dtype=float
        )
        predicted_knee = group[torque_columns["predicted_knee"]].to_numpy(
            dtype=float
        )
        residual_hip = true_hip - predicted_hip
        residual_knee = true_knee - predicted_knee
        combined_true = np.concatenate((true_hip, true_knee))
        combined_predicted = np.concatenate((predicted_hip, predicted_knee))
        combined_residual = combined_true - combined_predicted

        hip_rmse = _rmse(residual_hip)
        knee_rmse = _rmse(residual_knee)
        combined_rmse = _rmse(combined_residual)
        hip_nrmse, hip_range, hip_range_unreliable = _nrmse_percent(
            hip_rmse,
            true_hip,
            nrmse_epsilon_nm,
            minimum_nrmse_range_nm,
        )
        knee_nrmse, knee_range, knee_range_unreliable = _nrmse_percent(
            knee_rmse,
            true_knee,
            nrmse_epsilon_nm,
            minimum_nrmse_range_nm,
        )
        combined_nrmse, combined_range, combined_range_unreliable = (
            _nrmse_percent(
                combined_rmse,
                combined_true,
                nrmse_epsilon_nm,
                minimum_nrmse_range_nm,
            )
        )
        hip_peak_error, hip_peak_unreliable = _peak_error_percent(
            true_hip,
            predicted_hip,
            nrmse_epsilon_nm,
        )
        knee_peak_error, knee_peak_unreliable = _peak_error_percent(
            true_knee,
            predicted_knee,
            nrmse_epsilon_nm,
        )
        true_joint_norm = np.hypot(true_hip, true_knee)
        predicted_joint_norm = np.hypot(predicted_hip, predicted_knee)
        combined_peak_error, combined_peak_unreliable = _peak_error_percent(
            true_joint_norm,
            predicted_joint_norm,
            nrmse_epsilon_nm,
        )

        true_fx, true_fz, predicted_fx, predicted_fz, force_valid = (
            _provided_or_mapped_force(
                group,
                prediction_model,
                torque_columns,
                link_1_m,
                link_2_m,
            )
        )
        if np.any(force_valid):
            force_residual_magnitude = np.hypot(
                true_fx[force_valid] - predicted_fx[force_valid],
                true_fz[force_valid] - predicted_fz[force_valid],
            )
            force_rmse = _rmse(force_residual_magnitude)
            force_peak_error, _ = _peak_error_percent(
                np.hypot(true_fx[force_valid], true_fz[force_valid]),
                np.hypot(
                    predicted_fx[force_valid],
                    predicted_fz[force_valid],
                ),
                nrmse_epsilon_nm,
            )
        else:
            force_rmse = np.nan
            force_peak_error = np.nan

        q_hip = group["q_hip_rad"].to_numpy(dtype=float)
        q_knee = group["q_knee_rad"].to_numpy(dtype=float)
        angle_correlation = _strongest_signed_correlation(
            (residual_hip, residual_knee),
            (q_hip, q_knee),
        )
        velocity_features = tuple(
            group[column].to_numpy(dtype=float)
            for column in ("dq_hip_rad_s", "dq_knee_rad_s")
            if column in group
        )
        velocity_correlation = (
            _strongest_signed_correlation(
                (residual_hip, residual_knee),
                velocity_features,
            )
            if velocity_features
            else np.nan
        )

        any_small_range = bool(
            hip_range_unreliable
            or knee_range_unreliable
            or combined_range_unreliable
        )
        row.update(
            {
                "hip_torque_rmse_nm": hip_rmse,
                "knee_torque_rmse_nm": knee_rmse,
                "combined_torque_rmse_nm": combined_rmse,
                "hip_nrmse_percent": hip_nrmse,
                "knee_nrmse_percent": knee_nrmse,
                "combined_nrmse_percent": combined_nrmse,
                "hip_mae_nm": _mae(residual_hip),
                "knee_mae_nm": _mae(residual_knee),
                "hip_peak_error_percent": hip_peak_error,
                "knee_peak_error_percent": knee_peak_error,
                "combined_peak_error_percent": combined_peak_error,
                "hip_correlation": _safe_pearson(true_hip, predicted_hip),
                "knee_correlation": _safe_pearson(true_knee, predicted_knee),
                "hip_vaf": _safe_vaf_percent(true_hip, predicted_hip),
                "knee_vaf": _safe_vaf_percent(true_knee, predicted_knee),
                "endpoint_force_rmse_n": force_rmse,
                "endpoint_force_peak_error_percent": force_peak_error,
                "residual_mean_hip_nm": float(np.mean(residual_hip)),
                "residual_mean_knee_nm": float(np.mean(residual_knee)),
                "residual_angle_correlation": angle_correlation,
                "residual_velocity_correlation": velocity_correlation,
                "hip_true_torque_range_nm": hip_range,
                "knee_true_torque_range_nm": knee_range,
                "combined_true_torque_range_nm": combined_range,
                "nrmse_unreliable_small_range": any_small_range,
                "hip_nrmse_unreliable_small_range": hip_range_unreliable,
                "knee_nrmse_unreliable_small_range": knee_range_unreliable,
                "combined_nrmse_unreliable_small_range": (
                    combined_range_unreliable
                ),
                "peak_error_unreliable_small_true_peak": bool(
                    hip_peak_unreliable
                    or knee_peak_unreliable
                    or combined_peak_unreliable
                ),
                "valid_torque_sample_count": int(len(group)),
                "valid_endpoint_force_sample_count": int(np.sum(force_valid)),
                "metric_valid": True,
                "metric_invalid_reason": "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_all_model_metrics(
    predictions: pd.DataFrame,
    prediction_models: Sequence[str] = PREDICTION_MODELS,
    **metric_options: object,
) -> pd.DataFrame:
    """Compute trajectory metrics for generic and identified predictors."""

    tables = [
        compute_trajectory_metrics(
            predictions,
            prediction_model=model,
            **metric_options,
        )
        for model in prediction_models
    ]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def build_generic_vs_identified_comparison(
    trajectory_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build the required generic-baseline versus identified-model table."""

    required = {
        "prediction_model",
        "combined_torque_rmse_nm",
        "hip_peak_error_percent",
        "knee_peak_error_percent",
    }
    missing = required.difference(trajectory_metrics.columns)
    if missing:
        raise ValueError(
            "trajectory_metrics is missing columns: " f"{sorted(missing)}"
        )
    if trajectory_metrics.empty:
        return pd.DataFrame(
            columns=(
                "trajectory_id",
                "split",
                "generic_rmse",
                "identified_rmse",
                "improvement_percent",
                "generic_peak_error",
                "identified_peak_error",
            )
        )

    identity_columns = [
        column
        for column in _trajectory_group_columns(trajectory_metrics)
        if column != "prediction_model"
    ]
    value_columns = [
        "combined_torque_rmse_nm",
        "hip_peak_error_percent",
        "knee_peak_error_percent",
        "combined_peak_error_percent",
        "combined_nrmse_percent",
        "endpoint_force_rmse_n",
    ]
    available_values = [
        column for column in value_columns if column in trajectory_metrics.columns
    ]
    generic = trajectory_metrics.loc[
        trajectory_metrics["prediction_model"].eq("generic"),
        [*identity_columns, *available_values],
    ].copy()
    identified = trajectory_metrics.loc[
        trajectory_metrics["prediction_model"].eq("identified"),
        [*identity_columns, *available_values],
    ].copy()
    generic = generic.rename(
        columns={column: f"generic_{column}" for column in available_values}
    )
    identified = identified.rename(
        columns={column: f"identified_{column}" for column in available_values}
    )
    comparison = generic.merge(
        identified,
        on=identity_columns,
        how="outer",
        validate="one_to_one",
    )
    comparison["generic_rmse"] = comparison[
        "generic_combined_torque_rmse_nm"
    ]
    comparison["identified_rmse"] = comparison[
        "identified_combined_torque_rmse_nm"
    ]
    denominator = comparison["generic_rmse"].to_numpy(dtype=float)
    numerator = (
        comparison["generic_rmse"] - comparison["identified_rmse"]
    ).to_numpy(dtype=float)
    reliable_improvement_denominator = (
        np.isfinite(denominator)
        & (np.abs(denominator) > model_mismatch_nrmse_epsilon_nm)
    )
    improvement_percent = np.full(denominator.shape, np.nan, dtype=float)
    improvement_percent[reliable_improvement_denominator] = (
        100.0
        * numerator[reliable_improvement_denominator]
        / denominator[reliable_improvement_denominator]
    )
    comparison["improvement_percent"] = improvement_percent
    comparison["improvement_unreliable_small_generic_error"] = (
        ~reliable_improvement_denominator
    )
    if "generic_combined_peak_error_percent" in comparison:
        comparison["generic_peak_error"] = comparison[
            "generic_combined_peak_error_percent"
        ]
        comparison["identified_peak_error"] = comparison[
            "identified_combined_peak_error_percent"
        ]
    else:
        comparison["generic_peak_error"] = comparison[
            [
                "generic_hip_peak_error_percent",
                "generic_knee_peak_error_percent",
            ]
        ].max(axis=1)
        comparison["identified_peak_error"] = comparison[
            [
                "identified_hip_peak_error_percent",
                "identified_knee_peak_error_percent",
            ]
        ].max(axis=1)
    split_column = next(
        (
            column
            for column in ("dataset_split", "split")
            if column in comparison
        ),
        None,
    )
    if split_column is not None:
        comparison["split"] = comparison[split_column]
    elif "split" not in comparison:
        comparison["split"] = ""
    if "trajectory_id" not in comparison:
        comparison["trajectory_id"] = comparison.get(
            "trajectory_family",
            pd.Series("", index=comparison.index),
        )

    required_first = [
        "trajectory_id",
        "split",
        "generic_rmse",
        "identified_rmse",
        "improvement_percent",
        "generic_peak_error",
        "identified_peak_error",
    ]
    remaining = [
        column for column in comparison.columns if column not in required_first
    ]
    return comparison.loc[:, [*required_first, *remaining]]


def compute_residual_feature_correlations(
    predictions: pd.DataFrame,
    prediction_model: str = "identified",
) -> pd.DataFrame:
    """Correlate hip/knee torque residuals with the six requested features."""

    if predictions.empty:
        return pd.DataFrame(
            columns=[
                *_trajectory_group_columns(predictions),
                "prediction_model",
                "residual_joint",
                "feature",
                "correlation",
                "absolute_correlation",
                "sample_count",
                "correlation_valid",
                "invalid_reason",
            ]
        )
    missing_features = set(RESIDUAL_FEATURE_COLUMNS).difference(
        predictions.columns
    )
    if missing_features:
        raise ValueError(
            "prediction dataframe is missing residual features: "
            f"{sorted(missing_features)}"
        )
    torque_columns = _torque_columns(predictions, prediction_model)
    group_columns = _trajectory_group_columns(predictions)
    rows: list[dict[str, object]] = []
    for identity, raw_group in _iter_groups(predictions, group_columns):
        valid = _valid_prediction_mask(
            raw_group,
            prediction_model,
            torque_columns,
        )
        group = raw_group.loc[valid]
        residuals = {
            "hip": (
                group[torque_columns["true_hip"]].to_numpy(dtype=float)
                - group[torque_columns["predicted_hip"]].to_numpy(dtype=float)
            ),
            "knee": (
                group[torque_columns["true_knee"]].to_numpy(dtype=float)
                - group[torque_columns["predicted_knee"]].to_numpy(dtype=float)
            ),
        }
        for joint, residual in residuals.items():
            for feature in RESIDUAL_FEATURE_COLUMNS:
                feature_values = group[feature].to_numpy(dtype=float)
                finite = np.isfinite(residual) & np.isfinite(feature_values)
                correlation = _safe_pearson(residual, feature_values)
                if int(np.sum(finite)) < 2:
                    invalid_reason = "insufficient_finite_samples"
                elif (
                    np.ptp(residual[finite]) <= np.finfo(float).eps
                    or np.ptp(feature_values[finite]) <= np.finfo(float).eps
                ):
                    invalid_reason = "constant_residual_or_feature"
                else:
                    invalid_reason = ""
                rows.append(
                    {
                        **identity,
                        "prediction_model": prediction_model,
                        "residual_joint": joint,
                        "feature": feature,
                        "correlation": correlation,
                        "absolute_correlation": (
                            abs(correlation) if np.isfinite(correlation) else np.nan
                        ),
                        "sample_count": int(np.sum(finite)),
                        "correlation_valid": bool(np.isfinite(correlation)),
                        "invalid_reason": invalid_reason,
                    }
                )
    return pd.DataFrame(rows)


def _orthogonal_nonlinear_component(
    base: np.ndarray,
    nonlinear: np.ndarray,
) -> np.ndarray:
    """Remove constant and linear base effects from a nonlinear feature."""

    finite = np.isfinite(base) & np.isfinite(nonlinear)
    result = np.full(base.shape, np.nan, dtype=float)
    if int(np.sum(finite)) < 3:
        return result
    design = np.column_stack((np.ones(np.sum(finite)), base[finite]))
    fitted = design @ np.linalg.lstsq(design, nonlinear[finite], rcond=None)[0]
    result[finite] = nonlinear[finite] - fitted
    return result


def _maximum_absolute_finite(values: Iterable[float]) -> float:
    finite = [abs(float(value)) for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else np.nan


def compute_residual_diagnostics(
    predictions: pd.DataFrame,
    prediction_model: str = "identified",
    *,
    correlation_threshold: float = (
        model_mismatch_diagnostic_correlation_threshold
    ),
) -> pd.DataFrame:
    """Return conservative residual-pattern flags for model-mismatch review.

    The flags are screening diagnostics only.  Correlation cannot establish a
    physiological mechanism or uniquely distinguish stiffness, damping and
    coupling terms.
    """

    if (
        not np.isfinite(correlation_threshold)
        or correlation_threshold < 0.0
        or correlation_threshold > 1.0
    ):
        raise ValueError("correlation_threshold must lie in [0, 1].")
    missing_features = set(RESIDUAL_FEATURE_COLUMNS).difference(
        predictions.columns
    )
    if missing_features:
        raise ValueError(
            "prediction dataframe is missing residual features: "
            f"{sorted(missing_features)}"
        )
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                *_diagnostic_group_columns(predictions),
                "prediction_model",
                "nonlinear_stiffness_score",
                "nonlinear_damping_score",
                "joint_coupling_score",
                "possible_nonlinear_stiffness_mismatch",
                "possible_nonlinear_damping_mismatch",
                "possible_joint_coupling_mismatch",
                "diagnostic_only_not_mechanism_proof",
                "diagnostic_note",
            ]
        )

    torque_columns = _torque_columns(predictions, prediction_model)
    group_columns = _diagnostic_group_columns(predictions)
    rows: list[dict[str, object]] = []
    for identity, raw_group in _iter_groups(predictions, group_columns):
        valid = _valid_prediction_mask(
            raw_group,
            prediction_model,
            torque_columns,
        )
        group = raw_group.loc[valid]
        if group.empty:
            nonlinear_stiffness_score = np.nan
            nonlinear_damping_score = np.nan
            coupling_score = np.nan
        else:
            residual_hip = (
                group[torque_columns["true_hip"]].to_numpy(dtype=float)
                - group[torque_columns["predicted_hip"]].to_numpy(dtype=float)
            )
            residual_knee = (
                group[torque_columns["true_knee"]].to_numpy(dtype=float)
                - group[torque_columns["predicted_knee"]].to_numpy(dtype=float)
            )
            q_hip = group["q_hip_rad"].to_numpy(dtype=float)
            q_knee = group["q_knee_rad"].to_numpy(dtype=float)
            dq_hip = group["dq_hip_rad_s"].to_numpy(dtype=float)
            dq_knee = group["dq_knee_rad_s"].to_numpy(dtype=float)
            q_hip_nonlinear = _orthogonal_nonlinear_component(
                q_hip,
                np.power(q_hip, 3),
            )
            q_knee_nonlinear = _orthogonal_nonlinear_component(
                q_knee,
                np.power(q_knee, 3),
            )
            dq_hip_nonlinear = _orthogonal_nonlinear_component(
                dq_hip,
                np.abs(dq_hip) * dq_hip,
            )
            dq_knee_nonlinear = _orthogonal_nonlinear_component(
                dq_knee,
                np.abs(dq_knee) * dq_knee,
            )
            nonlinear_stiffness_score = _maximum_absolute_finite(
                (
                    _safe_pearson(residual_hip, q_hip_nonlinear),
                    _safe_pearson(residual_knee, q_knee_nonlinear),
                )
            )
            nonlinear_damping_score = _maximum_absolute_finite(
                (
                    _safe_pearson(residual_hip, dq_hip_nonlinear),
                    _safe_pearson(residual_knee, dq_knee_nonlinear),
                )
            )
            coupling_score = _maximum_absolute_finite(
                (
                    _safe_pearson(residual_hip, q_knee),
                    _safe_pearson(residual_knee, q_hip),
                )
            )
        rows.append(
            {
                **identity,
                "prediction_model": prediction_model,
                "nonlinear_stiffness_score": nonlinear_stiffness_score,
                "nonlinear_damping_score": nonlinear_damping_score,
                "joint_coupling_score": coupling_score,
                "possible_nonlinear_stiffness_mismatch": bool(
                    np.isfinite(nonlinear_stiffness_score)
                    and nonlinear_stiffness_score >= correlation_threshold
                ),
                "possible_nonlinear_damping_mismatch": bool(
                    np.isfinite(nonlinear_damping_score)
                    and nonlinear_damping_score >= correlation_threshold
                ),
                "possible_joint_coupling_mismatch": bool(
                    np.isfinite(coupling_score)
                    and coupling_score >= correlation_threshold
                ),
                "diagnostic_only_not_mechanism_proof": True,
                "diagnostic_note": (
                    "Residual correlation is a diagnostic flag only; it does "
                    "not prove a physiological mechanism."
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model_mismatch_predictions(
    predictions: pd.DataFrame,
    *,
    link_1_m: float = L1,
    link_2_m: float = L2,
) -> MismatchMetricBundle:
    """Evaluate both predictors and return all CSV-ready metric tables."""

    trajectory_metrics = compute_all_model_metrics(
        predictions,
        link_1_m=link_1_m,
        link_2_m=link_2_m,
    )
    correlations = pd.concat(
        [
            compute_residual_feature_correlations(predictions, model)
            for model in PREDICTION_MODELS
        ],
        ignore_index=True,
    )
    diagnostics = compute_residual_diagnostics(
        predictions,
        prediction_model="identified",
    )
    return MismatchMetricBundle(
        trajectory_metrics=trajectory_metrics,
        generic_vs_identified=build_generic_vs_identified_comparison(
            trajectory_metrics
        ),
        residual_feature_correlations=correlations,
        residual_diagnostics=diagnostics,
    )


# Descriptive aliases used by experiment orchestration and external notebooks.
compute_mismatch_metrics = compute_all_model_metrics
compare_generic_and_identified = build_generic_vs_identified_comparison
residual_feature_correlations = compute_residual_feature_correlations
