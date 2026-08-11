"""Leakage-aware kinematic observation modes for stage 4.5D.

The three modes intentionally have different input contracts:

``oracle_true_joint_state``
    Reads simulated joint position, velocity, and acceleration.  This is an
    upper-bound reference only.
``tcp_inverse_kinematics``
    Reads measured pull-point coordinates and assumed geometry, reconstructs
    angles with IK, then estimates both derivatives from those angles.
``independent_joint_measurement``
    Reads already measured/noisy joint angles (camera/IMU proxy), then estimates
    both derivatives from those angles.

The two non-oracle paths never select or correct a result with true joint state,
and never read simulated true velocity or acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .angle_reconstruction import reconstruct_joint_angles_from_pull_point


ORACLE_TRUE_JOINT_STATE = "oracle_true_joint_state"
TCP_INVERSE_KINEMATICS = "tcp_inverse_kinematics"
INDEPENDENT_JOINT_MEASUREMENT = "independent_joint_measurement"
KINEMATIC_OBSERVATION_MODES = (
    ORACLE_TRUE_JOINT_STATE,
    TCP_INVERSE_KINEMATICS,
    INDEPENDENT_JOINT_MEASUREMENT,
)

CANONICAL_STATE_COLUMNS = (
    "q_hip_rad",
    "q_knee_rad",
    "dq_hip_rad_s",
    "dq_knee_rad_s",
    "ddq_hip_rad_s2",
    "ddq_knee_rad_s2",
)
DEFAULT_ORACLE_STATE_COLUMNS = (
    "q_hip_true_rad",
    "q_knee_true_rad",
    "dq_hip_true_rad_s",
    "dq_knee_true_rad_s",
    "ddq_hip_true_rad_s2",
    "ddq_knee_true_rad_s2",
)
DEFAULT_TCP_POSITION_COLUMNS = (
    "x_pull_measured_m",
    "z_pull_measured_m",
)
DEFAULT_INDEPENDENT_ANGLE_COLUMNS = (
    "q_hip_measured_rad",
    "q_knee_measured_rad",
)

# These are safe observation/context fields.  In particular, no true state or
# true geometry field can leak through a non-oracle projection.
SAFE_CONTEXT_COLUMNS = (
    "subject_id",
    "trajectory_id",
    "trajectory_family",
    "trajectory_name",
    "speed_profile",
    "dataset_split",
    "phase",
    "time_s",
    "trajectory_sample_index",
    "fx_observed_n",
    "fz_observed_n",
    "force_mapping_valid",
    "wrench_is_stale",
)


@dataclass(frozen=True)
class KinematicObservationResult:
    dataframe: pd.DataFrame
    metadata: dict[str, object]

    def __getitem__(self, key: str) -> pd.Series:
        return self.dataframe[key]

    def __len__(self) -> int:
        return len(self.dataframe)


def _require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in dataframe]
    if missing:
        raise ValueError(f"kinematic observation is missing columns: {missing}")


def _validate_two_columns(
    columns: Sequence[str],
    name: str,
) -> tuple[str, str]:
    if len(columns) != 2 or any(not isinstance(column, str) for column in columns):
        raise ValueError(f"{name} must contain exactly two column names.")
    return str(columns[0]), str(columns[1])


def _reject_nonoracle_column_names(
    columns: Sequence[str] | None,
    name: str,
) -> None:
    if columns is None:
        return
    forbidden = [
        column
        for column in columns
        if "true" in str(column).lower()
        or "ground_truth" in str(column).lower()
        or str(column).lower().startswith(("dq_", "ddq_"))
    ]
    if forbidden:
        raise ValueError(
            f"{name} cannot select true or derivative state columns: "
            f"{forbidden}"
        )


def _safe_context(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SAFE_CONTEXT_COLUMNS if column in dataframe]
    result = dataframe.loc[:, columns].reset_index(drop=True).copy()
    if "time_s" not in result:
        raise ValueError("kinematic observation requires time_s.")
    return result


def _append_invalid_reason(
    reasons: np.ndarray,
    mask: np.ndarray,
    reason: str,
) -> None:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return
    current = reasons[selected].astype(str)
    reasons[selected] = np.where(
        current == "",
        reason,
        np.char.add(np.char.add(current, ";"), reason),
    )


def _group_labels(dataframe: pd.DataFrame) -> np.ndarray:
    candidates = (
        "trajectory_id",
        "trajectory_family",
        "trajectory_name",
        "speed_profile",
        "dataset_split",
        "phase",
    )
    columns = [column for column in candidates if column in dataframe]
    if not columns:
        return np.full(len(dataframe), "trajectory", dtype=object)
    labels = dataframe[columns[0]].fillna("").astype(str)
    for column in columns[1:]:
        labels = labels.str.cat(dataframe[column].fillna("").astype(str), sep="/")
    return labels.to_numpy(dtype=object)


def _call_derivative_estimator(
    angle_dataframe: pd.DataFrame,
    *,
    derivative_method: str,
    derivative_config: object | None,
    valid_column: str,
    group_columns: Sequence[str] | None,
) -> object:
    # Local import makes the input boundary here explicit and avoids coupling
    # angle reconstruction to the filtering implementation.
    from .derivative_estimation import estimate_joint_derivatives

    kwargs: dict[str, object] = {
        "method": derivative_method,
        "angle_columns": ("q_hip_est_rad", "q_knee_est_rad"),
        "valid_column": valid_column,
    }
    if derivative_config is not None:
        kwargs["config"] = derivative_config
    if group_columns is not None:
        kwargs["group_columns"] = tuple(group_columns)
    return estimate_joint_derivatives(angle_dataframe, **kwargs)


def synthesize_independent_joint_measurements(
    q_hip_source_rad: Sequence[float] | np.ndarray,
    q_knee_source_rad: Sequence[float] | np.ndarray,
    *,
    noise_standard_deviation_rad: float,
    random_seed: int,
) -> pd.DataFrame:
    """Create a reproducible camera/IMU-like angle observation.

    This helper represents sensor-data generation.  Its output, rather than the
    source arrays, is the only angle input accepted by the independent observer.
    """

    q_hip = np.asarray(q_hip_source_rad, dtype=float)
    q_knee = np.asarray(q_knee_source_rad, dtype=float)
    if q_hip.ndim != 1 or q_hip.shape != q_knee.shape:
        raise ValueError("source joint angles must be equal-length 1-D arrays.")
    standard_deviation = float(noise_standard_deviation_rad)
    if not np.isfinite(standard_deviation) or standard_deviation < 0.0:
        raise ValueError("noise_standard_deviation_rad must be non-negative.")
    if not np.isfinite(q_hip).all() or not np.isfinite(q_knee).all():
        raise ValueError("source joint angles must be finite.")
    rng = np.random.default_rng(int(random_seed))
    return pd.DataFrame(
        {
            "q_hip_measured_rad": q_hip
            + rng.normal(0.0, standard_deviation, len(q_hip)),
            "q_knee_measured_rad": q_knee
            + rng.normal(0.0, standard_deviation, len(q_knee)),
        }
    )


def synthesize_tcp_position_measurements(
    x_pull_source_m: Sequence[float] | np.ndarray,
    z_pull_source_m: Sequence[float] | np.ndarray,
    *,
    noise_standard_deviation_m: float,
    random_seed: int,
) -> pd.DataFrame:
    """Create reproducible measured TCP positions without modifying truth data."""

    x_pull = np.asarray(x_pull_source_m, dtype=float)
    z_pull = np.asarray(z_pull_source_m, dtype=float)
    if x_pull.ndim != 1 or x_pull.shape != z_pull.shape:
        raise ValueError("source pull points must be equal-length 1-D arrays.")
    standard_deviation = float(noise_standard_deviation_m)
    if not np.isfinite(standard_deviation) or standard_deviation < 0.0:
        raise ValueError("noise_standard_deviation_m must be non-negative.")
    if not np.isfinite(x_pull).all() or not np.isfinite(z_pull).all():
        raise ValueError("source pull points must be finite.")
    rng = np.random.default_rng(int(random_seed))
    return pd.DataFrame(
        {
            "x_pull_measured_m": x_pull
            + rng.normal(0.0, standard_deviation, len(x_pull)),
            "z_pull_measured_m": z_pull
            + rng.normal(0.0, standard_deviation, len(z_pull)),
        }
    )


def _oracle_observation(
    dataframe: pd.DataFrame,
    oracle_state_columns: Sequence[str] | None,
) -> KinematicObservationResult:
    columns = tuple(oracle_state_columns or DEFAULT_ORACLE_STATE_COLUMNS)
    if len(columns) != 6:
        raise ValueError("oracle_state_columns must contain six column names.")
    if oracle_state_columns is None and not set(columns).issubset(dataframe.columns):
        # Existing stage-4 datasets use the canonical names.  That fallback is
        # intentionally available only in the explicitly labelled oracle mode.
        columns = CANONICAL_STATE_COLUMNS
    _require_columns(dataframe, columns)
    output = _safe_context(dataframe)
    state = dataframe.loc[:, list(columns)].to_numpy(dtype=float)
    finite = np.isfinite(state).all(axis=1)
    for destination, source_index in zip(CANONICAL_STATE_COLUMNS, range(6)):
        output[destination] = state[:, source_index]
    output["q_hip_est_rad"] = state[:, 0]
    output["q_knee_est_rad"] = state[:, 1]
    output["observation_valid"] = finite
    output["sample_valid"] = finite
    output["observation_reason"] = np.where(
        finite, "", "nonfinite_oracle_state"
    )
    output["derivative_valid"] = finite
    output["derivative_reason"] = np.where(
        finite, "", "nonfinite_oracle_state"
    )
    output["filter_delay_s"] = 0.0
    output["uses_future_samples"] = False
    metadata: dict[str, object] = {
        "observation_mode": ORACLE_TRUE_JOINT_STATE,
        "oracle_upper_bound_only": True,
        "practical_observation_mode": False,
        "input_columns_accessed": list(columns),
        "reads_true_joint_position": True,
        "reads_true_joint_velocity": True,
        "reads_true_joint_acceleration": True,
        "derivatives_reconstructed_from_observed_angles": False,
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "valid_samples": int(finite.sum()),
    }
    return KinematicObservationResult(output, metadata)


def _tcp_observation(
    dataframe: pd.DataFrame,
    *,
    assumed_geometry: object,
    derivative_method: str,
    derivative_config: object | None,
    tcp_position_columns: Sequence[str],
    derivative_group_columns: Sequence[str] | None,
    reconstruction_options: Mapping[str, object] | None,
) -> KinematicObservationResult:
    _reject_nonoracle_column_names(tcp_position_columns, "tcp_position_columns")
    _reject_nonoracle_column_names(
        derivative_group_columns, "derivative_group_columns"
    )
    x_column, z_column = _validate_two_columns(
        tcp_position_columns, "tcp_position_columns"
    )
    _require_columns(dataframe, ("time_s", x_column, z_column))
    options = dict(reconstruction_options or {})
    forbidden_options = {
        key for key in options if "true" in key.lower() or key.startswith("q_")
    }
    if forbidden_options:
        raise ValueError(
            "reconstruction_options cannot contain true angles/geometry: "
            f"{sorted(forbidden_options)}"
        )
    reconstructed = reconstruct_joint_angles_from_pull_point(
        dataframe[x_column].to_numpy(dtype=float),
        dataframe[z_column].to_numpy(dtype=float),
        assumed_geometry=assumed_geometry,
        time_s=dataframe["time_s"].to_numpy(dtype=float),
        trajectory_ids=_group_labels(dataframe),
        **options,
    )
    output = _safe_context(dataframe)
    audit = reconstructed.dataframe.drop(
        columns=[
            "time_s",
            "trajectory_id",
            "x_pull_measured_m",
            "z_pull_measured_m",
        ]
    )
    output[x_column] = dataframe[x_column].to_numpy(dtype=float)
    output[z_column] = dataframe[z_column].to_numpy(dtype=float)
    output = pd.concat((output, audit.reset_index(drop=True)), axis=1)
    derivative_result = _call_derivative_estimator(
        output,
        derivative_method=derivative_method,
        derivative_config=derivative_config,
        valid_column="ik_valid",
        group_columns=derivative_group_columns,
    )
    derived = derivative_result.dataframe
    output = derived.copy()
    output["q_hip_rad"] = output["q_hip_est_rad"]
    output["q_knee_rad"] = output["q_knee_est_rad"]
    output["dq_hip_rad_s"] = output["dq_hip_est_rad_s"]
    output["dq_knee_rad_s"] = output["dq_knee_est_rad_s"]
    output["ddq_hip_rad_s2"] = output["ddq_hip_est_rad_s2"]
    output["ddq_knee_rad_s2"] = output["ddq_knee_est_rad_s2"]
    state = output.loc[:, list(CANONICAL_STATE_COLUMNS)].to_numpy(dtype=float)
    finite = np.isfinite(state).all(axis=1)
    valid = (
        output["ik_valid"].to_numpy(dtype=bool)
        & output["derivative_valid"].to_numpy(dtype=bool)
        & finite
    )
    reasons = output["ik_reason"].fillna("").astype(str).to_numpy(dtype=object)
    derivative_invalid = ~output["derivative_valid"].to_numpy(dtype=bool)
    derivative_reason = (
        output["derivative_reason"].fillna("").astype(str).to_numpy(dtype=object)
    )
    for reason in pd.unique(derivative_reason[derivative_invalid]):
        if reason:
            _append_invalid_reason(
                reasons,
                derivative_invalid & (derivative_reason == reason),
                str(reason),
            )
    _append_invalid_reason(reasons, ~finite, "nonfinite_reconstructed_state")
    output["observation_valid"] = valid
    output["sample_valid"] = valid
    output["observation_reason"] = np.where(valid, "", reasons)
    metadata = {
        "observation_mode": TCP_INVERSE_KINEMATICS,
        "oracle_upper_bound_only": False,
        "practical_observation_mode": True,
        "input_columns_accessed": ["time_s", x_column, z_column],
        "reads_true_joint_position": False,
        "reads_true_joint_velocity": False,
        "reads_true_joint_acceleration": False,
        "true_geometry_accessed": False,
        "derivatives_reconstructed_from_observed_angles": True,
        "derivative_method": derivative_method,
        "uses_future_samples": bool(
            output["uses_future_samples"].astype(bool).any()
        ),
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "ik_metadata": reconstructed.metadata,
        "derivative_metadata": derivative_result.metadata,
        "valid_samples": int(valid.sum()),
        "invalid_samples": int((~valid).sum()),
    }
    return KinematicObservationResult(output, metadata)


def _independent_observation(
    dataframe: pd.DataFrame,
    *,
    derivative_method: str,
    derivative_config: object | None,
    independent_angle_columns: Sequence[str],
    derivative_group_columns: Sequence[str] | None,
) -> KinematicObservationResult:
    _reject_nonoracle_column_names(
        independent_angle_columns, "independent_angle_columns"
    )
    _reject_nonoracle_column_names(
        derivative_group_columns, "derivative_group_columns"
    )
    hip_column, knee_column = _validate_two_columns(
        independent_angle_columns, "independent_angle_columns"
    )
    _require_columns(dataframe, ("time_s", hip_column, knee_column))
    output = _safe_context(dataframe)
    output[hip_column] = dataframe[hip_column].to_numpy(dtype=float)
    output[knee_column] = dataframe[knee_column].to_numpy(dtype=float)
    output["q_hip_est_rad"] = output[hip_column]
    output["q_knee_est_rad"] = output[knee_column]
    angle_valid = np.isfinite(
        output[["q_hip_est_rad", "q_knee_est_rad"]].to_numpy(dtype=float)
    ).all(axis=1)
    output["angle_measurement_valid"] = angle_valid
    derivative_result = _call_derivative_estimator(
        output,
        derivative_method=derivative_method,
        derivative_config=derivative_config,
        valid_column="angle_measurement_valid",
        group_columns=derivative_group_columns,
    )
    output = derivative_result.dataframe.copy()
    output["q_hip_rad"] = output["q_hip_est_rad"]
    output["q_knee_rad"] = output["q_knee_est_rad"]
    output["dq_hip_rad_s"] = output["dq_hip_est_rad_s"]
    output["dq_knee_rad_s"] = output["dq_knee_est_rad_s"]
    output["ddq_hip_rad_s2"] = output["ddq_hip_est_rad_s2"]
    output["ddq_knee_rad_s2"] = output["ddq_knee_est_rad_s2"]
    state = output.loc[:, list(CANONICAL_STATE_COLUMNS)].to_numpy(dtype=float)
    finite = np.isfinite(state).all(axis=1)
    derivative_valid = output["derivative_valid"].to_numpy(dtype=bool)
    valid = angle_valid & derivative_valid & finite
    reasons = np.full(len(output), "", dtype=object)
    _append_invalid_reason(reasons, ~angle_valid, "invalid_angle_measurement")
    derivative_reason = (
        output["derivative_reason"].fillna("").astype(str).to_numpy(dtype=object)
    )
    for reason in pd.unique(derivative_reason[~derivative_valid]):
        if reason:
            _append_invalid_reason(
                reasons,
                ~derivative_valid & (derivative_reason == reason),
                str(reason),
            )
    _append_invalid_reason(reasons, ~finite, "nonfinite_reconstructed_state")
    output["observation_valid"] = valid
    output["sample_valid"] = valid
    output["observation_reason"] = np.where(valid, "", reasons)
    metadata = {
        "observation_mode": INDEPENDENT_JOINT_MEASUREMENT,
        "oracle_upper_bound_only": False,
        "practical_observation_mode": True,
        "input_columns_accessed": ["time_s", hip_column, knee_column],
        "reads_true_joint_position": False,
        "reads_true_joint_velocity": False,
        "reads_true_joint_acceleration": False,
        "true_geometry_accessed": False,
        "derivatives_reconstructed_from_observed_angles": True,
        "derivative_method": derivative_method,
        "uses_future_samples": bool(
            output["uses_future_samples"].astype(bool).any()
        ),
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "derivative_metadata": derivative_result.metadata,
        "valid_samples": int(valid.sum()),
        "invalid_samples": int((~valid).sum()),
    }
    return KinematicObservationResult(output, metadata)


def build_kinematic_observation(
    dataframe: pd.DataFrame,
    mode: str,
    *,
    assumed_geometry: object | None = None,
    derivative_method: str = "savitzky_golay_offline",
    derivative_config: object | None = None,
    oracle_state_columns: Sequence[str] | None = None,
    tcp_position_columns: Sequence[str] = DEFAULT_TCP_POSITION_COLUMNS,
    independent_angle_columns: Sequence[str] = DEFAULT_INDEPENDENT_ANGLE_COLUMNS,
    derivative_group_columns: Sequence[str] | None = None,
    reconstruction_options: Mapping[str, object] | None = None,
) -> KinematicObservationResult:
    """Build one of the three explicitly labelled observation chains."""

    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("dataframe must be a non-empty pandas DataFrame.")
    if mode not in KINEMATIC_OBSERVATION_MODES:
        raise ValueError(
            f"unknown observation mode {mode!r}; expected "
            f"{KINEMATIC_OBSERVATION_MODES}."
        )
    if mode == ORACLE_TRUE_JOINT_STATE:
        return _oracle_observation(dataframe, oracle_state_columns)
    if mode == TCP_INVERSE_KINEMATICS:
        if assumed_geometry is None:
            raise ValueError("tcp_inverse_kinematics requires assumed_geometry.")
        return _tcp_observation(
            dataframe,
            assumed_geometry=assumed_geometry,
            derivative_method=derivative_method,
            derivative_config=derivative_config,
            tcp_position_columns=tcp_position_columns,
            derivative_group_columns=derivative_group_columns,
            reconstruction_options=reconstruction_options,
        )
    if assumed_geometry is not None:
        raise ValueError(
            "independent_joint_measurement does not use assumed geometry."
        )
    return _independent_observation(
        dataframe,
        derivative_method=derivative_method,
        derivative_config=derivative_config,
        independent_angle_columns=independent_angle_columns,
        derivative_group_columns=derivative_group_columns,
    )


def observe_oracle_true_joint_state(
    dataframe: pd.DataFrame,
    *,
    oracle_state_columns: Sequence[str] | None = None,
) -> KinematicObservationResult:
    """Explicit upper-bound wrapper for the oracle observation mode."""

    return build_kinematic_observation(
        dataframe,
        ORACLE_TRUE_JOINT_STATE,
        oracle_state_columns=oracle_state_columns,
    )


def observe_tcp_inverse_kinematics(
    dataframe: pd.DataFrame,
    assumed_geometry: object,
    **options: object,
) -> KinematicObservationResult:
    """Practical TCP-only wrapper; ``options`` are observation settings."""

    return build_kinematic_observation(
        dataframe,
        TCP_INVERSE_KINEMATICS,
        assumed_geometry=assumed_geometry,
        **options,
    )


def observe_independent_joint_measurement(
    dataframe: pd.DataFrame,
    **options: object,
) -> KinematicObservationResult:
    """Practical camera/IMU-like measured-angle wrapper."""

    return build_kinematic_observation(
        dataframe,
        INDEPENDENT_JOINT_MEASUREMENT,
        **options,
    )


# Concise alias for runners.
observe_kinematics = build_kinematic_observation


__all__ = [
    "CANONICAL_STATE_COLUMNS",
    "DEFAULT_INDEPENDENT_ANGLE_COLUMNS",
    "DEFAULT_ORACLE_STATE_COLUMNS",
    "DEFAULT_TCP_POSITION_COLUMNS",
    "INDEPENDENT_JOINT_MEASUREMENT",
    "KINEMATIC_OBSERVATION_MODES",
    "KinematicObservationResult",
    "ORACLE_TRUE_JOINT_STATE",
    "TCP_INVERSE_KINEMATICS",
    "build_kinematic_observation",
    "observe_independent_joint_measurement",
    "observe_kinematics",
    "observe_oracle_true_joint_state",
    "observe_tcp_inverse_kinematics",
    "synthesize_independent_joint_measurements",
    "synthesize_tcp_position_measurements",
]
