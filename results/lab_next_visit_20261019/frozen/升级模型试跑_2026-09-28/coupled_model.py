"""Independent offline five/six-parameter effective dynamics comparison.

The extension adds a nonnegative hip-knee coupling stiffness, with fixed rho=1:
V_c = 0.5 * k_c * ((q_h-q0_h) - (q_k-q0_k))**2.
The required inverse-dynamics torque is its positive potential gradient,
not the negative restoring torque. No controller or robot interfaces are used.

The original five-parameter fit is delegated unchanged to the repository's
estimator. The [0, 80] Nm/rad coupling bounds are an offline design choice,
not measured human limits. Rank/condition and standard errors are local
numerical diagnostics, not certificates of structural or practical
identifiability. Correlated time samples invalidate naive iid uncertainty.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Make this standalone workspace artifact importable without changing the
# source repository. Call with python -B to avoid repository bytecode writes.
if importlib.util.find_spec("lower_limb_sim") is None:
    repository = Path(os.environ.get("REHAB_REPO_DIR", r"D:\rehab_robot"))
    if not (repository / "lower_limb_sim" / "parameter_estimator.py").is_file():
        raise ImportError("Set REHAB_REPO_DIR to the existing rehab_robot repository.")
    sys.path.insert(0, str(repository))

from lower_limb_sim import config
from lower_limb_sim.parameter_estimator import (
    BaselineSubjectTemplate,
    estimate_subject_parameters,
    measured_joint_torque,
    predict_joint_torque,
    valid_observations,
)

BASE_PARAMETER_NAMES = tuple(config.identification_parameter_names)
COUPLING_PARAMETER = "k_coupling_nm_per_rad"
EXTENDED_PARAMETER_NAMES = BASE_PARAMETER_NAMES + (COUPLING_PARAMETER,)
COUPLING_BOUNDS = (0.0, 80.0)
COUPLING_SCALE = 20.0
SENSITIVITY_RANK_RELATIVE_TOLERANCE = 1e-8
BOUNDARY_RELATIVE_TOLERANCE = 1e-5
STATE_COLUMNS = (
    "q_hip_rad", "q_knee_rad", "dq_hip_rad_s", "dq_knee_rad_s",
    "ddq_hip_rad_s2", "ddq_knee_rad_s2",
)


def _parameter_dict(parameters: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(parameters, Mapping):
        missing = set(BASE_PARAMETER_NAMES) - set(parameters)
        if missing:
            raise ValueError(f"Missing parameters: {sorted(missing)}")
        result = {name: float(parameters[name]) for name in BASE_PARAMETER_NAMES}
        if COUPLING_PARAMETER in parameters:
            result[COUPLING_PARAMETER] = float(parameters[COUPLING_PARAMETER])
    else:
        values = np.asarray(parameters, dtype=float)
        if values.shape not in ((5,), (6,)):
            raise ValueError("parameters must contain five or six values.")
        result = dict(zip(EXTENDED_PARAMETER_NAMES[:len(values)], map(float, values)))
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("All parameters must be finite.")
    if result.get(COUPLING_PARAMETER, 0.0) < 0:
        raise ValueError("Coupling stiffness must be nonnegative.")
    return result


def predict_model(
    frame: pd.DataFrame,
    template: BaselineSubjectTemplate,
    parameters: Mapping[str, float] | Sequence[float],
    L1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict required hip/knee torque in Nm, retaining frame row order.

    Force columns are unnecessary for prediction. Omitting k_coupling or
    supplying zero gives exactly the existing five-parameter predictions.
    """
    missing = set(STATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing kinematic columns: {sorted(missing)}")
    if not np.isfinite(frame.loc[:, STATE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Prediction states must all be finite.")
    values = _parameter_dict(parameters)
    hip, knee = predict_joint_torque(frame, template, values, L1)
    coupling = values.get(COUPLING_PARAMETER, 0.0)
    if coupling == 0.0:
        return hip, knee
    displacement = (
        frame["q_hip_rad"].to_numpy(dtype=float) - template.q0_hip_rad
        - frame["q_knee_rad"].to_numpy(dtype=float) + template.q0_knee_rad
    )
    addition = coupling * displacement
    return hip + addition, knee - addition


def _specification(extended: bool):
    names = EXTENDED_PARAMETER_NAMES if extended else BASE_PARAMETER_NAMES
    initial = dict(config.identification_initial_guess)
    lower = dict(config.identification_lower_bounds)
    upper = dict(config.identification_upper_bounds)
    scales = dict(config.identification_parameter_scales)
    if extended:
        initial[COUPLING_PARAMETER] = 5.0
        lower[COUPLING_PARAMETER], upper[COUPLING_PARAMETER] = COUPLING_BOUNDS
        scales[COUPLING_PARAMETER] = COUPLING_SCALE
    return names, initial, lower, upper, scales


def _sensitivity_diagnostics(data, template, parameters, L1, torque_scales, extended):
    names, _, lower, upper, scales = _specification(extended)
    columns = []
    # This dynamics model is affine in all fitted parameters. A secant across
    # valid bounds is its exact sensitivity (up to floating point), and avoids
    # invalid negative values at a lower bound during numerical differencing.
    for name in names:
        low = dict(parameters)
        high = dict(parameters)
        low[name], high[name] = lower[name], upper[name]
        low_torque = predict_model(data, template, low, L1)
        high_torque = predict_model(data, template, high, L1)
        derivative = np.concatenate([
            (high_torque[j] - low_torque[j])
            / ((upper[name] - lower[name]) * torque_scales[j])
            for j in range(2)
        ])
        columns.append(derivative * scales[name])
    scaled = np.column_stack(columns)
    singular = np.linalg.svd(scaled, compute_uv=False)
    largest = float(singular[0]) if len(singular) else 0.0
    tolerance = largest * max(
        SENSITIVITY_RANK_RELATIVE_TOLERANCE,
        max(scaled.shape) * np.finfo(float).eps,
    )
    rank = int(np.sum(singular > tolerance))
    full_rank = rank == len(names)
    condition = float(largest / singular[-1]) if full_rank else float("inf")
    return {
        "scaled_sensitivity_singular_values": singular.tolist(),
        "scaled_sensitivity_rank": rank,
        "scaled_sensitivity_parameter_count": len(names),
        "scaled_sensitivity_condition": condition,
        "scaled_sensitivity_full_rank": full_rank,
        "scaled_sensitivity_rank_tolerance": tolerance,
        "scaled_sensitivity_rank_relative_tolerance": SENSITIVITY_RANK_RELATIVE_TOLERANCE,
        "scaled_sensitivity_parameter_scales": {name: scales[name] for name in names},
        "diagnostic_note": (
            "Torque-normalized prediction sensitivity with parameter-scale columns; "
            "rank and condition are local numerical diagnostics, not a structural "
            "identifiability certificate or a real-data sufficiency threshold. "
            "Standard errors are local least-squares approximations; temporal "
            "correlation, bounds and model mismatch can make them optimistic."
        ),
    }


def fit_model(
    training_frame: pd.DataFrame,
    template: BaselineSubjectTemplate,
    L1: float,
    L2: float,
    extended: bool = False,
    *,
    initial_guess: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, object]:
    """Fit only valid force/state observations; return parameters and diagnostics.

    Five parameters: call the repository estimator with its original defaults.
    Six parameters: same data, torque scales, robust soft_l1 loss and original
    parameter bounds, plus nonnegative coupling stiffness in [0, 80] Nm/rad.
    success means optimizer convergence only, never identification sufficiency.
    train_rmse is combined hip/knee torque RMSE in Nm. boundary_hits maps each
    near-bound parameter to 'lower' or 'upper' (1e-5 of its bounded range).
    Standard errors become None for all parameters on numerical rank loss;
    the unfiltered approximate values remain in raw_standard_errors.
    Optional initial_guess supports warm starts; a five-parameter warm start
    for the six-parameter model retains the default coupling start of 5 Nm/rad.
    """
    data = valid_observations(training_frame)
    measured = measured_joint_torque(data, L1, L2)
    torque_scales = tuple(max(float(np.std(torque)), 1.0) for torque in measured)
    names, initial, lower, upper, scales = _specification(extended)
    if initial_guess is not None:
        provided = _parameter_dict(initial_guess)
        initial.update({name: provided[name] for name in names if name in provided})
    if any(not lower[name] <= initial[name] <= upper[name] for name in names):
        raise ValueError("initial_guess must lie within parameter bounds.")

    if not extended:
        original = estimate_subject_parameters(
            training_frame, template, L1, L2, initial_guess=initial,
        )
        parameters = dict(original.estimated_parameters)
        success = original.optimizer_success
        message = original.optimizer_message
        cost = original.cost
        evaluations = original.number_of_function_evaluations
        raw_errors = dict(original.parameter_standard_errors)
        covariance = original.parameter_covariance
        loss = original.loss
        torque_scales = (original.torque_scales_nm["hip"], original.torque_scales_nm["knee"])
    else:
        def residual(vector):
            parameters_here = dict(zip(names, map(float, vector)))
            prediction = predict_model(data, template, parameters_here, L1)
            return np.concatenate([
                (measured[j] - prediction[j]) / torque_scales[j] for j in range(2)
            ])

        optimized = least_squares(
            residual,
            np.array([initial[name] for name in names]),
            bounds=(
                np.array([lower[name] for name in names]),
                np.array([upper[name] for name in names]),
            ),
            x_scale=np.array([scales[name] for name in names]),
            loss="soft_l1", f_scale=1.0, max_nfev=500,
        )
        parameters = dict(zip(names, map(float, optimized.x)))
        success = bool(optimized.success)
        message = str(optimized.message)
        cost = float(optimized.cost)
        evaluations = int(optimized.nfev)
        jacobian = np.asarray(optimized.jac, dtype=float)
        raw_residual = residual(optimized.x)
        variance = float(raw_residual @ raw_residual) / max(len(raw_residual) - len(names), 1)
        covariance_array = variance * np.linalg.pinv(jacobian.T @ jacobian)
        covariance = covariance_array.tolist()
        raw_errors = dict(zip(names, map(float, np.sqrt(np.maximum(np.diag(covariance_array), 0.0)))))
        loss = "soft_l1"

    prediction = predict_model(data, template, parameters, L1)
    residuals = [measured[j] - prediction[j] for j in range(2)]
    rmse_by_joint = dict(zip(("hip", "knee"), [float(np.sqrt(np.mean(r**2))) for r in residuals]))
    boundary_hits = {}
    for name in names:
        tolerance = BOUNDARY_RELATIVE_TOLERANCE * (upper[name] - lower[name])
        if parameters[name] - lower[name] <= tolerance:
            boundary_hits[name] = "lower"
        elif upper[name] - parameters[name] <= tolerance:
            boundary_hits[name] = "upper"
    diagnostics = _sensitivity_diagnostics(data, template, parameters, L1, torque_scales, extended)
    errors = raw_errors if diagnostics["scaled_sensitivity_full_rank"] else {name: None for name in names}
    return {
        "model": "six_parameter_coupled" if extended else "five_parameter_original",
        "parameters": parameters,
        "success": bool(success),
        "optimizer_message": message,
        "optimizer_cost": float(cost),
        "optimizer_nfev": int(evaluations),
        "loss": loss,
        "train_rmse": float(np.sqrt(np.mean(np.concatenate(residuals)**2))),
        "train_rmse_by_joint": rmse_by_joint,
        "valid_training_samples": len(data),
        "boundary_hits": boundary_hits,
        "boundary_relative_tolerance": BOUNDARY_RELATIVE_TOLERANCE,
        "standard_errors": errors,
        "raw_standard_errors": raw_errors,
        "parameter_covariance": covariance,
        "torque_scales_nm": dict(zip(("hip", "knee"), torque_scales)),
        "parameter_bounds": {name: [lower[name], upper[name]] for name in names},
        "coupling_rho": 1.0,
        "coupling_bounds_status": "offline_design_choice_not_human_limits",
        **diagnostics,
    }
