"""No-truth-leakage residual-GP BO baseline on the frozen alpha lattice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


STAGE_ID = "EQUAL_BUDGET_MODEL_INFORMED_BO_BASELINE_V1"
DEFAULT_ENABLED = False
OFFLINE_ONLY = "OFFLINE_ONLY"
NOT_HUMAN_READY = "NOT_HUMAN_READY"
NOT_ROBOT_APPROVED = "NOT_ROBOT_APPROVED"
PRIMARY_VARIANT = "BO_A_FULL_FEASIBLE"
SECONDARY_VARIANT = "BO_B_MODEL_SCREENED"
BO_VARIANTS = (PRIMARY_VARIANT, SECONDARY_VARIANT)
BUDGETS = (1, 2, 3, 5)
ALPHA_LOWER = np.asarray([-5.0, -5.0, -0.03], dtype=float)
ALPHA_UPPER = np.asarray([2.0, 2.0, 0.03], dtype=float)
GP_RANDOM_BASE_SEED = 20260830
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_REPEATS = 20_000
TIE_TOLERANCE = 1e-12
EI_XI = 0.0
POSTERIOR_STD_EPSILON = 1e-12
KERNEL_SPEC = {
    "kernel": "ConstantKernel * Matern(nu=2.5, ARD) + WhiteKernel",
    "constant_initial": 1e-4,
    "constant_bounds": [1e-8, 1e-1],
    "length_scale_initial": [0.5, 0.5, 0.5],
    "length_scale_bounds": [0.05, 2.0],
    "noise_initial": 1e-8,
    "noise_bounds": [1e-12, 1e-3],
    "optimizer": "fmin_l_bfgs_b",
    "optimizer_restarts": 0,
    "normalize_y": False,
}


def deterministic_seed(case_id: str, variant: str) -> int:
    payload = f"{GP_RANDOM_BASE_SEED}|{case_id}|{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def normalize_alpha(table: pd.DataFrame) -> np.ndarray:
    values = table.loc[:, ["hip_delta", "knee_delta", "phase_delta"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("BO alpha contains non-finite values")
    if (values < ALPHA_LOWER - 1e-12).any() or (
        values > ALPHA_UPPER + 1e-12
    ).any():
        raise ValueError("BO alpha is outside frozen bounds")
    return 2.0 * (values - ALPHA_LOWER) / (ALPHA_UPPER - ALPHA_LOWER) - 1.0


def expected_improvement(
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    *,
    incumbent: float,
) -> np.ndarray:
    mean = np.asarray(posterior_mean, dtype=float)
    std = np.asarray(posterior_std, dtype=float)
    improvement = float(incumbent) - mean - EI_XI
    output = np.zeros_like(mean)
    active = std > POSTERIOR_STD_EPSILON
    z = np.zeros_like(mean)
    z[active] = improvement[active] / std[active]
    output[active] = improvement[active] * norm.cdf(z[active]) + std[active] * norm.pdf(
        z[active]
    )
    output[~active] = np.maximum(improvement[~active], 0.0)
    return output


def _kernel() -> Any:
    return ConstantKernel(
        KERNEL_SPEC["constant_initial"],
        tuple(KERNEL_SPEC["constant_bounds"]),
    ) * Matern(
        length_scale=np.asarray(KERNEL_SPEC["length_scale_initial"], dtype=float),
        length_scale_bounds=tuple(KERNEL_SPEC["length_scale_bounds"]),
        nu=2.5,
    ) + WhiteKernel(
        noise_level=KERNEL_SPEC["noise_initial"],
        noise_level_bounds=tuple(KERNEL_SPEC["noise_bounds"]),
    )


def fit_residual_gp(observations: pd.DataFrame, *, seed: int) -> GaussianProcessRegressor:
    required = {"hip_delta", "knee_delta", "phase_delta", "residual"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"queried observations missing columns: {sorted(missing)}")
    if not observations.get("truth_was_queried", pd.Series(False, index=observations.index)).astype(bool).all():
        raise PermissionError("GP may fit only explicitly queried truth")
    model = GaussianProcessRegressor(
        kernel=_kernel(),
        alpha=0.0,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(normalize_alpha(observations), observations["residual"].to_numpy(dtype=float))
    return model


def acquisition_table(
    candidate_pool: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    if "J_truth" in candidate_pool.columns or any(
        "truth" in str(column).lower() for column in candidate_pool.columns
    ):
        raise PermissionError("acquisition candidate pool must not contain truth")
    required = {
        "trajectory_id",
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "J_pred",
    }
    missing = required.difference(candidate_pool.columns)
    if missing:
        raise ValueError(f"BO pool missing columns: {sorted(missing)}")
    queried = set(observations["trajectory_id"].astype(str))
    available = candidate_pool.loc[
        ~candidate_pool["trajectory_id"].astype(str).isin(queried)
    ].copy()
    if available.empty:
        raise RuntimeError("BO acquisition exhausted the candidate pool")
    gp = fit_residual_gp(observations, seed=seed)
    residual_mean, std = gp.predict(normalize_alpha(available), return_std=True)
    available["posterior_residual_mean"] = residual_mean
    available["posterior_std"] = std
    available["posterior_total_J_mean"] = available["J_pred"].to_numpy(dtype=float) + residual_mean
    incumbent = min(1.0, float(observations["J_truth"].min()))
    available["acquisition_EI"] = expected_improvement(
        available["posterior_total_J_mean"].to_numpy(dtype=float),
        std,
        incumbent=incumbent,
    )
    available = available.sort_values(
        ["acquisition_EI", "posterior_total_J_mean", "J_pred", "trajectory_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return available, str(gp.kernel_)


@dataclass
class SequentialQueryTruthGate:
    config_sha256: str = ""
    persisted: bool = False
    pending: str | None = None
    revealed: tuple[str, ...] = ()

    def mark_persisted(self, config_sha256: str) -> None:
        if len(config_sha256) != 64:
            raise ValueError("BO config SHA-256 is invalid")
        if self.pending or self.revealed:
            raise RuntimeError("cannot alter config after truth access")
        self.config_sha256 = str(config_sha256)
        self.persisted = True

    def authorize(self, trajectory_id: str) -> str:
        identifier = str(trajectory_id)
        if not self.persisted:
            raise PermissionError("BO_PROTOCOL_INTEGRITY = FAIL: config not persisted")
        if self.pending is not None or identifier in self.revealed:
            raise PermissionError("BO truth must be exposed once in query order")
        self.pending = identifier
        payload = f"{self.config_sha256}|{identifier}|{len(self.revealed)+1}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, trajectory_id: str, token: str) -> None:
        identifier = str(trajectory_id)
        expected = hashlib.sha256(
            f"{self.config_sha256}|{identifier}|{len(self.revealed)+1}".encode("utf-8")
        ).hexdigest()
        if self.pending != identifier or token != expected:
            raise PermissionError("BO truth completion token mismatch")
        self.revealed = (*self.revealed, identifier)
        self.pending = None


def run_bo_sequence(
    candidate_pool: pd.DataFrame,
    *,
    case_id: str,
    variant: str,
    first_trajectory_id: str,
    config_sha256: str,
    truth_query: Callable[[Mapping[str, Any], int, str], float],
    max_budget: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run sequential queries; no oracle or unqueried truth enters this function."""

    if variant not in BO_VARIANTS or max_budget not in BUDGETS:
        raise ValueError("unfrozen BO variant or budget")
    if "J_truth" in candidate_pool.columns:
        raise PermissionError("BO sequence received unqueried truth")
    if candidate_pool["trajectory_id"].astype(str).duplicated().any():
        raise ValueError("BO candidate identities are not unique")
    first = candidate_pool.loc[
        candidate_pool["trajectory_id"].astype(str).eq(str(first_trajectory_id))
    ]
    if len(first) != 1:
        raise RuntimeError("frozen model Top-1 is absent from BO pool")
    gate = SequentialQueryTruthGate()
    gate.mark_persisted(config_sha256)
    observations: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    seed = deterministic_seed(case_id, variant)
    for query_index in range(1, max_budget + 1):
        if query_index == 1:
            selected = first.iloc[0].copy()
            posterior_residual_mean = 0.0
            posterior_std = math.sqrt(float(KERNEL_SPEC["constant_initial"]))
            posterior_total = float(selected["J_pred"])
            acquisition = float("nan")
            fitted_kernel = "ZERO_RESIDUAL_PRIOR_BEFORE_FIRST_QUERY"
            query_rule = "FIXED_MODEL_PREDICTED_C1"
        else:
            observation_table = pd.DataFrame(observations)
            acquisition_map, fitted_kernel = acquisition_table(
                candidate_pool, observation_table, seed=seed
            )
            selected = acquisition_map.iloc[0].copy()
            posterior_residual_mean = float(selected["posterior_residual_mean"])
            posterior_std = float(selected["posterior_std"])
            posterior_total = float(selected["posterior_total_J_mean"])
            acquisition = float(selected["acquisition_EI"])
            query_rule = "EXPECTED_IMPROVEMENT_MAXIMUM"
        trajectory_id = str(selected["trajectory_id"])
        token = gate.authorize(trajectory_id)
        truth_j = float(truth_query(selected.to_dict(), query_index, token))
        gate.complete(trajectory_id, token)
        residual = truth_j - float(selected["J_pred"])
        observation = {
            "trajectory_id": trajectory_id,
            "hip_delta": float(selected["hip_delta"]),
            "knee_delta": float(selected["knee_delta"]),
            "phase_delta": float(selected["phase_delta"]),
            "J_pred": float(selected["J_pred"]),
            "J_truth": truth_j,
            "residual": residual,
            "truth_was_queried": True,
        }
        observations.append(observation)
        logs.append(
            {
                "case_id": case_id,
                "bo_variant": variant,
                "query_index": query_index,
                **observation,
                "posterior_residual_mean_before_query": posterior_residual_mean,
                "posterior_std_before_query": posterior_std,
                "posterior_total_J_mean_before_query": posterior_total,
                "acquisition_EI": acquisition,
                "query_rule": query_rule,
                "incumbent_before_query": min(
                    [1.0, *[float(row["J_truth"]) for row in observations[:-1]]]
                ),
                "fitted_kernel_before_query": fitted_kernel,
                "truth_authorization_token": token,
                "unqueried_truth_used": False,
            }
        )
    observations_frame = pd.DataFrame(observations)
    posterior, kernel = acquisition_table(candidate_pool, observations_frame, seed=seed)
    recommendation = posterior.iloc[0]
    posterior_row = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "bo_variant": variant,
                "budget": max_budget,
                "posterior_recommended_trajectory_id": str(recommendation["trajectory_id"]),
                "posterior_recommended_J_mean": float(recommendation["posterior_total_J_mean"]),
                "posterior_recommended_std": float(recommendation["posterior_std"]),
                "fitted_kernel": kernel,
                "recommendation_is_primary_physical_decision": False,
                "unqueried_truth_used": False,
            }
        ]
    )
    return pd.DataFrame(logs), posterior_row


def bootstrap_mean_ci(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(BOOTSTRAP_REPEATS, len(array)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


__all__ = [name for name in globals() if name.isupper()] + [
    "SequentialQueryTruthGate",
    "acquisition_table",
    "bootstrap_mean_ci",
    "deterministic_seed",
    "expected_improvement",
    "fit_residual_gp",
    "normalize_alpha",
    "run_bo_sequence",
]
