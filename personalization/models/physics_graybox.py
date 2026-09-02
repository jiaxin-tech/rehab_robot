"""Adapters around the existing five-effective-parameter gray-box semantics."""

from __future__ import annotations

import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import least_squares

from lower_limb_sim.config import (
    identification_initial_guess,
    identification_lower_bounds,
    identification_parameter_names,
    identification_parameter_scales,
    identification_upper_bounds,
)
from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.full_dynamics import inverse_dynamics
from lower_limb_sim.parameter_estimator import (
    baseline_template_from_dynamic_subject,
    candidate_subject_from_parameters,
)

from ..candidates import Candidate
from ..observations import EpisodeObservation, valid_observations
from .base import Prediction


EFFECTIVE_GRAY_BOX_PARAMETER_NAMES = tuple(identification_parameter_names)
COMPATIBLE_OFFLINE_ENDPOINT = "offline_joint_torque_vector_rms"


class EndpointPredictionAdapter(Protocol):
    def fit(self, history: list[EpisodeObservation]) -> None: ...

    def predict_value(self, candidate: Candidate) -> float: ...

    def metadata(self) -> dict[str, Any]: ...


class FullDynamicsGrayBoxEndpointAdapter:
    """Project gray-box dynamics projected to a documented scalar endpoint.

    The projection is joint-torque-vector RMS over the frozen V3 episode.  It
    is an offline compatible mechanical endpoint only; it is not the future
    validated measured endpoint.  No observation is interpreted as comfort or
    as currently unvalidated wrench RMS.
    """

    endpoint_name = COMPATIBLE_OFFLINE_ENDPOINT
    endpoint_unit = "N_m"

    def __init__(self, *, regularization_weight: float = 1.0) -> None:
        from external_simulation.myoleg_v3_trajectory_parameterization_design_v1.parameterization import (
            generate_v3_trajectory,
        )

        self._reference = self._load_frozen_reference()
        self._generate = generate_v3_trajectory
        self._template = baseline_template_from_dynamic_subject(
            DYNAMIC_SUBJECTS["baseline"]
        )
        self._theta = np.asarray(
            [identification_initial_guess[name] for name in EFFECTIVE_GRAY_BOX_PARAMETER_NAMES],
            dtype=float,
        )
        self._lower = np.asarray(
            [identification_lower_bounds[name] for name in EFFECTIVE_GRAY_BOX_PARAMETER_NAMES],
            dtype=float,
        )
        self._upper = np.asarray(
            [identification_upper_bounds[name] for name in EFFECTIVE_GRAY_BOX_PARAMETER_NAMES],
            dtype=float,
        )
        self._scales = np.asarray(
            [identification_parameter_scales[name] for name in EFFECTIVE_GRAY_BOX_PARAMETER_NAMES],
            dtype=float,
        )
        self.regularization_weight = float(regularization_weight)
        self._fit_count = 0

    @staticmethod
    def _load_frozen_reference() -> dict[str, Any]:
        """Read the frozen V2 parent without simulator or audit-builder imports."""

        root = Path(__file__).resolve().parents[2]
        v2_path = (
            root
            / "external_simulation_audits"
            / "myoleg_knee_rom_compatibility_audit_v1"
            / "NATIVE_ROM_REFERENCE_CANDIDATE.csv"
        )
        formal_path = root / "reference_release" / "reference_measured_asymmetric_closed_slow.csv"
        with v2_path.open(newline="", encoding="utf-8") as stream:
            v2 = list(csv.DictReader(stream))
        with formal_path.open(newline="", encoding="utf-8") as stream:
            formal = list(csv.DictReader(stream))
        if len(v2) != 401 or len(formal) != 401:
            raise RuntimeError("frozen reference sample count changed")
        time_s = np.asarray([float(row["time_s"]) for row in v2])
        global_phase = np.asarray([float(row["global_phase"]) for row in v2])
        segment_phase = np.asarray([float(row["segment_phase"]) for row in v2])
        phases = np.asarray([row["cycle_phase"] for row in v2])
        q = np.asarray(
            [[float(row["q_hip_rad"]), float(row["q_knee_rad"])] for row in v2]
        )
        dq = np.asarray(
            [
                [float(row["dq_hip_rad_s"]), float(row["dq_knee_rad_s"])]
                for row in v2
            ]
        )
        ddq = np.asarray(
            [
                [float(row["ddq_hip_rad_s2"]), float(row["ddq_knee_rad_s2"])]
                for row in v2
            ]
        )
        phase_rate = np.asarray(
            [float(row["minimum_jerk_phase_rate_s_inv"]) for row in formal]
        )
        phase_accel = np.asarray(
            [float(row["minimum_jerk_phase_acceleration_s_inv2"]) for row in formal]
        )
        peak = float(global_phase[phases == "flexion"][-1])
        return {
            "time_s": time_s,
            "global_phase": global_phase,
            "segment_phase": segment_phase,
            "phases": phases,
            "q": q,
            "dq": dq,
            "ddq": ddq,
            "phase_rate": phase_rate,
            "phase_accel": phase_accel,
            "peak": peak,
            "hip_spline": make_interp_spline(
                global_phase, q[:, 0], k=3, bc_type="periodic"
            ),
            "knee_spline": make_interp_spline(
                global_phase, q[:, 1], k=3, bc_type="periodic"
            ),
        }

    def _predict_with_theta(self, candidate: Candidate, theta: np.ndarray) -> float:
        trajectory = self._generate(
            self._reference, candidate.beta_flex, candidate.beta_extend
        )
        subject = candidate_subject_from_parameters(self._template, theta)
        dynamics = inverse_dynamics(
            trajectory.q[:, 0],
            trajectory.q[:, 1],
            trajectory.dq[:, 0],
            trajectory.dq[:, 1],
            trajectory.ddq[:, 0],
            trajectory.ddq[:, 1],
            subject,
            DYNAMIC_SUBJECTS["baseline"].com_thigh_m * 2.0,
        )
        combined_squared = np.asarray(dynamics.tau_total_hip_nm) ** 2 + np.asarray(
            dynamics.tau_total_knee_nm
        ) ** 2
        time_s = np.asarray(self._reference["time_s"], dtype=float)
        return float(
            np.sqrt(np.trapezoid(combined_squared, time_s) / (time_s[-1] - time_s[0]))
        )

    def fit(self, history: list[EpisodeObservation]) -> None:
        usable = valid_observations(history)
        self._fit_count = len(usable)
        if not usable:
            return
        candidates = [
            Candidate(
                item.candidate_id,
                item.beta_flex,
                item.beta_extend,
                index,
            )
            for index, item in enumerate(usable)
        ]
        targets = np.asarray([float(item.endpoint_value) for item in usable])
        uncertainties = np.asarray(
            [max(float(item.endpoint_uncertainty or 0.0), 0.05) for item in usable]
        )
        prior = np.asarray(
            [identification_initial_guess[name] for name in EFFECTIVE_GRAY_BOX_PARAMETER_NAMES],
            dtype=float,
        )

        def residual(theta: np.ndarray) -> np.ndarray:
            data = np.asarray(
                [self._predict_with_theta(candidate, theta) for candidate in candidates]
            )
            regularizer = (
                math.sqrt(self.regularization_weight) * (theta - prior) / self._scales
            )
            return np.concatenate(((data - targets) / uncertainties, regularizer))

        result = least_squares(
            residual,
            self._theta,
            bounds=(self._lower, self._upper),
            x_scale=self._scales,
            loss="linear",
            max_nfev=40,
        )
        self._theta = np.asarray(result.x, dtype=float)

    def predict_value(self, candidate: Candidate) -> float:
        return self._predict_with_theta(candidate, self._theta)

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "FullDynamicsGrayBoxEndpointAdapter",
            "parameter_semantics": "effective_gray_box_parameters",
            "parameter_names": list(EFFECTIVE_GRAY_BOX_PARAMETER_NAMES),
            "estimated_parameters": dict(zip(EFFECTIVE_GRAY_BOX_PARAMETER_NAMES, self._theta)),
            "fit_valid_episode_count": self._fit_count,
            "physics_prediction": "existing full inverse-dynamics joint torques",
            "endpoint_derivation": "time RMS of [tau_hip,tau_knee] vector magnitude",
            "endpoint_status": "COMPATIBLE_OFFLINE_MECHANICAL_ENDPOINT_NOT_FUTURE_REAL_ENDPOINT",
            "real_trial_inputs": "only valid EpisodeObservation values up to current trial",
        }


class AnalyticDevelopmentPhysicsAdapter:
    """Fast, explicit benchmark-only stand-in for the scalar endpoint adapter.

    This adapter exists because the real scalar endpoint is not finalized.  It
    does not alter or physiologically reinterpret the five gray-box parameter
    semantics and must never be used as real-subject evidence.
    """

    def __init__(
        self,
        *,
        optimum_beta: tuple[float, float],
        prior_quality: str,
        landscape: str,
    ) -> None:
        if prior_quality not in {"P0", "P1", "P2", "P3"}:
            raise ValueError("prior_quality must be P0, P1, P2, or P3")
        self.optimum_beta = optimum_beta
        self.prior_quality = prior_quality
        self.landscape = landscape
        self._fit_count = 0

    def fit(self, history: list[EpisodeObservation]) -> None:
        # The benchmark prior is preregistered and fixed.  Subject adaptation is
        # performed by the residual GP; fitting here only audits causal inputs.
        self._fit_count = len(valid_observations(history))

    def predict_value(self, candidate: Candidate) -> float:
        ox, oz = self.optimum_beta
        if self.prior_quality == "P1":
            ox += 0.005
            oz -= 0.005
        elif self.prior_quality == "P2":
            ox += 0.01
            oz += 0.0025
        elif self.prior_quality == "P3":
            ox, oz = -ox, -oz
        x = (candidate.beta_flex - ox) / 0.03
        z = (candidate.beta_extend - oz) / 0.03
        if self.prior_quality == "P3":
            return -0.35 * x * x - 0.25 * z * z + 0.5
        if self.prior_quality == "P2":
            return 0.38 * x * x + 0.22 * z * z - 0.2
        if self.landscape == "anisotropic":
            return 1.05 * x * x + 0.18 * z * z - 0.3
        if self.landscape == "rotated":
            u, v = (x + z) / math.sqrt(2.0), (x - z) / math.sqrt(2.0)
            return 0.8 * u * u + 0.22 * v * v - 0.28
        return 0.5 * x * x + 0.38 * z * z - 0.25

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "AnalyticDevelopmentPhysicsAdapter",
            "classification": "OFFLINE_ALGORITHM_TEST_CASE",
            "prior_quality": self.prior_quality,
            "parameter_semantics_reinterpreted": False,
            "effective_parameter_names": list(EFFECTIVE_GRAY_BOX_PARAMETER_NAMES),
            "fit_valid_episode_count": self._fit_count,
            "endpoint_status": "TEMPORARY_COMPATIBLE_SYNTHETIC_MECHANICAL_ENDPOINT_ADAPTER",
        }


class PhysicsSubjectModel:
    """Causal physics prediction interface used by greedy and residual-GP BO."""

    def __init__(self, adapter: EndpointPredictionAdapter) -> None:
        self.adapter = adapter

    def fit(self, history: list[EpisodeObservation]) -> None:
        self.adapter.fit(list(history))

    def predict(self, candidate: Candidate) -> Prediction:
        value = float(self.adapter.predict_value(candidate))
        return Prediction(
            mean=value,
            std=0.0,
            valid=math.isfinite(value),
            metadata={"physics_valid": math.isfinite(value), **self.adapter.metadata()},
        )

    def state_summary(self) -> dict[str, Any]:
        return self.adapter.metadata()
