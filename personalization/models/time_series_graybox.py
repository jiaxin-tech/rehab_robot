"""Explicit time-series ID adapters; legacy scalar optimizer is not changed."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd

from lower_limb_sim import parameter_estimator as estimator
from lower_limb_sim.config import identification_initial_guess, identification_lower_bounds, identification_upper_bounds
from lower_limb_sim.full_dynamics import inverse_dynamics
from lower_limb_sim.mechanical_endpoints import (
    E0, E2, BranchRMSReference, MechanicalReferenceContext,
    branch_rms_components, branch_balanced_reference_normalized_rms,
    full_cycle_dual_joint_rms,
)


class TimeSeriesFiveParameterGrayBoxAdapter:
    """E0 predictions with the existing sample-level five-parameter estimator.

    Already-derived episode frames are concatenated only for pointwise residuals.
    No derivatives, interpolation, integration or time weighting across episodes.
    """
    endpoint = E0
    parameter_semantics = "EFFECTIVE_FIVE_PARAMETER_GRAY_BOX"

    def __init__(self, domain, *, baseline_template, L1: float, L2: float):
        if not domain.profile.frozen:
            raise ValueError("FROZEN_SUBJECT_ROM_REQUIRED")
        if not np.isfinite([L1, L2]).all() or L1 <= 0 or L2 <= 0:
            raise ValueError("POSITIVE_PROJECT_GEOMETRY_REQUIRED")
        self.domain, self.baseline_template = domain, baseline_template
        self.L1, self.L2 = float(L1), float(L2)
        self._rom_identity = (domain.profile.profile_id, domain.profile.version,
                              domain.profile.fingerprint, domain.subject_reference.reference_version)
        self._theta = np.asarray([identification_initial_guess[n] for n in estimator.PARAMETER_NAMES])
        self._accepted_payloads = ()
        self._estimate = None
        self._last_attempt = None
        self._last_error = None
        self._fit_updates = 0
        self._sample_counts = {}
        self._reference_cache = None
        self._reference_recomputations = 0

    @property
    def endpoint_name(self):
        return self.endpoint.endpoint_name

    @property
    def endpoint_unit(self):
        return self.endpoint.unit

    @property
    def theta_hat(self):
        return dict(zip(estimator.PARAMETER_NAMES, map(float, self._theta)))

    def _check_domain(self):
        p = self.domain.profile
        if (p.profile_id, p.version, p.fingerprint,
                self.domain.subject_reference.reference_version) != self._rom_identity:
            raise ValueError("FROZEN_ROM_OR_REFERENCE_CHANGED_NEW_ADAPTER_REQUIRED")

    def _candidate(self, candidate):
        self._check_domain()
        actual = self.domain.by_id(candidate.candidate_id)
        if actual.beta != (candidate.beta_flex, candidate.beta_extend):
            raise ValueError("CANDIDATE_BETA_IDENTITY_MISMATCH")
        return actual

    def validate_history(self, history):
        self._check_domain()
        for observation in history:
            self.endpoint.require(observation.endpoint_name, observation.endpoint_unit)
            self._candidate(observation)

    def _payload_frame(self, observation):
        payload = observation.identification_payload
        if not observation.valid or payload is None or not payload.valid:
            return None
        identity = (payload.rom_profile_id, payload.rom_version,
                    payload.rom_fingerprint, payload.reference_version)
        if identity != self._rom_identity:
            raise ValueError("IDENTIFICATION_ROM_IDENTITY_MISMATCH")
        if (payload.episode_id != observation.episode_id
                or payload.candidate_id != observation.candidate_id
                or (payload.beta_flex, payload.beta_extend) != (observation.beta_flex, observation.beta_extend)):
            raise ValueError("IDENTIFICATION_CANDIDATE_OR_EPISODE_MISMATCH")
        if (payload.L1, payload.L2) != (self.L1, self.L2):
            raise ValueError("IDENTIFICATION_GEOMETRY_MISMATCH")
        frame = payload.to_frame()
        try:
            return estimator.valid_observations(frame)
        except ValueError as error:
            if "no valid finite samples" in str(error):
                return None
            raise

    def fit(self, history):
        self.validate_history(history)
        accepted, frames = [], []
        for observation in history:
            frame = self._payload_frame(observation)
            if frame is not None:
                accepted.append(observation.identification_payload)
                frames.append(frame)
        if not accepted:
            raise RuntimeError("TIMESERIES_ID_REQUIRES_VALID_IDENTIFICATION_EPISODE")
        if len({p.episode_id for p in accepted}) != len(accepted):
            raise ValueError("DUPLICATE_IDENTIFICATION_EPISODE")
        if tuple(accepted) == self._accepted_payloads:
            return
        # time_s remains episode-local; the estimator never uses it.
        training = pd.concat(frames, ignore_index=True)
        self._last_attempt = None
        try:
            estimate = estimator.estimate_subject_parameters(
                training, self.baseline_template, self.L1, self.L2,
                initial_guess=self.theta_hat,
            )
            self._last_attempt = estimate
            values = np.asarray([estimate.estimated_parameters[n] for n in estimator.PARAMETER_NAMES])
            bounded = all(identification_lower_bounds[n] <= estimate.estimated_parameters[n]
                          <= identification_upper_bounds[n] for n in estimator.PARAMETER_NAMES)
            if not estimate.optimizer_success or not np.isfinite(values).all() or not bounded:
                raise RuntimeError("TIMESERIES_IDENTIFICATION_FAILED: " + estimate.optimizer_message)
        except Exception as error:
            self._last_error = str(error)
            raise
        # Commit the model update only after a successful, finite, bounded estimate.
        self._theta, self._estimate = values.copy(), estimate
        self._accepted_payloads = tuple(accepted)
        self._sample_counts = {p.episode_id: len(f) for p, f in zip(accepted, frames)}
        self._fit_updates += 1
        self._last_error = None
        self._reference_cache = None

    def predict_torques(self, candidate):
        actual = self._candidate(candidate)
        if self._estimate is None or self._last_error is not None:
            raise RuntimeError("SUCCESSFUL_TIMESERIES_IDENTIFICATION_REQUIRED")
        trajectory = actual.trajectory
        subject = estimator.candidate_subject_from_parameters(self.baseline_template, self._theta)
        dynamics = inverse_dynamics(trajectory.q[:, 0], trajectory.q[:, 1],
                                    trajectory.dq[:, 0], trajectory.dq[:, 1],
                                    trajectory.ddq[:, 0], trajectory.ddq[:, 1], subject, self.L1)
        return np.asarray(dynamics.tau_total_hip_nm), np.asarray(dynamics.tau_total_knee_nm)

    def predict_value(self, candidate):
        hip, knee = self.predict_torques(candidate)
        return full_cycle_dual_joint_rms(hip, knee, self.domain.subject_reference.time_s)

    def metadata(self):
        diagnostics = self._estimate.as_serializable_dict() if self._estimate else None
        singular = self._estimate.jacobian_singular_values if self._estimate else []
        condition = float(max(singular) / min(singular)) if singular and min(singular) > 0 else None
        if condition is not None and not np.isfinite(condition):
            condition = None
        return {
            "adapter": type(self).__name__, "parameter_semantics": self.parameter_semantics,
            "identification_mode": "TIME_SERIES_TORQUE_RESIDUALS",
            "endpoint_name": self.endpoint_name, "endpoint_unit": self.endpoint_unit,
            "endpoint_status": "MODEL_DERIVED_NOT_REAL_MEASUREMENT_VALIDATED",
            "estimated_parameters": self.theta_hat,
            "fit_valid_episode_count": len(self._accepted_payloads),
            "valid_sample_count": sum(self._sample_counts.values()),
            "valid_samples_by_episode": dict(self._sample_counts),
            "fit_update_count": self._fit_updates, "identification_diagnostics": diagnostics,
            "parameter_bounds_status": "WITHIN_BOUNDS" if self._estimate else "NOT_FITTED",
            "jacobian_condition_number": condition,
            "conditioning_is_identifiability_certificate": False,
            "last_update_error": self._last_error,
            "last_attempt_diagnostics": self._last_attempt.as_serializable_dict() if self._last_attempt else None,
            "rom_profile_fingerprint": self._rom_identity[2],
            "reference_recomputation_count": self._reference_recomputations,
        }


class BranchBalancedE2GrayBoxEndpointAdapter(TimeSeriesFiveParameterGrayBoxAdapter):
    """E2 using this subject's reference predicted under the same current theta."""
    endpoint = E2

    def reference_context(self):
        self._check_domain()
        state = (*map(float, self._theta), self.L1, self.L2,
                 *map(float, asdict(self.baseline_template).values()))
        return MechanicalReferenceContext(*self._rom_identity, state)

    def predicted_reference(self):
        context = self.reference_context()
        if self._reference_cache is None or self._reference_cache.context != context:
            hip, knee = self.predict_torques(self.domain.reference)
            reference = self.domain.subject_reference
            components = branch_rms_components(hip, knee, reference.time_s, reference.phases)
            cached = BranchRMSReference(context, components)
            branch_balanced_reference_normalized_rms(components, cached, context=context)
            self._reference_cache = cached
            self._reference_recomputations += 1
        return self._reference_cache

    def predict_value(self, candidate):
        hip, knee = self.predict_torques(candidate)
        reference = self.domain.subject_reference
        components = branch_rms_components(hip, knee, reference.time_s, reference.phases)
        return branch_balanced_reference_normalized_rms(
            components, self.predicted_reference(), context=self.reference_context())
