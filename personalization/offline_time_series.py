"""Model-consistent offline integration provider, not a patient/cohort generator."""
from __future__ import annotations

import numpy as np
from lower_limb_sim.parameter_estimator import candidate_subject_from_parameters
from lower_limb_sim.full_dynamics import inverse_dynamics
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.mechanical_endpoints import (
    E0, E2, MechanicalReferenceContext, BranchRMSReference, branch_rms_components,
    full_cycle_dual_joint_rms, branch_balanced_reference_normalized_rms,
)
from .environment import PersonalizationEnvironment
from .identification import TimeSeriesIdentificationPayload, PROJECT_FORCE_MAPPING
from .observations import EpisodeObservation


class ModelConsistentTimeSeriesEnvironment(PersonalizationEnvironment):
    """Only evaluate the requested full trajectory; synthetic theta stays environment-side.

    J is derived from joint torques. Identification force is constructed by solving
    the existing project Jacobian relation; no arbitrary robot-axis mapping occurs.
    This provider is for software tests, never for personalization-necessity evidence.
    """
    offline_only = True

    def __init__(self, domain, *, baseline_template, theta, L1, L2, endpoint="E0"):
        if endpoint not in ("E0", "E2"):
            raise ValueError("endpoint must be E0 or E2")
        self.domain, self.baseline_template = domain, baseline_template
        self._subject = candidate_subject_from_parameters(baseline_template, theta)
        self._theta = tuple(float(theta[n]) for n in (
            "mass_scale", "k_hip_nm_per_rad", "k_knee_nm_per_rad",
            "b_hip_nm_s_per_rad", "b_knee_nm_s_per_rad"))
        self.L1, self.L2 = float(L1), float(L2)
        self.endpoint = E0 if endpoint == "E0" else E2
        self.revealed_candidate_ids = []
        self._reference = None

    def _torques(self, candidate):
        t = candidate.trajectory
        result = inverse_dynamics(t.q[:, 0], t.q[:, 1], t.dq[:, 0], t.dq[:, 1],
                                  t.ddq[:, 0], t.ddq[:, 1], self._subject, self.L1)
        return np.asarray(result.tau_total_hip_nm), np.asarray(result.tau_total_knee_nm)

    def evaluate(self, candidate, trial_index):
        actual = self.domain.by_id(candidate.candidate_id)
        if actual.beta != candidate.beta:
            raise ValueError("CANDIDATE_BETA_IDENTITY_MISMATCH")
        hip, knee = self._torques(actual)
        ref = self.domain.subject_reference
        profile = self.domain.profile
        context = MechanicalReferenceContext(profile.profile_id, profile.version,
                    profile.fingerprint, ref.reference_version, (*self._theta, self.L1, self.L2))
        if self.endpoint == E2:
            if self._reference is None:
                if candidate.candidate_id != self.domain.reference.candidate_id:
                    raise RuntimeError("E2_SYNTHETIC_PROVIDER_REQUIRES_REFERENCE_FIRST")
                self._reference = BranchRMSReference(context, branch_rms_components(hip, knee, ref.time_s, ref.phases))
            value = branch_balanced_reference_normalized_rms(
                branch_rms_components(hip, knee, ref.time_s, ref.phases), self._reference, context=context)
        else:
            value = full_cycle_dual_joint_rms(hip, knee, ref.time_s)
        t = actual.trajectory
        jacobian = np.swapaxes(leg_jacobian(t.q[:, 0], t.q[:, 1], self.L1, self.L2), -1, -2)
        force = np.full((len(t.q), 2), np.nan)
        valid = np.zeros(len(t.q), dtype=bool)
        for index, matrix in enumerate(jacobian):
            # Numerical full-rank check only, not an experimental safety threshold.
            if np.linalg.matrix_rank(matrix) == 2:
                force[index] = np.linalg.solve(matrix, [hip[index], knee[index]])
                valid[index] = np.isfinite(force[index]).all()
        episode_id = f"OFFLINE_TIMESERIES:{trial_index}"
        payload = TimeSeriesIdentificationPayload(
            episode_id=episode_id, candidate_id=actual.candidate_id,
            rom_profile_id=profile.profile_id, rom_version=profile.version,
            rom_fingerprint=profile.fingerprint, reference_version=ref.reference_version,
            beta_flex=actual.beta_flex, beta_extend=actual.beta_extend,
            time_s=ref.time_s, q=t.q, dq=t.dq, ddq=t.ddq, planar_force_n=force,
            sample_valid=valid, L1=self.L1, L2=self.L2, force_mapping=PROJECT_FORCE_MAPPING,
            mapping_provenance="existing leg_jacobian: tau=J.T@F; synthetic model-consistent solve",
            classification="OFFLINE_SYNTHETIC_SOFTWARE_TEST_ONLY")
        self.revealed_candidate_ids.append(actual.candidate_id)
        return EpisodeObservation(episode_id, trial_index, actual.candidate_id,
                                  actual.beta_flex, actual.beta_extend, self.endpoint.endpoint_name,
                                  value, self.endpoint.unit, 0.0, True,
                                  metadata={"classification": "OFFLINE_SYNTHETIC_SOFTWARE_TEST_ONLY",
                                            "endpoint_status": "MODEL_DERIVED_NOT_REAL_MEASUREMENT_VALIDATED"},
                                  identification_payload=payload)
