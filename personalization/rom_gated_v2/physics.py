"""Gray-box adapter that consumes subject-specific V3 q/dq/ddq arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.full_dynamics import inverse_dynamics
from lower_limb_sim.parameter_estimator import candidate_subject_from_parameters

from ..candidates import Candidate
from ..models.physics_graybox import FullDynamicsGrayBoxEndpointAdapter
from .reference import SubjectSpecificV3CandidateDomain


class SubjectSpecificFullDynamicsGrayBoxEndpointAdapter(
    FullDynamicsGrayBoxEndpointAdapter
):
    """V1 five-effective-parameter physics evaluated on the V2 trajectory.

    The inherited fit rule and effective parameter meanings are unchanged.  The
    only substitution is the trajectory lookup: it is now keyed by the frozen
    subject ROM plus beta rather than regenerated from one absolute reference.
    """

    def __init__(
        self,
        domain: SubjectSpecificV3CandidateDomain,
        *,
        regularization_weight: float = 1.0,
    ) -> None:
        super().__init__(regularization_weight=regularization_weight)
        if not domain.profile.frozen:
            raise RuntimeError("subject-specific gray-box requires a frozen ROM")
        self.domain = domain
        self._consumed_candidate_ids: set[str] = set()
        self._last_trajectory_shapes: dict[str, tuple[int, ...]] | None = None

    def _predict_with_theta(self, candidate: Candidate, theta: np.ndarray) -> float:
        trajectory = self.domain.trajectory_for(candidate.candidate_id)
        self._consumed_candidate_ids.add(candidate.candidate_id)
        self._last_trajectory_shapes = {
            "q": trajectory.q.shape,
            "dq": trajectory.dq.shape,
            "ddq": trajectory.ddq.shape,
        }
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
        time_s = self.domain.subject_reference.time_s
        return float(
            np.sqrt(
                np.trapezoid(combined_squared, time_s)
                / (time_s[-1] - time_s[0])
            )
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "adapter": "SubjectSpecificFullDynamicsGrayBoxEndpointAdapter",
            "trajectory_source": "SubjectROMProfile + beta -> actual q/dq/ddq",
            "rom_profile_id": self.domain.profile.profile_id,
            "rom_profile_fingerprint": self.domain.profile.fingerprint,
            "subject_reference_version": self.domain.subject_reference.reference_version,
            "consumed_subject_candidate_count": len(self._consumed_candidate_ids),
            "last_trajectory_shapes": self._last_trajectory_shapes,
            "V1_fit_and_parameter_semantics_unchanged": True,
        }
