"""Development-only composition of MyoLeg, external mechanics and assistance.

This adapter changes the required-drive force balance along a prescribed
trajectory. It does not run a forward controller or validate a patient model.
Only requested candidates are cached; trace_for never executes an experiment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from .controlled_actuation import (
    ControlledActuationConfig,
    ControlledActuationDomain,
    ControlledActuationPoint,
    assistance_waveform,
)
from .physical_interaction import (
    PHYSICAL_INTERACTION_FORMULA_VERSION,
    PhysicalInteractionConfig,
    PhysicalInteractionProfile,
    compute_physical_resistance,
    net_required_torque,
)


class PhysicalInteractionBackend:
    """Apply a fixed external mechanical profile to the unchanged V3 domain.

    ``tau_net = tau_native - tau_interaction_external - tau_assistance``.
    Profile coefficients do not depend on candidate assistance timing/share.
    Actual rad/s velocities already contain the duration transformation.
    """

    def __init__(
        self,
        native_backend: Any,
        domain: ControlledActuationDomain,
        profile: PhysicalInteractionProfile,
        config: PhysicalInteractionConfig | None = None,
        assistance_config: ControlledActuationConfig | None = None,
    ) -> None:
        self.native_backend = native_backend
        self.domain = domain
        self.profile = profile
        self.config = PhysicalInteractionConfig() if config is None else config
        self.assistance_config = ControlledActuationConfig() if assistance_config is None else assistance_config
        self._traces: dict[str, dict[str, Any]] = {}
        self.provenance = {
            "native": deepcopy(getattr(native_backend, "provenance", {})),
            "candidate_family": domain.family,
            "physical_interaction": {
                "formula_version": PHYSICAL_INTERACTION_FORMULA_VERSION,
                "profile": asdict(profile),
                "config": asdict(self.config),
                "interaction_equation": "tau_external = -scale * (K * (q - q0) + B * dq)",
                "net_equation": "tau_net = tau_native - tau_interaction_external - tau_assistance",
                "coefficient_units": {"K": "Nm/rad", "B": "Nm s/rad", "q0": "rad"},
                "scope": "declared additional external spring/damper; development mechanical model only",
                "native_passive_counted_once": True,
                "candidate_independent_subject_mechanics": True,
            },
            "controlled_actuation": self.assistance_config.as_dict(),
        }

    def _validate_point(self, point: ControlledActuationPoint) -> None:
        try:
            same_point = self.domain.by_id(point.candidate_id) is point
        except (KeyError, AttributeError):
            same_point = False
        if not same_point:
            raise ValueError("PHYSICAL_INTERACTION_CANDIDATE_IDENTITY_MISMATCH")
        time_s = np.asarray(point.time_s, dtype=float)
        q = np.asarray(point.trajectory.q, dtype=float)
        dq = np.asarray(point.trajectory.dq, dtype=float)
        if (time_s.ndim != 1 or len(time_s) < 6 or q.shape != (len(time_s), 2)
                or dq.shape != q.shape or not all(np.isfinite(value).all() for value in (time_s, q, dq))
                or np.any(np.diff(time_s) <= 0)):
            raise ValueError("INVALID_PHYSICAL_INTERACTION_KINEMATICS")

    def requested(self, point: ControlledActuationPoint) -> np.ndarray:
        self._validate_point(point)
        if point.candidate_id not in self._traces:
            q = np.asarray(point.trajectory.q, dtype=float)
            dq = np.asarray(point.trajectory.dq, dtype=float)
            time_s = np.asarray(point.time_s, dtype=float)
            interaction = compute_physical_resistance(q, dq, self.profile, self.config)
            if not interaction.valid:
                raise ValueError(interaction.invalid_reason)
            wave = assistance_waveform(point, self.domain.subject_reference.phases, self.assistance_config)
            native = np.asarray(self.native_backend.requested(point), dtype=float)
            assistance = wave["assistance_tau_nm"]
            net = net_required_torque(native, interaction, assistance)
            peak = float(np.max(np.sum(np.abs(assistance), axis=1)))
            if not np.isclose(peak, self.assistance_config.assist_peak_nm, rtol=0, atol=1e-12):
                raise ValueError("FIXED_ASSISTANCE_PEAK_VIOLATION")
            # These work diagnostics integrate executed dq against physical
            # time. A spring can deliver positive return-branch work; only
            # the viscous component must have nonpositive instantaneous power.
            elastic_work = np.trapezoid(interaction.elastic_component_nm * dq, time_s, axis=0)
            viscous_work = np.trapezoid(interaction.viscous_component_nm * dq, time_s, axis=0)
            potential_change = interaction.potential_energy_j[-1] - interaction.potential_energy_j[0]
            diagnostics = {
                "interaction_valid": True,
                "actuation_valid": True,
                "constraint_violations": [],
                "assist_peak_l1_nm": peak,
                "joint_assist_peak_nm": np.max(np.abs(assistance), axis=0).tolist(),
                "interaction_peak_nm": np.max(np.abs(interaction.external_resistance_tau_nm), axis=0).tolist(),
                "native_peak_nm": np.max(np.abs(native), axis=0).tolist(),
                "net_peak_nm": np.max(np.abs(net), axis=0).tolist(),
                "joint_elastic_work_j": elastic_work.tolist(),
                "joint_viscous_work_j": viscous_work.tolist(),
                "joint_potential_energy_change_j": potential_change.tolist(),
                "joint_spring_energy_balance_residual_j": (elastic_work + potential_change).tolist(),
                "joint_assistance_work_j": np.trapezoid(assistance * dq, time_s, axis=0).tolist(),
                "joint_abs_assistance_impulse_nm_s": np.trapezoid(np.abs(assistance), time_s, axis=0).tolist(),
                "force_balance_max_abs_error_nm": float(np.max(np.abs(
                    net + interaction.external_resistance_tau_nm + assistance - native))),
            }
            numeric_diagnostics = [value for value in diagnostics.values() if not isinstance(value, bool)]
            if any(not np.isfinite(value).all() for value in numeric_diagnostics):
                raise ValueError("NONFINITE_PHYSICAL_INTERACTION_DIAGNOSTICS")
            self._traces[point.candidate_id] = {
                "time_s": time_s.copy(),
                "q_rad": q.copy(),
                "dq_rad_s": dq.copy(),
                "native_tau_nm": native.copy(),
                "interaction_tau_nm": interaction.external_resistance_tau_nm.copy(),
                "assistance_tau_nm": assistance.copy(),
                "net_tau_nm": net.copy(),
                "elastic_tau_nm": interaction.elastic_component_nm.copy(),
                "viscous_tau_nm": interaction.viscous_component_nm.copy(),
                "potential_energy_j": interaction.potential_energy_j.copy(),
                "branch_phase": wave["branch_phase"].copy(),
                "diagnostics": diagnostics,
            }
        return self._traces[point.candidate_id]["net_tau_nm"].copy()

    def trace_for(self, point: ControlledActuationPoint) -> dict[str, Any]:
        """Audit an already requested candidate; never reveal unseen responses."""
        self._validate_point(point)
        if point.candidate_id not in self._traces:
            raise ValueError("PHYSICAL_INTERACTION_TRACE_NOT_REQUESTED")
        return deepcopy(self._traces[point.candidate_id])

    def unassisted_reference(self) -> np.ndarray:
        """Evaluator-only normalization baseline with the same added mechanics.

        Calling this function requests only the reference native trajectory.
        It does not insert an assisted candidate into the observation cache.
        """
        point = self.domain.reference
        self._validate_point(point)
        interaction = compute_physical_resistance(point.trajectory.q, point.trajectory.dq,
                                                  self.profile, self.config)
        if not interaction.valid:
            raise ValueError(interaction.invalid_reason)
        return net_required_torque(self.native_backend.requested(point), interaction)


__all__ = ["PhysicalInteractionBackend"]
