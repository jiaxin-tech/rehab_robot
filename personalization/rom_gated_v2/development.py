"""Explicitly synthetic Stage 0 cases and offline end-to-end environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..environment import PersonalizationEnvironment
from ..models.physics_graybox import PhysicsSubjectModel
from ..observations import EpisodeObservation
from .physics import SubjectSpecificFullDynamicsGrayBoxEndpointAdapter
from .reference import SubjectSpecificCandidate, SubjectSpecificV3CandidateDomain
from .rom import (
    NOT_A_HUMAN_SAFETY_MODEL,
    OFFLINE_ALGORITHM_DEVELOPMENT_ONLY,
    ROMBoundaryObservation,
    ROMDeterminationController,
    SubjectROMProfile,
    SyntheticThresholdGate,
)


ALGORITHM_DEVELOPMENT_ONLY = "ALGORITHM_DEVELOPMENT_ONLY"


@dataclass(frozen=True)
class OfflineROMDevelopmentCase:
    case_id: str
    hip_min_rad: float
    hip_max_rad: float
    knee_min_rad: float
    knee_max_rad: float
    optimum_beta: tuple[float, float]


def make_offline_rom_development_cases() -> tuple[OfflineROMDevelopmentCase, ...]:
    """Algorithm cases only; these are not clinical or population evidence."""

    radians = math.radians
    return (
        OfflineROMDevelopmentCase(
            "ROM_SMALL", radians(35.0), radians(85.0), radians(25.0), radians(80.0),
            (0.0125, -0.0100),
        ),
        OfflineROMDevelopmentCase(
            "ROM_MEDIUM", radians(29.0), radians(112.0), radians(18.5), radians(119.5),
            (-0.0150, 0.0175),
        ),
        OfflineROMDevelopmentCase(
            "ROM_LARGE", radians(20.0), radians(125.0), radians(10.0), radians(135.0),
            (0.0175, 0.0125),
        ),
        OfflineROMDevelopmentCase(
            "ROM_ASYMMETRIC", radians(15.0), radians(95.0), radians(30.0), radians(120.0),
            (-0.0125, -0.0175),
        ),
    )


def determine_synthetic_rom(
    case: OfflineROMDevelopmentCase,
    *,
    version: int = 1,
) -> tuple[ROMDeterminationController, SubjectROMProfile]:
    """Run SAFE, SAFE, SAFE, STOP with an explicit synthetic-only policy."""

    policy_id = f"SYNTHETIC_{case.case_id}_BOUNDARY_POLICY_V1"
    gate = SyntheticThresholdGate(
        stop_threshold=1.05,
        threshold_policy_id=policy_id,
    )
    controller = ROMDeterminationController(gate)
    episode_id = f"{case.case_id}_ROM_CALIBRATION_V{version}"
    for index, fraction in enumerate((0.0, 0.55, 1.0, 1.10), start=1):
        hip = case.hip_min_rad + fraction * (case.hip_max_rad - case.hip_min_rad)
        knee = case.knee_min_rad + fraction * (
            case.knee_max_rad - case.knee_min_rad
        )
        controller.observe(
            ROMBoundaryObservation(
                observation_id=f"{episode_id}:OBS{index}",
                calibration_episode_id=episode_id,
                configuration_id=f"{case.case_id}_Q{index}",
                hip_angle_rad=hip,
                knee_angle_rad=knee,
                valid=True,
                threshold_policy_id=policy_id,
                measurement_quality="SYNTHETIC_EXACT",
                metadata={
                    "classification": ALGORITHM_DEVELOPMENT_ONLY,
                    "synthetic_boundary_proxy": fraction,
                    "safety_model_status": NOT_A_HUMAN_SAFETY_MODEL,
                },
            )
        )
    profile = controller.freeze_profile(
        profile_id=f"{case.case_id}_PROFILE_V{version}",
        version=version,
        provenance=ALGORITHM_DEVELOPMENT_ONLY,
        safety_margin_policy=None,
        metadata={
            "development_case_id": case.case_id,
            "not_clinical_population_evidence": True,
            "not_a_human_safety_model": True,
        },
    )
    return controller, profile


class SubjectSpecificOfflineMechanicalEnvironment(PersonalizationEnvironment):
    """Executed-candidate-only mechanical development endpoint.

    The endpoint uses the same full inverse-dynamics implementation on the
    subject-specific q/dq/ddq, plus a small declared synthetic residual in beta
    space.  It is architecture smoke evidence, not a human or robot model.
    """

    endpoint_name = "offline_subject_rom_joint_torque_rms_plus_synthetic_residual"
    endpoint_unit = "N_m_development_proxy"

    def __init__(
        self,
        domain: SubjectSpecificV3CandidateDomain,
        case: OfflineROMDevelopmentCase,
    ) -> None:
        if domain.profile.metadata.get("development_case_id") != case.case_id:
            raise ValueError("development case and ROM domain do not match")
        self.domain = domain
        self.case = case
        self._truth_adapter = SubjectSpecificFullDynamicsGrayBoxEndpointAdapter(domain)
        self.revealed_candidate_ids: list[str] = []
        self.future_oracle_access_count = 0

    def evaluate(
        self, candidate: SubjectSpecificCandidate, trial_index: int
    ) -> EpisodeObservation:
        actual = self.domain.by_id(candidate.candidate_id)
        self.revealed_candidate_ids.append(actual.candidate_id)
        physics_value = self._truth_adapter.predict_value(actual)
        dx = (actual.beta_flex - self.case.optimum_beta[0]) / 0.03
        dz = (actual.beta_extend - self.case.optimum_beta[1]) / 0.03
        synthetic_residual = 0.08 * (0.6 * dx * dx + 0.4 * dz * dz)
        return EpisodeObservation(
            episode_id=f"{self.case.case_id}:STAGE1:{trial_index}",
            trial_index=trial_index,
            candidate_id=actual.candidate_id,
            beta_flex=actual.beta_flex,
            beta_extend=actual.beta_extend,
            endpoint_name=self.endpoint_name,
            endpoint_value=physics_value + synthetic_residual,
            endpoint_unit=self.endpoint_unit,
            endpoint_uncertainty=0.0,
            valid=True,
            metadata={
                "classification": OFFLINE_ALGORITHM_DEVELOPMENT_ONLY,
                "rom_profile_id": self.domain.profile.profile_id,
                "rom_profile_fingerprint": self.domain.profile.fingerprint,
                "actual_trajectory_consumed": ["q", "dq", "ddq"],
                "truth_hidden_from_selector": True,
                "not_a_human_or_robot_endpoint": True,
            },
        )


def make_subject_physics_model(
    domain: SubjectSpecificV3CandidateDomain,
) -> PhysicsSubjectModel:
    return PhysicsSubjectModel(SubjectSpecificFullDynamicsGrayBoxEndpointAdapter(domain))
