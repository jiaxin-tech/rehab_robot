"""Explicit offline configurations. Does not authorize a scientific comparison."""
from __future__ import annotations

from dataclasses import dataclass, replace
from lower_limb_sim.mechanical_endpoints import E0, E2
from .environment import PersonalizationEnvironment
from .models.physics_graybox import PhysicsSubjectModel
from .models.time_series_graybox import TimeSeriesFiveParameterGrayBoxAdapter, BranchBalancedE2GrayBoxEndpointAdapter
from .rom_gated_v2.reference import SubjectSpecificV3CandidateDomain
from .sequential import run_sequential_personalization


MODEL_INFORMED_BO_EI_TIMESERIES_ID = "MODEL_INFORMED_BO_EI_TIMESERIES_ID"
V2_METHODS = (MODEL_INFORMED_BO_EI_TIMESERIES_ID, "MODEL_INFORMED_BO_EI", "MODEL_INFORMED_BO_LCB",
              "MODEL_ONLY_GREEDY", "PURE_BO_EI", "PURE_BO_LCB")


@dataclass(frozen=True)
class OfflineBOConfiguration:
    endpoint: str  # E0 or E2, never inferred from numerical values
    method: str = MODEL_INFORMED_BO_EI_TIMESERIES_ID
    budget: int = 4  # Includes the reference; K_additional = budget - 1.
    xi: float = 0.0
    kappa: float = 1.5

    def __post_init__(self):
        import math
        if self.endpoint not in ("E0", "E2") or self.method not in V2_METHODS:
            raise ValueError("EXPLICIT_V2_ENDPOINT_AND_METHOD_REQUIRED")
        if not isinstance(self.budget, int) or isinstance(self.budget, bool) or self.budget < 1:
            raise ValueError("K_TOTAL_MUST_BE_POSITIVE_INTEGER")
        if not all(math.isfinite(x) and x >= 0 for x in (self.xi, self.kappa)):
            raise ValueError("INVALID_ACQUISITION_PARAMETERS")


class _CheckedOfflineEnvironment(PersonalizationEnvironment):
    def __init__(self, source, domain, endpoint, pure):
        self.source, self.domain, self.endpoint, self.pure = source, domain, endpoint, pure

    def evaluate(self, candidate, trial_index):
        observation = self.source.evaluate(candidate, trial_index)
        self.endpoint.require(observation.endpoint_name, observation.endpoint_unit)
        if (observation.candidate_id != candidate.candidate_id
                or (observation.beta_flex, observation.beta_extend) != candidate.beta
                or observation.trial_index != trial_index):
            raise ValueError("OBSERVATION_EXECUTION_IDENTITY_MISMATCH")
        # Pure BO receives only scalar-channel observations, including in its ledger.
        return replace(observation, identification_payload=None) if self.pure else observation


def run_offline_configuration(environment, domain, *, configuration: OfflineBOConfiguration,
                              baseline_template=None, L1=None, L2=None, seed=0):
    """Run canonical subject-specific 625 V3 offline only; all defaults are explicit."""
    if getattr(environment, "offline_only", False) is not True:
        raise ValueError("EXPLICIT_OFFLINE_ENVIRONMENT_REQUIRED")
    if not isinstance(domain, SubjectSpecificV3CandidateDomain) or not domain.profile.frozen or len(domain) != 625:
        raise ValueError("FROZEN_SUBJECT_625_V3_DOMAIN_REQUIRED")
    if len({candidate.beta for candidate in domain}) != 625:
        raise ValueError("UNIQUE_V3_BETA_PAIRS_REQUIRED")
    if configuration.budget > len(domain):
        raise ValueError("BUDGET_EXCEEDS_UNIQUE_CANDIDATE_DOMAIN")
    endpoint = E0 if configuration.endpoint == "E0" else E2
    pure = configuration.method.startswith("PURE_BO")
    physics = None
    if not pure:
        if baseline_template is None or L1 is None or L2 is None:
            raise ValueError("EXPLICIT_BASELINE_TEMPLATE_AND_GEOMETRY_REQUIRED")
        adapter_type = TimeSeriesFiveParameterGrayBoxAdapter if endpoint == E0 else BranchBalancedE2GrayBoxEndpointAdapter
        physics = PhysicsSubjectModel(adapter_type(domain, baseline_template=baseline_template, L1=L1, L2=L2))
    checked = _CheckedOfflineEnvironment(environment, domain, endpoint, pure)
    return run_sequential_personalization(
        checked, domain, method=configuration.method, budget=configuration.budget,
        seed=seed, physics_model=physics, xi=configuration.xi, kappa=configuration.kappa)
