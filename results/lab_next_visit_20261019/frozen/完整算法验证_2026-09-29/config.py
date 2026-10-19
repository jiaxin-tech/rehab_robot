"""Frozen exploratory thresholds for the first complete offline architecture.

These are declared engineering settings, not calibrated clinical acceptability
or true chance-constraint coverage. Sensitivity comparisons retain all outcomes.
"""
from dataclasses import dataclass, asdict, replace

@dataclass(frozen=True)
class AlgorithmConfig:
    total_budget: int = 8
    max_probes: int = 2
    reserve_trials: int = 1
    min_model_trials: int = 2
    max_condition: float = 10000.0
    max_normalized_train_rmse: float = 0.25
    max_prequential_normalized_torque_rmse: float = 0.30
    max_prequential_E3_error: float = 0.03
    max_scaled_parameter_change: float = 0.35
    feasibility_probability: float = 0.80
    E2_max: float = 1.01
    peak_ratio_max: float = 1.10
    optimism_z: float = 1.96
    minimum_gain: float = 0.005
    gp_signal_floors: tuple = (0.01,0.01,0.02)
    gp_length_scale: float = 0.7
    parameter_scaled_sd_floor: float = 0.005
    cluster_blocks_per_episode: int = 8
    uncertainty_multiplier: float = 1.0
    use_physics: bool = True
    use_gp: bool = True
    use_parameter_uncertainty: bool = True
    def __post_init__(self):
        if not isinstance(self.total_budget,int) or self.total_budget<1:
            raise ValueError('POSITIVE_INTEGER_TOTAL_BUDGET_REQUIRED')
        if self.max_probes<0 or self.reserve_trials<1 or self.min_model_trials<1:
            raise ValueError('INVALID_PROBE_OR_RESERVE_CONFIG')
        if not 0.5<self.feasibility_probability<1 or self.minimum_gain<0:
            raise ValueError('INVALID_PROBABILITY_OR_GAIN')
        if self.uncertainty_multiplier<=0:
            raise ValueError('UNCERTAINTY_MULTIPLIER_MUST_BE_POSITIVE')
    def as_dict(self): return asdict(self)
    def for_method(self,method):
        if method=='PURE_BO': return replace(self,use_physics=False,use_parameter_uncertainty=False)
        if method=='NO_PARAM_UNCERTAINTY': return replace(self,use_parameter_uncertainty=False)
        return self

Config=AlgorithmConfig
METHODS=('FULL','NO_ID','NO_STOP','NO_CONSTRAINT','GREEDY','PURE_BO','COMMON','NO_PARAM_UNCERTAINTY')
