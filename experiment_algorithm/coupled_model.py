"""Frozen parameter order, bounds and scales for the six-parameter learner."""
from lower_limb_sim import config

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
