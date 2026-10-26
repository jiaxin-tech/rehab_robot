"""SDK defaults; experiment parameters come from experiment.json."""
from config.bundle import configuration

_parameters = configuration()
_robot = _parameters["robot"]
_motion = _parameters["motion"]
ROBOT_IP = _robot["ip"]
ROBOT_LOCAL_IP = _robot["local_ip"]
ROBOT_CLASS = _robot["robot_class"]
ROBOT_STATE_MS = 8
ROBOT_MAX_LINEAR_SPEED_M_S = 1.0
ROBOT_CMD_CACHE = 1
# Constructor defaults only. Motion preparation uses explicit experiment RT settings.
ROBOT_RT_NETWORK_TOLERANCE = 20
ROBOT_RT_FILTER_HZ = 50.0

ROBOT_FORCE_SOURCE = "rokae_force_control_get_end_torque"
ROBOT_FORCE_RAW_FRAME = "world"
ROBOT_FORCE_HZ = _parameters["acquisition"]["wrench_hz"]
ROBOT_FORCE_BIAS_SAMPLES = 50
ROBOT_FORCE_STALE_S = _motion["max_wrench_age_s"]
# Manual Base +X/+Z checks on 2026-10-18; recheck after mounting/frame changes.
BASE_WRENCH_ROTATION_VERIFIED = True

COLLECT_HZ = ROBOT_FORCE_HZ
MAX_ROBOT_STATE_AGE_S = _motion["max_state_age_s"]
MAX_FORCE_SAMPLE_AGE_S = _motion["max_wrench_age_s"]
MAX_INTERNAL_STATE_SKEW_S = _motion["max_state_wrench_skew_s"]
