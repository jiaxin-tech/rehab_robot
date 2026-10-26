"""Shared coordinate and range conventions for the current reference."""
from utils.assets import artifact_path
ROM_PROTOCOL_VERSION = "ROM_PROTOCOL_V2"
FORMAL_HIP_ROM_DEG = (0.0, 120.0)
FORMAL_KNEE_ROM_DEG = (5.0, 145.0)
THETA_SHANK_DEFINITION = "q_hip - q_knee"
ACTIVE_REFERENCE_ID = "reference_measured_asymmetric_closed_slow"
ACTIVE_REFERENCE_PATH = artifact_path("reference_release/reference_measured_asymmetric_closed_slow.csv")
SOURCE_ACTIVE_REFERENCE_PATH = ACTIVE_REFERENCE_PATH
