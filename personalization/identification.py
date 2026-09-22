"""Immutable, already-mapped time-series channel, separate from BO scalar J."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


PROJECT_FORCE_MAPPING = "PROJECT_DEFINED_SYNTHETIC_PLANAR_ROBOT_ON_LEG"
VALIDATED_FORCE_MAPPING = "EXPLICITLY_VALIDATED_PROJECT_PLANAR_ROBOT_ON_LEG"


@dataclass(frozen=True)
class TimeSeriesIdentificationPayload:
    episode_id: str
    candidate_id: str
    rom_profile_id: str
    rom_version: int
    rom_fingerprint: str
    reference_version: str
    beta_flex: float
    beta_extend: float
    time_s: tuple[float, ...]
    q: tuple[tuple[float, float], ...]
    dq: tuple[tuple[float, float], ...]
    ddq: tuple[tuple[float, float], ...]
    planar_force_n: tuple[tuple[float, float], ...]
    sample_valid: tuple[bool, ...]
    L1: float
    L2: float
    force_mapping: str
    mapping_provenance: str
    classification: str
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self):
        time = np.asarray(self.time_s, dtype=float)
        if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
                or not np.all(np.diff(time) > 0)):
            raise ValueError("IDENTIFICATION_TIME_MUST_INCREASE_WITHIN_EPISODE")
        object.__setattr__(self, "time_s", tuple(float(x) for x in time))
        mask = np.asarray(self.sample_valid)
        if mask.shape != time.shape or mask.dtype != np.dtype(bool):
            raise ValueError("IDENTIFICATION_SAMPLE_VALID_REQUIRES_BOOLEAN_VECTOR")
        object.__setattr__(self, "sample_valid", tuple(bool(x) for x in mask))
        for name in ("q", "dq", "ddq", "planar_force_n"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (len(time), 2):
                raise ValueError(f"IDENTIFICATION_Nx2_REQUIRED: {name}")
            object.__setattr__(self, name, tuple(tuple(float(x) for x in row) for row in array))
        if self.force_mapping not in (PROJECT_FORCE_MAPPING, VALIDATED_FORCE_MAPPING):
            raise ValueError("EXPLICIT_PROJECT_FORCE_MAPPING_REQUIRED")
        if not all((self.episode_id, self.candidate_id, self.rom_profile_id,
                    self.rom_fingerprint, self.reference_version,
                    self.mapping_provenance, self.classification)):
            raise ValueError("IDENTIFICATION_PROVENANCE_REQUIRED")
        if (self.rom_version < 1 or not np.isfinite([self.L1, self.L2, self.beta_flex,
                                                   self.beta_extend]).all()
                or self.L1 <= 0 or self.L2 <= 0):
            raise ValueError("INVALID_IDENTIFICATION_GEOMETRY_OR_IDENTITY")
        if self.valid == bool(self.invalid_reason):
            raise ValueError("IDENTIFICATION_VALIDITY_REASON_MISMATCH")

    def to_frame(self) -> pd.DataFrame:
        q, dq, ddq, force = (np.asarray(x) for x in (self.q, self.dq, self.ddq, self.planar_force_n))
        return pd.DataFrame({
            "time_s": self.time_s, "episode_id": self.episode_id,
            "q_hip_rad": q[:, 0], "q_knee_rad": q[:, 1],
            "dq_hip_rad_s": dq[:, 0], "dq_knee_rad_s": dq[:, 1],
            "ddq_hip_rad_s2": ddq[:, 0], "ddq_knee_rad_s2": ddq[:, 1],
            "fx_observed_n": force[:, 0], "fz_observed_n": force[:, 1],
            "sample_valid": self.sample_valid,
        })

    def as_dict(self):
        from dataclasses import asdict
        result = asdict(self)
        for name in ("q", "dq", "ddq", "planar_force_n"):
            result[name] = [[value if np.isfinite(value) else None for value in row]
                            for row in result[name]]
        return result
