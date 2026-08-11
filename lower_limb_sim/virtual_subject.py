"""用于验证准静态算法的虚拟受试者参数。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import L1, virtual_shank_length_m


@dataclass(frozen=True)
class VirtualSubject:
    """虚拟腿段质量、质心和线性被动刚度参数。

    这些参数只用于软件与算法验证，不是医学参考值，也不代表真实患者。
    """

    subject_id: str
    mass_thigh_kg: float
    mass_shank_kg: float
    com_thigh_m: float
    com_shank_m: float
    k_hip_nm_per_rad: float
    k_knee_nm_per_rad: float
    q0_hip_rad: float
    q0_knee_rad: float
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must not be empty.")

        positive_fields = {
            "mass_thigh_kg": self.mass_thigh_kg,
            "mass_shank_kg": self.mass_shank_kg,
            "com_thigh_m": self.com_thigh_m,
            "com_shank_m": self.com_shank_m,
            "gravity_m_s2": self.gravity_m_s2,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        stiffness_fields = {
            "k_hip_nm_per_rad": self.k_hip_nm_per_rad,
            "k_knee_nm_per_rad": self.k_knee_nm_per_rad,
        }
        for name, value in stiffness_fields.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

        for name, value in {
            "q0_hip_rad": self.q0_hip_rad,
            "q0_knee_rad": self.q0_knee_rad,
        }.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if self.com_thigh_m > L1:
            raise ValueError("com_thigh_m cannot exceed the thigh segment length L1.")
        if self.com_shank_m > virtual_shank_length_m:
            raise ValueError(
                "com_shank_m cannot exceed the configured full shank length."
            )


BASELINE_SUBJECT = VirtualSubject(
    subject_id="baseline",
    mass_thigh_kg=7.0,
    mass_shank_kg=4.0,
    com_thigh_m=0.18,
    com_shank_m=0.16,
    k_hip_nm_per_rad=15.0,
    k_knee_nm_per_rad=12.0,
    q0_hip_rad=float(np.deg2rad(10.0)),
    q0_knee_rad=float(np.deg2rad(10.0)),
)

HIP_STIFF_SUBJECT = replace(
    BASELINE_SUBJECT,
    subject_id="hip_stiff",
    k_hip_nm_per_rad=30.0,
)

KNEE_STIFF_SUBJECT = replace(
    BASELINE_SUBJECT,
    subject_id="knee_stiff",
    k_knee_nm_per_rad=30.0,
)

HEAVY_LEG_SUBJECT = replace(
    BASELINE_SUBJECT,
    subject_id="heavy_leg",
    mass_thigh_kg=BASELINE_SUBJECT.mass_thigh_kg * 1.3,
    mass_shank_kg=BASELINE_SUBJECT.mass_shank_kg * 1.3,
)

VIRTUAL_SUBJECTS = {
    subject.subject_id: subject
    for subject in (
        BASELINE_SUBJECT,
        HIP_STIFF_SUBJECT,
        KNEE_STIFF_SUBJECT,
        HEAVY_LEG_SUBJECT,
    )
}


def get_virtual_subject(subject_id: str) -> VirtualSubject:
    """按 ID 返回示例虚拟受试者。"""

    try:
        return VIRTUAL_SUBJECTS[subject_id]
    except KeyError as exc:
        choices = ", ".join(VIRTUAL_SUBJECTS)
        raise ValueError(
            f"Unknown virtual subject {subject_id!r}; choose one of: {choices}."
        ) from exc
