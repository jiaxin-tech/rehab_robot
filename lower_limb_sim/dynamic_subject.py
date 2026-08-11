"""带独立惯量和线性阻尼参数的动态虚拟受试者。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .virtual_subject import VIRTUAL_SUBJECTS, VirtualSubject


@dataclass(frozen=True)
class DynamicVirtualSubject:
    """完整动态仿真使用的虚拟参数。

    转动惯量是独立参数，绝不由牵引点长度 L2 推算。这些数值只用于软件
    验证，不是医学标准、患者辨识结果或机器人安全阈值。
    """

    subject_id: str
    mass_thigh_kg: float
    mass_shank_kg: float
    com_thigh_m: float
    com_shank_m: float
    inertia_thigh_kg_m2: float
    inertia_shank_kg_m2: float
    b_hip_nm_s_per_rad: float
    b_knee_nm_s_per_rad: float
    k_hip_nm_per_rad: float
    k_knee_nm_per_rad: float
    q0_hip_rad: float
    q0_knee_rad: float
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        # 复用第二阶段的质量、质心、刚度和中性角合法性检查。
        VirtualSubject(
            subject_id=self.subject_id,
            mass_thigh_kg=self.mass_thigh_kg,
            mass_shank_kg=self.mass_shank_kg,
            com_thigh_m=self.com_thigh_m,
            com_shank_m=self.com_shank_m,
            k_hip_nm_per_rad=self.k_hip_nm_per_rad,
            k_knee_nm_per_rad=self.k_knee_nm_per_rad,
            q0_hip_rad=self.q0_hip_rad,
            q0_knee_rad=self.q0_knee_rad,
            gravity_m_s2=self.gravity_m_s2,
        )
        for name, value in {
            "inertia_thigh_kg_m2": self.inertia_thigh_kg_m2,
            "inertia_shank_kg_m2": self.inertia_shank_kg_m2,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in {
            "b_hip_nm_s_per_rad": self.b_hip_nm_s_per_rad,
            "b_knee_nm_s_per_rad": self.b_knee_nm_s_per_rad,
        }.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def as_metadata_dict(self) -> dict[str, str | float]:
        """返回可直接写入 JSON 的全部参数。"""

        return asdict(self)


def _from_quasi_static_subject(
    subject: VirtualSubject,
) -> DynamicVirtualSubject:
    baseline = VIRTUAL_SUBJECTS["baseline"]
    mass_scale = subject.mass_thigh_kg / baseline.mass_thigh_kg
    if not np.isclose(
        subject.mass_shank_kg / baseline.mass_shank_kg,
        mass_scale,
    ):
        raise ValueError("virtual subject thigh/shank mass scales must match.")
    return DynamicVirtualSubject(
        subject_id=subject.subject_id,
        mass_thigh_kg=subject.mass_thigh_kg,
        mass_shank_kg=subject.mass_shank_kg,
        com_thigh_m=subject.com_thigh_m,
        com_shank_m=subject.com_shank_m,
        # 第四阶段使用一个共同 mass_scale 同时缩放质量和独立惯量。
        # 对 baseline/刚度变体 scale=1；heavy_leg scale=1.3。
        inertia_thigh_kg_m2=0.12 * mass_scale,
        inertia_shank_kg_m2=0.06 * mass_scale,
        b_hip_nm_s_per_rad=2.0,
        b_knee_nm_s_per_rad=1.5,
        k_hip_nm_per_rad=subject.k_hip_nm_per_rad,
        k_knee_nm_per_rad=subject.k_knee_nm_per_rad,
        q0_hip_rad=subject.q0_hip_rad,
        q0_knee_rad=subject.q0_knee_rad,
        gravity_m_s2=subject.gravity_m_s2,
    )


DYNAMIC_SUBJECTS = {
    subject_id: _from_quasi_static_subject(subject)
    for subject_id, subject in VIRTUAL_SUBJECTS.items()
}


def get_dynamic_subject(subject_id: str) -> DynamicVirtualSubject:
    """按 ID 返回动态虚拟受试者。"""

    try:
        return DYNAMIC_SUBJECTS[subject_id]
    except KeyError as exc:
        choices = ", ".join(DYNAMIC_SUBJECTS)
        raise ValueError(
            f"Unknown dynamic subject {subject_id!r}; choose one of: {choices}."
        ) from exc
