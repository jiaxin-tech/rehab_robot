"""阶段 4.5C 复杂虚拟受试者的纯软件生成参数。

这些附加参数只属于模型失配数据生成器。五参数估计器不得导入本模块，
也不得从这里读取非线性、耦合或残余力矩真值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Mapping

import numpy as np

from .dynamic_subject import DynamicVirtualSubject


MISMATCH_PARAMETER_NAMES = (
    "k3_hip_nm_per_rad3",
    "k3_knee_nm_per_rad3",
    "k_coupling_nm_per_rad",
    "k_coupling_asymmetry",
    "b2_hip_nm_s2_per_rad2",
    "b2_knee_nm_s2_per_rad2",
    "residual_torque_scale_nm",
    "residual_torque_frequency",
)


@dataclass(frozen=True)
class MismatchVirtualSubject(DynamicVirtualSubject):
    """带可选结构失配项的虚拟动态受试者。

    基础质量、惯量、质心、线性刚度、线性阻尼和中性角与
    :class:`DynamicVirtualSubject` 完全相同。所有新增系数默认为零，因此
    默认实例严格退化为现有线性动力学。角度使用 rad，力矩使用 N·m。

    ``k_coupling_asymmetry`` 是耦合势能中的无量纲比值 ``r``：
    ``0.5 * k_coupling * (delta_q_hip - r * delta_q_knee)**2``。
    """

    k3_hip_nm_per_rad3: float = 0.0
    k3_knee_nm_per_rad3: float = 0.0
    k_coupling_nm_per_rad: float = 0.0
    k_coupling_asymmetry: float = 0.0
    b2_hip_nm_s2_per_rad2: float = 0.0
    b2_knee_nm_s2_per_rad2: float = 0.0
    residual_torque_scale_nm: float = 0.0
    residual_torque_frequency: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in MISMATCH_PARAMETER_NAMES:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def base_dynamic_subject(self) -> DynamicVirtualSubject:
        """返回不含任何复杂生成参数的新基础受试者对象。"""

        base_names = {field.name for field in fields(DynamicVirtualSubject)}
        values = asdict(self)
        return DynamicVirtualSubject(
            **{name: values[name] for name in base_names},
        )

    def mismatch_parameters_dict(self) -> dict[str, float]:
        """返回仅供生成器和事后审计使用的附加参数副本。"""

        return {
            name: float(getattr(self, name))
            for name in MISMATCH_PARAMETER_NAMES
        }

    def as_metadata_dict(self) -> dict[str, str | float]:
        """返回包含基础参数和生成参数的 JSON 兼容元数据。"""

        return asdict(self)


def mismatch_subject_from_dynamic_subject(
    subject: DynamicVirtualSubject,
    generator_parameters: Mapping[str, float] | None = None,
    **overrides: float,
) -> MismatchVirtualSubject:
    """从基础受试者纯函数式创建复杂生成受试者。

    原 ``subject`` 不会被修改。生成参数既可作为映射传入，也可使用关键字
    参数传入；重复定义或未知字段会被拒绝，避免静默拼写错误。
    """

    if not isinstance(subject, DynamicVirtualSubject):
        raise TypeError("subject must be a DynamicVirtualSubject.")
    supplied = dict(generator_parameters or {})
    duplicate = set(supplied).intersection(overrides)
    if duplicate:
        names = ", ".join(sorted(duplicate))
        raise ValueError(f"mismatch parameters supplied twice: {names}.")
    supplied.update(overrides)
    unknown = set(supplied).difference(MISMATCH_PARAMETER_NAMES)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown mismatch parameters: {names}.")

    values = asdict(subject)
    values.update({name: float(value) for name, value in supplied.items()})
    return MismatchVirtualSubject(**values)

