"""阶段 4.5C 的九种确定性动力学模型失配场景。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .dynamic_subject import DynamicVirtualSubject
from .mismatch_subject import (
    MISMATCH_PARAMETER_NAMES,
    MismatchVirtualSubject,
    mismatch_subject_from_dynamic_subject,
)


ESTIMATOR_MODEL_DESCRIPTION = (
    "five_parameter_linear_gray_box: mass_scale, k_hip, k_knee, b_hip, "
    "b_knee; no nonlinear, coupling, or residual terms"
)
DEFAULT_MISMATCH_RANDOM_SEED = 20260802


def _complete_parameters(**active: float) -> dict[str, float]:
    parameters = {name: 0.0 for name in MISMATCH_PARAMETER_NAMES}
    unknown = set(active).difference(parameters)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown scenario generator parameters: {names}.")
    parameters.update({name: float(value) for name, value in active.items()})
    return parameters


@dataclass(frozen=True)
class MismatchScenario:
    """一个可复现的软件虚拟受试者生成场景。"""

    scenario_name: str
    generator_parameters: Mapping[str, float]
    estimator_model_description: str
    random_seed: int
    model_mismatch_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.scenario_name.strip():
            raise ValueError("scenario_name must not be empty.")
        parameters = dict(self.generator_parameters)
        if set(parameters) != set(MISMATCH_PARAMETER_NAMES):
            raise ValueError(
                "generator_parameters must contain every mismatch parameter "
                "exactly once."
            )
        if not all(np.isfinite(value) and value >= 0.0 for value in parameters.values()):
            raise ValueError("generator parameters must be finite and non-negative.")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed,
            (int, np.integer),
        ):
            raise TypeError("random_seed must be an integer.")
        if len(set(self.model_mismatch_terms)) != len(self.model_mismatch_terms):
            raise ValueError("model_mismatch_terms must be unique.")
        object.__setattr__(
            self,
            "generator_parameters",
            MappingProxyType(parameters),
        )
        object.__setattr__(self, "random_seed", int(self.random_seed))

    def create_subject(
        self,
        base_subject: DynamicVirtualSubject,
    ) -> MismatchVirtualSubject:
        """不修改基础对象，创建本场景的复杂生成受试者。"""

        return mismatch_subject_from_dynamic_subject(
            base_subject,
            self.generator_parameters,
        )

    def as_metadata_dict(self) -> dict[str, object]:
        """返回完整且可直接序列化的场景元数据。"""

        return {
            "scenario_name": self.scenario_name,
            "generator_parameters": dict(self.generator_parameters),
            "estimator_model_description": self.estimator_model_description,
            "random_seed": self.random_seed,
            "model_mismatch_terms": list(self.model_mismatch_terms),
        }


def _scenario(
    name: str,
    terms: tuple[str, ...],
    *,
    seed_offset: int,
    **parameters: float,
) -> MismatchScenario:
    return MismatchScenario(
        scenario_name=name,
        generator_parameters=_complete_parameters(**parameters),
        estimator_model_description=ESTIMATOR_MODEL_DESCRIPTION,
        random_seed=DEFAULT_MISMATCH_RANDOM_SEED + seed_offset,
        model_mismatch_terms=terms,
    )


_SCENARIOS = (
    _scenario("matched_linear", (), seed_offset=0),
    _scenario(
        "nonlinear_stiffness_mild",
        ("nonlinear_stiffness",),
        seed_offset=1,
        k3_hip_nm_per_rad3=0.8,
        k3_knee_nm_per_rad3=0.6,
    ),
    _scenario(
        "nonlinear_stiffness_strong",
        ("nonlinear_stiffness",),
        seed_offset=2,
        k3_hip_nm_per_rad3=4.0,
        k3_knee_nm_per_rad3=3.5,
    ),
    _scenario(
        "hip_knee_coupling_mild",
        ("hip_knee_coupling",),
        seed_offset=3,
        k_coupling_nm_per_rad=1.5,
        k_coupling_asymmetry=0.40,
    ),
    _scenario(
        "hip_knee_coupling_strong",
        ("hip_knee_coupling",),
        seed_offset=4,
        k_coupling_nm_per_rad=7.0,
        k_coupling_asymmetry=0.60,
    ),
    _scenario(
        "nonlinear_damping_mild",
        ("nonlinear_damping",),
        seed_offset=5,
        b2_hip_nm_s2_per_rad2=0.15,
        b2_knee_nm_s2_per_rad2=0.12,
    ),
    _scenario(
        "structured_residual",
        ("structured_residual",),
        seed_offset=6,
        residual_torque_scale_nm=0.6,
        residual_torque_frequency=1.0,
    ),
    _scenario(
        "combined_mild",
        (
            "nonlinear_stiffness",
            "hip_knee_coupling",
            "nonlinear_damping",
            "structured_residual",
        ),
        seed_offset=7,
        k3_hip_nm_per_rad3=0.5,
        k3_knee_nm_per_rad3=0.4,
        k_coupling_nm_per_rad=1.0,
        k_coupling_asymmetry=0.60,
        b2_hip_nm_s2_per_rad2=0.08,
        b2_knee_nm_s2_per_rad2=0.06,
        residual_torque_scale_nm=0.3,
        residual_torque_frequency=1.0,
    ),
    _scenario(
        "combined_strong",
        (
            "nonlinear_stiffness",
            "hip_knee_coupling",
            "nonlinear_damping",
            "structured_residual",
        ),
        seed_offset=8,
        k3_hip_nm_per_rad3=3.5,
        k3_knee_nm_per_rad3=3.0,
        k_coupling_nm_per_rad=6.0,
        k_coupling_asymmetry=0.90,
        b2_hip_nm_s2_per_rad2=0.70,
        b2_knee_nm_s2_per_rad2=0.55,
        residual_torque_scale_nm=1.5,
        residual_torque_frequency=1.6,
    ),
)

MISMATCH_SCENARIO_DEFINITIONS: Mapping[str, MismatchScenario] = MappingProxyType(
    {scenario.scenario_name: scenario for scenario in _SCENARIOS},
)
MISMATCH_SCENARIOS = tuple(MISMATCH_SCENARIO_DEFINITIONS)
# 与其他阶段的小写配置命名兼容，但内容仍是只读场景映射。
mismatch_scenarios = MISMATCH_SCENARIO_DEFINITIONS


def get_mismatch_scenario(scenario_name: str) -> MismatchScenario:
    """按名称返回只读场景定义。"""

    try:
        return MISMATCH_SCENARIO_DEFINITIONS[scenario_name]
    except KeyError as exc:
        choices = ", ".join(MISMATCH_SCENARIOS)
        raise ValueError(
            f"Unknown mismatch scenario {scenario_name!r}; choose one of: "
            f"{choices}."
        ) from exc


def build_mismatch_subject(
    base_subject: DynamicVirtualSubject,
    scenario_name: str,
) -> MismatchVirtualSubject:
    """用给定基础受试者构建一个场景生成对象。"""

    return get_mismatch_scenario(scenario_name).create_subject(base_subject)
