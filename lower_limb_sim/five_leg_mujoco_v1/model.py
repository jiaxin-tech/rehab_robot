"""Frozen deterministic five-leg MuJoCo model definitions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco

from personalization.rom_gated_v2.rom import (
    FROZEN_SUBJECT_ROM_PROFILE_V1,
    SubjectROMProfile,
)


PACKAGE_DIR = Path(__file__).resolve().parent
FROZEN_DEFINITION_PATH = PACKAGE_DIR / "FROZEN_FIVE_LEG_MECHANICAL_PARAMETERS_V1.json"
EXPECTED_LEG_IDS = (
    "LEG_0_NOMINAL",
    "LEG_1_HEAVY_HIP_STIFF",
    "LEG_2_KNEE_DOMINANT",
    "LEG_3_NONLINEAR_COUPLED",
    "LEG_4_STRONG_STRUCTURAL_MISMATCH",
)


@dataclass(frozen=True)
class MechanicalLegDefinition:
    """One synthetic mechanical stress-test system fixed before analysis."""

    leg_id: str
    mechanical_regime: str
    hip_min_rad: float
    hip_max_rad: float
    knee_min_rad: float
    knee_max_rad: float
    mass_thigh_kg: float
    mass_shank_kg: float
    com_thigh_m: float
    com_shank_m: float
    inertia_thigh_kg_m2: float
    inertia_shank_kg_m2: float
    hip_stiffness_nm_per_rad: float
    knee_stiffness_nm_per_rad: float
    hip_damping_nm_s_per_rad: float
    knee_damping_nm_s_per_rad: float
    hip_neutral_rad: float
    knee_neutral_rad: float
    hip_cubic_stiffness_nm_per_rad3: float
    knee_cubic_stiffness_nm_per_rad3: float
    coupling_stiffness_nm_per_rad: float
    coupling_ratio: float

    def __post_init__(self) -> None:
        if self.leg_id not in EXPECTED_LEG_IDS:
            raise ValueError(f"unexpected benchmark leg_id {self.leg_id!r}")
        values = tuple(
            float(value)
            for name, value in vars(self).items()
            if name not in {"leg_id", "mechanical_regime"}
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all mechanical and ROM values must be finite")
        if self.hip_min_rad >= self.hip_max_rad:
            raise ValueError("hip ROM must be increasing")
        if self.knee_min_rad >= self.knee_max_rad:
            raise ValueError("knee ROM must be increasing")
        positive = (
            self.mass_thigh_kg,
            self.mass_shank_kg,
            self.com_thigh_m,
            self.com_shank_m,
            self.inertia_thigh_kg_m2,
            self.inertia_shank_kg_m2,
        )
        if not all(value > 0.0 for value in positive):
            raise ValueError("mass, COM, and inertia values must be positive")
        nonnegative = (
            self.hip_stiffness_nm_per_rad,
            self.knee_stiffness_nm_per_rad,
            self.hip_damping_nm_s_per_rad,
            self.knee_damping_nm_s_per_rad,
            self.hip_cubic_stiffness_nm_per_rad3,
            self.knee_cubic_stiffness_nm_per_rad3,
            self.coupling_stiffness_nm_per_rad,
        )
        if not all(value >= 0.0 for value in nonnegative):
            raise ValueError("passive coefficients must be non-negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MechanicalLegDefinition":
        rom = payload["rom_deg"]
        mechanics = payload["mechanics"]
        radians = math.radians
        return cls(
            leg_id=str(payload["leg_id"]),
            mechanical_regime=str(payload["mechanical_regime"]),
            hip_min_rad=radians(float(rom["hip_min"])),
            hip_max_rad=radians(float(rom["hip_max"])),
            knee_min_rad=radians(float(rom["knee_min"])),
            knee_max_rad=radians(float(rom["knee_max"])),
            mass_thigh_kg=float(mechanics["mass_thigh_kg"]),
            mass_shank_kg=float(mechanics["mass_shank_kg"]),
            com_thigh_m=float(mechanics["com_thigh_m"]),
            com_shank_m=float(mechanics["com_shank_m"]),
            inertia_thigh_kg_m2=float(mechanics["inertia_thigh_kg_m2"]),
            inertia_shank_kg_m2=float(mechanics["inertia_shank_kg_m2"]),
            hip_stiffness_nm_per_rad=float(
                mechanics["hip_stiffness_nm_per_rad"]
            ),
            knee_stiffness_nm_per_rad=float(
                mechanics["knee_stiffness_nm_per_rad"]
            ),
            hip_damping_nm_s_per_rad=float(
                mechanics["hip_damping_nm_s_per_rad"]
            ),
            knee_damping_nm_s_per_rad=float(
                mechanics["knee_damping_nm_s_per_rad"]
            ),
            hip_neutral_rad=radians(float(mechanics["hip_neutral_deg"])),
            knee_neutral_rad=radians(float(mechanics["knee_neutral_deg"])),
            hip_cubic_stiffness_nm_per_rad3=float(
                mechanics["hip_cubic_stiffness_nm_per_rad3"]
            ),
            knee_cubic_stiffness_nm_per_rad3=float(
                mechanics["knee_cubic_stiffness_nm_per_rad3"]
            ),
            coupling_stiffness_nm_per_rad=float(
                mechanics["coupling_stiffness_nm_per_rad"]
            ),
            coupling_ratio=float(mechanics["coupling_ratio"]),
        )

    def make_rom_profile(self) -> SubjectROMProfile:
        return SubjectROMProfile(
            profile_id=f"{self.leg_id}_SYNTHETIC_ROM_V1",
            version=1,
            hip_min_rad=self.hip_min_rad,
            hip_max_rad=self.hip_max_rad,
            knee_min_rad=self.knee_min_rad,
            knee_max_rad=self.knee_max_rad,
            rom_status=FROZEN_SUBJECT_ROM_PROFILE_V1,
            provenance="SYNTHETIC_MUJOCO_BENCHMARK_ROM_ONLY",
            frozen=True,
            metadata={
                "benchmark_id": "FIVE_LEG_MUJOCO_MECHANICAL_BENCHMARK_V1",
                "leg_id": self.leg_id,
                "not_clinical_population_evidence": True,
                "parameters_frozen_before_landscape_generation": True,
            },
        )

    def as_table_row(self) -> dict[str, Any]:
        row = dict(vars(self))
        row.update(
            {
                "hip_min_deg": math.degrees(self.hip_min_rad),
                "hip_max_deg": math.degrees(self.hip_max_rad),
                "knee_min_deg": math.degrees(self.knee_min_rad),
                "knee_max_deg": math.degrees(self.knee_max_rad),
                "hip_neutral_deg": math.degrees(self.hip_neutral_rad),
                "knee_neutral_deg": math.degrees(self.knee_neutral_rad),
            }
        )
        for name in (
            "hip_min_rad",
            "hip_max_rad",
            "knee_min_rad",
            "knee_max_rad",
            "hip_neutral_rad",
            "knee_neutral_rad",
        ):
            row.pop(name)
        return row


@dataclass(frozen=True)
class FrozenBenchmarkDefinition:
    benchmark_id: str
    classification: str
    thigh_length_m: float
    shank_length_m: float
    cuff_distance_m: float
    gravity_m_s2: float
    tracking_failure_rms_rad: float
    endpoint_name: str
    endpoint_unit: str
    reference_endpoint_uncertainty_nm: float
    analysis: Mapping[str, Any]
    legs: tuple[MechanicalLegDefinition, ...]


def load_frozen_benchmark_definition() -> FrozenBenchmarkDefinition:
    payload = json.loads(FROZEN_DEFINITION_PATH.read_text(encoding="utf-8"))
    geometry = payload["geometry"]
    replay = payload["replay"]
    endpoint = payload["endpoint"]
    gray_box = payload["gray_box"]
    legs = tuple(MechanicalLegDefinition.from_mapping(item) for item in payload["legs"])
    if tuple(item.leg_id for item in legs) != EXPECTED_LEG_IDS:
        raise RuntimeError("frozen benchmark must contain exactly the five ordered legs")
    if payload["coordinate_convention"]["theta_shank"] != "q_hip - q_knee":
        raise RuntimeError("theta_shank convention changed")
    if replay["mode"] != "EXACT_STATE_MJ_INVERSE":
        raise RuntimeError("unexpected replay mode")
    return FrozenBenchmarkDefinition(
        benchmark_id=str(payload["benchmark_id"]),
        classification=str(payload["classification"]),
        thigh_length_m=float(geometry["thigh_length_m"]),
        shank_length_m=float(geometry["shank_length_m"]),
        cuff_distance_m=float(
            geometry["cuff_interaction_point_distance_from_knee_m"]
        ),
        gravity_m_s2=float(replay["gravity_m_s2"]),
        tracking_failure_rms_rad=float(replay["tracking_failure_rms_rad"]),
        endpoint_name=str(endpoint["name"]),
        endpoint_unit=str(endpoint["unit"]),
        reference_endpoint_uncertainty_nm=float(
            gray_box["reference_endpoint_uncertainty_nm"]
        ),
        analysis=payload["analysis"],
        legs=legs,
    )


def build_mjcf(definition: FrozenBenchmarkDefinition, leg: MechanicalLegDefinition) -> str:
    """Create the fixed-pelvis two-hinge model for one frozen leg."""

    knee_range = (-leg.knee_max_rad, -leg.knee_min_rad)
    inertia_thigh = leg.inertia_thigh_kg_m2
    inertia_shank = leg.inertia_shank_kg_m2
    return f"""
<mujoco model="{leg.leg_id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 {-definition.gravity_m_s2:.17g}" integrator="RK4"/>
  <default>
    <joint armature="0" frictionloss="0"/>
    <geom contype="0" conaffinity="0" rgba="0.55 0.65 0.8 1"/>
  </default>
  <worldbody>
    <geom name="fixed_pelvis" type="box" pos="-0.04 0 0" size="0.04 0.08 0.05"/>
    <body name="thigh" pos="0 0 0">
      <joint name="hip" type="hinge" axis="0 -1 0"
             limited="true" range="{leg.hip_min_rad:.17g} {leg.hip_max_rad:.17g}"
             springref="{leg.hip_neutral_rad:.17g}"
             stiffness="{leg.hip_stiffness_nm_per_rad:.17g}"
             damping="{leg.hip_damping_nm_s_per_rad:.17g}"/>
      <inertial pos="{leg.com_thigh_m:.17g} 0 0" mass="{leg.mass_thigh_kg:.17g}"
                diaginertia="{inertia_thigh:.17g} {inertia_thigh:.17g} {inertia_thigh:.17g}"/>
      <geom name="thigh_segment" type="capsule" fromto="0 0 0 {definition.thigh_length_m:.17g} 0 0" size="0.035" mass="0"/>
      <body name="shank" pos="{definition.thigh_length_m:.17g} 0 0">
        <joint name="knee" type="hinge" axis="0 -1 0"
               limited="true" range="{knee_range[0]:.17g} {knee_range[1]:.17g}"
               springref="{-leg.knee_neutral_rad:.17g}"
               stiffness="{leg.knee_stiffness_nm_per_rad:.17g}"
               damping="{leg.knee_damping_nm_s_per_rad:.17g}"/>
        <inertial pos="{leg.com_shank_m:.17g} 0 0" mass="{leg.mass_shank_kg:.17g}"
                  diaginertia="{inertia_shank:.17g} {inertia_shank:.17g} {inertia_shank:.17g}"/>
        <geom name="shank_segment" type="capsule" fromto="0 0 0 {definition.shank_length_m:.17g} 0 0" size="0.03" mass="0"/>
        <site name="equivalent_cuff_interaction_point" pos="{definition.cuff_distance_m:.17g} 0 0" size="0.012" rgba="0.9 0.3 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_inverse_dynamics_drive" joint="hip" gear="1"/>
    <motor name="knee_inverse_dynamics_drive" joint="knee" gear="1"/>
  </actuator>
</mujoco>
""".strip()


def make_mujoco_model(
    definition: FrozenBenchmarkDefinition,
    leg: MechanicalLegDefinition,
) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(build_mjcf(definition, leg))
    if model.nq != 2 or model.nv != 2 or model.nu != 2:
        raise RuntimeError("benchmark model must have exactly two driven hinge DOFs")
    return model


__all__ = [
    "EXPECTED_LEG_IDS",
    "FROZEN_DEFINITION_PATH",
    "FrozenBenchmarkDefinition",
    "MechanicalLegDefinition",
    "build_mjcf",
    "load_frozen_benchmark_definition",
    "make_mujoco_model",
]
