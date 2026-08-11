"""Deterministic Stage 4.5D geometry and kinematic-observation scenarios.

The first 19 names reproduce the scenarios requested for the main experiment.
Additional explicitly named positive/negative variants support direction
sensitivity analysis.  Segment-length errors are static installation-time
calibration offsets; no scenario models time-varying strap slip.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .geometry_calibration import (
    AssumedGeometry,
    TrueGeometry,
    build_assumed_geometry,
)


ORACLE_TRUE_JOINT_STATE = "oracle_true_joint_state"
TCP_INVERSE_KINEMATICS = "tcp_inverse_kinematics"
INDEPENDENT_JOINT_MEASUREMENT = "independent_joint_measurement"
OBSERVATION_MODES = (
    ORACLE_TRUE_JOINT_STATE,
    TCP_INVERSE_KINEMATICS,
    INDEPENDENT_JOINT_MEASUREMENT,
)

DEFAULT_GEOMETRY_RANDOM_SEED = 20260803
_ALL_MODES = OBSERVATION_MODES
_TCP_ONLY = (TCP_INVERSE_KINEMATICS,)
_INDEPENDENT_ONLY = (INDEPENDENT_JOINT_MEASUREMENT,)


@dataclass(frozen=True, slots=True)
class GeometryErrorScenario:
    """A reproducible static calibration / measurement-error definition.

    Additive geometry errors always mean ``assumed - true``.  Measurement
    standard deviations do not alter geometry; they are consumed by the
    observation layer.  ``applicable_observation_modes`` lets orchestration
    reject meaningless comparisons, such as TCP noise in the oracle mode.
    """

    scenario_name: str
    scenario_category: str
    L1_error_m: float = 0.0
    L2_error_m: float = 0.0
    hip_center_x_error_m: float = 0.0
    hip_center_z_error_m: float = 0.0
    q0_hip_error_rad: float = 0.0
    q0_knee_error_rad: float = 0.0
    tcp_position_noise_std_m: float = 0.0
    independent_angle_noise_std_rad: float = 0.0
    random_seed: int = DEFAULT_GEOMETRY_RANDOM_SEED
    applicable_observation_modes: tuple[str, ...] = _ALL_MODES
    error_direction: str = "matched"
    static_calibration_only: bool = True
    l2_time_varying: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_name, str) or not self.scenario_name.strip():
            raise ValueError("scenario_name must not be empty.")
        if not isinstance(self.scenario_category, str) or not self.scenario_category:
            raise ValueError("scenario_category must not be empty.")

        signed_values = (
            "L1_error_m",
            "L2_error_m",
            "hip_center_x_error_m",
            "hip_center_z_error_m",
            "q0_hip_error_rad",
            "q0_knee_error_rad",
        )
        for name in signed_values:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)

        for name in (
            "tcp_position_noise_std_m",
            "independent_angle_noise_std_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)

        if isinstance(self.random_seed, (bool, np.bool_)) or not isinstance(
            self.random_seed,
            (int, np.integer),
        ):
            raise TypeError("random_seed must be an integer.")
        object.__setattr__(self, "random_seed", int(self.random_seed))

        modes = tuple(self.applicable_observation_modes)
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("applicable_observation_modes must be nonempty and unique.")
        unknown_modes = set(modes).difference(OBSERVATION_MODES)
        if unknown_modes:
            raise ValueError(
                "unknown observation modes: " + ", ".join(sorted(unknown_modes))
            )
        object.__setattr__(self, "applicable_observation_modes", modes)

        # Stage 4.5D is deliberately limited to static L2 calibration error.
        if not self.static_calibration_only or self.l2_time_varying:
            raise ValueError(
                "Stage 4.5D scenarios must remain static; dynamic L2/strap slip "
                "belongs to a later stage."
            )

    @property
    def randomized(self) -> bool:
        """Whether repeated seeds are meaningful for this scenario."""

        return bool(
            self.tcp_position_noise_std_m > 0.0
            or self.independent_angle_noise_std_rad > 0.0
        )

    @property
    def applicable_modes(self) -> tuple[str, ...]:
        """Short alias retained for experiment orchestration."""

        return self.applicable_observation_modes

    def is_applicable_to(self, observation_mode: str) -> bool:
        return observation_mode in self.applicable_observation_modes

    def create_assumed_geometry(
        self,
        true_geometry: TrueGeometry,
    ) -> AssumedGeometry:
        """Create a scalar-only assumption object without retaining truth."""

        return build_assumed_geometry(
            true_geometry,
            L1_error_m=self.L1_error_m,
            L2_error_m=self.L2_error_m,
            hip_center_x_error_m=self.hip_center_x_error_m,
            hip_center_z_error_m=self.hip_center_z_error_m,
            q0_hip_error_rad=self.q0_hip_error_rad,
            q0_knee_error_rad=self.q0_knee_error_rad,
        )

    # Compatibility name that reads naturally in experiment code.
    build_assumed_geometry = create_assumed_geometry

    def with_random_seed(self, random_seed: int) -> GeometryErrorScenario:
        """Return an immutable repeat definition for Monte-Carlo evaluation."""

        return replace(self, random_seed=random_seed)

    def make_random_generator(
        self,
        random_seed: int | None = None,
    ) -> np.random.Generator:
        """Create a fresh deterministic generator.

        Repeated calls with the same scenario/override intentionally reproduce
        the same observations instead of sharing mutable RNG state.
        """

        seed = self.random_seed if random_seed is None else random_seed
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed,
            (int, np.integer),
        ):
            raise TypeError("random_seed must be an integer.")
        return np.random.default_rng(int(seed))

    def sample_tcp_position_noise(
        self,
        size: int | tuple[int, ...],
        *,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample independent world-x/world-z TCP noise in metres."""

        generator = self.make_random_generator(random_seed)
        shape = (2,) + ((size,) if isinstance(size, int) else tuple(size))
        noise = generator.normal(
            loc=0.0,
            scale=self.tcp_position_noise_std_m,
            size=shape,
        )
        return noise[0], noise[1]

    def sample_independent_angle_noise(
        self,
        size: int | tuple[int, ...],
        *,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample independent hip/knee measurement noise in radians."""

        generator = self.make_random_generator(random_seed)
        shape = (2,) + ((size,) if isinstance(size, int) else tuple(size))
        noise = generator.normal(
            loc=0.0,
            scale=self.independent_angle_noise_std_rad,
            size=shape,
        )
        return noise[0], noise[1]

    def apply_tcp_position_noise(
        self,
        x_pull_m: float | np.ndarray,
        z_pull_m: float | np.ndarray,
        *,
        random_seed: int | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Apply reproducible noise while preserving broadcasted input shape."""

        x, z = np.broadcast_arrays(
            np.asarray(x_pull_m, dtype=float),
            np.asarray(z_pull_m, dtype=float),
        )
        noise_x, noise_z = self.sample_tcp_position_noise(
            x.shape,
            random_seed=random_seed,
        )
        x_noisy = x + noise_x
        z_noisy = z + noise_z
        if x.ndim == 0:
            return float(x_noisy), float(z_noisy)
        return x_noisy, z_noisy

    def apply_independent_angle_noise(
        self,
        q_hip_rad: float | np.ndarray,
        q_knee_rad: float | np.ndarray,
        *,
        random_seed: int | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Apply reproducible independent joint-angle noise."""

        q_hip, q_knee = np.broadcast_arrays(
            np.asarray(q_hip_rad, dtype=float),
            np.asarray(q_knee_rad, dtype=float),
        )
        hip_noise, knee_noise = self.sample_independent_angle_noise(
            q_hip.shape,
            random_seed=random_seed,
        )
        hip_noisy = q_hip + hip_noise
        knee_noisy = q_knee + knee_noise
        if q_hip.ndim == 0:
            return float(hip_noisy), float(knee_noisy)
        return hip_noisy, knee_noisy

    def as_metadata_dict(self) -> dict[str, object]:
        """Return JSON-safe scenario configuration, including applicability."""

        metadata = asdict(self)
        metadata["applicable_observation_modes"] = list(
            self.applicable_observation_modes
        )
        metadata["randomized"] = self.randomized
        metadata["angle_definition"] = "theta_shank = q_hip - q_knee"
        return metadata


def _scenario(
    scenario_name: str,
    scenario_category: str,
    *,
    seed_offset: int,
    applicable_observation_modes: tuple[str, ...] = _ALL_MODES,
    error_direction: str = "matched",
    **kwargs: float,
) -> GeometryErrorScenario:
    return GeometryErrorScenario(
        scenario_name=scenario_name,
        scenario_category=scenario_category,
        random_seed=DEFAULT_GEOMETRY_RANDOM_SEED + seed_offset,
        applicable_observation_modes=applicable_observation_modes,
        error_direction=error_direction,
        **kwargs,
    )


_DEG = np.pi / 180.0

# The exact 19 main scenario names from the Stage 4.5D specification.
_BASE_SCENARIOS = (
    _scenario("matched_geometry", "matched", seed_offset=0),
    _scenario(
        "L1_error_1cm", "segment_length", seed_offset=1,
        L1_error_m=0.01, error_direction="positive",
    ),
    _scenario(
        "L1_error_2cm", "segment_length", seed_offset=2,
        L1_error_m=0.02, error_direction="positive",
    ),
    _scenario(
        "L2_error_1cm", "pull_point_calibration", seed_offset=3,
        L2_error_m=0.01, error_direction="positive",
    ),
    _scenario(
        "L2_error_2cm", "pull_point_calibration", seed_offset=4,
        L2_error_m=0.02, error_direction="positive",
    ),
    _scenario(
        "L2_error_3cm", "pull_point_calibration", seed_offset=5,
        L2_error_m=0.03, error_direction="positive",
    ),
    _scenario(
        "hip_center_x_error_1cm", "hip_center", seed_offset=6,
        hip_center_x_error_m=0.01, error_direction="positive_x",
    ),
    _scenario(
        "hip_center_z_error_1cm", "hip_center", seed_offset=7,
        hip_center_z_error_m=0.01, error_direction="positive_z",
    ),
    _scenario(
        "hip_center_combined_error_2cm", "hip_center", seed_offset=8,
        hip_center_x_error_m=0.02, hip_center_z_error_m=0.02,
        error_direction="positive_x_positive_z",
    ),
    _scenario(
        "q0_error_3deg", "neutral_angle", seed_offset=9,
        q0_hip_error_rad=3.0 * _DEG, q0_knee_error_rad=3.0 * _DEG,
        error_direction="positive",
    ),
    _scenario(
        "q0_error_5deg", "neutral_angle", seed_offset=10,
        q0_hip_error_rad=5.0 * _DEG, q0_knee_error_rad=5.0 * _DEG,
        error_direction="positive",
    ),
    _scenario(
        "tcp_position_noise_low", "tcp_measurement_noise", seed_offset=11,
        tcp_position_noise_std_m=0.001,
        applicable_observation_modes=_TCP_ONLY, error_direction="stochastic",
    ),
    _scenario(
        "tcp_position_noise_medium", "tcp_measurement_noise", seed_offset=12,
        tcp_position_noise_std_m=0.003,
        applicable_observation_modes=_TCP_ONLY, error_direction="stochastic",
    ),
    _scenario(
        "tcp_position_noise_high", "tcp_measurement_noise", seed_offset=13,
        tcp_position_noise_std_m=0.005,
        applicable_observation_modes=_TCP_ONLY, error_direction="stochastic",
    ),
    _scenario(
        "independent_angle_noise_low", "independent_angle_noise", seed_offset=14,
        independent_angle_noise_std_rad=0.5 * _DEG,
        applicable_observation_modes=_INDEPENDENT_ONLY,
        error_direction="stochastic",
    ),
    _scenario(
        "independent_angle_noise_medium", "independent_angle_noise", seed_offset=15,
        independent_angle_noise_std_rad=1.0 * _DEG,
        applicable_observation_modes=_INDEPENDENT_ONLY,
        error_direction="stochastic",
    ),
    _scenario(
        "independent_angle_noise_high", "independent_angle_noise", seed_offset=16,
        independent_angle_noise_std_rad=2.0 * _DEG,
        applicable_observation_modes=_INDEPENDENT_ONLY,
        error_direction="stochastic",
    ),
    _scenario(
        "combined_geometry_mild", "combined", seed_offset=17,
        L1_error_m=0.005, L2_error_m=-0.005,
        hip_center_x_error_m=0.005, hip_center_z_error_m=-0.005,
        q0_hip_error_rad=3.0 * _DEG, q0_knee_error_rad=3.0 * _DEG,
        tcp_position_noise_std_m=0.001,
        independent_angle_noise_std_rad=0.5 * _DEG,
        error_direction="mixed",
    ),
    _scenario(
        "combined_geometry_strong", "combined", seed_offset=18,
        L1_error_m=0.02, L2_error_m=-0.03,
        hip_center_x_error_m=0.02, hip_center_z_error_m=-0.02,
        q0_hip_error_rad=5.0 * _DEG, q0_knee_error_rad=5.0 * _DEG,
        tcp_position_noise_std_m=0.005,
        independent_angle_noise_std_rad=2.0 * _DEG,
        error_direction="mixed",
    ),
)


# Explicit direction variants.  Base names above retain their specified,
# deterministic positive convention for stable CLI use.
_SENSITIVITY_VARIANTS = (
    _scenario(
        "L1_error_1cm_positive", "segment_length", seed_offset=101,
        L1_error_m=0.01, error_direction="positive",
    ),
    _scenario(
        "L1_error_1cm_negative", "segment_length", seed_offset=102,
        L1_error_m=-0.01, error_direction="negative",
    ),
    _scenario(
        "L1_error_2cm_positive", "segment_length", seed_offset=103,
        L1_error_m=0.02, error_direction="positive",
    ),
    _scenario(
        "L1_error_2cm_negative", "segment_length", seed_offset=104,
        L1_error_m=-0.02, error_direction="negative",
    ),
    _scenario(
        "L2_error_1cm_positive", "pull_point_calibration", seed_offset=105,
        L2_error_m=0.01, error_direction="positive",
    ),
    _scenario(
        "L2_error_1cm_negative", "pull_point_calibration", seed_offset=106,
        L2_error_m=-0.01, error_direction="negative",
    ),
    _scenario(
        "L2_error_2cm_positive", "pull_point_calibration", seed_offset=107,
        L2_error_m=0.02, error_direction="positive",
    ),
    _scenario(
        "L2_error_2cm_negative", "pull_point_calibration", seed_offset=108,
        L2_error_m=-0.02, error_direction="negative",
    ),
    _scenario(
        "L2_error_3cm_positive", "pull_point_calibration", seed_offset=109,
        L2_error_m=0.03, error_direction="positive",
    ),
    _scenario(
        "L2_error_3cm_negative", "pull_point_calibration", seed_offset=110,
        L2_error_m=-0.03, error_direction="negative",
    ),
    _scenario(
        "hip_center_x_error_1cm_positive", "hip_center", seed_offset=111,
        hip_center_x_error_m=0.01, error_direction="positive_x",
    ),
    _scenario(
        "hip_center_x_error_1cm_negative", "hip_center", seed_offset=112,
        hip_center_x_error_m=-0.01, error_direction="negative_x",
    ),
    _scenario(
        "hip_center_z_error_1cm_positive", "hip_center", seed_offset=113,
        hip_center_z_error_m=0.01, error_direction="positive_z",
    ),
    _scenario(
        "hip_center_z_error_1cm_negative", "hip_center", seed_offset=114,
        hip_center_z_error_m=-0.01, error_direction="negative_z",
    ),
    _scenario(
        "hip_center_combined_error_2cm_positive", "hip_center", seed_offset=115,
        hip_center_x_error_m=0.02, hip_center_z_error_m=0.02,
        error_direction="positive_x_positive_z",
    ),
    _scenario(
        "hip_center_combined_error_2cm_negative", "hip_center", seed_offset=116,
        hip_center_x_error_m=-0.02, hip_center_z_error_m=-0.02,
        error_direction="negative_x_negative_z",
    ),
)


BASE_GEOMETRY_ERROR_SCENARIOS = tuple(
    scenario.scenario_name for scenario in _BASE_SCENARIOS
)
GEOMETRY_SENSITIVITY_VARIANTS = tuple(
    scenario.scenario_name for scenario in _SENSITIVITY_VARIANTS
)

_ALL_SCENARIO_OBJECTS = _BASE_SCENARIOS + _SENSITIVITY_VARIANTS
GEOMETRY_ERROR_SCENARIO_DEFINITIONS: Mapping[str, GeometryErrorScenario] = (
    MappingProxyType(
        {scenario.scenario_name: scenario for scenario in _ALL_SCENARIO_OBJECTS}
    )
)
GEOMETRY_ERROR_SCENARIOS = tuple(GEOMETRY_ERROR_SCENARIO_DEFINITIONS)
# Lowercase compatibility name matching prior stage scenario modules.
geometry_error_scenarios = GEOMETRY_ERROR_SCENARIO_DEFINITIONS


def get_geometry_error_scenario(scenario_name: str) -> GeometryErrorScenario:
    """Return a read-only scenario definition by its stable CLI name."""

    try:
        return GEOMETRY_ERROR_SCENARIO_DEFINITIONS[scenario_name]
    except KeyError as exc:
        choices = ", ".join(GEOMETRY_ERROR_SCENARIOS)
        raise ValueError(
            f"Unknown geometry error scenario {scenario_name!r}; choose one of: "
            f"{choices}."
        ) from exc


def build_scenario_assumed_geometry(
    true_geometry: TrueGeometry,
    scenario_name: str,
) -> AssumedGeometry:
    """Create estimator-visible geometry for one named scenario."""

    return get_geometry_error_scenario(scenario_name).create_assumed_geometry(
        true_geometry
    )


# Concise alias for orchestration code.
build_assumed_geometry_for_scenario = build_scenario_assumed_geometry


__all__ = [
    "BASE_GEOMETRY_ERROR_SCENARIOS",
    "DEFAULT_GEOMETRY_RANDOM_SEED",
    "GEOMETRY_ERROR_SCENARIOS",
    "GEOMETRY_ERROR_SCENARIO_DEFINITIONS",
    "GEOMETRY_SENSITIVITY_VARIANTS",
    "INDEPENDENT_JOINT_MEASUREMENT",
    "OBSERVATION_MODES",
    "ORACLE_TRUE_JOINT_STATE",
    "TCP_INVERSE_KINEMATICS",
    "GeometryErrorScenario",
    "build_assumed_geometry_for_scenario",
    "build_scenario_assumed_geometry",
    "geometry_error_scenarios",
    "get_geometry_error_scenario",
]
