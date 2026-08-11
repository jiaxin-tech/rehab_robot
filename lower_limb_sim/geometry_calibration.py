"""Stage 4.5D geometry definitions with an explicit truth/assumption boundary.

The two geometry classes intentionally use different field names.  Simulation
code may create an :class:`AssumedGeometry` from a :class:`TrueGeometry`, but
the returned object contains copied scalar values only: it has no reference to
the truth object and exposes no ``*_true_*`` attributes.  Estimation code must
accept ``AssumedGeometry`` rather than ``TrueGeometry``.

All lengths and coordinates use metres; all angles use radians.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from .config import L1, L2


class _SubjectWithNeutralAngles(Protocol):
    """Structural type needed to build simulation truth geometry."""

    q0_hip_rad: float
    q0_knee_rad: float


def _finite_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _positive_length(name: str, value: float) -> float:
    value = _finite_scalar(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class TrueGeometry:
    """Generator-only human geometry used to create virtual observations.

    This object belongs on the simulation/evaluation side of the data boundary.
    It must never be supplied to angle reconstruction or parameter fitting.
    """

    L1_true_m: float
    L2_true_m: float
    hip_center_x_true_m: float
    hip_center_z_true_m: float
    q0_hip_true_rad: float
    q0_knee_true_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "L1_true_m",
            _positive_length("L1_true_m", self.L1_true_m),
        )
        object.__setattr__(
            self,
            "L2_true_m",
            _positive_length("L2_true_m", self.L2_true_m),
        )
        for name in (
            "hip_center_x_true_m",
            "hip_center_z_true_m",
            "q0_hip_true_rad",
            "q0_knee_true_rad",
        ):
            object.__setattr__(self, name, _finite_scalar(name, getattr(self, name)))

    # The neutral aliases let generator-side helpers share numerical code while
    # keeping the serialized truth field names explicit.
    @property
    def L1_m(self) -> float:
        return self.L1_true_m

    @property
    def L2_m(self) -> float:
        return self.L2_true_m

    @property
    def hip_center_x_m(self) -> float:
        return self.hip_center_x_true_m

    @property
    def hip_center_z_m(self) -> float:
        return self.hip_center_z_true_m

    @property
    def q0_hip_rad(self) -> float:
        return self.q0_hip_true_rad

    @property
    def q0_knee_rad(self) -> float:
        return self.q0_knee_true_rad

    def as_metadata_dict(self) -> dict[str, float]:
        """Return generator/evaluation metadata using explicit truth names."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class AssumedGeometry:
    """Geometry visible to reconstruction and the five-parameter estimator.

    No field, property, or hidden reference exposes the simulation truth.  The
    generic aliases below return only the assumed values stored in this object.
    """

    L1_assumed_m: float
    L2_assumed_m: float
    hip_center_x_assumed_m: float
    hip_center_z_assumed_m: float
    q0_hip_assumed_rad: float
    q0_knee_assumed_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "L1_assumed_m",
            _positive_length("L1_assumed_m", self.L1_assumed_m),
        )
        object.__setattr__(
            self,
            "L2_assumed_m",
            _positive_length("L2_assumed_m", self.L2_assumed_m),
        )
        for name in (
            "hip_center_x_assumed_m",
            "hip_center_z_assumed_m",
            "q0_hip_assumed_rad",
            "q0_knee_assumed_rad",
        ):
            object.__setattr__(self, name, _finite_scalar(name, getattr(self, name)))

    @property
    def L1_m(self) -> float:
        return self.L1_assumed_m

    @property
    def L2_m(self) -> float:
        return self.L2_assumed_m

    @property
    def hip_center_x_m(self) -> float:
        return self.hip_center_x_assumed_m

    @property
    def hip_center_z_m(self) -> float:
        return self.hip_center_z_assumed_m

    @property
    def q0_hip_rad(self) -> float:
        return self.q0_hip_assumed_rad

    @property
    def q0_knee_rad(self) -> float:
        return self.q0_knee_assumed_rad

    def as_metadata_dict(self) -> dict[str, float]:
        """Return only estimator-visible assumed geometry values."""

        return asdict(self)


def create_true_geometry(
    subject: _SubjectWithNeutralAngles,
    *,
    L1_true_m: float = L1,
    L2_true_m: float = L2,
    hip_center_x_true_m: float = 0.0,
    hip_center_z_true_m: float = 0.0,
) -> TrueGeometry:
    """Create generator truth geometry from an existing virtual subject.

    The function is intentionally generator-facing.  Downstream reconstruction
    should receive the separately created :class:`AssumedGeometry` only.
    """

    return TrueGeometry(
        L1_true_m=L1_true_m,
        L2_true_m=L2_true_m,
        hip_center_x_true_m=hip_center_x_true_m,
        hip_center_z_true_m=hip_center_z_true_m,
        q0_hip_true_rad=subject.q0_hip_rad,
        q0_knee_true_rad=subject.q0_knee_rad,
    )


# Descriptive alias used by experiment orchestration code.
true_geometry_from_dynamic_subject = create_true_geometry


def build_assumed_geometry(
    true_geometry: TrueGeometry,
    *,
    L1_error_m: float = 0.0,
    L2_error_m: float = 0.0,
    hip_center_x_error_m: float = 0.0,
    hip_center_z_error_m: float = 0.0,
    q0_hip_error_rad: float = 0.0,
    q0_knee_error_rad: float = 0.0,
) -> AssumedGeometry:
    """Copy truth scalars and apply additive calibration errors.

    Every error is defined as ``assumed - true``.  Positive hip-centre errors
    therefore move the assumed origin in the positive world ``x`` or ``z``
    direction.  The result does not retain ``true_geometry``.
    """

    errors = {
        "L1_error_m": L1_error_m,
        "L2_error_m": L2_error_m,
        "hip_center_x_error_m": hip_center_x_error_m,
        "hip_center_z_error_m": hip_center_z_error_m,
        "q0_hip_error_rad": q0_hip_error_rad,
        "q0_knee_error_rad": q0_knee_error_rad,
    }
    errors = {name: _finite_scalar(name, value) for name, value in errors.items()}
    return AssumedGeometry(
        L1_assumed_m=true_geometry.L1_true_m + errors["L1_error_m"],
        L2_assumed_m=true_geometry.L2_true_m + errors["L2_error_m"],
        hip_center_x_assumed_m=(
            true_geometry.hip_center_x_true_m
            + errors["hip_center_x_error_m"]
        ),
        hip_center_z_assumed_m=(
            true_geometry.hip_center_z_true_m
            + errors["hip_center_z_error_m"]
        ),
        q0_hip_assumed_rad=(
            true_geometry.q0_hip_true_rad + errors["q0_hip_error_rad"]
        ),
        q0_knee_assumed_rad=(
            true_geometry.q0_knee_true_rad + errors["q0_knee_error_rad"]
        ),
    )


def matched_assumed_geometry(true_geometry: TrueGeometry) -> AssumedGeometry:
    """Return a value-copy with no calibration error."""

    return build_assumed_geometry(true_geometry)


def calibration_error_from_geometries(
    true_geometry: TrueGeometry,
    assumed_geometry: AssumedGeometry,
) -> dict[str, float]:
    """Compute additive errors for final evaluation and audit only."""

    return {
        "L1_error_m": assumed_geometry.L1_assumed_m - true_geometry.L1_true_m,
        "L2_error_m": assumed_geometry.L2_assumed_m - true_geometry.L2_true_m,
        "hip_center_x_error_m": (
            assumed_geometry.hip_center_x_assumed_m
            - true_geometry.hip_center_x_true_m
        ),
        "hip_center_z_error_m": (
            assumed_geometry.hip_center_z_assumed_m
            - true_geometry.hip_center_z_true_m
        ),
        "q0_hip_error_rad": (
            assumed_geometry.q0_hip_assumed_rad - true_geometry.q0_hip_true_rad
        ),
        "q0_knee_error_rad": (
            assumed_geometry.q0_knee_assumed_rad - true_geometry.q0_knee_true_rad
        ),
    }


__all__ = [
    "AssumedGeometry",
    "TrueGeometry",
    "build_assumed_geometry",
    "calibration_error_from_geometries",
    "create_true_geometry",
    "matched_assumed_geometry",
    "true_geometry_from_dynamic_subject",
]
