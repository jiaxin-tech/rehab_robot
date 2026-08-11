"""Pure offline H -> B -> T coordinate transforms for Stage 6A.

Frames
------
H
    Human local sagittal frame.  Pull points are ``[x_pull, 0, z_pull]``.
B
    Robot Base frame.  Its laboratory placement must be supplied explicitly.
T
    TCP/tool frame.  The fixed tool offset is expressed in this frame.

No robot SDK, connection, control or safety module is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


MODEL_ANGLE_DEFINITION = "theta_shank = q_hip - q_knee"
DEFAULT_ORTHOGONALITY_TOLERANCE = 1e-6
MAXIMUM_ORTHOGONALITY_TOLERANCE = 1e-4
SUPPORTED_ORIENTATION_REPRESENTATIONS = (
    "rotation_vector_rad",
    "euler_xyz_rad",
)


def _finite_vector3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain exactly three finite values.")
    return tuple(float(value) for value in vector)


def _finite_increasing_range2(
    values: Sequence[float], name: str
) -> tuple[float, float]:
    limits = np.asarray(values, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all():
        raise ValueError(f"{name} must contain exactly two finite values.")
    if not bool(limits[0] < limits[1]):
        raise ValueError(f"{name} must be strictly increasing.")
    return float(limits[0]), float(limits[1])


@dataclass(frozen=True)
class TcpOrientation:
    """Explicit fixed TCP orientation and its mathematical representation."""

    representation: str
    values_rad: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.representation not in SUPPORTED_ORIENTATION_REPRESENTATIONS:
            raise ValueError(
                "tcp_orientation representation must be one of "
                f"{SUPPORTED_ORIENTATION_REPRESENTATIONS}."
            )
        object.__setattr__(
            self,
            "values_rad",
            _finite_vector3(self.values_rad, "tcp_orientation.values_rad"),
        )

    @property
    def rotation_base_from_tcp(self) -> np.ndarray:
        values = np.asarray(self.values_rad, dtype=float)
        if self.representation == "rotation_vector_rad":
            return Rotation.from_rotvec(values).as_matrix()
        return Rotation.from_euler("xyz", values, degrees=False).as_matrix()

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "values_rad": list(self.values_rad),
            "vendor_pose_semantics_verified": False,
        }


@dataclass(frozen=True)
class RobotFrameCalibration:
    """All experiment-specific Stage-6A transform values.

    ``tool_offset_m`` is the vector from the TCP origin to the strap connection
    point, expressed in T.  Consequently

    ``p_tcp_B = p_pull_B - R_B_T @ tool_offset_T``.
    """

    hip_center_in_base_m: tuple[float, float, float]
    human_x_axis_in_base: tuple[float, float, float]
    human_z_axis_in_base: tuple[float, float, float]
    tool_offset_m: tuple[float, float, float]
    tcp_orientation: TcpOrientation
    approved_hip_rom_deg: tuple[float, float]
    approved_knee_rom_deg: tuple[float, float]
    reviewed: bool
    notes: str = ""
    orthogonality_tolerance: float = DEFAULT_ORTHOGONALITY_TOLERANCE

    def __post_init__(self) -> None:
        # This is calibration review, not robot-execution approval.  Exact bool
        # checking deliberately rejects truthy strings and integers.
        if type(self.reviewed) is not bool or not self.reviewed:
            raise ValueError(
                "calibration reviewed must be the boolean true before Stage 6A "
                "may generate a Base/TCP CSV."
            )
        for field_name in (
            "hip_center_in_base_m",
            "human_x_axis_in_base",
            "human_z_axis_in_base",
            "tool_offset_m",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_vector3(getattr(self, field_name), field_name),
            )
        for field_name in ("approved_hip_rom_deg", "approved_knee_rom_deg"):
            object.__setattr__(
                self,
                field_name,
                _finite_increasing_range2(getattr(self, field_name), field_name),
            )
        if not isinstance(self.notes, str):
            raise ValueError("calibration notes must be a string.")
        tolerance = float(self.orthogonality_tolerance)
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0.0
            or tolerance > MAXIMUM_ORTHOGONALITY_TOLERANCE
        ):
            raise ValueError(
                "orthogonality_tolerance must be finite, positive, and no greater "
                f"than {MAXIMUM_ORTHOGONALITY_TOLERANCE:g}."
            )
        object.__setattr__(self, "orthogonality_tolerance", tolerance)

        x_axis = np.asarray(self.human_x_axis_in_base, dtype=float)
        z_axis = np.asarray(self.human_z_axis_in_base, dtype=float)
        norm_errors = np.abs(
            np.asarray((np.linalg.norm(x_axis), np.linalg.norm(z_axis))) - 1.0
        )
        perpendicular_error = abs(float(np.dot(x_axis, z_axis)))
        if bool(np.any(norm_errors > tolerance)):
            raise ValueError(
                "human x/z calibration axes must be unit vectors; they are not "
                "silently normalized."
            )
        if perpendicular_error > tolerance:
            raise ValueError(
                "human x/z calibration axes must be orthogonal; they are not "
                "silently orthogonalized."
            )
        rotation = self.rotation_base_from_human
        orthogonality_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
        )
        determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
        if orthogonality_error > tolerance or determinant_error > tolerance:
            raise ValueError("R_base_from_human is not a right-handed rotation matrix.")

    @property
    def human_y_axis_in_base(self) -> np.ndarray:
        # For a right-handed H frame: x_H cross y_H = z_H, hence y_H = z_H cross x_H.
        return np.cross(
            np.asarray(self.human_z_axis_in_base, dtype=float),
            np.asarray(self.human_x_axis_in_base, dtype=float),
        )

    @property
    def rotation_base_from_human(self) -> np.ndarray:
        return np.column_stack(
            (
                np.asarray(self.human_x_axis_in_base, dtype=float),
                self.human_y_axis_in_base,
                np.asarray(self.human_z_axis_in_base, dtype=float),
            )
        )

    @property
    def rotation_base_from_tcp(self) -> np.ndarray:
        return self.tcp_orientation.rotation_base_from_tcp

    @property
    def transform_orthogonality_error(self) -> float:
        rotation = self.rotation_base_from_human
        return float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))

    @property
    def transform_determinant(self) -> float:
        return float(np.linalg.det(self.rotation_base_from_human))

    @property
    def transform_is_orthogonal(self) -> bool:
        return bool(
            self.transform_orthogonality_error <= self.orthogonality_tolerance
            and abs(self.transform_determinant - 1.0) <= self.orthogonality_tolerance
        )

    def as_metadata_dict(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "hip_center_in_base_m": list(self.hip_center_in_base_m),
            "human_x_axis_in_base": list(self.human_x_axis_in_base),
            "human_y_axis_in_base_derived": self.human_y_axis_in_base.tolist(),
            "human_z_axis_in_base": list(self.human_z_axis_in_base),
            "R_base_from_human": self.rotation_base_from_human.tolist(),
            "tool_offset_m": list(self.tool_offset_m),
            "tool_offset_definition": (
                "vector_from_tcp_origin_to_strap_connection_point_expressed_in_T"
            ),
            "tcp_orientation": self.tcp_orientation.as_metadata_dict(),
            "R_base_from_tcp": self.rotation_base_from_tcp.tolist(),
            "orthogonality_tolerance": self.orthogonality_tolerance,
            "transform_orthogonality_error": self.transform_orthogonality_error,
            "transform_determinant": self.transform_determinant,
            "transform_is_orthogonal": self.transform_is_orthogonal,
            "reviewed": self.reviewed,
            "review_status_meaning": (
                "offline_H_B_T_calibration_review_only_not_robot_execution_approval"
            ),
            "notes": self.notes,
            "laboratory_coordinates_hardcoded": False,
        }
        metadata["approved_hip_rom_deg"] = list(self.approved_hip_rom_deg)
        metadata["approved_knee_rom_deg"] = list(self.approved_knee_rom_deg)
        return metadata


def validate_calibration_mapping(
    mapping: Mapping[str, object],
) -> RobotFrameCalibration:
    """Validate reviewed calibration without fallback laboratory values.

    A template containing ``null`` values or ``reviewed=false`` is intentionally
    not loadable.  This prevents both the JSON path and direct mapping callers
    from converting an unfinished worksheet into a trajectory export.
    """

    required = {
        "hip_center_in_base_m",
        "human_x_axis_in_base",
        "human_z_axis_in_base",
        "tool_offset_m",
        "tcp_orientation",
        "approved_hip_rom_deg",
        "approved_knee_rom_deg",
        "reviewed",
        "notes",
    }
    missing = required.difference(mapping)
    if missing:
        raise ValueError(f"calibration is missing required fields: {sorted(missing)}")
    if type(mapping["reviewed"]) is not bool or mapping["reviewed"] is not True:
        raise ValueError(
            "calibration reviewed must be the boolean true before Stage 6A "
            "may generate a Base/TCP CSV."
        )
    notes = mapping.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("calibration notes must be a string.")
    orientation_value = mapping["tcp_orientation"]
    if not isinstance(orientation_value, Mapping):
        raise ValueError(
            "tcp_orientation must be an object with representation/values_rad."
        )
    if (
        "representation" not in orientation_value
        or "values_rad" not in orientation_value
    ):
        raise ValueError("tcp_orientation requires representation and values_rad.")
    return RobotFrameCalibration(
        hip_center_in_base_m=_finite_vector3(
            mapping["hip_center_in_base_m"],  # type: ignore[arg-type]
            "hip_center_in_base_m",
        ),
        human_x_axis_in_base=_finite_vector3(
            mapping["human_x_axis_in_base"],  # type: ignore[arg-type]
            "human_x_axis_in_base",
        ),
        human_z_axis_in_base=_finite_vector3(
            mapping["human_z_axis_in_base"],  # type: ignore[arg-type]
            "human_z_axis_in_base",
        ),
        tool_offset_m=_finite_vector3(
            mapping["tool_offset_m"],  # type: ignore[arg-type]
            "tool_offset_m",
        ),
        tcp_orientation=TcpOrientation(
            representation=str(orientation_value["representation"]),
            values_rad=_finite_vector3(
                orientation_value["values_rad"],  # type: ignore[arg-type]
                "tcp_orientation.values_rad",
            ),
        ),
        approved_hip_rom_deg=_finite_increasing_range2(
            mapping["approved_hip_rom_deg"],  # type: ignore[arg-type]
            "approved_hip_rom_deg",
        ),
        approved_knee_rom_deg=_finite_increasing_range2(
            mapping["approved_knee_rom_deg"],  # type: ignore[arg-type]
            "approved_knee_rom_deg",
        ),
        reviewed=True,
        notes=notes,
        orthogonality_tolerance=float(
            mapping.get(
                "orthogonality_tolerance",
                DEFAULT_ORTHOGONALITY_TOLERANCE,
            )
        ),
    )


def calibration_from_mapping(mapping: Mapping[str, object]) -> RobotFrameCalibration:
    """Backward-compatible name for the strict calibration validator."""

    return validate_calibration_mapping(mapping)


def load_calibration_json(path: str | Path) -> RobotFrameCalibration:
    calibration_path = Path(path)
    with calibration_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("calibration JSON must contain an object.")
    return validate_calibration_mapping(payload)


def human_pull_points_to_base(
    x_pull_human_m: float | np.ndarray,
    z_pull_human_m: float | np.ndarray,
    calibration: RobotFrameCalibration,
) -> np.ndarray:
    """Map H-frame pull points to B with same-direction human axes.

    ``human_x_axis_in_base`` and ``human_z_axis_in_base`` are, by definition,
    the directions of positive H axes expressed in B.  Therefore the only
    consistent point transform is ``hip_B + R_B_H @ [x_H, 0, z_H]``.
    """

    x_values, z_values = np.broadcast_arrays(
        np.asarray(x_pull_human_m, dtype=float),
        np.asarray(z_pull_human_m, dtype=float),
    )
    points_human = np.stack((x_values, np.zeros_like(x_values), z_values), axis=-1)
    hip_center = np.asarray(calibration.hip_center_in_base_m, dtype=float)
    return hip_center + points_human @ calibration.rotation_base_from_human.T


def pull_points_base_to_tcp_origins(
    pull_points_base_m: np.ndarray,
    calibration: RobotFrameCalibration,
) -> np.ndarray:
    """Remove the fixed T-frame tool offset to obtain TCP origins in B."""

    points = np.asarray(pull_points_base_m, dtype=float)
    if points.shape[-1:] != (3,):
        raise ValueError("pull_points_base_m must end in a three-coordinate axis.")
    offset_base = calibration.rotation_base_from_tcp @ np.asarray(
        calibration.tool_offset_m, dtype=float
    )
    return points - offset_base


def tcp_origins_to_pull_points_base(
    tcp_origins_base_m: np.ndarray,
    calibration: RobotFrameCalibration,
) -> np.ndarray:
    """Reconstruct the strap point for tool-offset application auditing."""

    points = np.asarray(tcp_origins_base_m, dtype=float)
    if points.shape[-1:] != (3,):
        raise ValueError("tcp_origins_base_m must end in a three-coordinate axis.")
    offset_base = calibration.rotation_base_from_tcp @ np.asarray(
        calibration.tool_offset_m, dtype=float
    )
    return points + offset_base


__all__ = [
    "DEFAULT_ORTHOGONALITY_TOLERANCE",
    "MAXIMUM_ORTHOGONALITY_TOLERANCE",
    "MODEL_ANGLE_DEFINITION",
    "RobotFrameCalibration",
    "SUPPORTED_ORIENTATION_REPRESENTATIONS",
    "TcpOrientation",
    "calibration_from_mapping",
    "human_pull_points_to_base",
    "load_calibration_json",
    "pull_points_base_to_tcp_origins",
    "tcp_origins_to_pull_points_base",
    "validate_calibration_mapping",
]
