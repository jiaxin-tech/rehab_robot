"""Import an external bilateral-leg marker trajectory into the 2-D leg model.

Stage 5A treats a CSV marker trajectory as a kinematic *reference* only.  This
module validates its frame index and six lower-limb landmarks, performs an
explicit millimetre/metre conversion, constructs a subject-local sagittal
frame, and extracts hip/knee angles.  It does not import dynamics, force,
robot-control, acquisition, safety, or hardware modules.

The angle convention remains strictly::

    theta_shank = q_hip - q_knee

The tracked ankle is retained as an observed anatomical marker.  It is never
renamed or silently treated as the strap-equivalent pull point used by the
existing lower-limb simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .config import hip_range_deg, knee_range_deg


CoordinateUnit = Literal["mm", "m"]
MotionLeg = Literal["left", "right"]

LANDMARKS = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")
REQUIRED_COLUMNS = (
    "Frame",
    *(f"{landmark}_{axis}" for landmark in LANDMARKS for axis in "XYZ"),
)
UNIT_SCALE_TO_METRES: dict[str, float] = {"mm": 1e-3, "m": 1.0}


@dataclass(frozen=True)
class FrameContinuityAudit:
    first_frame: int
    last_frame: int
    sample_count: int
    strictly_increasing: bool
    continuous_unit_steps: bool
    gap_count: int
    missing_frame_count: int
    largest_step: int
    gap_after_frames: tuple[int, ...]

    def as_metadata_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentLengthSummary:
    mean_m: float
    median_m: float
    standard_deviation_m: float
    minimum_m: float
    maximum_m: float

    def as_metadata_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BilateralSegmentLengths:
    left_thigh: SegmentLengthSummary
    left_shank_to_ankle: SegmentLengthSummary
    right_thigh: SegmentLengthSummary
    right_shank_to_ankle: SegmentLengthSummary

    def for_leg(
        self,
        leg: MotionLeg,
    ) -> tuple[SegmentLengthSummary, SegmentLengthSummary]:
        if leg == "left":
            return self.left_thigh, self.left_shank_to_ankle
        if leg == "right":
            return self.right_thigh, self.right_shank_to_ankle
        raise ValueError("leg must be 'left' or 'right'.")

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "left_thigh": self.left_thigh.as_metadata_dict(),
            "left_shank_to_ankle": (
                self.left_shank_to_ankle.as_metadata_dict()
            ),
            "right_thigh": self.right_thigh.as_metadata_dict(),
            "right_shank_to_ankle": (
                self.right_shank_to_ankle.as_metadata_dict()
            ),
        }


@dataclass(frozen=True)
class MotionLegAudit:
    primary_motion_leg: MotionLeg
    selection_mode: str
    left_motion_score_m: float
    right_motion_score_m: float
    dominant_to_contralateral_score_ratio: float

    def as_metadata_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalSagittalFrame:
    """Static axes and fixed plane origin for an auditable sagittal frame."""

    x_axis_world: tuple[float, float, float]
    z_axis_world: tuple[float, float, float]
    lateral_axis_world: tuple[float, float, float]
    sagittal_plane_normal_world: tuple[float, float, float]
    reference_origin_world_m: tuple[float, float, float]
    origin_policy: str
    primary_motion_leg: MotionLeg
    contralateral_leg: MotionLeg
    z_axis_sign_flipped: bool
    x_axis_definition: str
    z_axis_definition: str

    @property
    def x_axis(self) -> np.ndarray:
        return np.asarray(self.x_axis_world, dtype=float)

    @property
    def z_axis(self) -> np.ndarray:
        return np.asarray(self.z_axis_world, dtype=float)

    @property
    def plane_normal(self) -> np.ndarray:
        return np.asarray(self.sagittal_plane_normal_world, dtype=float)

    def as_metadata_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceTrajectoryImportResult:
    """Validated source, SI markers, projected trajectory, and all audits."""

    source_dataframe: pd.DataFrame
    marker_dataframe_m: pd.DataFrame
    trajectory: pd.DataFrame
    frame_audit: FrameContinuityAudit
    segment_lengths: BilateralSegmentLengths
    motion_leg_audit: MotionLegAudit
    sagittal_frame: LocalSagittalFrame
    metadata: dict[str, object]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Convenient alias for the extracted/projected trajectory."""

        return self.trajectory

    @property
    def primary_motion_leg(self) -> MotionLeg:
        return self.motion_leg_audit.primary_motion_leg

    def __getitem__(self, key: str) -> pd.Series:
        return self.trajectory[key]

    def __len__(self) -> int:
        return len(self.trajectory)


def _normalise_coordinate_unit(coordinate_unit: str) -> CoordinateUnit:
    if not isinstance(coordinate_unit, str):
        raise TypeError("coordinate_unit must be explicitly 'mm' or 'm'.")
    unit = coordinate_unit.strip().lower()
    if unit not in UNIT_SCALE_TO_METRES:
        raise ValueError("coordinate_unit must be explicitly 'mm' or 'm'.")
    return unit  # type: ignore[return-value]


def _normalise_leg(value: str, *, allow_auto: bool) -> str:
    if not isinstance(value, str):
        raise TypeError("primary_motion_leg must be 'auto', 'left', or 'right'.")
    normalised = value.strip().lower()
    aliases = {"l": "left", "r": "right"}
    normalised = aliases.get(normalised, normalised)
    choices = {"left", "right"} | ({"auto"} if allow_auto else set())
    if normalised not in choices:
        expected = "'auto', 'left', or 'right'" if allow_auto else "'left' or 'right'"
        raise ValueError(f"primary_motion_leg must be {expected}.")
    return normalised


def _marker_columns(landmark: str) -> list[str]:
    return [f"{landmark}_{axis}" for axis in "XYZ"]


def _marker_array(dataframe_m: pd.DataFrame, landmark: str) -> np.ndarray:
    return dataframe_m[_marker_columns(landmark)].to_numpy(dtype=float)


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError(f"cannot construct {name}: degenerate 3-D vector.")
    return vector / norm


def validate_reference_trajectory_dataframe(
    dataframe: pd.DataFrame,
) -> FrameContinuityAudit:
    """Validate required marker fields and return a non-destructive frame audit.

    Frame indices must be finite integers and strictly increase.  Missing frame
    numbers are retained and reported as continuity gaps rather than silently
    reindexing or interpolating the trajectory.
    """

    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("reference trajectory must be a non-empty DataFrame.")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing_columns:
        raise ValueError(
            "reference trajectory is missing required columns: "
            f"{missing_columns}"
        )
    numeric = dataframe.loc[:, list(REQUIRED_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float))
    if nonfinite.any():
        locations = np.argwhere(nonfinite)
        preview = [
            f"row={int(row)}, column={REQUIRED_COLUMNS[int(column)]}"
            for row, column in locations[:8]
        ]
        raise ValueError(
            "required lower-limb data contain missing or non-finite values: "
            + ", ".join(preview)
        )
    frames_float = numeric["Frame"].to_numpy(dtype=float)
    if not np.equal(frames_float, np.round(frames_float)).all():
        raise ValueError("Frame values must be integer frame identifiers.")
    frames = np.round(frames_float).astype(np.int64)
    differences = np.diff(frames)
    if np.any(differences <= 0):
        raise ValueError("Frame must be strictly increasing without duplicates.")
    gaps = differences > 1
    return FrameContinuityAudit(
        first_frame=int(frames[0]),
        last_frame=int(frames[-1]),
        sample_count=len(frames),
        strictly_increasing=True,
        continuous_unit_steps=bool(np.all(differences == 1)),
        gap_count=int(gaps.sum()),
        missing_frame_count=int(np.sum(differences[gaps] - 1)),
        largest_step=int(differences.max()) if len(differences) else 0,
        gap_after_frames=tuple(int(value) for value in frames[:-1][gaps]),
    )


def convert_marker_coordinates_to_metres(
    dataframe: pd.DataFrame,
    *,
    coordinate_unit: str,
) -> pd.DataFrame:
    """Project required fields and explicitly convert XYZ values to metres."""

    unit = _normalise_coordinate_unit(coordinate_unit)
    validate_reference_trajectory_dataframe(dataframe)
    output = dataframe.loc[:, list(REQUIRED_COLUMNS)].copy()
    output["Frame"] = np.round(
        pd.to_numeric(output["Frame"], errors="raise").to_numpy(dtype=float)
    ).astype(np.int64)
    coordinate_columns = [column for column in REQUIRED_COLUMNS if column != "Frame"]
    output.loc[:, coordinate_columns] = (
        output.loc[:, coordinate_columns]
        .apply(pd.to_numeric, errors="raise")
        .astype(float)
        * UNIT_SCALE_TO_METRES[unit]
    )
    return output


def _segment_summary(lengths_m: np.ndarray) -> SegmentLengthSummary:
    values = np.asarray(lengths_m, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("segment lengths must be a non-empty finite array.")
    if np.any(values <= 0.0):
        raise ValueError("all observed segment lengths must be positive.")
    return SegmentLengthSummary(
        mean_m=float(np.mean(values)),
        median_m=float(np.median(values)),
        standard_deviation_m=float(np.std(values)),
        minimum_m=float(np.min(values)),
        maximum_m=float(np.max(values)),
    )


def estimate_bilateral_segment_lengths(
    marker_dataframe_m: pd.DataFrame,
) -> BilateralSegmentLengths:
    """Summarise bilateral hip-knee and knee-ankle marker distances."""

    def lengths(proximal: str, distal: str) -> np.ndarray:
        return np.linalg.norm(
            _marker_array(marker_dataframe_m, distal)
            - _marker_array(marker_dataframe_m, proximal),
            axis=1,
        )

    return BilateralSegmentLengths(
        left_thigh=_segment_summary(lengths("LHip", "LKnee")),
        left_shank_to_ankle=_segment_summary(lengths("LKnee", "LAnkle")),
        right_thigh=_segment_summary(lengths("RHip", "RKnee")),
        right_shank_to_ankle=_segment_summary(lengths("RKnee", "RAnkle")),
    )


def _leg_motion_score(marker_dataframe_m: pd.DataFrame, side: str) -> float:
    prefix = "L" if side == "left" else "R"
    hip = _marker_array(marker_dataframe_m, f"{prefix}Hip")
    relative_markers = np.concatenate(
        (
            _marker_array(marker_dataframe_m, f"{prefix}Knee") - hip,
            _marker_array(marker_dataframe_m, f"{prefix}Ankle") - hip,
        ),
        axis=1,
    )
    centred = relative_markers - np.median(relative_markers, axis=0)
    return float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))


def determine_primary_motion_leg(
    marker_dataframe_m: pd.DataFrame,
    *,
    primary_motion_leg: str = "auto",
    minimum_auto_score_ratio: float = 1.20,
) -> MotionLegAudit:
    """Select the moving leg from hip-relative knee/ankle motion amplitude."""

    requested = _normalise_leg(primary_motion_leg, allow_auto=True)
    ratio_threshold = float(minimum_auto_score_ratio)
    if not np.isfinite(ratio_threshold) or ratio_threshold < 1.0:
        raise ValueError("minimum_auto_score_ratio must be finite and >= 1.")
    left_score = _leg_motion_score(marker_dataframe_m, "left")
    right_score = _leg_motion_score(marker_dataframe_m, "right")
    if requested == "auto":
        selected: MotionLeg = "left" if left_score > right_score else "right"
        dominant = max(left_score, right_score)
        contralateral = min(left_score, right_score)
        ratio = dominant / max(contralateral, 1e-12)
        if dominant <= 1e-12 or ratio < ratio_threshold:
            raise ValueError(
                "primary motion leg is ambiguous; specify 'left' or 'right'."
            )
        selection_mode = "auto_motion_score"
    else:
        selected = requested  # type: ignore[assignment]
        selected_score = left_score if selected == "left" else right_score
        other_score = right_score if selected == "left" else left_score
        ratio = selected_score / max(other_score, 1e-12)
        selection_mode = "explicit"
    return MotionLegAudit(
        primary_motion_leg=selected,
        selection_mode=selection_mode,
        left_motion_score_m=left_score,
        right_motion_score_m=right_score,
        dominant_to_contralateral_score_ratio=float(ratio),
    )


def construct_local_sagittal_frame(
    marker_dataframe_m: pd.DataFrame,
    *,
    primary_motion_leg: MotionLeg,
) -> LocalSagittalFrame:
    """Construct +x toward the feet and +z toward flexion/knee elevation.

    The transverse/lateral axis points from the moving hip to the contralateral
    hip.  The contralateral hip-to-ankle vector is projected orthogonal to that
    axis to establish the bed/foot direction.  The remaining normal is signed
    so the moving knee lies predominantly in positive local z during flexion.
    """

    moving_prefix = "R" if primary_motion_leg == "right" else "L"
    contra_prefix = "L" if primary_motion_leg == "right" else "R"
    moving_hip = _marker_array(marker_dataframe_m, f"{moving_prefix}Hip")
    moving_knee = _marker_array(marker_dataframe_m, f"{moving_prefix}Knee")
    contra_hip = _marker_array(marker_dataframe_m, f"{contra_prefix}Hip")
    contra_ankle = _marker_array(marker_dataframe_m, f"{contra_prefix}Ankle")
    moving_hip_reference = np.median(moving_hip, axis=0)
    contra_hip_reference = np.median(contra_hip, axis=0)
    contra_ankle_reference = np.median(contra_ankle, axis=0)

    lateral_axis = _unit_vector(
        contra_hip_reference - moving_hip_reference,
        "left-right hip lateral axis",
    )
    toward_foot = contra_ankle_reference - contra_hip_reference
    toward_foot_in_sagittal_plane = toward_foot - lateral_axis * np.dot(
        toward_foot, lateral_axis
    )
    x_axis = _unit_vector(
        toward_foot_in_sagittal_plane,
        "contralateral hip-to-ankle foot axis",
    )
    z_axis = _unit_vector(np.cross(x_axis, lateral_axis), "sagittal z axis")

    knee_z = (moving_knee - moving_hip) @ z_axis
    flipped = bool(np.median(knee_z) < 0.0)
    if flipped:
        z_axis = -z_axis
        lateral_axis = -lateral_axis
    # Flipping z and lateral together preserves x and a right-handed basis.
    orthogonality = np.array(
        [
            np.dot(x_axis, z_axis),
            np.dot(x_axis, lateral_axis),
            np.dot(z_axis, lateral_axis),
        ]
    )
    if np.max(np.abs(orthogonality)) > 1e-10:
        raise RuntimeError("constructed sagittal axes are not orthogonal.")
    return LocalSagittalFrame(
        x_axis_world=tuple(float(value) for value in x_axis),
        z_axis_world=tuple(float(value) for value in z_axis),
        lateral_axis_world=tuple(float(value) for value in lateral_axis),
        sagittal_plane_normal_world=tuple(
            float(value) for value in lateral_axis
        ),
        reference_origin_world_m=tuple(
            float(value) for value in moving_hip_reference
        ),
        origin_policy=(
            "fixed_median_primary_hip_for_plane_distance; per-frame primary "
            "hip only for relative sagittal x/z coordinates"
        ),
        primary_motion_leg=primary_motion_leg,
        contralateral_leg=("left" if primary_motion_leg == "right" else "right"),
        z_axis_sign_flipped=flipped,
        x_axis_definition=(
            "contralateral hip-to-ankle vector projected orthogonal to the "
            "bilateral hip axis; positive toward the feet"
        ),
        z_axis_definition=(
            "cross(x_axis, lateral_axis), sign-selected so moving-knee "
            "elevation/flexion is positive"
        ),
    )


def _project_landmarks(
    marker_dataframe_m: pd.DataFrame,
    sagittal_frame: LocalSagittalFrame,
) -> pd.DataFrame:
    prefix = "L" if sagittal_frame.primary_motion_leg == "left" else "R"
    moving_hip_origins = _marker_array(marker_dataframe_m, f"{prefix}Hip")
    fixed_plane_origin = np.asarray(
        sagittal_frame.reference_origin_world_m,
        dtype=float,
    )
    x_axis = sagittal_frame.x_axis
    z_axis = sagittal_frame.z_axis
    plane_normal = sagittal_frame.plane_normal
    output = pd.DataFrame({"Frame": marker_dataframe_m["Frame"].to_numpy(dtype=int)})
    for landmark in LANDMARKS:
        marker = _marker_array(marker_dataframe_m, landmark)
        relative_to_moving_hip = marker - moving_hip_origins
        # Plane distance deliberately uses a fixed origin.  Using the moving
        # hip itself here would force the primary hip error to zero and hide
        # genuine out-of-plane translation.
        relative_to_fixed_plane = marker - fixed_plane_origin
        signed_out_of_plane = relative_to_fixed_plane @ plane_normal
        output[f"{landmark}_x_local_m"] = relative_to_moving_hip @ x_axis
        output[f"{landmark}_z_local_m"] = relative_to_moving_hip @ z_axis
        output[f"{landmark}_out_of_plane_m"] = signed_out_of_plane
        output[f"{landmark}_planarity_error_m"] = np.abs(
            signed_out_of_plane
        )
    return output


def extract_joint_angles_from_projected_markers(
    projected_dataframe: pd.DataFrame,
    *,
    primary_motion_leg: MotionLeg,
) -> pd.DataFrame:
    """Extract un-clipped hip/knee angles from projected marker segments."""

    prefix = "L" if primary_motion_leg == "left" else "R"
    hip_x = projected_dataframe[f"{prefix}Hip_x_local_m"].to_numpy(dtype=float)
    hip_z = projected_dataframe[f"{prefix}Hip_z_local_m"].to_numpy(dtype=float)
    knee_x = projected_dataframe[f"{prefix}Knee_x_local_m"].to_numpy(dtype=float)
    knee_z = projected_dataframe[f"{prefix}Knee_z_local_m"].to_numpy(dtype=float)
    ankle_x = projected_dataframe[f"{prefix}Ankle_x_local_m"].to_numpy(dtype=float)
    ankle_z = projected_dataframe[f"{prefix}Ankle_z_local_m"].to_numpy(dtype=float)

    thigh_x = knee_x - hip_x
    thigh_z = knee_z - hip_z
    shank_x = ankle_x - knee_x
    shank_z = ankle_z - knee_z
    thigh_norm = np.hypot(thigh_x, thigh_z)
    shank_norm = np.hypot(shank_x, shank_z)
    nondegenerate = (thigh_norm > 1e-12) & (shank_norm > 1e-12)
    q_hip = np.unwrap(np.arctan2(thigh_z, thigh_x))
    theta_shank_projected = np.unwrap(np.arctan2(shank_z, shank_x))

    # Knee flexion is the unsigned angle between the projected anatomical
    # segments: straight is 0 and flexion is positive.  The cross-product sign
    # is retained only as a branch audit; it never selects a branch using a
    # labelled answer or a subject identifier.
    denominator = np.where(nondegenerate, thigh_norm * shank_norm, 1.0)
    cosine = (thigh_x * shank_x + thigh_z * shank_z) / denominator
    q_knee = np.arccos(np.clip(cosine, -1.0, 1.0))
    theta_shank = q_hip - q_knee
    projected_signed_flexion = np.arctan2(
        thigh_x * shank_z - thigh_z * shank_x,
        thigh_x * shank_x + thigh_z * shank_z,
    )
    # With the model convention, positive flexion rotates the shank clockwise
    # from the thigh, hence the projected signed angle is non-positive.
    branch_tolerance_rad = float(np.deg2rad(0.5))
    flexion_branch_valid = projected_signed_flexion <= branch_tolerance_rad
    closure_error = np.arctan2(
        np.sin(theta_shank_projected - theta_shank),
        np.cos(theta_shank_projected - theta_shank),
    )
    closure_valid = np.abs(closure_error) <= branch_tolerance_rad
    finite = (
        np.isfinite(q_hip)
        & np.isfinite(q_knee)
        & np.isfinite(theta_shank)
        & np.isfinite(theta_shank_projected)
        & nondegenerate
    )
    angle_valid = finite & flexion_branch_valid & closure_valid
    hip_min, hip_max = np.deg2rad(hip_range_deg)
    knee_min, knee_max = np.deg2rad(knee_range_deg)
    tolerance = 1e-12
    hip_in_range = finite & (q_hip >= hip_min - tolerance) & (
        q_hip <= hip_max + tolerance
    )
    knee_in_range = finite & (q_knee >= knee_min - tolerance) & (
        q_knee <= knee_max + tolerance
    )
    reasons = np.full(len(q_hip), "", dtype=object)
    angle_reasons = np.full(len(q_hip), "", dtype=object)

    def append(mask: np.ndarray, reason: str) -> None:
        selected = np.asarray(mask, dtype=bool)
        current = reasons[selected].astype(str)
        reasons[selected] = np.where(
            current == "",
            reason,
            np.char.add(np.char.add(current, ";"), reason),
        )

    def append_angle(mask: np.ndarray, reason: str) -> None:
        selected = np.asarray(mask, dtype=bool)
        current = angle_reasons[selected].astype(str)
        angle_reasons[selected] = np.where(
            current == "",
            reason,
            np.char.add(np.char.add(current, ";"), reason),
        )

    append(~finite, "nonfinite_joint_angle")
    append(finite & ~hip_in_range, "q_hip_out_of_range")
    append(finite & ~knee_in_range, "q_knee_out_of_range")
    append_angle(~finite, "nonfinite_or_degenerate_projected_segment")
    append_angle(finite & ~flexion_branch_valid, "projected_knee_branch_opposes_flexion")
    append_angle(finite & ~closure_valid, "theta_shank_projection_closure_mismatch")
    output = projected_dataframe.copy()
    output["q_hip_rad"] = q_hip
    output["q_knee_rad"] = q_knee
    output["theta_shank_rad"] = theta_shank
    output["theta_shank_projected_rad"] = theta_shank_projected
    output["theta_shank_closure_error_rad"] = closure_error
    output["projected_signed_knee_rotation_rad"] = projected_signed_flexion
    output["q_hip_deg"] = np.rad2deg(q_hip)
    output["q_knee_deg"] = np.rad2deg(q_knee)
    output["theta_shank_deg"] = np.rad2deg(theta_shank)
    output["theta_shank_projected_deg"] = np.rad2deg(theta_shank_projected)
    output["angle_valid"] = angle_valid
    output["angle_invalid_reason"] = angle_reasons
    output["q_hip_within_configured_range"] = hip_in_range
    output["q_knee_within_configured_range"] = knee_in_range
    output["joint_range_valid"] = hip_in_range & knee_in_range
    output["joint_range_reason"] = reasons
    output["joint_angles_clipped"] = False
    output["theta_shank_definition"] = "q_hip - q_knee"
    # Preserve the anatomical ankle track under an explicit observed-marker
    # name.  No x_pull/z_pull alias is created here.
    output["observed_ankle_x_local_m"] = ankle_x
    output["observed_ankle_z_local_m"] = ankle_z
    output["observed_ankle_out_of_plane_m"] = projected_dataframe[
        f"{prefix}Ankle_out_of_plane_m"
    ].to_numpy(dtype=float)
    output["observed_ankle_is_pull_point"] = False
    return output


def import_reference_trajectory_dataframe(
    dataframe: pd.DataFrame,
    *,
    coordinate_unit: str,
    primary_motion_leg: str = "auto",
    source_name: str = "in_memory_reference_trajectory",
    minimum_auto_score_ratio: float = 1.20,
) -> ReferenceTrajectoryImportResult:
    """Validate, convert, frame, project, and extract a reference trajectory."""

    unit = _normalise_coordinate_unit(coordinate_unit)
    frame_audit = validate_reference_trajectory_dataframe(dataframe)
    markers_m = convert_marker_coordinates_to_metres(
        dataframe,
        coordinate_unit=unit,
    )
    segment_lengths = estimate_bilateral_segment_lengths(markers_m)
    motion_leg = determine_primary_motion_leg(
        markers_m,
        primary_motion_leg=primary_motion_leg,
        minimum_auto_score_ratio=minimum_auto_score_ratio,
    )
    sagittal_frame = construct_local_sagittal_frame(
        markers_m,
        primary_motion_leg=motion_leg.primary_motion_leg,
    )
    projected = _project_landmarks(markers_m, sagittal_frame)
    trajectory = extract_joint_angles_from_projected_markers(
        projected,
        primary_motion_leg=motion_leg.primary_motion_leg,
    )

    frames = markers_m["Frame"].to_numpy(dtype=int)
    frame_steps = np.r_[np.nan, np.diff(frames).astype(float)]
    trajectory.insert(1, "frame_step", frame_steps)
    trajectory.insert(
        2,
        "frame_contiguous_from_previous",
        np.r_[True, np.diff(frames) == 1],
    )
    prefix = "L" if motion_leg.primary_motion_leg == "left" else "R"
    primary_plane_error = np.concatenate(
        [
            trajectory[f"{prefix}{joint}_planarity_error_m"].to_numpy(dtype=float)
            for joint in ("Hip", "Knee", "Ankle")
        ]
    )
    metadata: dict[str, object] = {
        "source_name": str(source_name),
        "source_coordinate_unit": unit,
        "coordinate_scale_to_m": UNIT_SCALE_TO_METRES[unit],
        "required_columns": list(REQUIRED_COLUMNS),
        "frame_audit": frame_audit.as_metadata_dict(),
        "motion_leg_audit": motion_leg.as_metadata_dict(),
        "segment_lengths": segment_lengths.as_metadata_dict(),
        "local_sagittal_frame": sagittal_frame.as_metadata_dict(),
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "joint_angle_clipping_applied": False,
        "configured_hip_range_deg": list(hip_range_deg),
        "configured_knee_range_deg": list(knee_range_deg),
        "out_of_range_samples": int((~trajectory["joint_range_valid"]).sum()),
        "angle_invalid_samples": int((~trajectory["angle_valid"]).sum()),
        "theta_shank_projection_closure_max_abs_rad": float(
            np.nanmax(np.abs(trajectory["theta_shank_closure_error_rad"]))
        ),
        "primary_leg_planarity_rmse_m": float(
            np.sqrt(np.mean(primary_plane_error**2))
        ),
        "primary_leg_planarity_max_m": float(np.max(primary_plane_error)),
        "observed_ankle_retained": True,
        "observed_ankle_is_pull_point": False,
        "pull_point_reconstruction_performed": False,
        "dynamics_used": False,
        "hardware_used": False,
    }
    return ReferenceTrajectoryImportResult(
        source_dataframe=dataframe.copy(deep=True),
        marker_dataframe_m=markers_m,
        trajectory=trajectory,
        frame_audit=frame_audit,
        segment_lengths=segment_lengths,
        motion_leg_audit=motion_leg,
        sagittal_frame=sagittal_frame,
        metadata=metadata,
    )


def import_reference_trajectory_csv(
    csv_path: str | Path,
    *,
    coordinate_unit: str,
    primary_motion_leg: str = "auto",
    minimum_auto_score_ratio: float = 1.20,
) -> ReferenceTrajectoryImportResult:
    """Load a CSV, requiring the caller to state whether XYZ uses mm or m."""

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"reference trajectory CSV not found: {path}")
    dataframe = pd.read_csv(path)
    return import_reference_trajectory_dataframe(
        dataframe,
        coordinate_unit=coordinate_unit,
        primary_motion_leg=primary_motion_leg,
        source_name=str(path.resolve()),
        minimum_auto_score_ratio=minimum_auto_score_ratio,
    )


# Readable aliases for CLI/orchestration code.
load_reference_trajectory_csv = import_reference_trajectory_csv
import_marker_reference_trajectory = import_reference_trajectory_dataframe


__all__ = [
    "BilateralSegmentLengths",
    "CoordinateUnit",
    "FrameContinuityAudit",
    "LANDMARKS",
    "LocalSagittalFrame",
    "MotionLeg",
    "MotionLegAudit",
    "REQUIRED_COLUMNS",
    "ReferenceTrajectoryImportResult",
    "SegmentLengthSummary",
    "construct_local_sagittal_frame",
    "convert_marker_coordinates_to_metres",
    "determine_primary_motion_leg",
    "estimate_bilateral_segment_lengths",
    "extract_joint_angles_from_projected_markers",
    "import_marker_reference_trajectory",
    "import_reference_trajectory_csv",
    "import_reference_trajectory_dataframe",
    "load_reference_trajectory_csv",
    "validate_reference_trajectory_dataframe",
]
