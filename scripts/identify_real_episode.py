"""Offline adapter from one real five-file episode to the existing estimator.

No default anthropometry, wrench sign/frame transform, or delay is invented.
The command requires a reviewed ``identification_config.json`` in the episode
directory (or ``--config``).  Without it, or without valid real CSV rows, it
fails before creating parameter/metric outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lower_limb_sim.config import L1, L2
from lower_limb_sim.derivative_estimation import estimate_joint_derivatives
from lower_limb_sim.kinematics import forward_kinematics
from lower_limb_sim.parameter_estimator import (
    BaselineSubjectTemplate,
    compute_torque_metrics,
    estimate_subject_parameters,
)
from lower_limb_sim.formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG as APPROVED_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG as APPROVED_KNEE_ROM_DEG,
)
from lower_limb_sim.state_history_buffer import StateHistoryBuffer
from utils.provenance import current_git_commit


CONFIG_FILENAME = "identification_config.json"
PARAMETERS_FILENAME = "identified_parameters.json"
METRICS_FILENAME = "prediction_metrics.csv"


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _load_reviewed_config(path: Path) -> tuple[dict[str, Any], BaselineSubjectTemplate, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"reviewed real identification config not found: {path}; copy and review "
            "config/real_identification_config.json first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    expected = {
        "schema_version",
        "reviewed",
        "raw_wrench_frame",
        "R_rehab_from_raw_wrench",
        "force_sign_robot_on_leg",
        "assumed_wrench_delay_s",
        "baseline_subject_template",
        "notes",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("real identification config has unexpected or missing fields")
    if payload["schema_version"] != 1 or type(payload["reviewed"]) is not bool:
        raise ValueError("identification schema_version/reviewed types are invalid")
    if not payload["reviewed"]:
        raise PermissionError("real identification config is not reviewed")
    if payload["raw_wrench_frame"] not in {"world", "flange", "tool"}:
        raise ValueError("raw_wrench_frame must be reviewed as world, flange, or tool")
    sign = payload["force_sign_robot_on_leg"]
    if isinstance(sign, bool) or sign not in (-1, 1, -1.0, 1.0):
        raise ValueError("force_sign_robot_on_leg must be exactly -1 or 1")
    delay = _finite(payload["assumed_wrench_delay_s"], "assumed_wrench_delay_s")
    if delay < 0.0:
        raise ValueError("assumed_wrench_delay_s must be non-negative")
    rotation = np.asarray(payload["R_rehab_from_raw_wrench"], dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("R_rehab_from_raw_wrench must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("R_rehab_from_raw_wrench must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("R_rehab_from_raw_wrench must be right handed")
    baseline_payload = payload["baseline_subject_template"]
    baseline_fields = {
        "mass_thigh_kg",
        "mass_shank_kg",
        "com_thigh_m",
        "com_shank_m",
        "inertia_thigh_kg_m2",
        "inertia_shank_kg_m2",
        "q0_hip_rad",
        "q0_knee_rad",
        "gravity_m_s2",
    }
    if not isinstance(baseline_payload, Mapping) or set(baseline_payload) != baseline_fields:
        raise ValueError("baseline_subject_template fields are incomplete")
    baseline = BaselineSubjectTemplate(
        mass_thigh_kg=_positive(baseline_payload["mass_thigh_kg"], "mass_thigh_kg"),
        mass_shank_kg=_positive(baseline_payload["mass_shank_kg"], "mass_shank_kg"),
        com_thigh_m=_positive(baseline_payload["com_thigh_m"], "com_thigh_m"),
        com_shank_m=_positive(baseline_payload["com_shank_m"], "com_shank_m"),
        inertia_thigh_kg_m2=_positive(
            baseline_payload["inertia_thigh_kg_m2"], "inertia_thigh_kg_m2"
        ),
        inertia_shank_kg_m2=_positive(
            baseline_payload["inertia_shank_kg_m2"], "inertia_shank_kg_m2"
        ),
        q0_hip_rad=_finite(baseline_payload["q0_hip_rad"], "q0_hip_rad"),
        q0_knee_rad=_finite(baseline_payload["q0_knee_rad"], "q0_knee_rad"),
        gravity_m_s2=_positive(baseline_payload["gravity_m_s2"], "gravity_m_s2"),
    )
    payload["force_sign_robot_on_leg"] = float(sign)
    payload["assumed_wrench_delay_s"] = delay
    return payload, baseline, rotation


def _bool_series(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().isin(("true", "1")).to_numpy()


def _load_episode(episode: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    paths = {
        "metadata": episode / "metadata.json",
        "state": episode / "robot_state.csv",
        "wrench": episode / "robot_wrench.csv",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"episode is missing files: {missing}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("parent_reference_id") != ACTIVE_REFERENCE_ID:
        raise RuntimeError("episode parent_reference_id is missing or invalid")
    if metadata.get("parent_reference_sha256") != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("episode parent_reference_sha256 is missing or invalid")
    state = pd.read_csv(paths["state"])
    wrench = pd.read_csv(paths["wrench"])
    state_required = {
        "host_time_s", "tcp_x", "tcp_y", "tcp_z", "valid", "invalid_reason"
    }
    wrench_required = {
        "query_start_s", "query_end_s", "fx", "fy", "fz", "frame_type", "valid"
    }
    if state_required.difference(state.columns) or wrench_required.difference(wrench.columns):
        raise ValueError("episode state/wrench schema is incomplete")
    if state.empty or wrench.empty:
        raise ValueError("episode has no real state/wrench samples")
    return metadata, state, wrench


def _reviewed_geometry(metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    frame = metadata.get("rehab_frame")
    anchor = metadata.get("start_anchor")
    if not isinstance(frame, Mapping) or frame.get("reviewed") is not True:
        raise PermissionError("episode rehab frame is missing or not reviewed")
    if not isinstance(anchor, Mapping) or anchor.get("reviewed") is not True:
        raise PermissionError("episode start anchor is missing or not reviewed")
    rotation = np.asarray(frame.get("R_base_from_rehab"), dtype=float)
    pose = np.asarray(anchor.get("tcp_pose_base"), dtype=float)
    q_hip_start = _finite(anchor.get("reference_start_q_hip"), "reference_start_q_hip")
    q_knee_start = _finite(anchor.get("reference_start_q_knee"), "reference_start_q_knee")
    if rotation.shape != (3, 3) or pose.shape != (6,) or not np.isfinite(rotation).all() or not np.isfinite(pose).all():
        raise ValueError("episode reviewed frame/anchor geometry is invalid")
    return rotation, pose, q_hip_start, q_knee_start


def _reconstruct_angles(
    state: pd.DataFrame,
    rotation_base_from_rehab: np.ndarray,
    anchor_pose: np.ndarray,
    q_hip_start: float,
    q_knee_start: float,
) -> pd.DataFrame:
    time_s = state["host_time_s"].to_numpy(float)
    xyz = state[["tcp_x", "tcp_y", "tcp_z"]].to_numpy(float)
    if not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("robot_state host timestamps must be finite and strictly increasing")
    _, _, x0, z0 = forward_kinematics(q_hip_start, q_knee_start, L1, L2)
    delta_rehab = (xyz - anchor_pose[:3]) @ rotation_base_from_rehab
    x_pull = float(x0) + delta_rehab[:, 0]
    z_pull = float(z0) + delta_rehab[:, 2]
    distance_squared = x_pull**2 + z_pull**2
    cosine_knee = (distance_squared - L1**2 - L2**2) / (2.0 * L1 * L2)
    geometric = np.isfinite(cosine_knee) & (cosine_knee >= -1.0) & (cosine_knee <= 1.0)
    q_knee = np.arccos(np.clip(cosine_knee, -1.0, 1.0))
    q_hip = np.arctan2(z_pull, x_pull) + np.arctan2(
        L2 * np.sin(q_knee), L1 + L2 * np.cos(q_knee)
    )
    hip_bounds = np.deg2rad(APPROVED_HIP_ROM_DEG)
    knee_bounds = np.deg2rad(APPROVED_KNEE_ROM_DEG)
    approved = (
        (q_hip >= hip_bounds[0]) & (q_hip <= hip_bounds[1])
        & (q_knee >= knee_bounds[0]) & (q_knee <= knee_bounds[1])
    )
    source_valid = _bool_series(state["valid"])
    valid = geometric & approved & source_valid & np.isfinite(xyz).all(axis=1)
    angles = pd.DataFrame(
        {
            "time_s": time_s,
            "q_hip_measured_rad": np.where(valid, q_hip, np.nan),
            "q_knee_measured_rad": np.where(valid, q_knee, np.nan),
            "ik_valid": valid,
        }
    )
    derivatives = estimate_joint_derivatives(
        angles,
        method="savitzky_golay_offline",
        time_column="time_s",
        angle_columns=("q_hip_measured_rad", "q_knee_measured_rad"),
        valid_column="ik_valid",
        group_columns=(),
    ).dataframe
    return derivatives


def _aligned_observations(
    angles: pd.DataFrame,
    wrench: pd.DataFrame,
    config: Mapping[str, Any],
    wrench_rotation: np.ndarray,
) -> pd.DataFrame:
    valid_angles = angles[angles["derivative_valid"].astype(bool)].copy()
    if len(valid_angles) < 20:
        raise ValueError("insufficient valid reconstructed state derivatives")
    times = valid_angles["time_s"].to_numpy(float)
    median_dt = float(np.median(np.diff(times)))
    history = StateHistoryBuffer(
        history_duration_s=float(times[-1] - times[0] + 1.0),
        max_state_interval_s=max(2.5 * median_dt, 1e-6),
    )
    for row in valid_angles.itertuples(index=False):
        history.append(
            row.time_s,
            row.q_hip_measured_rad,
            row.q_knee_measured_rad,
            row.dq_hip_est_rad_s,
            row.dq_knee_est_rad_s,
            row.ddq_hip_est_rad_s2,
            row.ddq_knee_est_rad_s2,
        )

    wrench_valid = _bool_series(wrench["valid"])
    frame_valid = wrench["frame_type"].astype(str).eq(config["raw_wrench_frame"]).to_numpy()
    query_start = wrench["query_start_s"].to_numpy(float)
    query_end = wrench["query_end_s"].to_numpy(float)
    raw_force = wrench[["fx", "fy", "fz"]].to_numpy(float)
    query_time = 0.5 * (query_start + query_end)
    transformed = raw_force @ wrench_rotation.T
    transformed *= float(config["force_sign_robot_on_leg"])
    rows = []
    delay = float(config["assumed_wrench_delay_s"])
    for index, target_time in enumerate(query_time - delay):
        if not (
            wrench_valid[index]
            and frame_valid[index]
            and np.isfinite(raw_force[index]).all()
            and np.isfinite(target_time)
        ):
            continue
        match = history.query(float(target_time), method="linear_interpolation")
        if not match.valid:
            continue
        rows.append(
            {
                "state_timestamp_s": match.matched_timestamp_s,
                "wrench_timestamp_s": query_time[index],
                "q_hip_rad": match.q_hip_rad,
                "q_knee_rad": match.q_knee_rad,
                "dq_hip_rad_s": match.dq_hip_rad_s,
                "dq_knee_rad_s": match.dq_knee_rad_s,
                "ddq_hip_rad_s2": match.ddq_hip_rad_s2,
                "ddq_knee_rad_s2": match.ddq_knee_rad_s2,
                "fx_observed_n": transformed[index, 0],
                "fz_observed_n": transformed[index, 2],
                "sample_valid": True,
                "force_mapping_valid": True,
                "wrench_is_stale": False,
                "state_match_method": "StateHistoryBuffer.linear_interpolation",
                "state_time_error_s": match.time_error_s,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) < 50:
        raise ValueError(f"insufficient aligned real observations: {len(result)} < 50")
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def identify_real_episode(
    episode_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    episode = Path(episode_dir).expanduser().resolve()
    parameters_path = episode / PARAMETERS_FILENAME
    metrics_path = episode / METRICS_FILENAME
    if parameters_path.exists() or metrics_path.exists():
        raise FileExistsError("identification outputs already exist; refusing to overwrite")
    config_source = Path(config_path).expanduser().resolve() if config_path else episode / CONFIG_FILENAME
    config, baseline, wrench_rotation = _load_reviewed_config(config_source)
    metadata, state, wrench = _load_episode(episode)
    rotation, anchor_pose, q_hip_start, q_knee_start = _reviewed_geometry(metadata)
    angles = _reconstruct_angles(
        state, rotation, anchor_pose, q_hip_start, q_knee_start
    )
    observations = _aligned_observations(angles, wrench, config, wrench_rotation)
    split_index = max(30, int(math.floor(0.7 * len(observations))))
    if len(observations) - split_index < 20:
        split_index = len(observations) - 20
    training = observations.iloc[:split_index].copy()
    testing = observations.iloc[split_index:].copy()
    estimate = estimate_subject_parameters(training, baseline, L1, L2)
    if not estimate.optimizer_success:
        raise RuntimeError("five-parameter optimizer did not converge: " + estimate.optimizer_message)
    identification_git_commit = current_git_commit()
    metric_rows = []
    for split_name, dataframe in (
        ("train", training),
        ("test", testing),
        ("all", observations),
    ):
        metric_rows.append(
            {
                "split": split_name,
                "source_episode_git_commit": metadata.get("git_commit"),
                "identification_git_commit": identification_git_commit,
                **compute_torque_metrics(
                    dataframe,
                    baseline,
                    estimate.estimated_parameters,
                    L1,
                    L2,
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "source_episode": str(episode),
        "source_episode_git_commit": metadata.get("git_commit"),
        "parent_reference_id": ACTIVE_REFERENCE_ID,
        "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "identification_git_commit": identification_git_commit,
        "real_data_only": True,
        "model": "existing_five_parameter_lower_limb_model",
        "model_angle_definition": "theta_shank = q_hip - q_knee",
        "state_reconstruction": "start_anchor_tcp_delta_to_pull_point_then_approved_rom_ik",
        "derivative_method": "savitzky_golay_offline",
        "timestamp_matching": "StateHistoryBuffer.linear_interpolation",
        "identification_config": config,
        "baseline_subject_template": asdict(baseline),
        "aligned_observation_count": len(observations),
        "training_observation_count": len(training),
        "testing_observation_count": len(testing),
        "estimation": estimate.as_serializable_dict(),
    }
    _atomic_json(parameters_path, payload)
    try:
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    except Exception:
        parameters_path.unlink(missing_ok=True)
        raise
    return {
        "identified_parameters_json": str(parameters_path),
        "prediction_metrics_csv": str(metrics_path),
        "aligned_observation_count": len(observations),
        "optimizer_success": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Identify the existing five-parameter model from one real episode")
    parser.add_argument("episode_dir")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    result = identify_real_episode(args.episode_dir, config_path=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
