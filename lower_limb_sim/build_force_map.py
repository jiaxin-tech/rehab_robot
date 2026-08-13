"""构建虚拟受试者二维下肢准静态力地图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    L1,
    L2,
    force_map_data_dir,
    force_magnitude_limit_n,
    workspace_csv_path,
)
from .force_mapping import endpoint_force_from_joint_torque
from .formal_protocol import (
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
)
from .quasi_static_dynamics import quasi_static_joint_torque
from .virtual_subject import (
    VIRTUAL_SUBJECTS,
    VirtualSubject,
    get_virtual_subject,
)


def _true_mask(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
        .to_numpy()
    )


def load_workspace_atlas(path: str | Path = workspace_csv_path) -> pd.DataFrame:
    """加载第一阶段图谱并检查力地图所需字段。"""

    atlas_path = Path(path)
    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Workspace atlas not found: {atlas_path}. "
            "Run `python -m lower_limb_sim.workspace_atlas` first."
        )
    atlas = pd.read_csv(atlas_path)
    required = {
        "rom_protocol_version",
        "theta_shank_definition",
        "q_hip_rad",
        "q_knee_rad",
        "q_hip_deg",
        "q_knee_deg",
        "x_knee",
        "z_knee",
        "x_pull",
        "z_pull",
        "reachable",
        "near_singular",
        "jacobian_determinant",
        "jacobian_condition_number",
        "jacobian_near_singular",
        "jacobian_mapping_valid",
    }
    missing = required.difference(atlas.columns)
    if missing:
        raise ValueError(f"workspace atlas is missing columns: {sorted(missing)}")
    if not atlas["rom_protocol_version"].astype(str).eq(
        ROM_PROTOCOL_VERSION
    ).all():
        raise ValueError("workspace atlas is not ROM_PROTOCOL_V2")
    if not atlas["theta_shank_definition"].astype(str).eq(
        THETA_SHANK_DEFINITION
    ).all():
        raise ValueError("workspace atlas has an invalid theta_shank definition")
    return atlas


def build_force_map(
    subject: VirtualSubject,
    workspace_atlas: pd.DataFrame | None = None,
    force_limit_n: float = force_magnitude_limit_n,
) -> pd.DataFrame:
    """只对第一阶段 ``reachable=True`` 姿态建立准静态力地图。"""

    workspace = (
        load_workspace_atlas()
        if workspace_atlas is None
        else workspace_atlas.copy()
    )
    if "reachable" not in workspace:
        raise ValueError("workspace atlas is missing the reachable column.")
    reachable = workspace.loc[_true_mask(workspace["reachable"])].copy()
    if reachable.empty:
        raise ValueError("workspace atlas contains no reachable posture.")

    q_hip = reachable["q_hip_rad"].to_numpy(dtype=float)
    q_knee = reachable["q_knee_rad"].to_numpy(dtype=float)
    torque = quasi_static_joint_torque(q_hip, q_knee, subject, L1)
    force = endpoint_force_from_joint_torque(
        q_hip,
        q_knee,
        torque.tau_total_hip_nm,
        torque.tau_total_knee_nm,
        L1,
        L2,
        force_limit_n=force_limit_n,
    )

    result = reachable.reset_index(drop=True)
    result.insert(0, "subject_id", subject.subject_id)
    subject_columns = {
        "mass_thigh_kg": subject.mass_thigh_kg,
        "mass_shank_kg": subject.mass_shank_kg,
        "com_thigh_m": subject.com_thigh_m,
        "com_shank_m": subject.com_shank_m,
        "k_hip_nm_per_rad": subject.k_hip_nm_per_rad,
        "k_knee_nm_per_rad": subject.k_knee_nm_per_rad,
        "q0_hip_rad": subject.q0_hip_rad,
        "q0_knee_rad": subject.q0_knee_rad,
        "gravity_m_s2": subject.gravity_m_s2,
    }
    for column, value in subject_columns.items():
        result[column] = value

    result["tau_gravity_hip_nm"] = torque.tau_gravity_hip_nm
    result["tau_gravity_knee_nm"] = torque.tau_gravity_knee_nm
    result["tau_stiffness_hip_nm"] = torque.tau_stiffness_hip_nm
    result["tau_stiffness_knee_nm"] = torque.tau_stiffness_knee_nm
    result["tau_total_hip_nm"] = torque.tau_total_hip_nm
    result["tau_total_knee_nm"] = torque.tau_total_knee_nm
    result["fx_robot_on_leg_n"] = force.fx_robot_on_leg_n
    result["fz_robot_on_leg_n"] = force.fz_robot_on_leg_n
    result["force_magnitude_n"] = force.force_magnitude_n
    result["jacobian_determinant"] = force.jacobian_determinant
    result["jacobian_condition_number"] = force.jacobian_condition_number
    result["jacobian_near_singular"] = force.jacobian_near_singular
    result["force_mapping_valid"] = force.force_mapping_valid
    result["invalid_reason"] = force.invalid_reason
    return result


def save_force_map(
    force_map: pd.DataFrame,
    output_dir: str | Path = force_map_data_dir,
) -> tuple[Path, Path]:
    """保存 CSV 和按列压缩的 NPZ；NPZ 不依赖 pickle。"""

    subject_ids = force_map["subject_id"].astype(str).unique()
    if len(subject_ids) != 1:
        raise ValueError("force map must contain exactly one subject_id.")
    subject_id = subject_ids[0]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"force_map_{subject_id}.csv"
    npz_path = destination / f"force_map_{subject_id}.npz"
    force_map.to_csv(csv_path, index=False)

    arrays: dict[str, np.ndarray] = {}
    for column in force_map.columns:
        values = force_map[column].to_numpy()
        arrays[column] = values.astype(str) if values.dtype == object else values
    np.savez_compressed(npz_path, **arrays)
    return csv_path, npz_path


def _common_valid_keys(
    force_maps: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    key_columns = ["q_hip_deg", "q_knee_deg"]
    common: set[tuple[float, float]] | None = None
    for force_map in force_maps.values():
        valid = force_map.loc[_true_mask(force_map["force_mapping_valid"])]
        keys = {
            (float(row.q_hip_deg), float(row.q_knee_deg))
            for row in valid[key_columns].itertuples(index=False)
        }
        common = keys if common is None else common.intersection(keys)
    if not common:
        raise ValueError("No common valid posture exists across virtual subjects.")
    return pd.DataFrame(sorted(common), columns=key_columns)


def build_virtual_subject_comparison(
    force_maps: dict[str, pd.DataFrame],
    target_postures_deg: tuple[tuple[float, float], ...] = (
        (30.0, 30.0),
        (60.0, 60.0),
        (90.0, 100.0),
        (110.0, 120.0),
    ),
) -> pd.DataFrame:
    """选择靠近给定目标的共同有效姿态，形成跨受试者比较表。"""

    if set(force_maps) != set(VIRTUAL_SUBJECTS):
        raise ValueError("comparison requires all configured virtual subjects.")
    common = _common_valid_keys(force_maps)
    selected_keys: list[tuple[float, float]] = []
    for target_hip, target_knee in target_postures_deg:
        squared_error = (
            (common["q_hip_deg"] - target_hip) ** 2
            + (common["q_knee_deg"] - target_knee) ** 2
        )
        row = common.iloc[int(squared_error.argmin())]
        key = (float(row["q_hip_deg"]), float(row["q_knee_deg"]))
        if key not in selected_keys:
            selected_keys.append(key)

    rows: list[pd.Series] = []
    for subject_id in VIRTUAL_SUBJECTS:
        force_map = force_maps[subject_id]
        for q_hip_deg, q_knee_deg in selected_keys:
            match = force_map.loc[
                (force_map["q_hip_deg"] == q_hip_deg)
                & (force_map["q_knee_deg"] == q_knee_deg)
                & _true_mask(force_map["force_mapping_valid"])
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one common row for {subject_id} at "
                    f"({q_hip_deg}, {q_knee_deg})."
                )
            rows.append(match.iloc[0])
    comparison = pd.DataFrame(rows).reset_index(drop=True)
    validate_virtual_subject_differences(comparison)
    return comparison


def validate_virtual_subject_differences(comparison: pd.DataFrame) -> None:
    """验证刚度和质量变体在共同大屈曲姿态下呈现预期差异。"""

    high_hip = comparison.loc[
        comparison["q_hip_deg"] == comparison["q_hip_deg"].max()
    ]
    high_knee = comparison.loc[
        comparison["q_knee_deg"] == comparison["q_knee_deg"].max()
    ]

    def row(rows: pd.DataFrame, subject_id: str) -> pd.Series:
        selected = rows.loc[rows["subject_id"] == subject_id]
        if selected.empty:
            raise ValueError(f"Missing comparison row for {subject_id}.")
        return selected.iloc[0]

    if not (
        row(high_hip, "hip_stiff")["tau_stiffness_hip_nm"]
        > row(high_hip, "baseline")["tau_stiffness_hip_nm"]
    ):
        raise ValueError("hip_stiff comparison did not increase hip stiffness.")
    if not (
        row(high_knee, "knee_stiff")["tau_stiffness_knee_nm"]
        > row(high_knee, "baseline")["tau_stiffness_knee_nm"]
    ):
        raise ValueError("knee_stiff comparison did not increase knee stiffness.")

    baseline = row(high_hip, "baseline")
    heavy = row(high_hip, "heavy_leg")
    baseline_gravity_norm = np.hypot(
        baseline["tau_gravity_hip_nm"],
        baseline["tau_gravity_knee_nm"],
    )
    heavy_gravity_norm = np.hypot(
        heavy["tau_gravity_hip_nm"],
        heavy["tau_gravity_knee_nm"],
    )
    if not heavy_gravity_norm > baseline_gravity_norm:
        raise ValueError("heavy_leg comparison did not increase gravity torque.")


def save_virtual_subject_comparison(
    comparison: pd.DataFrame,
    output_dir: str | Path = force_map_data_dir,
) -> Path:
    """保存共同姿态比较 CSV。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "virtual_subject_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    return csv_path


def save_force_map_metadata(
    force_maps: dict[str, pd.DataFrame],
    workspace: pd.DataFrame,
    workspace_path: str | Path,
    output_dir: str | Path = force_map_data_dir,
) -> Path:
    """保存可复现的正式 ROM 协议与各受试者 force-map 摘要。"""

    summaries: dict[str, dict[str, int | float]] = {}
    for subject_id, force_map in force_maps.items():
        valid = _true_mask(force_map["force_mapping_valid"])
        valid_force = force_map.loc[valid, "force_magnitude_n"].to_numpy(
            dtype=float
        )
        summaries[subject_id] = {
            "row_count": int(len(force_map)),
            "valid_count": int(valid.sum()),
            "maximum_valid_force_magnitude_n": float(
                np.nanmax(valid_force)
            ),
        }

    source_path = Path(workspace_path)
    try:
        source_text = str(source_path.resolve().relative_to(Path.cwd()))
    except ValueError:
        source_text = str(source_path.resolve())
    metadata = {
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "source_workspace": source_text,
        "source_workspace_sample_count": int(len(workspace)),
        "source_workspace_reachable_sample_count": int(
            _true_mask(workspace["reachable"]).sum()
        ),
        "virtual_subjects": summaries,
        "force_magnitude_software_anomaly_limit_n": float(
            force_magnitude_limit_n
        ),
        "force_limit_interpretation": (
            "existing offline software anomaly gate, not a real-robot "
            "safety threshold"
        ),
        "legacy_force_maps_overwritten": False,
        "real_robot_connected": False,
        "hardware_code_modified": False,
        "safety_thresholds_modified": False,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "force_map_metadata.json"
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_one(
    subject: VirtualSubject,
    workspace: pd.DataFrame,
    output_dir: Path,
    make_plots: bool,
) -> pd.DataFrame:
    force_map = build_force_map(subject, workspace)
    csv_path, npz_path = save_force_map(force_map, output_dir)
    valid_count = int(_true_mask(force_map["force_mapping_valid"]).sum())
    print(
        f"{subject.subject_id}: rows={len(force_map)}, "
        f"valid={valid_count}, invalid={len(force_map) - valid_count}"
    )
    print(f"CSV: {csv_path}")
    print(f"NPZ: {npz_path}")
    if make_plots:
        from .visualize_force_map import generate_force_map_plots

        for path in generate_force_map_plots(
            force_map,
            output_dir / subject.subject_id,
        ):
            print(f"figure: {path}")
    return force_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subject_id",
        nargs="?",
        choices=tuple(VIRTUAL_SUBJECTS),
        help="要生成的虚拟受试者 ID。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="生成全部四名虚拟受试者，并输出共同姿态比较。",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=workspace_csv_path,
        help="第一阶段 workspace_atlas.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=force_map_data_dir,
        help="CSV、NPZ 和图片输出根目录。",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="只保存数据，不生成图片。",
    )
    args = parser.parse_args()
    if args.all and args.subject_id is not None:
        parser.error("subject_id and --all cannot be used together.")
    if not args.all and args.subject_id is None:
        parser.error("provide a subject_id or use --all.")

    workspace = load_workspace_atlas(args.workspace)
    subject_ids = tuple(VIRTUAL_SUBJECTS) if args.all else (args.subject_id,)
    force_maps = {
        subject_id: _build_one(
            get_virtual_subject(subject_id),
            workspace,
            args.output_dir,
            make_plots=not args.no_plots,
        )
        for subject_id in subject_ids
    }
    if args.all:
        comparison = build_virtual_subject_comparison(force_maps)
        csv_path = save_virtual_subject_comparison(
            comparison,
            args.output_dir,
        )
        print(f"comparison CSV: {csv_path}")
        if not args.no_plots:
            from .visualize_force_map import plot_virtual_subject_comparison

            png_path = plot_virtual_subject_comparison(
                comparison,
                args.output_dir / "virtual_subject_comparison.png",
            )
            print(f"comparison figure: {png_path}")
        metadata_path = save_force_map_metadata(
            force_maps,
            workspace,
            args.workspace,
            args.output_dir,
        )
        print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
