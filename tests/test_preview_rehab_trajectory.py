"""Pure-offline preview regressions."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from control.start_anchor import FixedTcpOrientation, StartAnchor
from control.start_anchored_relative_trajectory import (
    FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
)
from lower_limb_sim.reference_measured_asymmetric import (
    MEASURED_ASYMMETRIC_NOMINAL_ID,
)
from lower_limb_sim.run_robot_trajectory_export import (
    DEFAULT_REFERENCE_PATH,
    load_closed_reference_trajectory,
)
from scripts.preview_rehab_trajectory import preview_trajectory


def _anchor() -> StartAnchor:
    reference, _ = load_closed_reference_trajectory(DEFAULT_REFERENCE_PATH)
    first = reference.iloc[0]
    pose = (0.41, -0.12, 0.36, 0.1, -0.2, 0.3)
    return StartAnchor(
        capture_host_time_s=123.0,
        tcp_pose_base=pose,
        tcp_position_base_m=pose[:3],
        tcp_orientation=FixedTcpOrientation(pose[3:]),
        robot_joint_positions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        trajectory_id=FIRST_ROBOT_TRIAL_TRAJECTORY_ID,
        reference_start_q_hip=float(first["q_hip_rad"]),
        reference_start_q_knee=float(first["q_knee_rad"]),
        anchor_id="anchor_preview_test",
        reviewed=False,
        notes="offline test",
    )


def _frame_path(tmp_path: Path) -> Path:
    path = tmp_path / "frame.json"
    path.write_text(
        json.dumps(
            {
                "rehab_x_axis_in_base": [1.0, 0.0, 0.0],
                "rehab_z_axis_in_base": [0.0, 0.0, 1.0],
                "reviewed": False,
                "notes": "offline preview draft",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_preview_writes_csv_json_and_four_nonempty_plots(tmp_path: Path) -> None:
    output = tmp_path / "preview"
    summary = preview_trajectory(
        anchor=_anchor(),
        frame_config=_frame_path(tmp_path),
        output_dir=output,
    )
    assert summary["trajectory_valid"] is True
    assert summary["execution_approved"] is False
    assert summary["frame_reviewed"] is False
    assert summary["anchor_reviewed"] is False
    assert Path(summary["trajectory_csv"]).is_file()
    assert Path(summary["metadata_json"]).is_file()
    assert len(summary["plots"]) == 4
    assert all(Path(path).stat().st_size > 0 for path in summary["plots"])
    trajectory = pd.read_csv(summary["trajectory_csv"])
    assert len(trajectory) == 401
    assert trajectory["trajectory_valid"].all()
    metadata = json.loads(Path(summary["metadata_json"]).read_text(encoding="utf-8"))
    assert metadata["git_commit"]
    assert metadata["preview_only"] is True
    assert metadata["hardware_accessed"] is False
    assert metadata["robot_execution_approved"] is False


def test_preview_module_has_no_hardware_or_motion_dependency() -> None:
    import scripts.preview_rehab_trajectory as module

    source = inspect.getsource(module)
    assert "from hardware" not in source
    assert "import hardware" not in source
    for forbidden in ("enable_realtime", "move_l(", "move_j(", "setPowerState"):
        assert forbidden not in source


def test_preview_rejects_anchor_reference_mismatch(tmp_path: Path) -> None:
    anchor = _anchor()
    mismatched = StartAnchor(
        **{**anchor.__dict__, "trajectory_id": MEASURED_ASYMMETRIC_NOMINAL_ID}
    )
    with pytest.raises(ValueError, match="trajectory_id does not match"):
        preview_trajectory(
            anchor=mismatched,
            frame_config=_frame_path(tmp_path),
            output_dir=tmp_path / "preview",
        )
