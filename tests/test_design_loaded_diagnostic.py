"""Offline design preserves source evidence and cannot approve a rigid fixture."""
import copy
from datetime import datetime, timedelta, timezone
import json
import socket

import numpy as np
import pandas as pd
import pytest

from config.bundle import ROOT, configuration
from measurement_validation.loaded_trial import validate_config
from scripts import design_loaded_diagnostic as loaded

CONFIG = ROOT / "config/continuous_taught_90s.empty.json"
SOURCE = ROOT / "results/manual_lab/20261122_143700/continuous_empty_01"
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def design_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("loaded_design")
    path, output = root / "template.json", root / "preview"
    result = loaded.design(CONFIG, output, output_config=path)
    return path, output, result


def _fresh_anchor(tmp_path, *, age_s=60., same_id=False):
    anchor = json.loads((SOURCE / "anchor.json").read_text(encoding="utf-8"))
    if not same_id:
        anchor["anchor_id"] = "new_current_installed_anchor"
    anchor["created_at"] = (NOW - timedelta(seconds=age_s)).isoformat()
    path = tmp_path / "current_anchor.json"
    loaded.write_new_json(path, anchor)
    return path


def _review(path, anchor, output, **changes):
    values = dict(setup_id="rotational_free_fixture_replacement",
                  connection_type="passive_joint", tool_description="Actual swiveling fixture",
                  support_description="Actual thigh mounting and leg support",
                  review_notes="Complete small-path check performed; actual raw limits selected",
                  manual_path_checked=True, max_force_n=80., max_torque_nm=20., now=NOW)
    values.update(changes)
    return loaded.review(path, anchor, output, **values)


def test_design_is_offline_blocked_and_preserves_original_evidence(tmp_path, monkeypatch):
    files = [CONFIG, SOURCE / "anchor.json", SOURCE / "round_status.json",
             ROOT / configuration(CONFIG)["trajectory"]["geometry_path"]]
    original = {path: path.read_bytes() for path in files}
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("offline design used networking"))
    path, output = tmp_path / "template.json", tmp_path / "preview"
    result = loaded.design(CONFIG, output, output_config=path)
    assert result["status"] == "BLOCKED_RIGID_CONNECTION_REQUIRES_COUPLED_POSE"
    assert result["hardware_accessed"] is False and result["ready_for_replay"] is False
    assert result["formal_force_comparison_ready"] is False
    assert result["duration_s"] == 90 and result["command_count"] == 4501
    assert result["trajectory_audit_reasons"] == []
    assert result["audited_maximum_speed_m_s"] < .005
    assert result["audited_maximum_acceleration_m_s2"] < .01
    config = configuration(path)
    assert config["loaded_trial"]["connection_type"] == "rigid_fixture"
    assert config["loaded_trial"]["reviewed"] is False
    assert config["loaded_trial"]["manual_path_checked"] is False
    assert config["loaded_trial"]["reviewed_anchor_pose_base"] is None
    assert config["validation"]["capture_dummy_leg"] is True
    assert config["training"]["enabled"] is False and "force_training" not in config
    validate_config(config, require_review=False)
    with pytest.raises(ValueError):
        validate_config(config)
    table = pd.read_csv(output / "trajectory_preview.csv")
    xyz = table[[f"tcp_{axis}_base" for axis in "xyz"]].to_numpy(float)
    assert np.allclose(xyz[0], xyz[-1], atol=1e-12)
    assert np.ptp(table.tcp_z_base) == pytest.approx(.022531359, abs=1e-6)
    assert result["positive_excursion_base_m"]["y"] == pytest.approx(.02677649, abs=1e-6)
    assert set(table.trajectory_phase) == {"lift", "high_reach", "return"}
    assert result["internal_holds_s"] == 0
    assert all(path.read_bytes() == raw for path, raw in original.items())
    assert all(loaded.sha(output / name) == digest for name, digest in result["output_files_sha256"].items())


def test_rigid_connection_review_is_not_approved_even_after_manual_checkbox(design_files, tmp_path):
    path, _, _ = design_files
    anchor = _fresh_anchor(tmp_path)
    with pytest.raises(ValueError):
        _review(path, anchor, tmp_path / "blocked.json", connection_type="rigid_fixture")
    assert not (tmp_path / "blocked.json").exists()


def test_review_requires_explicit_actual_path_check_and_new_anchor(design_files, tmp_path):
    path, _, _ = design_files
    anchor = _fresh_anchor(tmp_path)
    with pytest.raises(ValueError, match="manual-path-checked"):
        _review(path, anchor, tmp_path / "unchecked.json", manual_path_checked=False)
    with pytest.raises(ValueError, match="historical empty-load anchor"):
        _review(path, SOURCE / "anchor.json", tmp_path / "historical.json")
    assert not (tmp_path / "unchecked.json").exists()
    assert not (tmp_path / "historical.json").exists()


@pytest.mark.parametrize("age", [1801., -6.])
def test_review_rejects_stale_or_future_capture(design_files, tmp_path, age):
    path, _, _ = design_files
    anchor = _fresh_anchor(tmp_path, age_s=age)
    with pytest.raises(ValueError, match="within 30 minutes"):
        _review(path, anchor, tmp_path / "old.json")


def test_real_rotationally_free_installation_review_creates_new_config(design_files, tmp_path):
    path, _, _ = design_files
    original = path.read_bytes()
    anchor = _fresh_anchor(tmp_path)
    output = tmp_path / "new_subdirectory/reviewed.json"
    result = _review(path, anchor, output)
    candidate = configuration(output)
    assert result["status"] == "MANUAL_REVIEW_RECORDED"
    assert result["hardware_accessed"] is False and result["motion_commanded"] is False
    assert candidate["motion"]["max_force_n"] == 80
    assert candidate["motion"]["max_torque_nm"] == 20
    assert candidate["loaded_trial"]["reviewed"] is True
    assert candidate["loaded_trial"]["reviewed_anchor_pose_base"] == configuration(anchor)["tcp_pose_base"]
    validate_config(candidate)
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        _review(path, anchor, output)


@pytest.mark.parametrize("changes", [{"max_force_n": 101.}, {"max_torque_nm": 201.},
                                     {"max_force_n": 0.}, {"max_torque_nm": float("nan")}])
def test_review_does_not_guess_or_expand_actual_force_thresholds(design_files, tmp_path, changes):
    path, _, _ = design_files
    anchor = _fresh_anchor(tmp_path)
    with pytest.raises(ValueError):
        _review(path, anchor, tmp_path / "bad_caps.json", **changes)


def test_design_refuses_bad_source_and_overwrite(design_files, tmp_path):
    path, output, _ = design_files
    with pytest.raises(FileExistsError):
        loaded.design(CONFIG, output, output_config=tmp_path / "different.json")
    with pytest.raises(FileExistsError):
        loaded.design(CONFIG, tmp_path / "different", output_config=path)
    bad = copy.deepcopy(configuration(CONFIG))
    bad["trajectory"]["geometry_sha256"] = "0" * 64
    source = tmp_path / "bad_source.json"
    loaded.write_new_json(source, bad)
    with pytest.raises(ValueError):
        loaded.design(source, tmp_path / "invalid", output_config=tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_review_cli_requires_declared_raw_force_limits_and_manual_check():
    args = ["review", "--config", "unused", "--anchor", "unused", "--setup-id", "test",
            "--connection-type", "passive_joint", "--tool-description", "test",
            "--support-description", "test", "--review-notes", "test", "--output", "unused"]
    with pytest.raises(SystemExit) as exc:
        loaded.main(args)
    assert exc.value.code == 2
