"""Offline protocol gates using copied evidence, never an SDK or network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from collection.twincat_leg import atomic_json
from config.bundle import ROOT, configuration
from control.continuous_taught_trajectory import build_continuous_taught_trajectory
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from measurement_validation import loaded_trial as protocol


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def recipe(tmp_path):
    source = ROOT / "results/manual_lab/20261122_143700/continuous_empty_01"
    copied = tmp_path / "copied_empty"
    copied.mkdir()
    for name in protocol.SOURCE_FILES:
        shutil.copyfile(source / name, copied / name)
    c = configuration(ROOT / "config/continuous_taught_90s.empty.json")
    geometry = tmp_path / "geometry.json"
    shutil.copyfile(ROOT / c["trajectory"]["geometry_path"], geometry)
    c["trajectory"].update(geometry_path=str(geometry), scale_x=.03, scale_y=.03, scale_z=.03)
    for section in ("motion", "empty_load"):
        c[section].update(max_tcp_speed_m_s=.005, max_tcp_acceleration_m_s2=.01)
    c["validation"]["capture_dummy_leg"] = True
    c["loaded_trial"] = dict(protocol=protocol.PROTOCOL, setup_id=None,
        connection_type=None, tool_description=None, support_description=None,
        reviewed=False, manual_path_checked=False, reviewed_anchor_pose_base=None,
        reviewed_geometry_sha256=c["trajectory"]["geometry_sha256"],
        reviewed_scale_xyz=[.03]*3, review_notes=None, reviewed_at=None,
        strict_wrench_freshness=True, source_empty_session=str(copied),
        source_empty_status_sha256=hashlib.sha256((copied / "round_status.json").read_bytes()).hexdigest())
    return c


def reviewed(c):
    c = deepcopy(c)
    source_plan = read(Path(c["loaded_trial"]["source_empty_session"]) / "tcp_plan.json")
    c["loaded_trial"].update(setup_id="synthetic unit-test setup", connection_type="rope_or_strap",
        tool_description="synthetic test tool", support_description="synthetic test support",
        reviewed=True, manual_path_checked=True,
        reviewed_anchor_pose_base=source_plan["anchor"]["tcp_pose_base"],
        review_notes="Synthetic review fixture only; no laboratory review is claimed.",
        reviewed_at="2026-11-22T16:00:00+08:00")
    return c


def plan_for(c, directory):
    config_path = directory / "fixture.json"
    atomic_json(config_path, c)
    anchor = read(Path(c["loaded_trial"]["source_empty_session"]) / "tcp_plan.json")["anchor"]
    spec = {key: value for key, value in c["trajectory"].items() if key not in ("mode", "fixed_cycle_duration_s")}
    _, _, plan = build_continuous_taught_trajectory(spec,
        current_tcp_start_pose=anchor["tcp_pose_base"], rehab_frame=load_rehab_frame_config(config_path))
    plan["anchor"] = anchor
    atomic_json(directory / "leg_start_origin.json", {"usable_for_motion": True,
        "axes": {name: {"physical_angle_deg": 0.} for name in ("hip", "knee")},
        "notes": "Synthetic protocol test origin only; not a laboratory capture."})
    return plan


def test_template_is_offline_only_until_real_manual_review(recipe, tmp_path):
    assert protocol.active(recipe)
    assert not protocol.active({"loaded_trial": {"protocol": "another"}})
    assert not protocol.active({})
    assert protocol.validate_config(recipe, require_review=False) is None
    with pytest.raises(ValueError, match="review"):
        protocol.validate_config(recipe)
    with pytest.raises(ValueError):
        protocol.freeze_binding(tmp_path, recipe, plan_for(recipe, tmp_path))
    assert not (tmp_path / "loaded_trial_binding.json").exists()


@pytest.mark.parametrize("section,key,value", [
    ("trajectory", "scale_x", .03001), ("trajectory", "scale_y", 0),
    ("trajectory", "scale_z", True), ("trajectory", "scale_z", float("nan")),
    ("trajectory", "cycle_duration_s", 89), ("trajectory", "ramp_duration_s", 6),
    ("trajectory", "sample_period_s", .01), ("trajectory", "fixed_cycle_duration_s", True),
    ("motion", "max_tcp_speed_m_s", .00501), ("empty_load", "max_tcp_speed_m_s", .006),
    ("motion", "max_tcp_acceleration_m_s2", .011), ("empty_load", "max_tcp_acceleration_m_s2", False),
    ("motion", "max_force_n", 101), ("motion", "max_torque_nm", 201),
    ("motion", "max_wrench_age_s", .1), ("training", "enabled", True),
    ("training", "enabled", 0), ("training", "mode", "force_only"),
    ("validation", "capture_dummy_leg", False), ("validation", "capture_dummy_leg", 1),
    ("loaded_trial", "strict_wrench_freshness", False), ("loaded_trial", "reviewed", "true"),
    ("loaded_trial", "reviewed_scale_xyz", [.03, .02, .03]),
    ("loaded_trial", "reviewed_geometry_sha256", "a"*64),
    ("loaded_trial", "reviewed_anchor_pose_base", [0, 0, 0, 0, 0, True]),
    ("loaded_trial", "reviewed_at", "2026-11-22T16:00:00"),
])
def test_invalid_recipe_never_passes_even_for_preview(recipe, section, key, value):
    recipe[section][key] = value
    with pytest.raises(ValueError):
        protocol.validate_config(recipe, require_review=False)


def test_source_status_and_geometry_are_rehashed_each_time(recipe):
    protocol.validate_config(recipe, require_review=False)
    source = Path(recipe["loaded_trial"]["source_empty_session"]) / "round_status.json"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="status SHA"):
        protocol.validate_config(recipe, require_review=False)
    recipe["loaded_trial"]["source_empty_status_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    geometry = Path(recipe["trajectory"]["geometry_path"])
    geometry.write_bytes(geometry.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="geometry SHA"):
        protocol.validate_config(recipe, require_review=False)


def test_missing_geometry_is_a_value_error(recipe):
    geometry = Path(recipe["trajectory"]["geometry_path"])
    geometry.unlink()
    with pytest.raises(ValueError, match="geometry file"):
        protocol.validate_config(recipe, require_review=False)


def test_missing_source_evidence_is_a_value_error(recipe):
    source = Path(recipe["loaded_trial"]["source_empty_session"])
    (source / "dispatch.csv").unlink()
    with pytest.raises(ValueError, match="missing loaded diagnostic evidence"):
        protocol.validate_config(recipe, require_review=False)


@pytest.mark.parametrize("section,key,value", [
    (None, "empty_load", False), (None, "status", "CAPTURE_COMPLETE"),
    ("execution", "completed", False), ("execution", "commands_dispatched", 4500),
    ("execution", "same_session_realtime_cleanup_completed", False),
    ("tcp_plan", "geometry_sha256", "b"*64),
])
def test_failed_or_incompatible_empty_motion_cannot_be_a_source(recipe, section, key, value):
    path = Path(recipe["loaded_trial"]["source_empty_session"]) / "round_status.json"
    state = read(path)
    (state if section is None else state[section])[key] = value
    atomic_json(path, state)
    recipe["loaded_trial"]["source_empty_status_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        protocol.validate_config(recipe, require_review=False)


def test_frozen_binding_retains_motion_provenance_without_quality_claim(recipe, tmp_path):
    c = reviewed(recipe)
    plan = plan_for(c, tmp_path)
    binding = protocol.freeze_binding(tmp_path, c, plan)
    state = {"tcp_plan": plan, "loaded_trial_binding": binding,
             "leg_start_origin": read(tmp_path / "leg_start_origin.json")}
    assert protocol.require_binding(tmp_path, c, state) == binding
    assert binding["source_empty_motion_completed"] is True
    assert "not asserted" in binding["source_empty_quality"]
    assert set(binding["source_empty_sha256"]) == set(protocol.SOURCE_FILES)
    assert set(binding["source_code_sha256"]) == set(protocol.SOURCE_CODE)
    assert c["robot_observation"]["unloaded_wrench_bias"] is None
    assert c["robot_observation"]["compensation_verified"] is False


@pytest.mark.parametrize("kind", ["file", "state", "configuration", "plan", "source_dispatch", "source_config",
    "origin_file", "origin_state", "origin_missing", "origin_state_missing"])
def test_frozen_binding_rejects_mutations(recipe, tmp_path, kind):
    c = reviewed(recipe)
    plan = plan_for(c, tmp_path)
    binding = protocol.freeze_binding(tmp_path, c, plan)
    state = {"tcp_plan": deepcopy(plan), "loaded_trial_binding": deepcopy(binding),
             "leg_start_origin": read(tmp_path / "leg_start_origin.json")}
    if kind == "file":
        saved = deepcopy(binding)
        saved["status"] = "UNBOUND"
        atomic_json(tmp_path / "loaded_trial_binding.json", saved)
    elif kind == "state":
        state["loaded_trial_binding"]["status"] = "UNBOUND"
    elif kind == "configuration":
        c["loaded_trial"]["review_notes"] += " changed"
    elif kind == "plan":
        state["tcp_plan"]["path_length_m"] += .001
    elif kind == "origin_file":
        origin = read(tmp_path / "leg_start_origin.json")
        origin["axes"]["hip"]["physical_angle_deg"] = 1.
        atomic_json(tmp_path / "leg_start_origin.json", origin)
    elif kind == "origin_state":
        state["leg_start_origin"]["axes"]["hip"]["physical_angle_deg"] = 1.
    elif kind == "origin_missing":
        (tmp_path / "leg_start_origin.json").unlink()
    elif kind == "origin_state_missing":
        del state["leg_start_origin"]
    else:
        source = Path(c["loaded_trial"]["source_empty_session"])
        target = source / ("dispatch.csv" if kind == "source_dispatch" else "prepared_config.json")
        target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        protocol.require_binding(tmp_path, c, state)


def test_freeze_requires_an_actual_usable_leg_origin(recipe, tmp_path):
    c = reviewed(recipe)
    plan = plan_for(c, tmp_path)
    origin_path = tmp_path / "leg_start_origin.json"
    origin = read(origin_path)
    origin_path.unlink()
    with pytest.raises(ValueError, match="invalid loaded diagnostic evidence"):
        protocol.freeze_binding(tmp_path, c, plan)
    assert not (tmp_path / "loaded_trial_binding.json").exists()
    origin["usable_for_motion"] = False
    atomic_json(origin_path, origin)
    with pytest.raises(ValueError, match="usable captured leg start"):
        protocol.freeze_binding(tmp_path, c, plan)


@pytest.mark.parametrize("axis,offset", [(0, .006), (3, .1)])
def test_freeze_rejects_anchor_outside_reviewed_position_or_rotation(recipe, tmp_path, axis, offset):
    c = reviewed(recipe)
    plan = plan_for(c, tmp_path)
    c["loaded_trial"]["reviewed_anchor_pose_base"][axis] += offset
    with pytest.raises(ValueError, match="reviewed pose"):
        protocol.freeze_binding(tmp_path, c, plan)


def test_anchor_rotation_comparison_accepts_equivalent_full_turn(recipe, tmp_path):
    c = reviewed(recipe)
    plan = plan_for(c, tmp_path)
    c["loaded_trial"]["reviewed_anchor_pose_base"][3] += 2*np.pi
    assert protocol.freeze_binding(tmp_path, c, plan)["status"] == "BOUND"


def test_schema_missing_or_extra_fields_fail_closed(recipe):
    del recipe["loaded_trial"]["manual_path_checked"]
    with pytest.raises(ValueError):
        protocol.validate_config(recipe, require_review=False)
    recipe["loaded_trial"]["manual_path_checked"] = False
    recipe["loaded_trial"]["skip_checks"] = True
    with pytest.raises(ValueError):
        protocol.validate_config(recipe, require_review=False)


def test_rigid_shank_cannot_be_enabled_by_review_flags(recipe, tmp_path):
    c = reviewed(recipe)
    c["loaded_trial"]["connection_type"] = "rigid_fixture"
    protocol.validate_config(c, require_review=False)
    with pytest.raises(ValueError, match="coupled position/orientation"):
        protocol.validate_config(c)
    with pytest.raises(ValueError, match="coupled position/orientation"):
        protocol.freeze_binding(tmp_path, c, plan_for(c, tmp_path))
