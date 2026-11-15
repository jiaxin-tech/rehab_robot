"""Single-clock reference, source integrity and physical checks, fully offline."""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.bundle import ROOT
from control import continuous_taught_trajectory as continuous
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from control.taught_tcp_trajectory import _load_geometry as load_segmented


SOURCE_SETTINGS = ROOT/"results/manual_lab/20261115_taught_70s_v1/settings.json"
POSE = (.73,-.11,.27,3.08,-.05,.75)


def write_geometry(path, geometry):
    path.write_text(json.dumps(geometry,allow_nan=False),encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated():
    settings=json.loads(SOURCE_SETTINGS.read_text(encoding="utf-8"))
    return continuous.geometry_from_taught(settings)


@pytest.fixture
def spec(tmp_path,generated):
    path=tmp_path/"geometry.json"
    return dict(geometry_path=str(path),geometry_sha256=write_geometry(path,generated[0]),
        scale_x=1.,scale_y=1.,scale_z=1.,cycle_duration_s=90.,ramp_duration_s=5.,sample_period_s=.02)


@pytest.fixture
def frame():
    return load_rehab_frame_config(ROOT/"config/taught_tcp_70s.empty.json")


@pytest.fixture
def safety():
    return SimpleNamespace(max_tcp_speed_m_s=.13,max_tcp_acceleration_m_s2=.45,
        workspace_min_base_m=[0.,-.15,.1],workspace_max_base_m=[1.,1.,1.2])


def build(spec,frame,pose=POSE):
    return continuous.build_continuous_taught_trajectory(spec,current_tcp_start_pose=pose,rehab_frame=frame)


def test_global_rounding_preserves_peak_amplitudes_and_records_changes(generated):
    geometry,diagnostics=generated
    old=json.loads(SOURCE_SETTINGS.read_text(encoding="utf-8"))
    assert geometry["source"]["parent_geometry_sha256"] == old["geometry_sha256"]
    control=np.array(geometry["control_points_delta_base_m"])
    model=continuous._curve(control)
    assert model["extrema"][1]["maximum"]==pytest.approx(.8925496429423121,abs=1e-12)
    assert model["extrema"][2]["maximum"]==pytest.approx(.7510453004226585,abs=1e-12)
    assert geometry["source"]["processing"]["phase_joins_rounded"] is True
    assert diagnostics["maximum_nearest_distance_m"]<.01
    assert geometry["source"]["processing"]["no_two_dimensional_projection"] is True
    assert np.array_equal(control[0],np.zeros(3)) and np.array_equal(control[-1],np.zeros(3))


def test_exact_90_cycle_has_no_hold_or_boundary_stop(spec,frame,safety):
    table,audit,meta=build(spec,frame)
    assert audit.trajectory_id=="drag_taught_continuous_v1" and audit.trajectory_valid
    assert table.time_s.iloc[0]==0 and table.time_s.iloc[-1]==90
    assert len(table)==4501 and set(table.trajectory_phase)=={"lift","high_reach","return"}
    assert meta["holds_s"]==0
    speed=np.linalg.norm(table[[f"tcp_v{a}_base" for a in "xyz"]],axis=1)
    assert np.array_equal(speed[[0,-1]],np.zeros(2))
    assert (speed[1:-1]>0).all()
    boundaries=[meta["phase_boundaries"][phase]["end_s"] for phase in ("lift","high_reach")]
    for boundary in boundaries:
        row=int(np.argmin(np.abs(table.time_s-boundary)))
        assert speed[row]>meta["constant_cartesian_speed_m_s"]*.999
    assert continuous.audit_continuous_taught_trajectory(table,safety,frame=frame)[0]==()


def test_middle_speed_is_verified_from_positions_not_only_declared_velocity(spec,frame):
    table,_,meta=build(spec,frame)
    xyz=table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
    clock=table.time_s.to_numpy(float)
    middle=(clock[:-1]>=5)&(clock[1:]<=85)
    chord_speed=np.linalg.norm(np.diff(xyz,axis=0),axis=1)/np.diff(clock)
    target=meta["constant_cartesian_speed_m_s"]
    assert np.max(np.abs(chord_speed[middle]/target-1))<.002
    speed=np.linalg.norm(table[[f"tcp_v{a}_base" for a in "xyz"]],axis=1)
    assert np.max(np.abs(speed[(clock>=5)&(clock<=85)]/target-1))<1e-5
    assert meta["inverse_arc_speed_relative_error"]<1e-5
    assert meta["sampled_midcourse_chord_speed_relative_error"]<.002
    # The ramp has smooth zero acceleration at the very first and last sample.
    acceleration=table[[f"tcp_a{a}_base" for a in "xyz"]].to_numpy(float)
    assert np.array_equal(acceleration[[0,-1]],np.zeros((2,3)))


def test_closure_fixed_current_orientation_and_base_coordinate_transform(spec,frame):
    table,_,_=build(spec,frame)
    xyz=table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
    assert np.array_equal(xyz[0],POSE[:3]) and np.array_equal(xyz[0],xyz[-1])
    assert np.allclose(table[[f"tcp_r{a}" for a in "xyz"]],POSE[3:],atol=0,rtol=0)
    delta=table[[f"delta_{a}_R" for a in "xyz"]].to_numpy(float)
    assert np.allclose(delta@frame.rotation_base_from_rehab.T,xyz-POSE[:3],atol=1e-12,rtol=0)
    scaled,_,meta=build(dict(spec,scale_y=.9,scale_z=.8),frame)
    assert meta["geometric_extrema_delta_base_m"][1]["maximum"]==pytest.approx(.8925496429423121*.9,abs=1e-12)
    assert meta["geometric_extrema_delta_base_m"][2]["maximum"]==pytest.approx(.7510453004226585*.8,abs=1e-12)
    assert scaled.time_s.iloc[-1]==90


@pytest.mark.parametrize("column,value",[("tcp_x_base",.3),("tcp_rz",0),("delta_y_R",10),
    ("tcp_vz_base",100),("time_s",0),("trajectory_phase","return"),("trajectory_valid",False)])
def test_exact_reconstruction_rejects_table_tampering(spec,frame,safety,column,value):
    table,_,_=build(spec,frame)
    table.loc[100,column]=value
    assert continuous.audit_continuous_taught_trajectory(table,safety,frame=frame)[0]


def test_geometry_is_rehashed_for_build_and_audit(spec,frame,safety):
    from pathlib import Path
    table,_,_=build(spec,frame)
    path=Path(spec["geometry_path"]);path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError,match="SHA-256 mismatch"):build(spec,frame)
    assert continuous.audit_continuous_taught_trajectory(table,safety,frame=frame)[0]==("tcp_plan_invalid",)


@pytest.mark.parametrize("key,value",[("scale_x",True),("scale_y",0),("cycle_duration_s",0),
    ("ramp_duration_s",45),("sample_period_s",.1),("sample_period_s",.0001),("cycle_duration_s",float("nan"))])
def test_invalid_settings_fail_closed(spec,frame,key,value):
    with pytest.raises(ValueError):build(dict(spec,**{key:value}),frame)


def test_geometry_schema_cannot_hide_open_loop_or_wrong_phase_locations(spec,frame,generated,tmp_path):
    for damaged in ("open_loop","phase","unknown"):
        geometry=deepcopy(generated[0])
        if damaged=="open_loop":geometry["control_points_delta_base_m"][-1][0]=.001
        if damaged=="phase":geometry["phase_boundaries_arc_fraction"]=[.1,.9]
        if damaged=="unknown":geometry["unused"]=1
        path=tmp_path/(damaged+".json")
        changed=dict(spec,geometry_path=str(path),geometry_sha256=write_geometry(path,geometry))
        with pytest.raises(ValueError):build(changed,frame)


def test_limits_frame_anchor_and_sparse_speed_audit_remain_binding(spec,frame,safety):
    table,_,_=build(spec,frame)
    assert continuous.audit_continuous_taught_trajectory(table,safety)[0]==("tcp_plan_invalid",)
    shifted=SimpleNamespace(tcp_pose_base=(POSE[0]+.01,*POSE[1:]))
    assert "tcp_plan_content_mismatch" in continuous.audit_continuous_taught_trajectory(table,safety,frame=frame,anchor=shifted)[0]
    constrained=SimpleNamespace(**vars(safety));constrained.max_tcp_speed_m_s=.01
    constrained.max_tcp_acceleration_m_s2=.01
    reasons,_,_=continuous.audit_continuous_taught_trajectory(table,constrained,frame=frame)
    assert "trajectory_speed_limit_exceeded" in reasons and "trajectory_acceleration_limit_exceeded" in reasons
    constrained.workspace_max_base_m=[.75,.5,.8]
    assert "trajectory_outside_reviewed_workspace" in continuous.audit_continuous_taught_trajectory(table,constrained,frame=frame)[0]
    coarse,_,_=build(dict(spec,sample_period_s=.05),frame)
    assert "continuous_speed_uniformity_failed" not in continuous.audit_continuous_taught_trajectory(coarse,safety,frame=frame)[0]
