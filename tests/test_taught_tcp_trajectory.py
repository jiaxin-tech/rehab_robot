"""Offline taught geometry invariants, source binding and execution limits."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from control import taught_tcp_trajectory as taught

POSE = [.6, -.1, .3, 3., -.1, .7]


@pytest.fixture
def frame():
    return SimpleNamespace(rotation_base_from_rehab=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]))


@pytest.fixture
def geometry():
    return {"schema_version":1,"recorded_anchor_pose":[.5,0.,.2,0.,0.,0.],"source":{"synthetic":True},
        "segments":[
        {"phase":"lift","control_points_delta_base_m":[[0.,0.,0.],[.01,.01,.05],[.02,.05,.12],[.03,.1,.2]]},
        {"phase":"high_reach","control_points_delta_base_m":[[.03,.1,.2],[.04,.15,.2],[.05,.25,.17],[.06,.3,.15]]},
        {"phase":"return","control_points_delta_base_m":[[.06,.3,.15],[.05,.2,.1],[.02,.03,.02],[0.,0.,0.]]}]}


def write_geometry(path, geometry):
    path.write_text(json.dumps(geometry),encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def spec(tmp_path, geometry):
    path=tmp_path/"geometry.json"
    return dict(geometry_path=str(path),geometry_sha256=write_geometry(path,geometry),
        scale_x=1.,scale_y=1.,scale_z=1.,lift_duration_s=5.,reach_duration_s=5.,return_duration_s=5.,
        start_hold_s=.2,turn_hold_s=.2,end_hold_s=.2,sample_period_s=.05)


@pytest.fixture
def safety():
    return SimpleNamespace(max_tcp_speed_m_s=5.,max_tcp_acceleration_m_s2=10.,
        workspace_min_base_m=[0.,-1.,0.],workspace_max_base_m=[2.,2.,2.])


def build(spec,frame):
    return taught.build_taught_tcp_trajectory(spec,current_tcp_start_pose=POSE,rehab_frame=frame)


def test_shape_phases_rest_closure_and_orientation(spec,frame,safety):
    table,audit,meta=build(spec,frame)
    xyz=table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy()
    assert audit.trajectory_id=="drag_taught_loop_v1" and audit.trajectory_valid
    assert set(table.trajectory_phase)=={"start_hold","lift","high_reach","turn_hold","return","end_hold"}
    assert meta["duration_s"]==pytest.approx(15.6)
    assert np.array_equal(xyz[0],xyz[-1]) and np.array_equal(xyz[0],POSE[:3])
    assert np.allclose(table[[f"tcp_r{a}" for a in "xyz"]],POSE[3:])
    assert np.allclose(table[[f"delta_{a}_R" for a in "xyz"]].to_numpy() @ frame.rotation_base_from_rehab.T,xyz-POSE[:3])
    for phase in taught.PHASES:
        row=table.loc[table.trajectory_phase.eq(phase)].iloc[-1]
        assert np.array_equal(row[[f"tcp_v{a}_base" for a in "xyz"]].to_numpy(float),np.zeros(3))
        assert np.array_equal(row[[f"tcp_a{a}_base" for a in "xyz"]].to_numpy(float),np.zeros(3))
    assert taught.audit_taught_tcp_trajectory(table,safety,frame=frame)[0]==()


def test_base_scaling_convex_hull_and_relative_path(spec,frame,geometry,tmp_path,monkeypatch):
    monkeypatch.setattr(taught,"ROOT",tmp_path)
    elsewhere=tmp_path/"elsewhere";elsewhere.mkdir();monkeypatch.chdir(elsewhere)
    table,_,_=build(dict(spec,geometry_path="geometry.json",scale_x=.5,scale_y=1.5,scale_z=2.),frame)
    controls=np.vstack([s["control_points_delta_base_m"] for s in geometry["segments"]])*[.5,1.5,2.]+POSE[:3]
    xyz=table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy()
    assert (xyz>=controls.min(axis=0)-1e-12).all() and (xyz<=controls.max(axis=0)+1e-12).all()
    assert table.tcp_z_base.max()==pytest.approx(.7)
    assert table.tcp_y_base.max()==pytest.approx(.35)


def test_source_rehashed_on_each_build_and_audit(spec,frame,safety):
    from pathlib import Path
    table,_,_=build(spec,frame)
    path=Path(spec["geometry_path"]);path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError,match="SHA-256 mismatch"):build(spec,frame)
    assert taught.audit_taught_tcp_trajectory(table,safety,frame=frame)[0]==("tcp_plan_invalid",)


@pytest.mark.parametrize("key,value",[("scale_x",True),("scale_y",0),("scale_z",-1),("scale_x",float("nan")),
    ("return_duration_s",0),("start_hold_s",-.1),("lift_duration_s",.0001),("sample_period_s",.0001),
    ("geometry_sha256","A"*64)])
def test_invalid_spec(spec,frame,key,value):
    with pytest.raises(ValueError):build(dict(spec,**{key:value}),frame)


@pytest.mark.parametrize("damage",["unknown","version_bool","bool_control","open","join","peak","phase"])
def test_invalid_geometry_even_when_rehashed(spec,frame,geometry,tmp_path,damage):
    g=deepcopy(geometry)
    if damage=="unknown":g["unknown"]=1
    if damage=="version_bool":g["schema_version"]=True
    if damage=="bool_control":g["segments"][0]["control_points_delta_base_m"][1][0]=True
    if damage=="open":g["segments"][-1]["control_points_delta_base_m"][-1][0]=.001
    if damage=="join":g["segments"][1]["control_points_delta_base_m"][0][0]=.031
    if damage=="peak":g["segments"][1]["control_points_delta_base_m"][1][2]=.21
    if damage=="phase":g["segments"][1]["phase"]="return"
    path=tmp_path/"damaged.json"
    with pytest.raises(ValueError):build(dict(spec,geometry_path=str(path),geometry_sha256=write_geometry(path,g)),frame)


@pytest.mark.parametrize("column,value",[("tcp_x_base",.71),("tcp_rx",0),("delta_x_R",10),("tcp_vy_base",100),
    ("time_s",0),("trajectory_phase","return"),("trajectory_valid",False)])
def test_table_tampering(spec,frame,safety,column,value):
    table,_,_=build(spec,frame);table.loc[20,column]=value
    assert taught.audit_taught_tcp_trajectory(table,safety,frame=frame)[0]


def test_limits_dense_peak_and_binding(spec,frame,safety):
    table,_,_=build(dict(spec,sample_period_s=1.),frame)
    sampled=np.linalg.norm(table[[f"tcp_v{a}_base" for a in "xyz"]],axis=1).max()
    _,dense,_=taught.audit_taught_tcp_trajectory(table,safety,frame=frame)
    assert dense>sampled+1e-6
    limit=SimpleNamespace(**vars(safety));limit.max_tcp_speed_m_s=(sampled+dense)/2
    assert "trajectory_speed_limit_exceeded" in taught.audit_taught_tcp_trajectory(table,limit,frame=frame)[0]
    limit.max_tcp_acceleration_m_s2=1e-6
    assert "trajectory_acceleration_limit_exceeded" in taught.audit_taught_tcp_trajectory(table,limit,frame=frame)[0]
    limit.workspace_max_base_m=[.61,2,2]
    assert "trajectory_outside_reviewed_workspace" in taught.audit_taught_tcp_trajectory(table,limit,frame=frame)[0]
    assert taught.audit_taught_tcp_trajectory(table,safety)[0]==("tcp_plan_invalid",)
    anchor=SimpleNamespace(tcp_pose_base=[.61,*POSE[1:]])
    assert "tcp_plan_content_mismatch" in taught.audit_taught_tcp_trajectory(table,safety,frame=frame,anchor=anchor)[0]
