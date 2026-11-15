"""Synthetic file design checks; no real device or network access."""
import csv
import hashlib
import json
import socket
import numpy as np
import pytest
from scripts import design_taught_trajectory as designer
from scripts.preview_recorded_trajectory import REQUIRED_FIELDS
from control.taught_tcp_trajectory import build_taught_tcp_trajectory, audit_taught_tcp_trajectory
from control.start_anchored_relative_trajectory import load_rehab_frame_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    recording=tmp_path/"recording";recording.mkdir()
    config=json.loads((designer.ROOT/"config/experiment.json").read_text(encoding="utf-8"))
    for key,value in (("max_tcp_speed_m_s",.004),("max_tcp_acceleration_m_s2",.001)):
        config["motion"][key]=config["empty_load"][key]=value
    config_path=tmp_path/"experiment.json";write_json(config_path,config)
    (recording/"config_snapshot.json").write_bytes(config_path.read_bytes())
    rows=[];anchor=np.array([.7,-.1,.3])
    for index,time in enumerate(np.arange(0,90.0001,.05)):
        if time<=12.5:delta=np.zeros(3)
        elif time<=35:delta=designer.blend7((time-12.5)/22.5)*np.array([-.02,.03,.08])
        elif time<=50:delta=np.array([-.02,.03,.08])+designer.blend7((time-35)/15)*np.array([.01,.04,-.01])
        elif time<=80:
            u=designer.blend7((time-50)/30);delta=(1-u)*np.array([-.01,.07,.07])+u*np.array([.003,.001,-.001])
        else:delta=np.array([.003,.001,-.001])
        row=dict(row_index=index,row_kind="state",host_time_s=100+float(time),query_start_s=100+float(time)+.001,
            query_end_s=100+float(time)+.002,sequence_id=index+1,frame_valid=True,valid=True,invalid_reason="")
        row.update(dict(zip(("tcp_x","tcp_y","tcp_z"),anchor+delta)));row.update(tcp_rx=3.1,tcp_ry=0.,tcp_rz=.2)
        rows.append(row)
    with (recording/"recorded_tcp.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=REQUIRED_FIELDS);writer.writeheader();writer.writerows(rows)
    write_json(recording/"recording_status.json",dict(schema_version=1,status="COMPLETE",partial=False,ready_for_replay=False))
    manifest=dict(schema_version=1,config_sha256=sha(config_path),config_snapshot_sha256=sha(config_path),
        units=dict(position="base_m",orientation="euler_xyz_rad",joints="rad"),source_sha256={"synthetic":"qa"},source_git={"commit":"fake"})
    manifest["output_files_sha256"]={name:sha(recording/name) for name in ("recorded_tcp.csv","recording_status.json")}
    write_json(recording/"recording_manifest.json",manifest)
    return recording,config_path,config


def rehash(recording):
    path=recording/"recording_manifest.json";manifest=json.loads(path.read_text())
    manifest["output_files_sha256"]={name:sha(recording/name) for name in ("recorded_tcp.csv","recording_status.json")}
    write_json(path,manifest)


def test_shape_extrema_closure_and_original_bytes(source):
    recording,_,_=source;original={p.name:p.read_bytes() for p in recording.iterdir()}
    geometry,_=designer.process_source(recording)
    lift,reach,ret=[np.array(s["control_points_delta_base_m"]) for s in geometry["segments"]]
    assert np.array_equal(lift[0],np.zeros(3)) and np.array_equal(ret[-1],np.zeros(3))
    assert np.array_equal(lift[-1],reach[0]) and np.array_equal(reach[-1],ret[0])
    assert lift[-1]==pytest.approx([-.02,.03,.08],abs=1e-7)
    assert reach[-1]==pytest.approx([-.01,.07,.07],abs=1e-7)
    lower=np.array(geometry["source"]["raw_envelope_delta_base_m"]["min"])
    upper=np.array(geometry["source"]["raw_envelope_delta_base_m"]["max"])
    assert all(((c>=lower-1e-12)&(c<=upper+1e-12)).all() for c in (lift,reach,ret))
    closure=geometry["source"]["closure"]
    assert closure["raw_closure_error_m"]==pytest.approx(np.linalg.norm([.003,.001,-.001]),abs=1e-7)
    assert closure["applied_only_to_phase"]=="return"
    assert {p.name:p.read_bytes() for p in recording.iterdir()}==original


@pytest.mark.parametrize("status",["FAILED","CANCELLED"])
def test_incomplete_source(source,status):
    recording,_,_=source;path=recording/"recording_status.json";value=json.loads(path.read_text())
    value.update(status=status,partial=True);write_json(path,value);rehash(recording)
    with pytest.raises(ValueError,match="complete recording"):designer.process_source(recording)


def test_source_tamper(source):
    recording,_,_=source
    with (recording/"recorded_tcp.csv").open("a",encoding="utf-8") as stream:stream.write(" ")
    with pytest.raises(ValueError,match="SHA-256 mismatch"):designer.process_source(recording)


@pytest.mark.parametrize("start,end",[(-1,80),(80,12.5),(12.5,95)])
def test_trim_bounds(source,start,end):
    with pytest.raises(ValueError,match="invalid explicit trim"):
        designer.process_source(source[0],trim_start_s=start,trim_end_s=end)


def test_offline_design_retime_plots_and_empty_template(source,tmp_path,monkeypatch):
    recording,config_path,config=source;original=sha(config_path)
    def forbidden(*a,**k):raise AssertionError("offline design tried network")
    monkeypatch.setattr(socket,"socket",forbidden)
    output=tmp_path/"design";new_config=tmp_path/"empty_template.json"
    summary=designer.design(recording,config_path,output,output_config=new_config)
    assert summary["status"]=="OFFLINE_DESIGN_COMPLETE",summary.get("error")
    assert summary["ready_for_replay"] is False and summary["hardware_accessed"] is False
    assert summary["timing"]["recommended_total_period_s"]>73.5
    assert summary["timing"]["audited_maximum_speed_m_s"]<=.004
    assert summary["timing"]["audited_maximum_acceleration_m_s2"]<=.001
    settings=json.loads((output/"settings.json").read_text());geometry=json.loads((output/"geometry.json").read_text())
    assert settings["geometry_sha256"]==sha(output/"geometry.json")
    frame=load_rehab_frame_config(config_path)
    table,_,_=build_taught_tcp_trajectory(settings,current_tcp_start_pose=geometry["recorded_anchor_pose"],rehab_frame=frame)
    assert audit_taught_tcp_trajectory(table,designer.effective_safety(config),frame=frame)[0]==()
    assert np.allclose(table[["tcp_rx","tcp_ry","tcp_rz"]],np.array(geometry["recorded_anchor_pose"])[3:])
    candidate=json.loads(new_config.read_text())
    assert sha(config_path)==original and candidate["motion"]==config["motion"]
    assert candidate["force_training"]==config["force_training"] and candidate["calibration"]==config["calibration"]
    assert candidate["trajectory"]["mode"]=="taught_tcp" and candidate["training"]["mode"]=="disabled"
    assert candidate["training"]["enabled"] is False and candidate["validation"]["capture_dummy_leg"] is False
    assert len(list(output.glob("*.png")))==6
    for name,digest in summary["output_files_sha256"].items():assert sha(output/name)==digest


def test_existing_paths_and_invalid_scales(source,tmp_path):
    recording,config_path,_=source;existing=tmp_path/"existing";existing.mkdir()
    with pytest.raises(FileExistsError):designer.design(recording,config_path,existing)
    summary=designer.design(recording,config_path,tmp_path/"invalid_scale",scale_x=0)
    assert summary["status"]=="FAILED" and "scales" in summary["error"]
    config=tmp_path/"preserve.json";config.write_bytes(b"preserve")
    summary=designer.design(recording,config_path,tmp_path/"existing_config",output_config=config)
    assert summary["status"]=="FAILED" and config.read_bytes()==b"preserve"
