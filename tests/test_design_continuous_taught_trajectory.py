"""File-level checks of the continuous preview, configuration and provenance."""
import json
import socket
import numpy as np
import pandas as pd
import pytest
from config.bundle import ROOT, configuration
from scripts import design_continuous_taught_trajectory as design
from scripts import lab_session

SETTINGS=ROOT/"results/manual_lab/20261115_taught_70s_v1/settings.json"
CONFIG=ROOT/"config/taught_tcp_70s.empty.json"
ANCHOR=ROOT/"results/manual_lab/20261115_161440/taught_empty_01/anchor.json"


def test_continuous_design_has_no_phase_stops_and_keeps_limits_and_inputs(tmp_path,monkeypatch):
    originals={p:p.read_bytes() for p in (SETTINGS,CONFIG,ANCHOR)}
    def forbidden(*a,**kw):raise AssertionError("offline generation attempted network access")
    monkeypatch.setattr(socket,"socket",forbidden)
    output=tmp_path/"continuous";config=tmp_path/"new_config.json"
    result=design.design(SETTINGS,CONFIG,output,output_config=config,anchor_path=ANCHOR)
    assert result["status"]=="OFFLINE_DESIGN_COMPLETE"
    assert result["hardware_accessed"] is False and result["physical_limits_changed"] is False
    assert result["settings"]["cycle_duration_s"]==90
    assert result["trajectory_metadata"]["holds_s"]==0
    assert result["speed_verification"]["constant_speed_interval_s"]==[5.,85.]
    assert result["speed_verification"]["commanded_chord_speed_maximum_relative_error"]<.002
    assert result["trajectory_audit_reasons"]==[] and result["execution_review"]["allowed"]
    table=pd.read_csv(output/"trajectory_preview.csv")
    xyz=table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
    assert table.time_s.iloc[-1]==90 and np.allclose(xyz[0],xyz[-1],atol=1e-12)
    assert set(table.trajectory_phase)=={"lift","high_reach","return"}
    for phase in ("lift","high_reach"):
        row=table.loc[table.trajectory_phase.eq(phase)].iloc[-1]
        assert np.linalg.norm(row[[f"tcp_v{a}_base" for a in "xyz"]].to_numpy(float))>.03
    candidate=configuration(config);previous=configuration(CONFIG)
    assert candidate["motion"]==previous["motion"] and candidate["empty_load"]==previous["empty_load"]
    assert candidate["trajectory"]["mode"]=="continuous_taught_tcp"
    assert candidate["trajectory"]["fixed_cycle_duration_s"]==90
    assert not any(name in candidate["trajectory"] for name in ("lift_duration_s","turn_hold_s","start_hold_s"))
    assert lab_session.check_config(config,empty_load=True)==[]
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    assert all(design.sha(output/name)==digest for name,digest in result["output_files_sha256"].items())
    assert len(list(output.glob("*.png")))==6


def test_continuous_design_does_not_overwrite_existing_paths(tmp_path):
    output=tmp_path/"existing";output.mkdir()
    with pytest.raises(FileExistsError):design.design(SETTINGS,CONFIG,output)
