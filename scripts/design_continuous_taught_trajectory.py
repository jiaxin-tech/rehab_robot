"""Create a continuous Cartesian-speed preview from a frozen taught shape."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config.bundle import ROOT, configuration
from control.continuous_taught_trajectory import geometry_from_taught, build_continuous_taught_trajectory, audit_continuous_taught_trajectory
from control.continuous_taught_timing import review_continuous_cycle
from control.start_anchor import load_start_anchor
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from safety.experiment_safety import ExperimentSafetyConfig
from scripts.preview_rehab_trajectory import _plot_preview


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def design(source_settings_path, config_path, output_dir, *, output_config=None,
           anchor_path=None, cycle_duration_s=90., ramp_duration_s=5., sample_period_s=.02):
    output = Path(output_dir).resolve()
    if output.exists() or (output_config is not None and Path(output_config).exists()):
        raise FileExistsError("choose new output paths; preserve previous designs and configurations")
    parent_settings = json.loads(Path(source_settings_path).read_text(encoding="utf-8-sig"))
    config = configuration(config_path)
    geometry, comparison = geometry_from_taught(parent_settings)
    geometry["source"]["design_source_sha256"] = {
        name:sha(ROOT/name) for name in ("scripts/design_continuous_taught_trajectory.py", "control/continuous_taught_timing.py")}
    output.mkdir(parents=True, exist_ok=False)
    geometry_path = output/"geometry.json"
    write_json(geometry_path, geometry)
    relative_path = geometry_path.relative_to(ROOT).as_posix() if geometry_path.is_relative_to(ROOT) else str(geometry_path)
    settings = dict(geometry_path=relative_path, geometry_sha256=sha(geometry_path),
        scale_x=1., scale_y=1., scale_z=1., cycle_duration_s=cycle_duration_s,
        ramp_duration_s=ramp_duration_s, sample_period_s=sample_period_s)
    pose = (load_start_anchor(anchor_path).tcp_pose_base if anchor_path is not None
            else geometry["recorded_anchor_pose"])
    frame = load_rehab_frame_config(config_path)
    table, audit, metadata = build_continuous_taught_trajectory(settings,
        current_tcp_start_pose=pose, rehab_frame=frame)
    motion = copy.deepcopy(config["motion"])
    for key in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
        motion[key] = min(motion[key], config["empty_load"][key])
    review = review_continuous_cycle(settings, cycle_duration_s,
        speed_limit_m_s=motion["max_tcp_speed_m_s"], acceleration_limit_m_s2=motion["max_tcp_acceleration_m_s2"],
        margin=config["empty_load"]["retiming_margin"], actual_duration_s=float(table.time_s.iloc[-1]))
    reasons, vmax, amax = audit_continuous_taught_trajectory(table, ExperimentSafetyConfig(**motion),
        expected_trajectory_id=audit.trajectory_id, frame=frame)
    write_json(output/"settings.json", settings)
    table.to_csv(output/"trajectory_preview.csv", index=False)
    figures = _plot_preview(table, output)
    old_xyz = comparison["source_dense_delta_base_m"]+np.asarray(pose[:3])
    new_xyz = comparison["new_dense_delta_base_m"]+np.asarray(pose[:3])
    fig, axes = plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
    for axis,(a,b) in zip(axes,((0,1),(0,2),(1,2))):
        axis.plot(old_xyz[:,a],old_xyz[:,b],color=".65",label="Previous segmented path")
        axis.plot(new_xyz[:,a],new_xyz[:,b],color="#247e9b",label="Continuous rounded path")
        axis.set(xlabel=f"Base {'XYZ'[a]} (m)",ylabel=f"Base {'XYZ'[b]} (m)")
        axis.set_aspect("equal",adjustable="datalim");axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Geometry comparison | offline preview")
    comparison_path = output/"shape_comparison.png"
    fig.savefig(comparison_path,dpi=145);plt.close(fig);figures.append(comparison_path)
    times = table.time_s.to_numpy(float)
    xyz = table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
    interval_speed = np.linalg.norm(np.diff(xyz,axis=0),axis=1)/np.diff(times)
    mid = (times[:-1]+times[1:])/2
    cruise = (mid>=ramp_duration_s)&(mid<=cycle_duration_s-ramp_duration_s)
    target_speed = metadata["constant_cartesian_speed_m_s"]
    verification = dict(constant_speed_interval_s=[ramp_duration_s,cycle_duration_s-ramp_duration_s],
        commanded_chord_speed_min_m_s=float(interval_speed[cruise].min()),
        commanded_chord_speed_max_m_s=float(interval_speed[cruise].max()),
        commanded_chord_speed_maximum_relative_error=float(np.max(np.abs(interval_speed[cruise]/target_speed-1))),
        phase_boundary_speeds_m_s={name:target_speed for name in ("lift_to_high_reach","high_reach_to_return")},
        internal_holds_s=0., whole_cycle_s=float(times[-1]))
    fig,axis=plt.subplots(figsize=(10,4),constrained_layout=True)
    axis.plot(mid,interval_speed*100,label="Commanded interval speed",color="#247e9b")
    axis.axvspan(0,ramp_duration_s,color=".9",label="Smooth start / finish")
    axis.axvspan(cycle_duration_s-ramp_duration_s,cycle_duration_s,color=".9")
    for phase,label in (("lift","Highest point"),("high_reach","Farthest point")):
        boundary=metadata["phase_boundaries"][phase]["end_s"]
        axis.axvline(boundary,color=".45",linestyle="--",linewidth=1)
        axis.text(boundary+.6,target_speed*105,label,fontsize=8,rotation=0)
    axis.set(xlabel="Time (s)",ylabel="Cartesian speed (cm/s)",ylim=(0,target_speed*120),
             title=f"Continuous {cycle_duration_s:g} s path: no internal stop; constant middle speed")
    axis.grid(alpha=.2);axis.legend(loc="lower center",fontsize=8)
    profile_path=output/"speed_profile.png"
    fig.savefig(profile_path,dpi=145);plt.close(fig);figures.append(profile_path)
    candidate = copy.deepcopy(config)
    candidate["trajectory"] = dict(mode="continuous_taught_tcp", fixed_cycle_duration_s=float(cycle_duration_s), **settings)
    candidate["training"].update(enabled=False,mode="disabled")
    candidate["validation"]["capture_dummy_leg"] = False
    if output_config is not None:
        write_json(output_config,candidate)
    summary = dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        status="OFFLINE_DESIGN_COMPLETE" if review["allowed"] and not reasons else "OFFLINE_REVIEW_REQUIRED",
        ready_for_replay=False,hardware_accessed=False,physical_limits_changed=False,
        source_settings_sha256=sha(source_settings_path),source_config_sha256=sha(config_path),
        preview_anchor_source=str(Path(anchor_path).resolve()) if anchor_path is not None else "original_recorded_pose",
        preview_anchor_source_sha256=sha(anchor_path) if anchor_path is not None else None,
        preview_anchor_pose=list(pose),settings=settings,trajectory_metadata=metadata,
        execution_review=review,trajectory_audit_reasons=list(reasons),
        audited_maximum_speed_m_s=vmax,audited_maximum_acceleration_m_s2=amax,
        geometry_processing=comparison["processing"],speed_verification=verification,
        output_config=str(Path(output_config).resolve()) if output_config is not None else None,
        output_config_sha256=sha(output_config) if output_config is not None else None,
        processing_source_sha256={name:sha(ROOT/name) for name in (
            "control/continuous_taught_trajectory.py","control/continuous_taught_timing.py","scripts/design_continuous_taught_trajectory.py")},
        output_files_sha256={p.name:sha(p) for p in [geometry_path,output/"settings.json",output/"trajectory_preview.csv",*figures]})
    write_json(output/"summary.json",summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-settings",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--output-config",type=Path)
    parser.add_argument("--anchor",type=Path)
    parser.add_argument("--cycle-duration-s",type=float,default=90.)
    parser.add_argument("--ramp-duration-s",type=float,default=5.)
    args=parser.parse_args(argv)
    result=design(args.source_settings,args.config,args.output_dir,output_config=args.output_config,
        anchor_path=args.anchor,cycle_duration_s=args.cycle_duration_s,ramp_duration_s=args.ramp_duration_s)
    print(json.dumps({name:result[name] for name in ("status","execution_review","speed_verification")}))
    return 0 if result["status"]=="OFFLINE_DESIGN_COMPLETE" else 1


if __name__=="__main__":
    raise SystemExit(main())
