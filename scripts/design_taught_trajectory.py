"""Derive an explicitly processed, offline-only TCP shape from verified teaching."""
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
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from scripts.preview_recorded_trajectory import _verify_inputs, _samples, _orientation_angles_deg
from control.taught_tcp_trajectory import build_taught_tcp_trajectory, audit_taught_tcp_trajectory, stage_kinematic_extrema
from control.start_anchored_relative_trajectory import load_rehab_frame_config
from safety.experiment_safety import ExperimentSafetyConfig

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("lift", "high_reach", "return")
COLORS = ("#1b7897", "#be7a13", "#914f8b")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def blend7(u):
    return u**4 * (35 - 84*u + 70*u**2 - 20*u**3)


def arc_coordinates(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def arc_resample(points, spacing_m=.005):
    distance = arc_coordinates(points)
    keep = np.r_[True, np.diff(distance) > 1e-12]
    if distance[-1] <= 1e-8:
        raise ValueError("each taught stage must contain nonzero spatial motion")
    grid = np.linspace(0, distance[-1], max(4, int(np.ceil(distance[-1]/spacing_m))+1))
    result = np.column_stack([np.interp(grid, distance[keep], points[keep, axis]) for axis in range(3)])
    result[0], result[-1] = points[0], points[-1]
    return result


def process_source(session, *, trim_start_s=12.5, trim_end_s=80., filter_window_s=.5,
                   control_spacing_m=.02):
    """Validate source; expose every geometry change, retaining the initial anchor."""
    manifest, status, rows, hashes, config_sha = _verify_inputs(Path(session).resolve(strict=True))
    samples, dropped = _samples(rows)
    if status["status"] != "COMPLETE" or status.get("partial") is not False or dropped:
        raise ValueError("geometry design requires a complete recording with no invalid rows")
    pose = np.array([sample["pose"] for sample in samples])
    time = np.array([sample["time"] for sample in samples]); time = time-time[0]
    if np.diff(time).max() > .1:
        raise ValueError("source has a receive gap above 0.1 s; do not smooth across it")
    if not np.isfinite([trim_start_s, trim_end_s, filter_window_s, control_spacing_m]).all():
        raise ValueError("finite processing parameters required")
    if not 0 <= trim_start_s < trim_end_s <= time[-1] or filter_window_s <= 0 or control_spacing_m <= 0:
        raise ValueError("invalid explicit trim/filter/control spacing")
    indices = np.flatnonzero((time >= trim_start_s) & (time <= trim_end_s))
    if len(indices) < 15:
        raise ValueError("trim requires at least fifteen source points")
    raw_time, raw = time[indices], pose[indices, :3]
    top = int(np.argmax(raw[:, 2])); far = int(np.argmax(raw[:, 1]))
    if not 3 <= top < far < len(raw)-3:
        raise ValueError("maximum Z then maximum base Y must define three nonempty stages; manual review required")
    step = float(np.median(np.diff(raw_time)))
    grid = np.linspace(raw_time[0], raw_time[-1], int(np.ceil((raw_time[-1]-raw_time[0])/step))+1)
    actual_step = float(grid[1]-grid[0])
    window = max(5, int(round(filter_window_s/actual_step)))
    window += 1-window % 2
    window = min(window, len(grid) if len(grid) % 2 else len(grid)-1)
    uniform = np.column_stack([np.interp(grid, raw_time, raw[:, i]) for i in range(3)])
    filtered = savgol_filter(uniform, window, 3, axis=0, mode="interp")
    smoothed = np.column_stack([np.interp(raw_time, grid, filtered[:, i]) for i in range(3)])
    smoothing_error = np.linalg.norm(smoothed-raw, axis=1)
    anchor = pose[0].copy(); restorations = []
    for name, i, target in (("initial_recorded_anchor", 0, anchor[:3]),
                            ("source_maximum_Z", top, raw[top]),
                            ("source_maximum_base_Y", far, raw[far]),
                            ("source_trimmed_end", len(raw)-1, raw[-1])):
        restorations.append(dict(name=name, trimmed_index=i, change_m=(target-smoothed[i]).tolist(),
                                 norm_m=float(np.linalg.norm(target-smoothed[i]))))
        smoothed[i] = target
    lower = np.minimum(raw.min(axis=0), anchor[:3])-anchor[:3]
    upper = np.maximum(raw.max(axis=0), anchor[:3])-anchor[:3]
    closure = raw[-1]-anchor[:3]
    cuts = (0, top, far, len(raw)-1)
    segments, processed, clamp_reports = [], [], []
    for phase, begin, end in zip(PHASES, cuts[:-1], cuts[1:]):
        part = smoothed[begin:end+1]-anchor[:3]
        if phase == "return":
            distance = arc_coordinates(part)
            if distance[-1] <= 0:
                raise ValueError("return stage has no arc length")
            part = part-blend7(distance/distance[-1])[:, None]*closure
            part[-1] = 0.
        before = part.copy(); part = np.clip(part, lower, upper)
        control = arc_resample(part, control_spacing_m)
        if phase == "lift":
            control[0] = 0.; control[-1] = raw[top]-anchor[:3]
        elif phase == "high_reach":
            control[0] = raw[top]-anchor[:3]; control[-1] = raw[far]-anchor[:3]
        else:
            control[0] = raw[far]-anchor[:3]; control[-1] = 0.
        clamped = np.linalg.norm(part-before, axis=1)
        clamp_reports.append(dict(phase=phase, point_count=int(np.count_nonzero(clamped > 1e-12)),
                                  maximum_change_m=float(clamped.max()), control_count=len(control)))
        segments.append(dict(phase=phase, control_points_delta_base_m=control.tolist()))
        processed.append(part+anchor[:3])
    source = dict(trajectory_id="drag_taught_loop_v1", coordinate_system="robot_base_delta_m",
        source_session=str(Path(session).resolve()), recording_manifest_sha256=sha(Path(session)/"recording_manifest.json"),
        verified_output_sha256=hashes, recording_config_sha256=config_sha,
        recorded_source_sha256=manifest.get("source_sha256"), recorded_source_git=manifest.get("source_git"),
        trim=dict(requested_start_s=trim_start_s, requested_end_s=trim_end_s,
                  selected_start_s=float(raw_time[0]), selected_end_s=float(raw_time[-1]),
                  source_csv_lines=[int(indices[0]+2), int(indices[-1]+2)]),
        stage_definition=dict(method="explicit draft design: maximum base Z, then maximum base Y",
            top_time_s=float(raw_time[top]), top_source_csv_line=int(indices[top]+2),
            far_time_s=float(raw_time[far]), far_source_csv_line=int(indices[far]+2),
            human_shape_review_required=True),
        filter=dict(method="uniform-time interpolation then Savitzky-Golay and interpolation to source times",
                    polynomial_order=3, requested_window_s=filter_window_s, uniform_step_s=actual_step,
                    window_points=window, actual_window_s=window*actual_step,
                    maximum_smoothing_displacement_m=float(smoothing_error.max()),
                    rms_smoothing_displacement_m=float(np.sqrt(np.mean(smoothing_error**2)))),
        endpoint_restorations=restorations,
        closure=dict(raw_trimmed_end_minus_initial_anchor_m=closure.tolist(),
                     raw_closure_error_m=float(np.linalg.norm(closure)), applied_only_to_phase="return",
                     method="subtract closure vector times seventh-order rest-to-rest blend of normalized return arc length",
                     maximum_intended_correction_m=float(np.linalg.norm(closure)),
                     subsequent_envelope_clamp_report=clamp_reports),
        control_resampling=dict(method="uniform polyline arc-length control spacing", target_spacing_m=control_spacing_m,
                                interpolation="clamped uniform cubic B-spline inside control convex hull"),
        raw_envelope_delta_base_m=dict(min=lower.tolist(), max=upper.tolist()),
        orientation=dict(method="fixed at original recorded anchor; recorded orientation changes deliberately discarded",
                         original_maximum_change_deg=float(_orientation_angles_deg(pose[:, 3:]).max())),
        ready_for_replay=False, hardware_accessed=False)
    geometry = dict(schema_version=1, recorded_anchor_pose=anchor.tolist(), segments=segments, source=source)
    return geometry, dict(raw_xyz=raw, raw_time=raw_time, source_pose=pose, processed_stages=processed,
                          cuts=cuts, source_durations=np.diff(raw_time[list(cuts)]))


def effective_safety(config):
    motion = copy.deepcopy(config["motion"])
    for name in ("max_tcp_speed_m_s", "max_tcp_acceleration_m_s2"):
        motion[name] = min(motion[name], config["empty_load"][name])
        if not np.isfinite(motion[name]) or motion[name] <= 0:
            raise ValueError("positive finite physical limits required")
    return ExperimentSafetyConfig(**motion)


def numeric_kinematics(table):
    time = table.time_s.to_numpy(float)
    xyz = table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
    velocity = np.gradient(xyz, time, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
    return np.linalg.norm(velocity, axis=1), np.linalg.norm(acceleration, axis=1)


def retime(settings, anchor, config, *, frame):
    safety = effective_safety(config)
    margin = max(1.01, float(config["empty_load"].get("retiming_margin", 1.01)))
    duration_keys = ("lift_duration_s", "reach_duration_s", "return_duration_s")
    result, iterations = dict(settings), []
    for iteration in range(8):
        dense = dict(result, sample_period_s=.01)
        table, _, _ = build_taught_tcp_trajectory(dense, current_tcp_start_pose=anchor, rehab_frame=frame)
        v, a = numeric_kinematics(table); clock = table.time_s.to_numpy(float)
        analytic = stage_kinematic_extrema(result)
        start = result["start_hold_s"]; multipliers, stage_metrics = [], []
        for i, key in enumerate(duration_keys):
            end = start+result[key]
            selected = (clock >= start-1e-9) & (clock <= end+1e-9)
            vmax = max(float(v[selected].max()), analytic[i][0])
            amax = max(float(a[selected].max()), analytic[i][1])
            scale = max(1., margin*vmax/safety.max_tcp_speed_m_s,
                        np.sqrt(margin*amax/safety.max_tcp_acceleration_m_s2))
            multipliers.append(scale)
            stage_metrics.append(dict(phase=PHASES[i], duration_s=result[key],
                                      numerical_speed_m_s=vmax, numerical_acceleration_m_s2=amax,
                                      duration_scale=scale))
            start = end+(result["turn_hold_s"] if i == 1 else 0.)
        iterations.append(stage_metrics)
        if max(multipliers) > 1+1e-7:
            for key, scale in zip(duration_keys, multipliers): result[key] = float(np.ceil(result[key]*scale))
            continue
        reasons, vmax, amax = audit_taught_tcp_trajectory(table, safety, frame=frame)
        if reasons:
            ratio = max(1., margin*float(vmax or 0)/safety.max_tcp_speed_m_s,
                        np.sqrt(margin*float(amax or 0)/safety.max_tcp_acceleration_m_s2))
            if ratio <= 1+1e-7:
                raise ValueError("processed trajectory audit failed: "+";".join(reasons))
            for key in duration_keys: result[key] = float(np.ceil(result[key]*ratio))
            continue
        return result, dict(margin=margin, numerical_sample_period_s=.01,
            speed_limit_m_s=safety.max_tcp_speed_m_s, acceleration_limit_m_s2=safety.max_tcp_acceleration_m_s2,
            iterations=iterations, audited_maximum_speed_m_s=float(vmax),
            audited_maximum_acceleration_m_s2=float(amax),
            recommended_total_period_s=sum(result[k] for k in (*duration_keys, "start_hold_s", "turn_hold_s", "end_hold_s")))
    raise ValueError("retiming did not converge within eight offline iterations")


def make_plots(output, table, geometry, source):
    anchor=np.array(geometry["recorded_anchor_pose"]); raw=source["raw_xyz"]
    xyz=table[[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float); time=table.time_s.to_numpy(float)
    phase=table.trajectory_phase.astype(str).to_numpy(); files=[]
    def save(fig, name):
        fig.suptitle("OFFLINE DESIGN DRAFT | NOT APPROVED FOR REPLAY", fontsize=10)
        path=output/name; fig.savefig(path,dpi=145); plt.close(fig); files.append(path)
    def stages(axis, a, b=None, three=False):
        for name,color in zip(PHASES,COLORS):
            mask=phase==name
            if three: axis.plot(*xyz[mask].T,color=color,label=name)
            elif b is not None: axis.plot(xyz[mask,a],xyz[mask,b],color=color,label=name)
    fig=plt.figure(figsize=(9,7),constrained_layout=True); ax=fig.add_subplot(111,projection="3d")
    ax.plot(*raw.T,color="0.6",linewidth=.8,label="raw trimmed source"); stages(ax,0,three=True)
    span=max(np.ptp(np.vstack([raw,xyz]),axis=0).max()*1.1,.01); center=(np.vstack([raw,xyz]).min(axis=0)+np.vstack([raw,xyz]).max(axis=0))/2
    ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),zlim=(center[2]-span/2,center[2]+span/2),xlabel="Base X (m)",ylabel="Base Y (m)",zlabel="Base Z (m)")
    ax.set_box_aspect((1,1,1)); ax.legend(); save(fig,"trajectory_3d.png")
    fig,axes=plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
    for ax,(a,b) in zip(axes,((0,1),(0,2),(1,2))):
        ax.plot(raw[:,a],raw[:,b],color="0.6",linewidth=.8,label="raw"); stages(ax,a,b)
        ax.set(xlabel=f"Base {'XYZ'[a]} (m)",ylabel=f"Base {'XYZ'[b]} (m)"); ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);save(fig,"projections.png")
    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,constrained_layout=True)
    for i,a in enumerate("XYZ"):
        axes[i,0].plot(time,xyz[:,i]); axes[i,0].set_ylabel(f"Base {a} (m)")
        axes[i,1].plot(time,(xyz[:,i]-anchor[i])*1000); axes[i,1].set_ylabel(f"Delta {a} (mm)")
    for ax in axes[-1]:ax.set_xlabel("Designed time (s)")
    save(fig,"xyz_time.png")
    speed,accel=numeric_kinematics(table)
    for values,label,name in ((speed,"Speed (m/s)","speed_time.png"),(accel,"Acceleration (m/s2)","acceleration_time.png")):
        fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True);ax.plot(time,values);ax.set(xlabel="Designed time (s)",ylabel=label);ax.grid(alpha=.2);save(fig,name)
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    raw_angles=_orientation_angles_deg(source["source_pose"][:,3:])
    axes[0].plot(np.arange(len(raw_angles)),raw_angles,color="0.5");axes[0].set(xlabel="Original valid sample index",ylabel="Recorded rotation from anchor (deg)")
    axes[1].plot(time,np.zeros_like(time));axes[1].set(xlabel="Designed time (s)",ylabel="Designed rotation change (deg)",title="Fixed original anchor orientation: an explicit design change")
    save(fig,"orientation_time.png")
    return files


def design(session, config_path, output_dir, *, scale_x=1., scale_y=1., scale_z=1.,
           output_config=None, trim_start_s=12.5, trim_end_s=80., control_spacing_m=.02):
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=False)
    summary=dict(schema_version=1,status="FAILED",ready_for_replay=False,hardware_accessed=False,
                 created_utc=datetime.now(timezone.utc).isoformat())
    try:
        if output_config is not None and Path(output_config).exists():
            raise FileExistsError("output config already exists; original configurations are never overwritten")
        if not np.isfinite([scale_x,scale_y,scale_z]).all() or min(scale_x,scale_y,scale_z)<=0:
            raise ValueError("scales must be positive finite values")
        config=json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
        frame=load_rehab_frame_config(config_path)
        geometry,source=process_source(session,trim_start_s=trim_start_s,trim_end_s=trim_end_s,
                                       control_spacing_m=control_spacing_m)
        geometry["source"]["processing_source_sha256"]={name:sha(ROOT/name) for name in
            ("scripts/design_taught_trajectory.py","scripts/preview_recorded_trajectory.py","control/taught_tcp_trajectory.py")}
        geometry_path=output/"geometry.json";write_json(geometry_path,geometry)
        path_value=geometry_path.relative_to(ROOT).as_posix() if geometry_path.is_relative_to(ROOT) else str(geometry_path)
        settings=dict(geometry_path=path_value,geometry_sha256=sha(geometry_path),scale_x=float(scale_x),scale_y=float(scale_y),scale_z=float(scale_z),
            lift_duration_s=float(source["source_durations"][0]),reach_duration_s=float(source["source_durations"][1]),return_duration_s=float(source["source_durations"][2]),
            start_hold_s=2.,turn_hold_s=2.,end_hold_s=2.,sample_period_s=float(config["trajectory"]["sample_period_s"]))
        settings,timing=retime(settings,geometry["recorded_anchor_pose"],config,frame=frame)
        table,_,metadata=build_taught_tcp_trajectory(settings,current_tcp_start_pose=geometry["recorded_anchor_pose"],rehab_frame=frame)
        reasons,_,_=audit_taught_tcp_trajectory(table,effective_safety(config),frame=frame)
        if reasons:raise ValueError("final design audit: "+";".join(reasons))
        write_json(output/"settings.json",settings);table.to_csv(output/"trajectory_preview.csv",index=False)
        figures=make_plots(output,table,geometry,source);shape=[]
        for i,name in enumerate(PHASES):
            begin,end=source["cuts"][i:i+2];raw=source["raw_xyz"][begin:end+1]
            selected=table.trajectory_phase.eq(name);curve=table.loc[selected,[f"tcp_{a}_base" for a in "xyz"]].to_numpy(float)
            shape.append(dict(phase=name,raw_to_designed_max_nearest_distance_m=float(cKDTree(curve).query(raw)[0].max()),
                              designed_to_raw_max_nearest_distance_m=float(cKDTree(raw).query(curve)[0].max())))
        config_output=None
        if output_config is not None:
            candidate=copy.deepcopy(config);candidate["trajectory"]={"mode":"taught_tcp",**settings}
            candidate["training"]["enabled"]=False;candidate["training"]["mode"]="disabled"
            candidate["validation"]["capture_dummy_leg"]=False
            write_json(output_config,candidate);config_output=dict(path=str(Path(output_config).resolve()),sha256=sha(output_config),scope="EMPTY_LOAD_TEMPLATE_ONLY")
        summary.update(status="OFFLINE_DESIGN_COMPLETE",input_config_sha256=sha(config_path),geometry_sha256=sha(geometry_path),
            scales=dict(x=scale_x,y=scale_y,z=scale_z),timing=timing,shape_comparison=shape,
            processing=geometry["source"],settings=settings,trajectory_metadata=metadata,output_config=config_output,
            limitations=["Processed design draft only; no motion or replay authorization.",
                         "Source manual orientation was replaced by fixed original anchor orientation.",
                         "Distinct source branches retained; maximum-Z and maximum-Y stage choices require human review.",
                         "Numeric retiming changes the manual cadence and does not establish contact safety."],
            output_files_sha256={p.name:sha(p) for p in [geometry_path,output/"settings.json",output/"trajectory_preview.csv",*figures]})
    except (OSError,ValueError,TypeError,KeyError,OverflowError) as exc:
        summary["error"]=f"{type(exc).__name__}: {exc}"
    write_json(output/"summary.json",summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session",type=Path,required=True);parser.add_argument("--config",type=Path,default=ROOT/"config/experiment.json")
    parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--output-config",type=Path)
    parser.add_argument("--control-spacing-m",type=float,default=.02)
    for axis in "xyz":parser.add_argument("--scale-"+axis,type=float,default=1.)
    args=parser.parse_args(argv)
    result=design(args.session,args.config,args.output_dir,scale_x=args.scale_x,scale_y=args.scale_y,scale_z=args.scale_z,
                  output_config=args.output_config,control_spacing_m=args.control_spacing_m)
    print(json.dumps({k:result.get(k) for k in ("status","ready_for_replay","error")},ensure_ascii=False))
    print(str(args.output_dir.resolve()/"summary.json"))
    return 0 if result["status"]=="OFFLINE_DESIGN_COMPLETE" else 1


if __name__=="__main__":
    raise SystemExit(main())
