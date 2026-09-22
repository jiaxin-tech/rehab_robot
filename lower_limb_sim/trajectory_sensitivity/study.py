"""Wider coordination and fixed-cycle branch-duration exploration."""
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
import mujoco
from external_simulation.myoleg_v3_trajectory_parameterization_design_v1.parameterization import generate_v3_trajectory, interior_warp_basis
from external_simulation.myoleg_reference_trajectory_replay_v1.build_and_replay import prescribed_truth
from lower_limb_sim.five_leg_mujoco_v1 import load_frozen_benchmark_definition, build_leg_domain, make_mujoco_model, replay_trajectory
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain, MODEL_PATH
from lower_limb_sim.mechanical_endpoints import MechanicalReferenceContext, BranchRMSReference, branch_rms_components, branch_balanced_reference_normalized_rms, full_cycle_dual_joint_rms

OUT=Path('outputs/trajectory_sensitivity')
RANGES=(.03,.06,.09,.12)
GRID1=np.round(np.arange(-8,9)*.015,8)
GRID2=np.round(np.arange(-4,5)*.03,8)
SHIFTS=(-.10,-.05,0.,.05,.10)
COMPONENTS=('hip_flex','hip_extend','knee_flex','knee_extend')


def trajectory(reference, mapping, bf, be, share_shift=0.):
    """Reuse V3 geometry, then apply piecewise linear time scaling.

    Source branch boundary has zero velocity/acceleration, so this is C2.
    q(t_new)=q_old(t_old); dq_new=dq_old/a; ddq_new=ddq_old/a^2.
    """
    dense=np.linspace(0,1,10001)
    derivative=interior_warp_basis(dense)[1]
    minimum=min(np.min(1+bf*derivative),np.min(1+be*derivative))
    if minimum<=0: raise ValueError('NONMONOTONE_PHASE_WARP')
    v=generate_v3_trajectory(mapping,bf,be)
    time=reference.time_s.copy(); origin=time[0]; duration=time[-1]-origin
    turn=int(np.flatnonzero(reference.phases=='flexion')[-1])
    tf=time[turn]-origin; share=tf/duration; new_share=share+share_shift
    if not 0<new_share<1: raise ValueError('INVALID_BRANCH_DURATION')
    if np.max(np.abs(v.dq[turn]))>1e-9 or np.max(np.abs(v.ddq[turn]))>1e-8:
        raise ValueError('NONSTATIONARY_TURN_CANNOT_RETIME_C2')
    scales=np.where(np.arange(len(time))<=turn,new_share/share,(1-new_share)/(1-share))
    time=np.where(np.arange(len(time))<=turn,origin+(time-origin)*scales,
                  origin+duration*new_share+(time-origin-tf)*scales)
    q=v.q.copy();dq=v.dq/scales[:,None];ddq=v.ddq/scales[:,None]**2
    lo,hi=reference.q.min(axis=0),reference.q.max(axis=0)
    outside=max(float(np.max(lo-q)),float(np.max(q-hi)),0.)
    if outside>2e-5: raise ValueError('OUTSIDE_ROM_NO_CLIPPING')
    if not all(np.isfinite(a).all() for a in (q,dq,ddq,time)):
        raise ValueError('NONFINITE_TRAJECTORY')
    return dict(time_s=time,q=q,dq=dq,ddq=ddq,phases=reference.phases),dict(
        flexion_share=new_share,flexion_duration_s=duration*new_share,
        extension_duration_s=duration*(1-new_share),min_warp_derivative=float(minimum),
        ROM_excess_rad=outside,
        hip_speed_peak_deg_s=float(np.rad2deg(np.max(np.abs(dq[:,0])))),
        knee_speed_peak_deg_s=float(np.rad2deg(np.max(np.abs(dq[:,1])))),
        hip_accel_peak_deg_s2=float(np.rad2deg(np.max(np.abs(ddq[:,0])))),
        knee_accel_peak_deg_s2=float(np.rad2deg(np.max(np.abs(ddq[:,1])))),
        knee_angle_change_peak_deg=float(np.rad2deg(np.max(np.abs(q[:,1]-reference.q[:,1])))))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    # Written before any response is evaluated; descriptive offline grids, no tuning.
    (OUT/'settings.json').write_text(json.dumps(dict(beta_ranges=RANGES,
        step1_spacing=.015,step2_spacing=.03,share_shifts=SHIFTS,
        cycle_duration_s=24,normalization='original same-model (0,0,0) reference',
        interpretation='offline sensitivity, grid minima only; no robot limits validated'),indent=2))
    definition=load_frozen_benchmark_definition()
    sources=[(leg.leg_id,build_leg_domain(leg),make_mujoco_model(definition,leg),leg)
             for leg in definition.legs]
    sources.append(('MYOLEG_NATIVE_P0',native_domain(),mujoco.MjModel.from_xml_path(str(MODEL_PATH)),None))
    rows=[]
    for name,domain,model,leg in sources:
        r=domain.subject_reference;mapping=r.as_v3_mapping();p=domain.profile
        context=MechanicalReferenceContext(p.profile_id,p.version,p.fingerprint,r.reference_version,())
        points1={(float(a),float(b),0.) for a,b in itertools.product(GRID1,repeat=2)}
        points2={(float(a),float(b),float(s)) for a,b,s in itertools.product(GRID2,GRID2,SHIFTS)}
        points=[(0.,0.,0.)]+sorted((points1|points2)-{(0.,0.,0.)})
        reference=None; traces=[];valid_keys=[]
        for i,(bf,be,shift) in enumerate(points):
            base=dict(model=name,beta_flex=bf,beta_extend=be,share_shift=shift,
                      step1=(bf,be,shift) in points1,step2=(bf,be,shift) in points2)
            try:
                t,kin=trajectory(r,mapping,bf,be,shift)
                if leg is not None:
                    result=replay_trajectory(model=model,definition=definition,leg=leg,
                        time_s=t['time_s'],q_project=t['q'],dq_project=t['dq'],ddq_project=t['ddq'])
                    if not result.valid: raise ValueError(result.invalid_reason)
                    tau=np.column_stack((result.tau_hip_nm,result.tau_knee_nm))
                    residual=np.nan;warning=np.nan
                else:
                    arrays,_=prescribed_truth(model,t);tau=arrays['tau_truth_nm']
                    residual=float(np.max(np.abs(arrays['decomposition_residual_nm'])))
                    warning=int(np.max(arrays['warning_count']))
                    if residual>1e-8 or warning: raise ValueError('MYOLEG_DYNAMICS_DIAGNOSTIC')
                comp=branch_rms_components(*tau.T,t['time_s'],t['phases'])
                if reference is None:
                    if (bf,be,shift)!=(0.,0.,0.): raise RuntimeError('REFERENCE_REQUIRED')
                    reference=BranchRMSReference(context,comp)
                row=dict(base,valid=True,invalid_reason='',**kin,
                         E0=full_cycle_dual_joint_rms(*tau.T,t['time_s']),
                         E2=branch_balanced_reference_normalized_rms(comp,reference,context=context),
                         hip_torque_peak_Nm=float(np.max(np.abs(tau[:,0]))),
                         knee_torque_peak_Nm=float(np.max(np.abs(tau[:,1]))),
                         decomposition_residual_Nm=residual,warning_count=warning)
                row.update({key+'_RMS_Nm':float(x) for key,x in zip(COMPONENTS,comp)})
                row.update({key+'_relative':float(x/y) for key,x,y in zip(COMPONENTS,comp,reference.components)})
                traces.append(tau);valid_keys.append((bf,be,shift))
            except ValueError as exc:
                if i==0: raise
                row=dict(base,valid=False,invalid_reason=str(exc))
            rows.append(row)
            if i==0 or (i+1)%100==0:
                print(name,i+1,'/',len(points),'valid=',row['valid'],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'all_candidates.csv',index=False)
        np.savez_compressed(OUT/(name+'_torques.npz'),parameters=np.array(valid_keys),
                            tau_Nm=np.array(traces),source_time_s=r.time_s,phases=r.phases)
    summarize()


def summarize():
    data=pd.read_csv(OUT/'all_candidates.csv');summaries=[]
    for name,all_rows in data.groupby('model',sort=False):
        g=all_rows[all_rows.valid];ref=g[(g.beta_flex==0)&(g.beta_extend==0)&(g.share_shift==0)].iloc[0]
        for step in ('step1','step2'):
            for radius in (RANGES if step=='step1' else (.12,)):
                subset=g[g[step] & (g.beta_flex.abs()<=radius+1e-9)&(g.beta_extend.abs()<=radius+1e-9)]
                for endpoint in ('E0','E2'):
                    best=subset.loc[subset[endpoint].idxmin()]
                    summaries.append(dict(model=name,step=step,beta_radius=radius,endpoint=endpoint,
                        count=len(subset),reference=float(ref[endpoint]),minimum=float(best[endpoint]),
                        improvement_pct=100*(ref[endpoint]-best[endpoint])/ref[endpoint],
                        landscape_range_pct=100*(subset[endpoint].max()-subset[endpoint].min())/ref[endpoint],
                        beta_flex=best.beta_flex,beta_extend=best.beta_extend,share_shift=best.share_shift,
                        flexion_share=best.flexion_share,
                        beta_boundary=bool(max(abs(best.beta_flex),abs(best.beta_extend))>=radius-1e-9),
                        share_boundary=bool(abs(best.share_shift)>=.10-1e-9),
                        near_minimum_0p1pct=int((subset[endpoint]<=best[endpoint]+.001*ref[endpoint]).sum()),
                        best_hip_speed_ratio=best.hip_speed_peak_deg_s/ref.hip_speed_peak_deg_s,
                        best_knee_speed_ratio=best.knee_speed_peak_deg_s/ref.knee_speed_peak_deg_s,
                        best_hip_accel_ratio=best.hip_accel_peak_deg_s2/ref.hip_accel_peak_deg_s2,
                        best_knee_accel_ratio=best.knee_accel_peak_deg_s2/ref.knee_accel_peak_deg_s2))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__': run()
