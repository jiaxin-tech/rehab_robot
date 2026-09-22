from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline,BPoly
import mujoco
from lower_limb_sim.trajectory_sensitivity.study import trajectory,COMPONENTS
from lower_limb_sim.five_leg_mujoco_v1 import load_frozen_benchmark_definition,build_leg_domain,make_mujoco_model,replay_trajectory
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain,MODEL_PATH
from external_simulation.myoleg_reference_trajectory_replay_v1.build_and_replay import prescribed_truth
from lower_limb_sim.mechanical_endpoints import branch_rms_components,full_cycle_dual_joint_rms

OUT=Path('outputs/e3_candidate_comparison')
GRID=np.round(np.arange(-4,5)*.03,8)
SHIFTS=(-.10,-.05,0.,.05,.10)
_BUMP=make_interp_spline([0,.5,1],[0,1,0],k=5,bc_type=([(1,0),(2,0)],[(1,0),(2,0)]))


def BUMP(s,nu=0):
    # C2 compact support protects the source's long flat endpoint regions.
    s=np.asarray(s); inside=(s>=.2)&(s<=.8)
    return np.where(inside,_BUMP(np.clip((s-.2)/.6,0,1),nu=nu)/.6**nu,0.)


def e3(components,reference):
    c,r=np.asarray(components),np.asarray(reference)
    if c.shape!=(4,) or r.shape!=(4,) or not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(c<0) or np.any(r<=0):
        raise ValueError('INVALID_E3_COMPONENTS')
    return float(np.mean(c/r))


def key_posture(r,mapping,af,ae,shift):
    # Offset knee angle at each branch's temporal midpoint, relative to that
    # leg's knee ROM span. Endpoint derivatives vanish; hip geometry unchanged.
    base,kin=trajectory(r,mapping,0,0,shift)
    old=r.time_s;turn=np.flatnonzero(r.phases=='flexion')[-1]
    bounds=(old[0],old[turn],old[-1]);span=np.ptp(r.q[:,1])
    q=r.q.copy();dq=r.dq.copy();ddq=r.ddq.copy()
    for j,alpha in enumerate((af,ae)):
        ids=np.arange(0,turn+1) if j==0 else np.arange(turn,len(old))
        duration=bounds[j+1]-bounds[j];s=(old[ids]-bounds[j])/duration
        for target,order in ((q,0),(dq,1),(ddq,2)):
            target[ids,1]+=alpha*span*BUMP(s,nu=order)/duration**order
    # Inspect dense interpolation including original q/dq/ddq; no clipping.
    reverse=0.
    for j in (0,1):
        ids=np.arange(0,turn+1) if j==0 else np.arange(turn,len(old))
        interp=BPoly.from_derivatives(old[ids],[[q[i,1],dq[i,1],ddq[i,1]] for i in ids])
        dense=np.linspace(bounds[j],bounds[j+1],2001)
        values=interp(dense);speed=interp(dense,nu=1)
        if min(values)<r.q[:,1].min()-2e-5 or max(values)>r.q[:,1].max()+2e-5:
            raise ValueError('KEY_POSTURE_OUTSIDE_ROM')
        reverse+=float(np.trapezoid(np.maximum(-speed*(1 if j==0 else -1),0),dense))
    f=(bounds[1]-bounds[0])/(bounds[-1]-bounds[0]);scales=np.where(np.arange(len(old))<=turn,(f+shift)/f,(1-f-shift)/(1-f))
    dq/=scales[:,None];ddq/=scales[:,None]**2
    base.update(q=q,dq=dq,ddq=ddq)
    kin.update(reverse_motion_deg=float(np.rad2deg(reverse)),hip_speed_peak_deg_s=float(np.rad2deg(np.max(abs(dq[:,0])))),
               knee_speed_peak_deg_s=float(np.rad2deg(np.max(abs(dq[:,1])))),
               hip_accel_peak_deg_s2=float(np.rad2deg(np.max(abs(ddq[:,0])))),
               knee_accel_peak_deg_s2=float(np.rad2deg(np.max(abs(ddq[:,1])))),
               knee_angle_change_peak_deg=float(np.rad2deg(np.max(abs(q[:,1]-r.q[:,1])))),
               flex_midpoint_offset_deg=float(np.rad2deg(af*span)),extend_midpoint_offset_deg=float(np.rad2deg(ae*span)))
    return base,kin


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'settings.json').write_text(json.dumps(dict(endpoint='E3 equal mean of four same-model original-reference RMS ratios',
        key_posture_ROM_offsets=GRID.tolist(),time_share_shifts=SHIFTS,
        relative_comparison_limits=dict(speed=1.5,acceleration=2.,joint_peak_torque=1.1),
        limits_semantics='illustrative offline comparison only; not robot or clinical limits',
        candidates_semantics='grid optima; no equal-budget BO experiment'),indent=2))
    previous=pd.read_csv('outputs/trajectory_sensitivity/all_candidates.csv')
    rows=[]
    # Reuse all historical scalar responses; E3 recomputation does not change truth.
    for family,mask in (
        ('BETA_ORIGINAL',previous.step1&(previous.beta_flex.abs()<=.03000001)&(previous.beta_extend.abs()<=.03000001)),
        ('BETA_WIDE',previous.step1),
        ('BETA_TIMING',previous.step2)):
        f=previous[mask].copy();f['family']=family
        f['parameter_flex']=f.beta_flex;f['parameter_extend']=f.beta_extend
        f['E3']=f[[k+'_relative' for k in COMPONENTS]].mean(axis=1)
        rows.extend(f.to_dict('records'))
    # Also retain exact original 625-point domains, rather than calling a 25-point
    # subset the full frozen baseline.
    old=pd.read_csv('lower_limb_sim/five_leg_mujoco_v1/results_e2_necessity_v1/e2_full_landscapes.csv')
    myo=pd.read_csv('outputs/myoleg_trajectory_loads/trajectory_loads.csv')
    frozen=[]
    for name,g in old.groupby('leg_id'):
        for x in g.to_dict('records'):
            frozen.append(dict(model=name,beta_flex=x['beta_flex'],beta_extend=x['beta_extend'],E0=x['E0'],E2=x['E2'],
                E3=np.mean([x[k] for k in ('hip_flexion','hip_extension','knee_flexion','knee_extension')])))
    for x in myo.to_dict('records'):
        frozen.append(dict(model='MYOLEG_NATIVE_P0',beta_flex=x['beta_flex'],beta_extend=x['beta_extend'],E0=x['E0_Nm'],E2=x['E2'],
            E3=np.mean([x[k+'_relative'] for k in ('hip_flexion','hip_extension','knee_flexion','knee_extension')])))
    pd.DataFrame(frozen).to_csv(OUT/'frozen_625_rescored.csv',index=False)
    definition=load_frozen_benchmark_definition()
    sources=[(l.leg_id,build_leg_domain(l),make_mujoco_model(definition,l),l) for l in definition.legs]
    sources.append(('MYOLEG_NATIVE_P0',native_domain(),mujoco.MjModel.from_xml_path(str(MODEL_PATH)),None))
    points=[(0.,0.,0.)]+[x for x in itertools.product(GRID,GRID,SHIFTS) if x!=(0.,0.,0.)]
    for name,domain,model,leg in sources:
        r=domain.subject_reference;mapping=r.as_v3_mapping()
        refrow=previous[(previous.model==name)&(previous.beta_flex==0)&(previous.beta_extend==0)&(previous.share_shift==0)].iloc[0]
        ref=np.array([refrow[k+'_RMS_Nm'] for k in COMPONENTS]);traces=[];keys=[]
        for i,(af,ae,shift) in enumerate(points):
            row=dict(model=name,family='KEY_POSTURE_TIMING',parameter_flex=af,parameter_extend=ae,share_shift=shift)
            try:
                t,kin=key_posture(r,mapping,af,ae,shift)
                if leg:
                    a=replay_trajectory(model=model,definition=definition,leg=leg,time_s=t['time_s'],q_project=t['q'],dq_project=t['dq'],ddq_project=t['ddq'])
                    if not a.valid:raise ValueError(a.invalid_reason)
                    tau=np.column_stack((a.tau_hip_nm,a.tau_knee_nm))
                else:
                    a,_=prescribed_truth(model,t);tau=a['tau_truth_nm']
                    if np.max(a['warning_count']) or np.max(abs(a['decomposition_residual_nm']))>1e-8:raise ValueError('MYOLEG_DYNAMICS_DIAGNOSTIC')
                c=branch_rms_components(*tau.T,t['time_s'],t['phases'])
                row.update(valid=True,invalid_reason='',**kin,E0=full_cycle_dual_joint_rms(*tau.T,t['time_s']),E2=float(max(c/ref)),E3=e3(c,ref),
                    hip_torque_peak_Nm=float(np.max(abs(tau[:,0]))),knee_torque_peak_Nm=float(np.max(abs(tau[:,1]))))
                row.update({k+'_RMS_Nm':v for k,v in zip(COMPONENTS,c)})
                row.update({k+'_relative':v for k,v in zip(COMPONENTS,c/ref)})
                if i==0:
                    np.testing.assert_allclose(c,ref,rtol=1e-12,atol=1e-10)
                traces.append(tau);keys.append((af,ae,shift))
            except ValueError as exc:
                row.update(valid=False,invalid_reason=str(exc))
            rows.append(row)
            if i==0 or (i+1)%100==0:print(name,i+1,'/405',row['valid'],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'candidates.csv',index=False)
        np.savez_compressed(OUT/(name+'_key_posture_torques.npz'),parameters=np.array(keys),tau_Nm=np.array(traces),source_time_s=r.time_s,phases=r.phases)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
