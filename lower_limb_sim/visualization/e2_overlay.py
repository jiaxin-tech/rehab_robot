"""Read-only frozen E2 data and branch-prefix visualization semantics."""
import csv
import json
from pathlib import Path
import numpy as np
from lower_limb_sim.five_leg_mujoco_v1.replay import replay_trajectory
from lower_limb_sim.mechanical_endpoints import (
    time_rms, branch_rms_components, BranchRMSReference, MechanicalReferenceContext,
    branch_balanced_reference_normalized_rms, full_cycle_dual_joint_rms,
)
from .mujoco_rehab_video import select

# Explicit artifact component order is the shared helper's Hf/He/Kf/Ke order.
KEYS=('hip_flexion','hip_extension','knee_flexion','knee_extension')
LABELS=('HIP FLEXION','HIP EXTENSION','KNEE FLEXION','KNEE EXTENSION')


class FrozenE2Study:
    def __init__(self):
        self.root=Path(__file__).resolve().parents[1]/'five_leg_mujoco_v1'/'results_e2_necessity_v1'
        self.summary=json.loads((self.root/'study_summary.json').read_text())
        with (self.root/'e2_full_landscapes.csv').open() as f:self.rows=list(csv.DictReader(f))
        self.lookup={(r['leg_id'],float(r['beta_flex']),float(r['beta_extend'])):r for r in self.rows}
        if len(self.lookup)!=3125:raise ValueError('expected frozen 5 x 625 landscape')
        if self.summary['E2_PERSONALIZATION_NECESSITY']!='NOT_SUPPORTED':raise ValueError('unexpected frozen conclusion')
        self.oracles={r['leg_id']:r for r in self.summary['E2_oracle_characterization']}
        self.common=self.summary['E2_common_candidate']

    def row(self,leg,beta):return self.lookup[(leg,float(beta[0]),float(beta[1]))]


def limiting(values,compact=False):
    values=np.asarray(values)
    ids=np.flatnonzero(np.isclose(values,values.max(),atol=1e-12,rtol=0))
    if len(ids)==4:return 'ALL FOUR (TIE)'
    return ' / '.join(('HF','HE','KF','KE')[i] if compact else LABELS[i] for i in ids)


class E2Episode:
    def __init__(self,renderer,beta,study):
        self.renderer=renderer;self.beta=tuple(beta);self.candidate=select(renderer.domain,beta)
        self.row=study.row(renderer.leg.leg_id,beta)
        ref=renderer.reference
        def replay(candidate):
            t=candidate.trajectory
            result=replay_trajectory(model=renderer.model,definition=renderer.definition,leg=renderer.leg,
                time_s=ref.time_s,q_project=t.q,dq_project=t.dq,ddq_project=t.ddq)
            if not result.valid:raise ValueError(result.invalid_reason)
            return np.asarray(result.tau_hip_nm),np.asarray(result.tau_knee_nm)
        reference_torque=replay(renderer.domain.reference)
        self.torque=replay(self.candidate)
        self.denominator=np.asarray(branch_rms_components(*reference_torque,ref.time_s,ref.phases))
        self.components=np.asarray(branch_rms_components(*self.torque,ref.time_s,ref.phases))
        self.normalized=self.components/self.denominator
        p=renderer.domain.profile
        context=MechanicalReferenceContext(p.profile_id,p.version,p.fingerprint,ref.reference_version,
            tuple(float(v) for k,v in vars(renderer.leg).items() if isinstance(v,(int,float))))
        self.e2=branch_balanced_reference_normalized_rms(self.components,BranchRMSReference(context,tuple(self.denominator)),context=context)
        self.e0=full_cycle_dual_joint_rms(*self.torque,ref.time_s)
        self.frozen_error=abs(self.e2-float(self.row['E2']))
        np.testing.assert_allclose(self.normalized,[float(self.row[k]) for k in KEYS],atol=1e-12,rtol=0)
        np.testing.assert_allclose([self.e2,self.e0],[float(self.row['E2']),float(self.row['E0'])],atol=1e-10,rtol=1e-12)
        self.running=np.full((len(ref.time_s),4),np.nan)
        self.finalized=np.zeros((len(ref.time_s),4),bool)
        # Prefix estimates reuse shared time_rms; no instantaneous E2 is defined.
        for i in range(len(ref.time_s)):
            for j,(joint,branch) in enumerate(((0,'flexion'),(0,'extension'),(1,'flexion'),(1,'extension'))):
                available=np.flatnonzero((ref.phases==branch)&(np.arange(len(ref.time_s))<=i))
                if len(available)>=2:
                    self.running[i,j]=time_rms(self.torque[joint][available],ref.time_s[available])/self.denominator[j]
                    self.finalized[i,j]=i>=np.flatnonzero(ref.phases==branch)[-1]
        np.testing.assert_allclose(self.running[-1],self.normalized,atol=1e-12)

    def record(self):
        return dict(leg=self.renderer.leg.leg_id,beta=self.beta,E0=self.e0,E2=self.e2,
            components=dict(zip(KEYS,map(float,self.normalized))),limiting_branch=limiting(self.normalized),
            frozen_E2_error=self.frozen_error)


def draw_branch_panel(draw,text,episode,index,box,compact=False):
    """Four bars; pending branches have no fabricated zero value."""
    x,y,w,h=box
    final=index==len(episode.running)-1
    values=episode.running[index]
    valid=np.isfinite(values)
    maximum=np.nanmax(values) if valid.any() else None
    text((x,y),'FINAL COMPONENTS' if final else 'RUNNING COMPONENT ESTIMATE',14 if compact else 23,'#65d9d7')
    spacing=43 if compact else 104
    for j,label in enumerate(LABELS):
        yy=y+30+j*spacing if compact else y+55+j*spacing
        color='#efb860' if valid[j] and np.isclose(values[j],maximum,atol=1e-12,rtol=0) else '#65d9d7'
        value=f'{values[j]:.6f}' if valid[j] else '--'
        status='F' if episode.finalized[index,j] else ('R' if valid[j] else 'pending')
        text((x,yy),f'{("HF","HE","KF","KE")[j]} {value} {status}' if compact else label,17 if compact else 22,color)
        if not compact:text((x,yy+31),f'{value}  '+('FINAL' if status=='F' else 'RUNNING' if status=='R' else 'PENDING'),25,color)
        bar_y=yy+24 if compact else yy+66
        draw.rectangle((x,bar_y,x+w,bar_y+7),fill='#283a49')
        # Same zero-origin 0..2.5 ratio axis for every running and final panel.
        if valid[j]:draw.rectangle((x,bar_y,x+w*min(values[j]/2.5,1),bar_y+7),fill=color)
        draw.line((x+w/2.5,bar_y-3,x+w/2.5,bar_y+10),fill='#dbe5ef',width=1)
    bottom=y+215 if compact else y+492
    text((x,bottom),f'FINAL E2 = {episode.e2:.6f}' if final else 'E2: pending full episode',19 if compact else 28,'#e4edf4')
    text((x,bottom+(27 if compact else 47)),('LIMIT: ' if compact else 'LIMITING BRANCH: ')+limiting(values,compact) if final else 'F=final; R=running' if compact else 'No instantaneous E2(t)',16 if compact else 20,'#9aaec0')
    if not compact:text((x,bottom+88),'Bar axis: 0–2.5; tick = reference 1.0',18,'#9aaec0')
