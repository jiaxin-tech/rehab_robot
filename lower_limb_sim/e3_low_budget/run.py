from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import json,math
import numpy as np,pandas as pd
from lower_limb_sim.e3_candidate_comparison.run import key_posture,e3,COMPONENTS
from lower_limb_sim.trajectory_sensitivity.study import trajectory
from lower_limb_sim.five_leg_mujoco_v1 import load_frozen_benchmark_definition,build_leg_domain
from lower_limb_sim.visualization.myoleg_robot_scene import native_domain
from lower_limb_sim.mechanical_endpoints import EndpointIdentity,branch_rms_components
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject
from personalization.models.time_series_graybox import TimeSeriesFiveParameterGrayBoxAdapter
from personalization.models.residual_gp import ResidualGaussianProcess
from personalization.models.base import Prediction
from personalization.observations import EpisodeObservation
from personalization.identification import TimeSeriesIdentificationPayload,PROJECT_FORCE_MAPPING
from personalization.selectors.bo import ExpectedImprovementSelector
from personalization.selectors.greedy import ModelOnlyGreedySelector
from personalization.selectors.random import RandomSelector

OUT=Path('outputs/e3_low_budget');SOURCE=Path('outputs/e3_candidate_comparison')
TIERS=(0.,.01,.02);FAMILIES=('BETA_TIMING','KEY_POSTURE_TIMING')
METHODS=('REFERENCE','RANDOM','ADAPTIVE_GREEDY','PURE_BO_EI','MODEL_INFORMED_BO_EI')
E3=EndpointIdentity('E3_BRANCH_MEAN_REFERENCE_NORMALIZED_RMS','dimensionless')

@dataclass
class Point:
    candidate_id:str
    candidate_index:int
    beta_flex:float # Legacy payload identity slot: alpha for KEY_POSTURE, beta otherwise.
    beta_extend:float
    share_shift:float
    trajectory:object
    time_s:object
    @property
    def beta(self):return (self.beta_flex,self.beta_extend)
    @property
    def features(self):return np.array([self.beta_flex/.03,self.beta_extend/.03,self.share_shift/.05])

class Domain:
    def __init__(self,source,rows,family):
        self.profile=source.profile;self.subject_reference=source.subject_reference
        r=self.subject_reference;m=r.as_v3_mapping();self.points=[]
        # Only kinematic columns influence the domain; no force-based mask.
        for x in rows.sort_values(['parameter_flex','parameter_extend','share_shift']).itertuples():
            kin_ok=all(getattr(x,j+'_speed_peak_deg_s_ratio')<=1.5+1e-12 and getattr(x,j+'_accel_peak_deg_s2_ratio')<=2+1e-12 for j in ('hip','knee'))
            if not kin_ok:continue
            f=key_posture if family=='KEY_POSTURE_TIMING' else trajectory
            t,_=f(r,m,x.parameter_flex,x.parameter_extend,x.share_shift)
            i=len(self.points);p=Point(f'{family}:{i}',i,x.parameter_flex,x.parameter_extend,x.share_shift,SimpleNamespace(q=t['q'],dq=t['dq'],ddq=t['ddq']),t['time_s'])
            self.points.append(p)
        self.lookup={p.candidate_id:p for p in self.points}
        self.reference=next(p for p in self.points if (*p.beta,p.share_shift)==(0.,0.,0.))
    def __iter__(self):return iter(self.points)
    def __len__(self):return len(self.points)
    def by_id(self,key):return self.lookup[key]

class Adapter(TimeSeriesFiveParameterGrayBoxAdapter):
    endpoint=E3
    def predict_value(self,candidate):
        p=self._candidate(candidate);h,k=self.predict_torques(p)
        comp=branch_rms_components(h,k,p.time_s,self.domain.subject_reference.phases)
        if self._reference_cache is None:
            r=self.domain.reference;rh,rk=self.predict_torques(r)
            self._reference_cache=branch_rms_components(rh,rk,r.time_s,self.domain.subject_reference.phases)
        return e3(comp,self._reference_cache)

class GP3D:
    """Same fixed Matérn-5/2 covariance/noise/jitter; explicit third coordinate."""
    def __init__(self):self.core=ResidualGaussianProcess();self.x=np.empty((0,3));self.chol=None;self.alpha=None
    def fit(self,x,y):
        self.x=np.asarray(x,dtype=float).reshape(-1,3);y=np.asarray(y)
        if not len(y):self.chol=self.alpha=None;return
        cov=self.core._kernel(self.x,self.x)+np.eye(len(y))*(1e-12+self.core.jitter)
        extra=self.core.jitter
        for _ in range(6):
            try:self.chol=np.linalg.cholesky(cov+extra*np.eye(len(y)));break
            except np.linalg.LinAlgError:extra*=10
        else:raise RuntimeError('GP_FACTORIZATION_FAILED')
        self.alpha=np.linalg.solve(self.chol.T,np.linalg.solve(self.chol,y))
    def predict(self,x):
        if self.chol is None:return 0.,self.core.signal_std
        c=self.core._kernel(self.x,np.asarray(x)[None,:])[:,0];v=np.linalg.solve(self.chol,c)
        return float(c@self.alpha),math.sqrt(max(self.core.signal_std**2-float(v@v),0))

class Model:
    def __init__(self,domain,method):
        self.domain,self.method=domain,method;self.gp=GP3D();self.mean=0.;self.physics={};self.adapter=None
        if method!='PURE_BO_EI':
            self.adapter=Adapter(domain,baseline_template=baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS['baseline']),L1=.42,L2=.30)
    def fit(self,history):
        valid=[h for h in history if h.valid]
        if self.adapter:
            self.adapter.fit(history)
            self.physics={p.candidate_id:self.adapter.predict_value(p) for p in self.domain}
        y=np.array([h.endpoint_value for h in valid]);x=[self.domain.by_id(h.candidate_id).features for h in valid]
        self.mean=float(y.mean()) if self.adapter is None else 0.
        target=y-self.mean if self.adapter is None else y-np.array([self.physics[h.candidate_id] for h in valid])
        if self.method!='ADAPTIVE_GREEDY':self.gp.fit(x,target)
    def predict(self,p):
        baseline=self.physics[p.candidate_id] if self.adapter else self.mean
        mean,std=(0.,0.) if self.method=='ADAPTIVE_GREEDY' else self.gp.predict(p.features)
        return Prediction(mean=baseline+mean,std=std,valid=True,metadata={})

class ReplayStore:
    """Private environment backend. Releases only the requested trajectory.

    Replays saved deterministic MuJoCo torque samples, not a fresh simulator run.
    No predictions, ranks or feasible-candidate list leave this object.
    """
    def __init__(self,name,family):
        path=(SOURCE/(name+'_key_posture_torques.npz') if family=='KEY_POSTURE_TIMING' else Path('outputs/trajectory_sensitivity')/(name+'_torques.npz'))
        a=np.load(path);self._values={tuple(np.round(k,8)):v for k,v in zip(a['parameters'],a['tau_Nm'])}
    def requested(self,p):return self._values[tuple(np.round([*p.beta,p.share_shift],8))].copy()

class Environment:
    def __init__(self,domain,store,tier,physics):
        self.domain,self.store,self.tier,self.physics=domain,store,tier,physics
        self.reference=None;self.peaks=None;self.calls=[]
    def evaluate(self,p,index):
        if not self.calls and p.candidate_id!=self.domain.reference.candidate_id:raise ValueError('REFERENCE_FIRST')
        self.calls.append(p.candidate_id);tau=self.store.requested(p);r=self.domain.subject_reference;profile=self.domain.profile
        c=np.array(branch_rms_components(*tau.T,p.time_s,r.phases));peaks=np.max(abs(tau),axis=0)
        if self.reference is None:self.reference=c;self.peaks=peaks
        feasible=bool(np.all(c/self.reference<=1+self.tier+1e-12) and np.all(peaks/self.peaks<=1.1+1e-12))
        raw=dict(E3=e3(c,self.reference),E2=float(max(c/self.reference)),peak_ratio=float(max(peaks/self.peaks)),feasible=feasible)
        payload=None;episode=f'episode:{index}'
        if feasible and self.physics:
            t=p.trajectory;jt=leg_jacobian(t.q[:,0],t.q[:,1],.42,.30).swapaxes(-1,-2)
            force=np.linalg.solve(jt,tau[...,None])[...,0]
            payload=TimeSeriesIdentificationPayload(episode_id=episode,candidate_id=p.candidate_id,rom_profile_id=profile.profile_id,rom_version=profile.version,rom_fingerprint=profile.fingerprint,reference_version=r.reference_version,
                beta_flex=p.beta_flex,beta_extend=p.beta_extend,time_s=p.time_s,q=t.q,dq=t.dq,ddq=t.ddq,planar_force_n=force,sample_valid=np.ones(len(p.time_s),bool),L1=.42,L2=.30,force_mapping=PROJECT_FORCE_MAPPING,
                mapping_provenance='Saved MuJoCo torque replay; algebraic equivalent force, not cuff measurement',classification='OFFLINE_E3_THREE_DIMENSIONAL_REPLAY')
        obs=EpisodeObservation(episode,index,p.candidate_id,*p.beta,E3.endpoint_name,raw['E3'] if feasible else None,E3.unit,0.,feasible,invalid_reason=None if feasible else 'OBSERVED_LOAD_CONSTRAINT_VIOLATION',identification_payload=payload)
        return obs,raw

def sequential(domain,store,tier,method,seed=0,budget=4):
    physics=method in ('ADAPTIVE_GREEDY','MODEL_INFORMED_BO_EI')
    model=Model(domain,method) if method not in ('REFERENCE','RANDOM') else None
    selector=RandomSelector(seed) if method=='RANDOM' else (ModelOnlyGreedySelector() if method=='ADAPTIVE_GREEDY' else ExpectedImprovementSelector(xi=0.))
    env=Environment(domain,store,tier,physics);history=[];records=[];diags=[];p=domain.reference;best=1.;bestid=p.candidate_id
    for k in range(1,(1 if method=='REFERENCE' else budget)+1):
        obs,raw=env.evaluate(p,k);history.append(obs)
        if obs.valid and obs.endpoint_value<best:best=obs.endpoint_value;bestid=p.candidate_id
        record=dict(trial_index=k-1,candidate_id=p.candidate_id,parameter_flex=p.beta_flex,parameter_extend=p.beta_extend,share_shift=p.share_shift,**raw,best_E3=best,best_candidate_id=bestid)
        records.append(record)
        if k<budget and method!='REFERENCE':
            if model:
                model.fit(history)
                if model.adapter:diags.append(dict(trial_index=k-1,**model.adapter.metadata()))
            selected=selector.select_next(history,domain,model)
            record['next_selection']=dict(candidate_id=selected.candidate.candidate_id,acquisition_value=selected.acquisition_value,metadata=selected.metadata)
            p=selected.candidate
    assert len(env.calls)==len(set(env.calls))
    return records,diags


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'settings.json').write_text(json.dumps(dict(tiers=TIERS,methods=METHODS,random_seeds=list(range(20)),budget_total=4,
        feature_scales=[.03,.03,.05],kernel='unchanged Matern52',length_scale=.7,signal_std=.6,xi=0.,
        force_constraints='revealed after requested replay, consume trial; infeasible observations excluded from GP and ID',
        observations='saved full-trajectory deterministic MuJoCo torque replay; not fresh simulation or hardware'),indent=2))
    data=pd.read_csv(SOURCE/'candidates_with_limits.csv');data=data[data.family.isin(FAMILIES)]
    group1=[]
    for (name,family),g in data.groupby(['model','family'],sort=False):
        for tier in TIERS:
            feasible=g[g.comparison_feasible & (g.E2<=1+tier+1e-12)]
            b=feasible.loc[feasible.E3.idxmin()]
            group1.append(dict(model=name,family=family,tier=tier,feasible_count=len(feasible),best_E3=b.E3,improvement_pct=100*(1-b.E3),E2=b.E2,parameter_flex=b.parameter_flex,parameter_extend=b.parameter_extend,share_shift=b.share_shift,boundary=bool(max(abs(b.parameter_flex),abs(b.parameter_extend))>=.12-1e-12)))
    pd.DataFrame(group1).to_csv(OUT/'constraint_sensitivity.csv',index=False)
    definition=load_frozen_benchmark_definition();sources={l.leg_id:build_leg_domain(l) for l in definition.legs};sources['MYOLEG_NATIVE_P0']=native_domain()
    histories=[];finals=[];diagnostics=[]
    for (name,family),g in data.groupby(['model','family'],sort=False):
        domain=Domain(sources[name],g,family);store=ReplayStore(name,family)
        for tier in TIERS:
            for method in METHODS:
                for seed in (range(20) if method=='RANDOM' else (0,)):
                    rows,diag=sequential(domain,store,tier,method,seed)
                    identity=dict(model=name,family=family,tier=tier,method=method,seed=seed)
                    # Evaluator-only truth access, after the sequential run returns.
                    feasible=g[g.comparison_feasible & (g.E2<=1+tier+1e-12)]
                    oracle=float(feasible.E3.min());headroom=1-oracle
                    for k in range(4):
                        row=rows[min(k,len(rows)-1)];prefix=rows[:k+1]
                        finals.append(dict(identity,K_additional=k,executed_trials=len(prefix),best_E3=row['best_E3'],improvement_pct=100*(1-row['best_E3']),oracle_E3=oracle,headroom=headroom,regret=max(row['best_E3']-oracle,0),normalized_regret=(max(row['best_E3']-oracle,0)/headroom if headroom>1e-12 else np.nan),constraint_violations=sum(not x['feasible'] for x in prefix)))
                    histories.extend(dict(identity,**x) for x in rows)
                    diagnostics.extend(dict(identity,**x) for x in diag)
            print(name,family,'tier',tier,'done',flush=True)
        pd.DataFrame(finals).to_csv(OUT/'algorithm_results.csv',index=False)
        pd.DataFrame(histories).to_csv(OUT/'trial_history.csv',index=False)
        (OUT/'identification_diagnostics.json').write_text(json.dumps(diagnostics,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
    print('COMPLETE',len(finals),'result rows',flush=True)

if __name__=='__main__':main()
