"""A small actual offline EI run on the frozen MyoLeg prescribed-state model.

The identification payload uses algebraically equivalent project-plane forces,
not a simulated cuff force sensor. No search over methods, seeds or parameters.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
from external_simulation.myoleg_reference_trajectory_replay_v1.build_and_replay import prescribed_truth
from lower_limb_sim.visualization.myoleg_robot_scene import MODEL_PATH,native_domain
from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.mechanical_endpoints import E0,full_cycle_dual_joint_rms
from personalization.environment import PersonalizationEnvironment
from personalization.identification import TimeSeriesIdentificationPayload,PROJECT_FORCE_MAPPING
from personalization.observations import EpisodeObservation
from personalization.integrated_v2 import OfflineBOConfiguration,run_offline_configuration

L1,L2=.42,.30


class MyoLegOfflineDemo(PersonalizationEnvironment):
    offline_only=True
    def __init__(self,domain):
        self.domain=domain
        self.model=mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.evaluated=[]
        self.arrays=[]

    def evaluate(self,candidate,trial_index):
        actual=self.domain.by_id(candidate.candidate_id)
        if actual.beta!=candidate.beta:raise ValueError('candidate identity mismatch')
        t=actual.trajectory;r=self.domain.subject_reference;p=self.domain.profile
        arrays,_=prescribed_truth(self.model,dict(time_s=r.time_s,q=t.q,dq=t.dq,ddq=t.ddq,phases=r.phases))
        torque=arrays['tau_truth_nm']
        j=leg_jacobian(t.q[:,0],t.q[:,1],L1,L2).swapaxes(-1,-2)
        force=np.linalg.solve(j,torque[...,None])[...,0]
        if not np.isfinite(force).all():raise ValueError('invalid equivalent force')
        np.testing.assert_allclose(np.einsum('nij,nj->ni',j,force),torque,rtol=1e-12,atol=1e-10)
        value=full_cycle_dual_joint_rms(torque[:,0],torque[:,1],r.time_s)
        episode=f'MYOLEG_VISUALIZATION_DEMO:{trial_index}'
        payload=TimeSeriesIdentificationPayload(episode_id=episode,candidate_id=actual.candidate_id,
            rom_profile_id=p.profile_id,rom_version=p.version,rom_fingerprint=p.fingerprint,
            reference_version=r.reference_version,beta_flex=actual.beta_flex,beta_extend=actual.beta_extend,
            time_s=r.time_s,q=t.q,dq=t.dq,ddq=t.ddq,planar_force_n=force,sample_valid=np.ones(len(t.q),bool),
            L1=L1,L2=L2,force_mapping=PROJECT_FORCE_MAPPING,
            mapping_provenance='MyoLeg prescribed_truth tau; F=solve(project leg_jacobian.T,tau); equivalent force, NOT cuff sensor or strap physics',
            classification='OFFLINE_MYOLEG_ALGEBRAIC_EQUIVALENT_FORCE_DEMO')
        self.evaluated.append(actual.candidate_id)
        self.arrays.append(dict(time_s=r.time_s,q=t.q,dq=t.dq,ddq=t.ddq,tau=torque,equivalent_force=force))
        print(f'Trial {trial_index-1}: beta={actual.beta}; E0={value:.9f} Nm',flush=True)
        return EpisodeObservation(episode,trial_index,actual.candidate_id,*actual.beta,E0.endpoint_name,
            value,E0.unit,0.,True,metadata={'source':'MyoLeg prescribed_truth','force_semantics':'algebraic equivalent force, not physical cuff measurement'},identification_payload=payload)


def run_demo(output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    domain=native_domain();environment=MyoLegOfflineDemo(domain)
    config=OfflineBOConfiguration(endpoint='E0',budget=4)
    run=run_offline_configuration(environment,domain,configuration=config,
        baseline_template=baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS['baseline']),L1=L1,L2=L2,seed=0)
    rows=[];best=float('inf')
    for k,entry in enumerate(run.ledger.entries):
        obs=entry.observation;best=min(best,obs.endpoint_value)
        rows.append(dict(trial=k,beta=list(entry.candidate.beta),objective=obs.endpoint_value,best=best,
            candidate_id=entry.candidate.candidate_id,acquisition='Expected Improvement' if k else 'Reference initialization',
            next_selection_metadata=entry.next_selection_metadata,physics=entry.physics_model_state_summary))
    result=dict(method=run.method,endpoint=E0.endpoint_name,unit=E0.unit,budget=4,seed=0,
        source_model=str(MODEL_PATH),classification='OFFLINE_ALGORITHM_DEMO_NOT_PERSONALIZATION_NECESSITY_EVIDENCE',
        force_payload='algebraically equivalent planar forces derived from actual MyoLeg inverse-dynamics torque; not cuff sensor data',
        trials=rows,reference_objective=rows[0]['objective'],best_observed_objective=best,
        relative_improvement_percent=100*(rows[0]['objective']-best)/rows[0]['objective'],
        selected_beta=list(run.best_observed_candidate.beta),
        necessity_conclusion='NOT_SUPPORTED')
    output.write_text(json.dumps(result,indent=2)+'\n')
    # Retain the actual executed full trajectories, not an oracle landscape.
    np.savez_compressed(output.with_suffix('.npz'),**{f'trial_{k}_{name}':value for k,a in enumerate(environment.arrays) for name,value in a.items()})
    print(json.dumps({k:result[k] for k in ('reference_objective','best_observed_objective','relative_improvement_percent','selected_beta')},indent=2))
    return result
