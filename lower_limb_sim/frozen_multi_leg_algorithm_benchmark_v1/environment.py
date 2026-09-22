"""Requested-candidate-only five-leg observations; no landscape or oracle access."""
from dataclasses import replace
import numpy as np
from lower_limb_sim.five_leg_mujoco_v1.model import make_mujoco_model
from lower_limb_sim.five_leg_mujoco_v1.replay import replay_trajectory
from lower_limb_sim.jacobian import leg_jacobian
from lower_limb_sim.mechanical_endpoints import (
    E0,E2,MechanicalReferenceContext,BranchRMSReference,branch_rms_components,
    branch_balanced_reference_normalized_rms,full_cycle_dual_joint_rms,
)
from personalization.environment import PersonalizationEnvironment
from personalization.identification import TimeSeriesIdentificationPayload,PROJECT_FORCE_MAPPING
from personalization.observations import EpisodeObservation


class FrozenLegEnvironment(PersonalizationEnvironment):
    offline_only=True

    def __init__(self,definition,leg,domain,endpoint='E2',scalar_only=False):
        if endpoint not in ('E0','E2'):raise ValueError(endpoint)
        self.definition,self.leg,self.domain=definition,leg,domain
        self.endpoint=E2 if endpoint=='E2' else E0
        self.model=make_mujoco_model(definition,leg)
        self.scalar_only=scalar_only
        self.reference=None
        self.observations=[]
        p=domain.profile;r=domain.subject_reference
        self.context=MechanicalReferenceContext(p.profile_id,p.version,p.fingerprint,r.reference_version,
            tuple(float(v) for v in vars(leg).values() if isinstance(v,(float,int))))

    def evaluate(self,candidate,trial_index):
        actual=self.domain.by_id(candidate.candidate_id)
        if actual.beta!=candidate.beta:raise ValueError('candidate identity mismatch')
        if trial_index!=len(self.observations)+1:raise ValueError('nonsequential execution')
        if not self.observations and actual!=self.domain.reference:raise ValueError('reference must execute first')
        t=actual.trajectory;r=self.domain.subject_reference;p=self.domain.profile
        replay=replay_trajectory(model=self.model,definition=self.definition,leg=self.leg,
            time_s=r.time_s,q_project=t.q,dq_project=t.dq,ddq_project=t.ddq)
        identity=f'{self.leg.leg_id}:{self.endpoint.endpoint_name}:{trial_index}'
        missing_reference=(self.endpoint==E2 and self.reference is None and actual!=self.domain.reference)
        if not replay.valid or missing_reference:
            observation=EpisodeObservation(identity,trial_index,actual.candidate_id,*actual.beta,
                self.endpoint.endpoint_name,None,self.endpoint.unit,0.,False,invalid_reason='NO_VALID_SAME_LEG_REFERENCE' if missing_reference else replay.invalid_reason)
        else:
            torque=np.column_stack((replay.tau_hip_nm,replay.tau_knee_nm))
            components=branch_rms_components(torque[:,0],torque[:,1],r.time_s,r.phases)
            if self.reference is None:self.reference=BranchRMSReference(self.context,components)
            value=(branch_balanced_reference_normalized_rms(components,self.reference,context=self.context)
                if self.endpoint==E2 else full_cycle_dual_joint_rms(torque[:,0],torque[:,1],r.time_s))
            L1,L2=self.definition.thigh_length_m,self.definition.cuff_distance_m
            jt=leg_jacobian(t.q[:,0],t.q[:,1],L1,L2).swapaxes(-1,-2)
            force=np.linalg.solve(jt,torque[...,None])[...,0]
            if not np.isfinite(force).all():raise ValueError('nonfinite equivalent force')
            payload=TimeSeriesIdentificationPayload(episode_id=identity,candidate_id=actual.candidate_id,
                rom_profile_id=p.profile_id,rom_version=p.version,rom_fingerprint=p.fingerprint,
                reference_version=r.reference_version,beta_flex=actual.beta_flex,beta_extend=actual.beta_extend,
                time_s=r.time_s,q=t.q,dq=t.dq,ddq=t.ddq,planar_force_n=force,
                sample_valid=np.ones(len(r.time_s),bool),L1=L1,L2=L2,force_mapping=PROJECT_FORCE_MAPPING,
                mapping_provenance='Frozen MuJoCo required-drive torque; equivalent project force solve(J.T,F)=tau; not cuff sensor data',
                classification='OFFLINE_FROZEN_MECHANICAL_ALGORITHM_STRESS_TEST')
            observation=EpisodeObservation(identity,trial_index,actual.candidate_id,*actual.beta,
                self.endpoint.endpoint_name,value,self.endpoint.unit,0.,True,
                metadata={'leg_id':self.leg.leg_id,'rom_identity':p.profile_id,'branch_rms_nm':components,
                    'reference_branch_rms_nm':self.reference.components,'force_semantics':'algebraic equivalent force'},
                identification_payload=payload)
        self.observations.append(observation)
        return replace(observation,identification_payload=None) if self.scalar_only else observation
