"""Six-parameter candidate using one PUBLIC nominal mechanical basis.

No subject id, development manifest, delta, simulator response or outcome map
is accepted by this learner module. Its common baseline comes from the same
public nominal XML for every subject, with the existing supine coordinate map.

rigid_only (minimal change): m * public_rigid + K(q-q0) + B dq + coupling.
fixed_nonrigid (stronger common prior): add public nominal nonrigid drive.
In the latter mode K/B/coupling are NONNEGATIVE RESIDUAL INCREMENTS above that
nonlinear baseline. They are not absolute physiological stiffness/damping.
The mass bound remains [0.6,1.6]. Neither mode proves identifiability.
"""
from pathlib import Path
import hashlib
import numpy as np
import mujoco

from .coupled_model import EXTENDED_PARAMETER_NAMES, STATE_COLUMNS
from . import coordinates as replay
from lower_limb_sim.myoleg_benchmark.portable_model import load_model_with_provenance

MODE='rigid_only'
_BASIS=None


def set_mode(mode):
    global MODE
    if mode not in ('rigid_only','fixed_nonrigid'):raise ValueError('UNKNOWN_BASIS_MODE')
    MODE=mode


class PublicNominalBasis:
    def __init__(self):
        self.model,self.provenance=load_model_with_provenance()
        self.data=mujoco.MjData(self.model)
        self.mass=np.zeros((self.model.nv,self.model.nv))
        self.cache={}
        self.provenance.update(basis_source='common public nominal XML; no subject lookup',
            source_sha256=hashlib.sha256(Path(self.provenance['model_path']).read_bytes()).hexdigest(),
            coordinate_map='existing supine root pose and seven-knee-coordinate virtual-work tangent',
            includes_foot_and_auxiliary_inertia=True)

    def evaluate(self,frame):
        values=np.ascontiguousarray(frame.loc[:,STATE_COLUMNS].to_numpy(float))
        if not np.isfinite(values).all():raise ValueError('NONFINITE_KINEMATICS')
        rigid=np.empty((len(values),2));nonrigid=np.empty_like(rigid)
        for i,row in enumerate(values):
            key=row.tobytes()
            if key not in self.cache:
                model,data=self.model,self.data
                replay.reset_to_target_state(model,data,row[:2],row[2:4],row[4:6])
                acceleration=np.array(data.qacc)
                tangent=replay.independent_coordinate_tangent(model,data)
                mujoco.mj_forward(model,data)
                actuator=np.array(data.qfrc_actuator)
                data.qacc[:]=acceleration
                mujoco.mj_inverse(model,data)
                mujoco.mj_fullM(model,self.mass,data.qM)
                rigid_row=tangent.T@(self.mass@acceleration+np.array(data.qfrc_bias))
                nonrigid_row=-tangent.T@(np.array(data.qfrc_passive)+actuator+np.array(data.qfrc_constraint))
                self.cache[key]=(rigid_row,nonrigid_row)
            rigid[i],nonrigid[i]=self.cache[key]
        return rigid,nonrigid


def public_basis():
    global _BASIS
    if _BASIS is None:_BASIS=PublicNominalBasis()
    return _BASIS


def design_matrix(frame,template):
    rigid,nonrigid=public_basis().evaluate(frame)
    n=len(frame);matrix=np.zeros((2*n,6));matrix[:,0]=rigid.T.reshape(-1)
    qh=frame.q_hip_rad.to_numpy(float)-template.q0_hip_rad
    qk=frame.q_knee_rad.to_numpy(float)-template.q0_knee_rad
    matrix[:n,1]=qh;matrix[n:,2]=qk
    matrix[:n,3]=frame.dq_hip_rad_s.to_numpy(float)
    matrix[n:,4]=frame.dq_knee_rad_s.to_numpy(float)
    matrix[:n,5]=qh-qk;matrix[n:,5]=qk-qh
    offset=nonrigid.T.reshape(-1) if MODE=='fixed_nonrigid' else np.zeros(2*n)
    return matrix,offset


def predict_model(frame,template,parameters,L1):
    if not isinstance(parameters,dict):parameters=dict(zip(EXTENDED_PARAMETER_NAMES,parameters))
    matrix,offset=design_matrix(frame,template)
    values=np.array([parameters[name] for name in EXTENDED_PARAMETER_NAMES],dtype=float)
    if values.shape!=(6,) or not np.isfinite(values).all():raise ValueError('SIX_FINITE_PARAMETERS_REQUIRED')
    prediction=(matrix@values+offset).reshape(2,-1)
    return prediction[0],prediction[1]
