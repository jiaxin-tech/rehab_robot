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
import sys,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import mujoco

WORK=Path(__file__).resolve().parents[2]
if str(WORK) not in sys.path:sys.path.insert(0,str(WORK))
from coupled_model import EXTENDED_PARAMETER_NAMES,_specification,STATE_COLUMNS
from lower_limb_sim.parameter_estimator import valid_observations,measured_joint_torque
from lower_limb_sim.myoleg_benchmark.portable_model import load_model_with_provenance
from external_simulation.myoleg_reference_trajectory_replay_v1 import build_and_replay as replay

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


def fit_model(training_frame,template,L1,L2,extended=True,*,initial_guess=None):
    if not extended:raise ValueError('THIS_IS_A_SIX_PARAMETER_CANDIDATE')
    data=valid_observations(training_frame)
    measured=np.column_stack(measured_joint_torque(data,L1,L2))
    torque_scales=np.maximum(measured.std(axis=0),1.)
    row_scale=np.repeat(torque_scales,len(data))
    matrix,offset=design_matrix(data,template)
    y=measured.T.reshape(-1)
    names,initial,lower,upper,scales=_specification(True)
    if initial_guess is not None:
        supplied=initial_guess if isinstance(initial_guess,dict) else dict(zip(names,initial_guess))
        initial.update({k:float(v) for k,v in supplied.items() if k in names})
    lo=np.array([lower[n] for n in names]);hi=np.array([upper[n] for n in names])
    x0=np.array([initial[n] for n in names]);ps=np.array([scales[n] for n in names])
    if not ((x0>=lo)&(x0<=hi)).all():raise ValueError('INITIAL_GUESS_OUT_OF_BOUNDS')
    normalized=matrix/row_scale[:,None]
    residual=lambda x:(y-matrix@x-offset)/row_scale
    fit=least_squares(residual,x0,jac=lambda x:-normalized,bounds=(lo,hi),x_scale=ps,loss='soft_l1',f_scale=1.,max_nfev=500)
    difference=(y-matrix@fit.x-offset).reshape(2,-1)
    singular=np.linalg.svd(normalized*ps,compute_uv=False)
    tol=singular[0]*max(1e-8,max(matrix.shape)*np.finfo(float).eps)
    rank=int(np.sum(singular>tol))
    raw=residual(fit.x);variance=float(raw@raw)/max(len(raw)-len(names),1)
    cov=variance*np.linalg.pinv(fit.jac.T@fit.jac)
    se=np.sqrt(np.maximum(np.diag(cov),0.))
    boundary={}
    for i,name in enumerate(names):
        if fit.x[i]-lo[i]<=1e-5*(hi[i]-lo[i]):boundary[name]='lower'
        elif hi[i]-fit.x[i]<=1e-5*(hi[i]-lo[i]):boundary[name]='upper'
    return dict(model='six_parameter_public_nominal_'+MODE,parameters=dict(zip(names,map(float,fit.x))),
        success=bool(fit.success),optimizer_message=str(fit.message),optimizer_cost=float(fit.cost),
        optimizer_nfev=int(fit.nfev),loss='soft_l1',
        train_rmse=float(np.sqrt(np.mean(difference**2))),
        train_rmse_by_joint=dict(zip(['hip','knee'],np.sqrt(np.mean(difference**2,axis=1)).tolist())),
        valid_training_samples=len(data),boundary_hits=boundary,boundary_relative_tolerance=1e-5,
        standard_errors=dict(zip(names,se.tolist())) if rank==6 else {n:None for n in names},
        raw_standard_errors=dict(zip(names,se.tolist())),parameter_covariance=cov.tolist(),
        torque_scales_nm=dict(zip(['hip','knee'],torque_scales.tolist())),
        parameter_bounds={name:[lower[name],upper[name]] for name in names},
        scaled_sensitivity_singular_values=singular.tolist(),scaled_sensitivity_rank=rank,
        scaled_sensitivity_parameter_count=6,scaled_sensitivity_full_rank=rank==6,
        scaled_sensitivity_condition=float(singular[0]/singular[-1]) if rank==6 else float('inf'),
        scaled_sensitivity_rank_tolerance=float(tol),coupling_rho=1.,
        public_prior=dict(public_basis().provenance),
        parameter_semantics=('public-nominal rigid mechanics scale plus nonnegative linear impedance'
                            if MODE=='rigid_only' else 'public-nominal rigid scale plus fixed nonlinear nonrigid prior plus nonnegative residual stiffness/damping increments; not absolute human M/B/K'),
        diagnostic_note='Remaining bounds/model mismatch must be checked; rank does not certify practical identifiability; public nominal is common to all subjects, never subject-specific truth.')
