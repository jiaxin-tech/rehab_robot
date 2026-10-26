"""Observation-only corrected six-parameter prior, residual GP and diagnostics.

All tolerances and uncertainties are research heuristics. This module has no
backend/cohort/oracle inputs and cannot certify safety or identifiability.
Parameter uncertainty uses contiguous-time, joint-paired cluster sandwich
scores, not an iid covariance over the 401 correlated samples.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import hashlib

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.linalg import cho_solve, solve_triangular

from .coupled_model import _specification, STATE_COLUMNS
from . import coordinate_consistent_model as mechanical
from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject

DEFAULTS = dict(
    use_physics=True, use_gp=True, use_parameter_uncertainty=True,
    feature_scales=(.12, .12, .10), gp_length_scale=.7,
    gp_signal_floor=(.01, .01, .02), gp_noise_floor=(.001, .002, .004),
    uncertainty_multiplier=1., parameter_scaled_sd_floor=.005,
    covariance_blocks_per_episode=8, covariance_ridge=1e-9,
    information_ridge=.01, covariance_interval_z=1.96,
)


def _frame(point):
    c = point.trajectory
    return pd.DataFrame(dict(q_hip_rad=c.q[:,0], q_knee_rad=c.q[:,1],
        dq_hip_rad_s=c.dq[:,0], dq_knee_rad_s=c.dq[:,1],
        ddq_hip_rad_s2=c.ddq[:,0], ddq_knee_rad_s2=c.ddq[:,1]))


def _weights(time, mask):
    ids = np.flatnonzero(mask)
    if len(ids) < 2:
        raise ValueError('TWO_SAMPLES_PER_BRANCH_REQUIRED')
    t = np.asarray(time)[ids]
    if np.any(np.diff(t) <= 0):
        raise ValueError('TIME_MUST_INCREASE')
    w = np.zeros(len(time)); d = np.diff(t) / (t[-1]-t[0]) / 2
    np.add.at(w, ids[:-1], d); np.add.at(w, ids[1:], d)
    return w


class ModelContext:
    """Common public mechanics and candidate kinematics, safe to share across runs.

    matrix has shape [candidate, 2*T, 6], hip rows followed by knee rows.
    torque_design has shape [candidate, 2, T, 6]. No outcome is cached here.
    """
    def __init__(self, domain):
        self.domain = domain
        self.points = list(domain)
        self.index_by_id = {p.candidate_id:i for i,p in enumerate(self.points)}
        self.reference_index = self.index_by_id[domain.reference.candidate_id]
        self.template = baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS['baseline'])
        self.frames = {p.candidate_id:_frame(p) for p in self.points}
        lengths = {len(f) for f in self.frames.values()}
        if len(lengths) != 1:
            raise ValueError('EQUAL_TRAJECTORY_SAMPLE_COUNTS_REQUIRED')
        self.n_time = lengths.pop()
        self.raw_features = np.array([[p.beta_flex,p.beta_extend,p.share_shift] for p in self.points])
        self.features = self.raw_features / np.asarray(DEFAULTS['feature_scales'])
        self.initial_parameters = dict(mass_scale=1., k_hip_nm_per_rad=10.,
            k_knee_nm_per_rad=10., b_hip_nms_per_rad=1., b_knee_nms_per_rad=1.,
            k_coupling_nm_per_rad=1.)
        self.names, initial, low, high, scale = _specification(True)
        # Explicit public initial values, in repository parameter-name order.
        self.initial_vector = np.array([1.,10.,10.,1.,1.,1.])
        self.initial_parameters = dict(zip(self.names, self.initial_vector.tolist()))
        self.lower = np.array([low[n] for n in self.names])
        self.upper = np.array([high[n] for n in self.names])
        self.parameter_scales = np.array([scale[n] for n in self.names])
        mechanical.set_mode('rigid_only')
        # Cache only the public basis, bound to source and exact kinematics.
        basis = mechanical.public_basis()
        digest = hashlib.sha256()
        digest.update(Path(mechanical.__file__).read_bytes())
        digest.update(str(basis.provenance['source_sha256']).encode())
        digest.update(repr(self.template).encode())
        for p in self.points:
            digest.update(self.frames[p.candidate_id].to_numpy(float).tobytes())
        key = digest.hexdigest()
        self.matrix = np.stack([mechanical.design_matrix(self.frames[p.candidate_id], self.template)[0]
                                for p in self.points])
        if self.matrix.shape != (len(self.points),2*self.n_time,6) or not np.isfinite(self.matrix).all():
            raise ValueError('INVALID_PUBLIC_BASIS_CACHE')
        self.torque_design = self.matrix.reshape(len(self.points),2,self.n_time,6)
        self.offset = np.zeros(self.matrix.shape[:2])
        self.design_by_candidate = {p.candidate_id:self.matrix[i] for i,p in enumerate(self.points)}
        self.public_prior = dict(basis.provenance, basis_cache_sha256=key, mode='rigid_only')
        phases = np.asarray(domain.subject_reference.phases)
        self.weights = np.array([[_weights(p.time_s, phases == branch)
                                 for branch in ('flexion','extension')] for p in self.points])
        self.component_grams = np.einsum('njti,nbt,njtk->njbik',
                                        self.torque_design,self.weights,self.torque_design).reshape(-1,4,6,6)
        self.public_metrics = self.metrics(self.initial_vector)
        mask = (self.public_metrics[:,1] <= 1.01+1e-12) & (self.public_metrics[:,2] <= 1.1+1e-12)
        choices = np.flatnonzero(mask)
        common = choices[np.argmin(self.public_metrics[choices,0])] if len(choices) else self.reference_index
        self.common_candidate_id = self.points[int(common)].candidate_id

    def torques(self, theta):
        values = np.array([theta[n] for n in self.names]) if isinstance(theta,Mapping) else np.asarray(theta)
        return (self.matrix@values).reshape(len(self.points),2,self.n_time).transpose(0,2,1)

    def metrics_from_torque(self, torque):
        """All-candidate N*T*2 torques, normalized by the same array's reference."""
        tau = np.asarray(torque,float)
        if tau.shape != (len(self.points),self.n_time,2) or not np.isfinite(tau).all():
            raise ValueError('ALL_CANDIDATE_N_T_2_TORQUES_REQUIRED')
        comp = np.sqrt(np.maximum(np.einsum('ntj,nbt->njb',tau*tau,self.weights),0)).reshape(-1,4)
        peak = np.max(np.abs(tau),axis=1)
        ref = self.reference_index
        if np.any(comp[ref] <= 1e-12) or np.any(peak[ref] <= 1e-12):
            raise ValueError('NONZERO_PUBLIC_REFERENCE_REQUIRED')
        ratios = comp/comp[ref]
        return np.column_stack((ratios.mean(axis=1),ratios.max(axis=1),(peak/peak[ref]).max(axis=1)))

    def metrics(self, theta):
        return self.metrics_from_torque(self.torques(theta))

    def metrics_and_jacobian(self, theta):
        """Local delta propagation; active max branches can change near ties."""
        theta = np.asarray(theta,float)
        gtheta = self.component_grams@theta
        comp = np.sqrt(np.maximum(np.einsum('nbi,i->nb',gtheta,theta),1e-24))
        dc = gtheta/comp[:,:,None]
        ref = self.reference_index
        ratio = comp/comp[ref]
        dr = dc/comp[ref,:,None] - comp[:,:,None]*dc[ref]/comp[ref,:,None]**2
        tau = self.torques(theta).transpose(0,2,1)
        max_t = np.argmax(abs(tau),axis=2)
        n = np.arange(len(self.points))[:,None]; j = np.arange(2)[None,:]
        peak = abs(tau[n,j,max_t])
        dp = self.torque_design[n,j,max_t,:]*np.sign(tau[n,j,max_t,None])
        pr = peak/peak[ref]
        dpr = dp/peak[ref,:,None] - peak[:,:,None]*dp[ref]/peak[ref,:,None]**2
        e2pick = np.argmax(ratio,axis=1); ppick = np.argmax(pr,axis=1)
        rows = np.arange(len(self.points))
        values = np.column_stack((ratio.mean(axis=1),ratio[rows,e2pick],pr[rows,ppick]))
        jac = np.stack((dr.mean(axis=1),dr[rows,e2pick],dpr[rows,ppick]),axis=1)
        return values,jac


class MaternGP:
    def __init__(self, length_scale, signal_std):
        self.length_scale = float(length_scale)
        self.signal_std = float(signal_std)
        self.x = np.empty((0,3)); self.factor = None; self.alpha = None

    def kernel(self,a,b):
        d = np.linalg.norm((np.asarray(a)[:,None,:]-np.asarray(b)[None,:,:])/self.length_scale,axis=2)
        z = np.sqrt(5.)*d
        return self.signal_std**2*(1+z+z*z/3)*np.exp(-z)

    def fit(self,x,y,noise):
        self.x = np.asarray(x,float).reshape(-1,3)
        y, noise = np.asarray(y,float),np.asarray(noise,float)
        if len(self.x) != len(y) or noise.shape != y.shape or np.any(noise<0):
            raise ValueError('INVALID_GP_DATA')
        if not all(np.isfinite(v).all() for v in (self.x,y,noise)):
            raise ValueError('NONFINITE_GP_DATA')
        covariance = self.kernel(self.x,self.x)+np.diag(noise**2+1e-12)
        self.factor = np.linalg.cholesky(covariance)
        self.alpha = cho_solve((self.factor,True),y,check_finite=False)

    def predict(self,x):
        x = np.asarray(x,float)
        if self.alpha is None:
            return np.zeros(len(x)),np.full(len(x),self.signal_std)
        cross = self.kernel(self.x,x)
        proj = solve_triangular(self.factor,cross,lower=True,check_finite=False)
        return cross.T@self.alpha,np.sqrt(np.maximum(self.signal_std**2-(proj*proj).sum(axis=0),0))


def cluster_sandwich(matrix,residual,episode_lengths,blocks,ridge):
    """Covariance in scaled theta coordinates with paired-joint time blocks.

    matrix/residual are episode-concatenated (hip,knee) rows. Dependence within
    each block is retained; long-range dependence across blocks is not known.
    Robust soft_l1 estimating-equation derivative is used in the bread.
    """
    psi = residual/np.sqrt(1+residual**2)
    derivative = (1+residual**2)**-1.5
    bread = matrix.T@(matrix*derivative[:,None])
    inv = np.linalg.pinv(bread+ridge*np.eye(matrix.shape[1]),rcond=1e-12)
    scores=[]; offset=0
    for count in episode_lengths:
        for ids in np.array_split(np.arange(count),min(blocks,count)):
            paired=np.concatenate((ids+offset,ids+count+offset))
            scores.append(matrix[paired].T@psi[paired])
        offset += 2*count
    score=np.asarray(scores)
    meat=score.T@score
    groups=len(score)
    correction=groups/max(groups-1,1)*(len(residual)-1)/max(len(residual)-matrix.shape[1],1)
    covariance=inv@meat@inv*correction
    return (covariance+covariance.T)/2,groups


class Surrogate:
    def __init__(self,context,config=None):
        self.context=context; self.domain=context.domain
        self.config=dict(DEFAULTS)
        if config is not None:
            self.config.update(dict(config) if isinstance(config,Mapping) else vars(config))
        if 'gp_signal_floors' in self.config:
            self.config['gp_signal_floor']=self.config['gp_signal_floors']
        if 'cluster_blocks_per_episode' in self.config:
            self.config['covariance_blocks_per_episode']=self.config['cluster_blocks_per_episode']
        self.theta=context.initial_vector.copy()
        self.parameter_covariance=None; self.diagnostics={}; self._predictions=None
        self._episode_ids=(); self._prequential=[]
        self._gram=np.zeros((6,6)); self.torque_scales=np.ones(2)
        self._x=context.raw_features/np.asarray(self.config['feature_scales'])

    def _accepted(self, measurements):
        accepted, episodes, candidates = [], set(), set()
        for m in measurements:
            o = m.observation
            if not o.valid:
                raise ValueError("invalid observation excluded from model")
            if o.episode_id in episodes or o.candidate_id in candidates:
                raise ValueError("duplicate episode/candidate")
            episodes.add(o.episode_id)
            candidates.add(o.candidate_id)
            f = m.frame
            state = f.loc[:, STATE_COLUMNS].to_numpy(float)
            tau = f[["tau_hip_nm", "tau_knee_nm"]].to_numpy(float)
            scalar = np.array([o.endpoint_value, m.E2, m.peak_ratio], float)
            if not np.isfinite(state).all() or not np.isfinite(tau).all() or not np.isfinite(scalar).all():
                raise ValueError("nonfinite direct measurement")
            index = self.context.index_by_id[o.candidate_id]
            matrix = mechanical.design_matrix(f, self.context.template)[0]
            accepted.append((m, index, scalar, f, matrix, tau))
        if not accepted:
            raise ValueError("valid direct joint-torque observation required")
        return accepted

    def fit(self,measurements):
        data=self._accepted(measurements)
        ids=tuple(item[0].observation.episode_id for item in data)
        if ids==self._episode_ids: return self
        if self._episode_ids and ids[:len(self._episode_ids)] != self._episode_ids:
            raise ValueError('APPEND_ONLY_HISTORY_REQUIRED')
        # Evaluate an arriving complete episode before using it in any update.
        if self._predictions is not None:
            for m,i,scalar,frame,mat,tau in data[len(self._episode_ids):]:
                delta=scalar-self._predictions['mean'][i]
                noise=max(float(m.observation.endpoint_uncertainty or 0.),0.)
                sd=np.sqrt(self._predictions['std'][i]**2+noise**2)
                record=dict(episode_id=m.observation.episode_id,candidate_id=m.observation.candidate_id,
                    prior_training_episodes=len(self._episode_ids),endpoint_errors=delta.tolist(),
                    E3_abs_error=float(abs(delta[0])),E2_abs_error=float(abs(delta[1])),
                    peak_abs_error=float(abs(delta[2])),predicted_std=sd.tolist(),
                    covered_95=(abs(delta)<=self.config['covariance_interval_z']*sd).tolist(),
                    source='Prediction committed before this entire episode was observed')
                if self.config['use_physics'] and tau is not None and len(tau)==self.context.n_time:
                    difference=tau.T.reshape(-1)-mat@self.theta
                    scale=np.repeat(self.torque_scales,len(tau))
                    record.update(torque_rmse=float(np.sqrt(np.mean(difference**2))),
                        normalized_torque_rmse=float(np.sqrt(np.mean((difference/scale)**2))),
                        complete_trajectory=True)
                else:
                    record.update(normalized_torque_rmse=None,complete_trajectory=False)
                self._prequential.append(record)
        previous=self.theta.copy()
        if self.config['use_physics']:
            self._fit_physics(data)
            physics,jac=self.context.metrics_and_jacobian(self.theta)
            model_std=np.sqrt(np.maximum(np.einsum('nei,ij,nej->ne',jac,self.parameter_covariance,jac),0))
            if not self.config['use_parameter_uncertainty']: model_std[:]=0
        else:
            vals=np.array([row[2] for row in data])
            physics=np.tile(vals.mean(axis=0),(len(self.context.points),1))
            model_std=np.zeros_like(physics)
            self.diagnostics=dict(rank=None,condition=None,boundary_hits={},normalized_train_rmse=None,
                                  train_rmse=None,parameters=None)
        self.diagnostics['scaled_parameter_change']=float(np.linalg.norm((self.theta-previous)/self.context.parameter_scales)/np.sqrt(6))
        indices=np.array([row[1] for row in data]); y=np.array([row[2] for row in data])
        residual=y-physics[indices]
        floors=np.asarray(self.config['gp_signal_floor'],float)
        amplitudes=np.maximum(floors,np.sqrt(np.mean(residual**2,axis=0)))
        noise_floor=np.asarray(self.config['gp_noise_floor'],float)
        obs_noise=np.array([float(row[0].observation.endpoint_uncertainty or 0.) for row in data])
        gp_mean=np.zeros_like(physics); gp_std=np.zeros_like(physics)
        if self.config['use_gp']:
            for j in range(3):
                gp=MaternGP(self.config['gp_length_scale'],amplitudes[j])
                noise=np.sqrt(obs_noise**2+noise_floor[j]**2)
                gp.fit(self._x[indices],residual[:,j],noise)
                gp_mean[:,j],gp_std[:,j]=gp.predict(self._x)
        multiplier=float(self.config['uncertainty_multiplier'])
        std=np.sqrt(model_std**2+gp_std**2)*multiplier
        self._predictions=dict(mean=physics+gp_mean,std=std,physics_mean=physics,
            model_std=model_std*multiplier,gp_std=gp_std*multiplier)
        self._episode_ids=ids
        covered=np.array([r['covered_95'] for r in self._prequential],bool)
        self.diagnostics.update(
            fit_episode_ids=list(ids),training_episodes=len(ids),parameter_count=6,
            latest_prequential=self._prequential[-1] if self._prequential else None,
            prequential=list(self._prequential),prequential_count=len(self._prequential),
            prequential_coverage_95=covered.mean(axis=0).tolist() if len(covered) else None,
            gp_signal_std=amplitudes.tolist(),gp_length_scale=self.config['gp_length_scale'],
            gp_feature_scales=list(self.config['feature_scales']),
            uncertainty_method='paired-joint contiguous-time cluster sandwich + local endpoint Jacobian + residual GP; quadrature sum heuristic',
            uncertainty_caveats='Within-block dependence retained; cross-block dependence, shared reference-denominator noise, model discrepancy and parameter/GP dependence are not fully modeled. Max-endpoint Jacobian is local. Intervals are not safety guarantees.',
            public_prior=self.context.public_prior,
            heuristic_config=dict(self.config))
        return self

    def _fit_physics(self,data):
        mats=[r[4] for r in data]; torques=[r[5] for r in data]
        lengths=[len(t) for t in torques]
        self.torque_scales=np.maximum(np.concatenate(torques).std(axis=0),1.)
        matrix=np.concatenate(mats)
        y=np.concatenate([t.T.reshape(-1) for t in torques])
        row_scale=np.concatenate([np.repeat(self.torque_scales,n) for n in lengths])
        normalized=matrix/row_scale[:,None]
        residual=lambda x:(y-matrix@x)/row_scale
        fit=least_squares(residual,self.theta,jac=lambda x:-normalized,
            bounds=(self.context.lower,self.context.upper),x_scale=self.context.parameter_scales,
            loss='soft_l1',f_scale=1.,max_nfev=500)
        if not fit.success or not np.isfinite(fit.x).all(): raise RuntimeError('SIX_PARAMETER_FIT_FAILED')
        self.theta=fit.x.copy()
        scaled=normalized*self.context.parameter_scales
        singular=np.linalg.svd(scaled,compute_uv=False)
        tol=singular[0]*max(1e-8,max(scaled.shape)*np.finfo(float).eps)
        rank=int(np.sum(singular>tol))
        raw=residual(self.theta)
        covariance,groups=cluster_sandwich(scaled,raw,lengths,
            int(self.config['covariance_blocks_per_episode']),float(self.config['covariance_ridge']))
        # A stated regularizing floor avoids a zero covariance from one perfect
        # deterministic trajectory; it is not a confidence statement.
        floor=float(self.config['parameter_scaled_sd_floor'])
        eig,vec=np.linalg.eigh(covariance)
        covariance=(vec*np.maximum(eig,floor**2))@vec.T
        self.parameter_covariance=covariance*self.context.parameter_scales[:,None]*self.context.parameter_scales[None,:]
        bound={}
        for i,name in enumerate(self.context.names):
            span=self.context.upper[i]-self.context.lower[i]
            if self.theta[i]-self.context.lower[i]<=1e-5*span: bound[name]='lower'
            elif self.context.upper[i]-self.theta[i]<=1e-5*span: bound[name]='upper'
        self._gram=np.zeros((6,6))
        for mat,n in zip(mats,lengths):
            z=mat/np.repeat(self.torque_scales,n)[:,None]*self.context.parameter_scales
            self._gram += z.T@z/n
        self.diagnostics=dict(rank=rank,condition=float(singular[0]/singular[-1]) if rank==6 else float('inf'),
            boundary_hits=bound,parameters=dict(zip(self.context.names,self.theta.tolist())),
            normalized_train_rmse=float(np.sqrt(np.mean(raw**2))),
            train_rmse=float(np.sqrt(np.mean((y-matrix@self.theta)**2))),
            optimizer_success=True,optimizer_cost=float(fit.cost),optimizer_nfev=int(fit.nfev),
            scaled_sensitivity_singular_values=singular.tolist(),scaled_sensitivity_rank=rank,
            valid_training_samples=int(sum(lengths)),covariance_cluster_count=groups,
            covariance_blocks_per_episode=int(self.config['covariance_blocks_per_episode']),
            parameter_covariance=self.parameter_covariance.tolist(),
            parameter_standard_errors=np.sqrt(np.diag(self.parameter_covariance)).tolist(),
            torque_scales_nm=self.torque_scales.tolist(),coupling_zero_allowed=True,
            parameter_bounds={n:[float(l),float(u)] for n,l,u in zip(self.context.names,self.context.lower,self.context.upper)})

    def predict_all(self):
        if self._predictions is None: raise RuntimeError('FIT_REQUIRED')
        return {key:val.copy() for key,val in self._predictions.items()}

    def information_scores(self,available_indices):
        """Per-trajectory scaled log-det geometry gain; no outcome or iid claim."""
        ids=np.asarray(available_indices,dtype=int)
        if np.any(ids<0) or np.any(ids>=len(self.context.points)): raise ValueError('INVALID_CANDIDATE_INDEX')
        base=self._gram+float(self.config['information_ridge'])*np.eye(6)
        old=np.linalg.slogdet(base)[1]
        scaled=self.context.torque_design[ids]/self.torque_scales[None,:,None,None]*self.context.parameter_scales
        grams=np.einsum('njti,njtk->nik',scaled,scaled)/self.context.n_time
        return np.linalg.slogdet(base[None,:,:]+grams)[1]-old
