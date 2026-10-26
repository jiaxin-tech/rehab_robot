"""Observation-only finite-budget identification and constrained EI policy.

All full candidate arrays come from common mechanics and acquired measurements.
No backend, oracle, subject parameters or outcome table enters this controller.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.special import ndtr
from .config import AlgorithmConfig, Config, METHODS

@dataclass(frozen=True)
class Action:
    candidate_id: str | None
    role: str
    stop_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


def expected_improvement(mean,std,best):
    mean,std=np.asarray(mean,float),np.asarray(std,float)
    improvement=best-mean
    safe=np.maximum(std,1e-15)
    z=improvement/safe
    ei=improvement*ndtr(z)+safe*np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
    return np.where(std>1e-14,np.maximum(ei,0.),np.maximum(improvement,0.))


def constraint_probability(mean,std,limits):
    """Gaussian marginal feasibility and dependence-agnostic union lower bound.

This is a model-based score. Gaussian marginals and calibration are hypotheses,
not a certified physical safety probability.
"""
    mean,std=np.asarray(mean,float),np.asarray(std,float)
    limits=np.asarray(limits,float)
    p=ndtr((limits-mean)/np.maximum(std,1e-15))
    p=np.where(std>1e-14,p,(mean<=limits).astype(float))
    return p,np.maximum(0.,p.sum(axis=1)-1.)


class Policy:
    def __init__(self,context,config=None,method='FULL',surrogate=None):
        if method not in METHODS:raise ValueError('UNKNOWN_POLICY_METHOD')
        self.context=context
        self.domain=context.domain
        self.config=AlgorithmConfig() if config is None else config
        if isinstance(self.config,dict):self.config=AlgorithmConfig(**self.config)
        self.method=method
        self.method_config=self.config.for_method(method)
        if surrogate is None:
            from .surrogate import Surrogate
            surrogate=Surrogate(context,self.method_config.as_dict())
        self.surrogate=surrogate
        self.latest_prediction=None
        self.roles={}
        self.decisions=[]
        self._last_fit_count=0
        self._last_action_count=None
        self._last_action=None
        self._stopped=None
        self.ids=[p.candidate_id for p in self.domain]
        self.index={cid:i for i,cid in enumerate(self.ids)}
        self.common_candidate_id=context.common_candidate_id

    def recommend(self,measurements):
        eligible=[]
        for m in measurements:
            o=m.observation
            if not o.valid or not m.feasible: continue
            try: values=np.asarray([o.endpoint_value,m.E2,m.peak_ratio],float)
            except (ValueError,TypeError): continue
            if not np.isfinite(values).all(): continue
            if values[1]>self.config.E2_max+1e-12 or values[2]>self.config.peak_ratio_max+1e-12: continue
            eligible.append(o)
        if not eligible:return None
        return min(eligible,key=lambda o:(float(o.endpoint_value),int(o.trial_index))).candidate_id

    def _finish(self,reason,diagnostics):
        result=Action(None,'STOP',reason,diagnostics)
        self._stopped=result
        self.decisions.append(result)
        return result

    def _choose(self,index,role,diagnostics,count):
        cid=self.ids[int(index)]
        diagnostics=dict(diagnostics,selected_candidate_id=cid,selected_role=role)
        if self.latest_prediction is not None:
            diagnostics.update(selected_predicted_mean=np.asarray(self.latest_prediction['mean'])[index].tolist(),
                               selected_predicted_std=np.asarray(self.latest_prediction['std'])[index].tolist())
        result=Action(cid,role,None,diagnostics)
        self.roles[cid]=role
        self._last_action_count=count
        self._last_action=result
        self.decisions.append(result)
        return result

    def _reliability(self,measurements):
        c=self.config
        d=dict(self.surrogate.diagnostics)
        failures=[]
        if not self.method_config.use_physics:
            return True,[],dict(d,identification_status='NOT_APPLICABLE_MODEL_FREE')
        if len(measurements)<c.min_model_trials:failures.append('NEED_CROSS_TRAJECTORY_CHECK')
        if not d.get('success',True):failures.append('OPTIMIZER_FAILED')
        rank=d.get('rank',d.get('scaled_sensitivity_rank',0))
        cond=d.get('condition',d.get('scaled_sensitivity_condition',float('inf')))
        if rank<6:failures.append('RANK_DEFICIENT')
        if not np.isfinite(cond) or cond>c.max_condition:failures.append('ILL_CONDITIONED')
        bounds=d.get('boundary_hits',{})
        if bounds.get('mass_scale') is not None:failures.append('MASS_BOUNDARY')
        # Zero stiffness/damping is allowed in a nested passive effective model.
        if any(v=='upper' for k,v in bounds.items() if k!='mass_scale'):failures.append('IMPEDANCE_UPPER_BOUNDARY')
        rmse=d.get('normalized_train_rmse',float('inf'))
        if not np.isfinite(rmse) or rmse>c.max_normalized_train_rmse:failures.append('MODEL_MISMATCH')
        change=d.get('scaled_parameter_change',None)
        if len(measurements)>=c.min_model_trials and (change is None or not np.isfinite(change) or change>c.max_scaled_parameter_change):
            failures.append('PARAMETERS_UNSTABLE')
        pre=d.get('latest_prequential')
        if len(measurements)>=c.min_model_trials:
            if not pre:failures.append('NO_PREQUENTIAL_VALIDATION')
            else:
                torque=pre.get('normalized_torque_rmse',float('inf'))
                endpoint=pre.get('E3_abs_error',float('inf'))
                if torque is None or not np.isfinite(torque) or torque>c.max_prequential_normalized_torque_rmse:failures.append('PREQUENTIAL_TORQUE_ERROR')
                if endpoint is None or not np.isfinite(endpoint) or endpoint>c.max_prequential_E3_error:failures.append('PREQUENTIAL_ENDPOINT_ERROR')
        return not failures,failures,d

    def next_action(self,measurements):
        """Propose one action; one acquired episode consumes one total-budget slot."""
        if self._stopped is not None:return self._stopped
        n=len(measurements)
        if self._last_action_count==n:return self._last_action
        c=self.config
        info={'method':self.method,'acquired_trials':n,'total_budget':c.total_budget,
              'remaining_trials':max(0,c.total_budget-n),'thresholds_status':'DECLARED_OFFLINE_HEURISTICS_NOT_CLINICAL',
              'constraint_probability_semantics':'Gaussian marginal union lower bound; calibration not guaranteed'}
        if n>c.total_budget:raise ValueError('HISTORY_EXCEEDS_BUDGET')
        if n==0:return self._choose(self.index[self.domain.reference.candidate_id],'REFERENCE',info,n)
        used=[m.observation.candidate_id for m in measurements]
        if len(set(used))!=n:raise ValueError('DUPLICATE_CANDIDATE_NO_REPEAT_NOISE_MODEL')
        if used[0]!=self.domain.reference.candidate_id:raise ValueError('REFERENCE_MUST_BE_FIRST')
        if any(cid not in self.index for cid in used):raise ValueError('UNKNOWN_HISTORY_CANDIDATE')
        if any(not m.observation.valid for m in measurements):
            return self._finish('INVALID_OBSERVATION',info)
        if not measurements[0].feasible:return self._finish('REFERENCE_INFEASIBLE',info)
        for m in measurements:
            vals=[m.observation.endpoint_value,m.E2,m.peak_ratio,m.observation.endpoint_uncertainty or 0.]
            if not np.isfinite(np.asarray(vals,float)).all():return self._finish('INVALID_OBSERVATION',info)
        if n>=c.total_budget:return self._finish('TOTAL_BUDGET_REACHED',info)
        available=np.array([i for i,cid in enumerate(self.ids) if cid not in used],dtype=int)
        if not len(available):return self._finish('DOMAIN_EXHAUSTED',info)
        if n!=self._last_fit_count:
            try:
                self.surrogate.fit(measurements)
                self.latest_prediction=self.surrogate.predict_all()
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                return self._finish('MODEL_UPDATE_FAILED',dict(info,error=f'{type(exc).__name__}: {exc}'))
            self._last_fit_count=n
        pred=self.latest_prediction
        mean=np.asarray(pred['mean'],float)
        std=np.asarray(pred['std'],float)
        if mean.shape!=(len(self.ids),3) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<0):
            return self._finish('INVALID_PREDICTION',info)
        marginal,joint=constraint_probability(mean[:,1:],std[:,1:],[c.E2_max,c.peak_ratio_max])
        prospective=joint>=c.feasibility_probability
        eligible=available if self.method=='NO_CONSTRAINT' else available[prospective[available]]
        info.update(available_count=len(available),eligible_count=len(eligible),
                    parameter_uncertainty_propagated=bool(self.method_config.use_parameter_uncertainty))
        ready,failures,diagnostics=self._reliability(measurements)
        info.update(identification_ready=ready,identification_failures=failures,model_diagnostics=diagnostics)
        if self.method=='COMMON':
            if self.common_candidate_id in used:return self._finish('PUBLIC_COMMON_EVALUATED',info)
            idx=self.index[self.common_candidate_id]
            if not prospective[idx]:return self._finish('COMMON_NOT_PREDICTED_FEASIBLE',info)
            info['selected_joint_feasibility_lower']=float(joint[idx])
            return self._choose(idx,'COMMON_CONFIRMATION',info,n)
        gate_on=self.method not in ('NO_ID','PURE_BO')
        probes=sum(self.roles.get(cid)=='INFORMATION_PROBE' for cid in used)
        info['probes_used']=probes
        if gate_on and not ready:
            # Do not use up the last slot to collect calibration with no option
            # to search or confirm an observed/comparable public solution.
            can_probe=probes<c.max_probes and c.total_budget-n>c.reserve_trials
            if can_probe and len(eligible):
                scores=np.asarray(self.surrogate.information_scores(eligible),float)
                if scores.shape!=(len(eligible),) or not np.isfinite(scores).all():
                    return self._finish('INVALID_INFORMATION_SCORES',info)
                idx=int(eligible[np.argmax(scores)])
                info.update(information_score=float(scores.max()),selected_joint_feasibility_lower=float(joint[idx]))
                return self._choose(idx,'INFORMATION_PROBE',info,n)
            idx=self.index[self.common_candidate_id]
            if self.common_candidate_id not in used and prospective[idx]:
                info['selected_joint_feasibility_lower']=float(joint[idx])
                return self._choose(idx,'COMMON_CONFIRMATION',info,n)
            return self._finish('IDENTIFICATION_INSUFFICIENT' if len(eligible) else 'NO_PREDICTED_FEASIBLE_PROBE',info)
        if not len(eligible):return self._finish('NO_PREDICTED_FEASIBLE_CANDIDATE',info)
        incumbent_id=self.recommend(measurements)
        if incumbent_id is None:return self._finish('NO_OBSERVED_FEASIBLE_INCUMBENT',info)
        incumbent=next(m.observation for m in measurements if m.observation.candidate_id==incumbent_id)
        best=float(incumbent.endpoint_value)
        incumbent_noise=max(0.,float(incumbent.endpoint_uncertainty or 0.))
        # Include uncertainty in the incumbent as well as the unobserved values.
        # All potentially feasible candidates participate in the stop check,
        # including those not yet confidently feasible for execution.
        possible=np.all(mean[:,1:]-c.optimism_z*std[:,1:]<=np.array([c.E2_max,c.peak_ratio_max]),axis=1)
        optimistic_pool=available if self.method=='NO_CONSTRAINT' else available[possible[available]]
        optimistic_gain=0. if not len(optimistic_pool) else max(0.,best+c.optimism_z*incumbent_noise-float(np.min(mean[optimistic_pool,0]-c.optimism_z*std[optimistic_pool,0])))
        info.update(incumbent_id=incumbent_id,incumbent_observed_E3=best,optimistic_gain=optimistic_gain,
                    minimum_gain=c.minimum_gain,optimistic_pool_count=len(optimistic_pool))
        if self.method!='NO_STOP' and optimistic_gain<=c.minimum_gain:
            return self._finish('NO_CREDIBLE_ADDITIONAL_GAIN',info)
        ei=expected_improvement(mean[eligible,0],std[eligible,0],best)
        if self.method=='GREEDY':scores=-mean[eligible,0]
        elif self.method=='NO_CONSTRAINT':scores=ei
        else:scores=ei*joint[eligible]
        idx=int(eligible[np.argmax(scores)])
        info.update(selected_joint_feasibility_lower=float(joint[idx]),selected_marginal_feasibility=marginal[idx].tolist(),
                    acquisition_value=float(np.max(scores)),acquisition='CONSTRAINT_WEIGHTED_EI_UNION_BOUND' if self.method not in ('GREEDY','NO_CONSTRAINT') else self.method)
        return self._choose(idx,'SEARCH',info,n)
