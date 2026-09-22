"""Predetermined frozen algorithm runs and post-run evaluator exports."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from lower_limb_sim.five_leg_mujoco_v1.model import load_frozen_benchmark_definition
from lower_limb_sim.five_leg_mujoco_v1.benchmark import build_leg_domain
from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject
from personalization.integrated_v2 import OfflineBOConfiguration,run_offline_configuration
from personalization.sequential import run_sequential_personalization
from personalization.models.time_series_graybox import TimeSeriesFiveParameterGrayBoxAdapter,BranchBalancedE2GrayBoxEndpointAdapter
from .environment import FrozenLegEnvironment

OUT=Path(__file__).resolve().parent/'results'
METHODS={'ADAPTIVE_GREEDY':'MODEL_ONLY_GREEDY','PURE_BO_EI':'PURE_BO_EI',
    'MODEL_INFORMED_BO_EI':'MODEL_INFORMED_BO_EI_TIMESERIES_ID','MODEL_INFORMED_BO_LCB':'MODEL_INFORMED_BO_LCB'}
SEEDS=tuple(range(20))
LOW_HEADROOM_RELATIVE=.01


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,tuple,dict)) else v for k,v in row.items()})


class Evaluator:
    """Constructed only after sequential execution; never supplied to algorithms."""
    def __init__(self):
        path=Path(__file__).resolve().parents[1]/'five_leg_mujoco_v1/results_e2_necessity_v1/e2_full_landscapes.csv'
        with path.open() as f:self.rows=list(csv.DictReader(f))
        self.by_id={(r['leg_id'],r['candidate_id']):r for r in self.rows}

    def truth(self,leg,candidate,endpoint):return float(self.by_id[(leg,candidate.candidate_id)][endpoint])

    def headroom(self,leg,domain,endpoint):
        # Frozen deterministic tie-break: ordered candidate index.
        row=min((r for r in self.rows if r['leg_id']==leg),key=lambda r:(float(r[endpoint]),int(r['candidate_index'])))
        oracle=domain.by_id(row['candidate_id']);oj=float(row[endpoint]);ref=self.truth(leg,domain.reference,endpoint)
        headroom=ref-oj
        return dict(LEG_ID=leg,endpoint=endpoint,reference_J=ref,oracle_J=oj,oracle_beta=oracle.beta,
            optimization_headroom=headroom,relative_headroom=headroom/oj,
            headroom_status='REFERENCE_ALREADY_ORACLE' if headroom==0 else 'LOW_OPTIMIZATION_HEADROOM' if headroom/oj<LOW_HEADROOM_RELATIVE else 'POSITIVE_OPTIMIZATION_HEADROOM')


def execute(definition,leg,domain,endpoint,method,budget,seed=0):
    env=FrozenLegEnvironment(definition,leg,domain,endpoint,scalar_only=method in ('REFERENCE','RANDOM'))
    if method in ('REFERENCE','RANDOM'):
        run=run_sequential_personalization(env,domain,method='Reference' if method=='REFERENCE' else 'Random',budget=budget,seed=seed)
    else:
        config=OfflineBOConfiguration(endpoint=endpoint,method=METHODS[method],budget=budget)
        run=run_offline_configuration(env,domain,configuration=config,baseline_template=baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS['baseline']),
            L1=definition.thigh_length_m,L2=definition.cuff_distance_m,seed=seed)
    assert len(run.ledger.entries)==budget
    assert run.ledger.entries[0].candidate==domain.reference
    assert len(set(run.ledger.executed_candidate_ids))==budget
    assert {e.observation.endpoint_name for e in run.ledger.entries}=={env.endpoint.endpoint_name}
    return run,env


def primary_row(evaluator,run,leg,domain,endpoint,method,seed,k,entries=None,recommendation=None):
    entries=entries or run.ledger.entries
    valid=[e for e in entries if e.observation.valid]
    winner=min(valid,key=lambda e:(e.observation.endpoint_value,e.trial_index))
    h=evaluator.headroom(leg,domain,endpoint)
    truth=evaluator.truth(leg,winner.candidate,endpoint);regret=truth-h['oracle_J']
    rec=recommendation if recommendation is not None else run.model_recommended_final_candidate
    if method in ('REFERENCE','RANDOM'):rec=None
    return dict(**h,METHOD=method,seed=seed,K_additional=k,actual_full_trajectory_evaluations=len(entries),
        selected_measured_beta=winner.candidate.beta,selected_measured_J=winner.observation.endpoint_value,
        final_regret=regret,normalized_regret=(regret/h['optimization_headroom'] if h['optimization_headroom']>0 else None),
        normalized_regret_status='DEFINED' if h['optimization_headroom']>0 else 'NO_OPTIMIZATION_HEADROOM',
        model_recommended_beta=rec.beta if rec else None,
        model_recommended_truth_regret=evaluator.truth(leg,rec,endpoint)-h['oracle_J'] if rec else None,
        best_found_trial=winner.trial_index-1,valid_trials=len(valid),invalid_trials=len(entries)-len(valid))


def histories(run,env,evaluator,leg,endpoint,method,seed):
    rows=[];diagnostics=[];best=None
    for i,e in enumerate(run.ledger.entries):
        obs=e.observation
        if obs.valid and (best is None or obs.endpoint_value<best.observation.endpoint_value):best=e
        metadata=run.ledger.entries[i-1].next_selection_metadata if i else {}
        physics=e.physics_model_state_summary
        truth=evaluator.truth(leg,e.candidate,endpoint)
        if obs.valid:np.testing.assert_allclose(obs.endpoint_value,truth,rtol=1e-12,atol=1e-12)
        rows.append(dict(leg_id=leg,method=method,endpoint=endpoint,seed=seed,trial_index=i,
            candidate_id=e.candidate.candidate_id,rom_identity=env.domain.profile.profile_id,
            beta_flex=e.candidate.beta_flex,beta_extend=e.candidate.beta_extend,
            measured_endpoint=obs.endpoint_value,true_endpoint_if_evaluator_only=truth,
            best_measured_endpoint=best.observation.endpoint_value,best_measured_beta=best.candidate.beta,
            theta_hat=physics.get('estimated_parameters'),
            predicted_mean_at_selected=metadata.get('predictive_mean'),predicted_std_at_selected=metadata.get('predictive_std'),
            acquisition_type=metadata.get('acquisition_type','REFERENCE_INITIALIZATION' if i==0 else 'RANDOM'),
            acquisition_value=metadata.get('acquisition_value'),valid=obs.valid,invalid_reason=obs.invalid_reason))
        diagnostics.append(dict(leg_id=leg,endpoint=endpoint,method=method,seed=seed,trial_index=i,
            physics=physics,residual_gp=e.residual_model_state_summary,next_selection_metadata=e.next_selection_metadata,
            torque_residual_l2_nm=(physics['identification_diagnostics']['residual_statistics']['torque_rmse_combined_nm'] * np.sqrt(2*physics['valid_sample_count']) if physics.get('identification_diagnostics') else None)))
    return rows,diagnostics


def prior_quality(definition,leg,domain,endpoint,reference_observation,evaluator):
    cls=BranchBalancedE2GrayBoxEndpointAdapter if endpoint=='E2' else TimeSeriesFiveParameterGrayBoxAdapter
    adapter=cls(domain,baseline_template=baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS['baseline']),
        L1=definition.thigh_length_m,L2=definition.cuff_distance_m)
    adapter.fit([reference_observation])
    predicted=np.array([adapter.predict_value(c) for c in domain])
    # Truth enters evaluator only after all sequential runs for this leg/endpoint.
    actual=np.array([evaluator.truth(leg.leg_id,c,endpoint) for c in domain])
    rmse=float(np.sqrt(np.mean((predicted-actual)**2)));span=float(np.ptp(actual))
    candidate=tuple(domain)[int(np.argmin(predicted))]
    h=evaluator.headroom(leg.leg_id,domain,endpoint)
    return dict(leg_id=leg.leg_id,endpoint=endpoint,rmse=rmse,nrmse_by_landscape_range=rmse/span if span else None,
        spearman=float(spearmanr(predicted,actual).statistic),predicted_best_beta=candidate.beta,
        true_oracle_beta=h['oracle_beta'],predicted_best_truth_regret=evaluator.truth(leg.leg_id,candidate,endpoint)-h['oracle_J'],
        theta_hat=adapter.theta_hat,identification_diagnostics=adapter.metadata(),evaluator_only=True)


def main(output=OUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    definition=load_frozen_benchmark_definition()
    protocol=dict(primary_endpoint='E2',secondary_endpoint='E0',primary_method=METHODS['MODEL_INFORMED_BO_EI'],
        method_mapping=METHODS,random_seeds=SEEDS,K_total=[1,2,3,4],xi=0.,kappa=1.5,
        low_headroom_relative_threshold=LOW_HEADROOM_RELATIVE,
        reference_policy='one evaluation, carried across budget display; not an equal-cost adaptive arm',
        deterministic_noise_semantics='zero added noise; each evaluation replays requested full trajectory',
        normalization='regret / (reference - oracle) only for positive headroom; otherwise missing',
        future_data_policy='independent K_total runs; assert identical prefixes; no evaluator object passed to algorithms',
        classification='OFFLINE_FROZEN_ALGORITHM_STRESS_TEST_NOT_PERSONALIZATION_NECESSITY_EVIDENCE')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    write_csv(output/'leg_parameters.csv',[dict(model_source='five_leg_mujoco_v1/FROZEN_FIVE_LEG_MECHANICAL_PARAMETERS_V1.json',**l.as_table_row()) for l in definition.legs])
    allrows=[];trialrows=[];diagnostics=[];quality=[];headrooms=[];failures=[];evaluator=None;start=time.perf_counter()
    for endpoint in ('E2','E0'):
        for leg in definition.legs:
            domain=build_leg_domain(leg)
            reference_run,reference_env=execute(definition,leg,domain,endpoint,'REFERENCE',1)
            if evaluator is None:evaluator=Evaluator()
            headrooms.append(evaluator.headroom(leg.leg_id,domain,endpoint))
            for k in range(4):allrows.append(primary_row(evaluator,reference_run,leg.leg_id,domain,endpoint,'REFERENCE',0,k))
            h,d=histories(reference_run,reference_env,evaluator,leg.leg_id,endpoint,'REFERENCE',0);trialrows+=h;diagnostics+=d
            for method in METHODS:
                runs=[]
                for budget in (1,2,3,4):
                    try:
                        run,env=execute(definition,leg,domain,endpoint,method,budget)
                        if runs:assert run.ledger.executed_candidate_ids[:budget-1]==runs[-1].ledger.executed_candidate_ids
                        runs.append(run)
                        allrows.append(primary_row(evaluator,run,leg.leg_id,domain,endpoint,method,0,budget-1))
                        if budget==4:
                            h,d=histories(run,env,evaluator,leg.leg_id,endpoint,method,0);trialrows+=h;diagnostics+=d
                    except Exception as error:
                        failures.append(dict(leg=leg.leg_id,endpoint=endpoint,method=method,budget=budget,error=repr(error)))
                        print('FAILED',failures[-1],flush=True);break
                print(f'{endpoint} {leg.leg_id} {method} budgets done; elapsed {time.perf_counter()-start:.1f}s',flush=True)
            for seed in SEEDS:
                run,env=execute(definition,leg,domain,endpoint,'RANDOM',4,seed)
                for k in range(4):
                    entries=run.ledger.entries[:k+1]
                    winner=min(entries,key=lambda e:(e.observation.endpoint_value,e.trial_index))
                    allrows.append(primary_row(evaluator,run,leg.leg_id,domain,endpoint,'RANDOM',seed,k,entries=entries,recommendation=winner.candidate))
                h,d=histories(run,env,evaluator,leg.leg_id,endpoint,'RANDOM',seed);trialrows+=h;diagnostics+=d
            quality.append(prior_quality(definition,leg,domain,endpoint,reference_env.observations[0],evaluator))
            write_csv(output/'main_results.csv',allrows);write_csv(output/'trial_history.csv',trialrows)
            write_csv(output/'headroom.csv',headrooms);write_csv(output/'prior_quality.csv',quality)
            (output/'identification_and_selection_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
            (output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
            print(f'FINISHED {endpoint} {leg.leg_id}; {len(trialrows)} trial history rows',flush=True)
    (output/'run_status.json').write_text(json.dumps(dict(status='COMPLETE' if not failures else 'COMPLETE_WITH_LIMITATIONS',
        elapsed_seconds=time.perf_counter()-start,main_rows=len(allrows),trial_history_rows=len(trialrows),failures=failures),indent=2)+'\n')

if __name__=='__main__':main()
