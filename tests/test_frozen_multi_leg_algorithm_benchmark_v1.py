from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from lower_limb_sim.five_leg_mujoco_v1.model import load_frozen_benchmark_definition
from lower_limb_sim.five_leg_mujoco_v1.benchmark import build_leg_domain
from lower_limb_sim.frozen_multi_leg_algorithm_benchmark_v1.environment import FrozenLegEnvironment
from lower_limb_sim.frozen_multi_leg_algorithm_benchmark_v1.run import execute,OUT


@pytest.fixture(scope='module')
def setup():
    definition=load_frozen_benchmark_definition();leg=definition.legs[-1]
    return definition,leg,build_leg_domain(leg)


def test_methods_domain_endpoint_budget_oracle_isolation(setup,monkeypatch):
    definition,leg,domain=setup
    original=Path.open
    def guarded(path,*a,**kw):
        if 'results_e2_necessity' in str(path) or 'landscape' in str(path):raise AssertionError('algorithm attempted oracle/landscape access')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    for method in ('RANDOM','ADAPTIVE_GREEDY','PURE_BO_EI','MODEL_INFORMED_BO_EI','MODEL_INFORMED_BO_LCB'):
        run,env=execute(definition,leg,domain,'E2',method,2)
        assert env.domain is domain and len(domain)==625
        assert len(run.ledger.entries)==len(set(run.ledger.executed_candidate_ids))==2
        assert run.ledger.entries[0].candidate==domain.reference
        assert all(o.endpoint_name=='E2_BRANCH_BALANCED_REFERENCE_NORMALIZED_RMS' for o in run.ledger.observations)
        if method in ('PURE_BO_EI','RANDOM'):
            assert all(o.identification_payload is None for o in run.ledger.observations)
            assert all(not e.physics_model_state_summary for e in run.ledger.entries)
        else:assert run.ledger.entries[-1].physics_model_state_summary['fit_valid_episode_count']==2
        assert len(env.observations[0].identification_payload.time_s)==401


def test_larger_future_budget_cannot_change_prefix(setup):
    args=(*setup,'E2','MODEL_INFORMED_BO_EI')
    short,_=execute(*args,2);long,_=execute(*args,4)
    assert short.ledger.executed_candidate_ids==long.ledger.executed_candidate_ids[:2]
    for a,b in zip(short.ledger.entries,long.ledger.entries):
        assert a.observation.endpoint_value==b.observation.endpoint_value
        assert a.physics_model_state_summary['estimated_parameters']==b.physics_model_state_summary['estimated_parameters']


def test_normalization_stays_per_leg(setup):
    definition,_,_=setup;denominators=[]
    for leg in definition.legs[:2]:
        domain=build_leg_domain(leg);env=FrozenLegEnvironment(definition,leg,domain)
        obs=env.evaluate(domain.reference,1)
        assert obs.endpoint_value==1
        assert env.reference.context.rom_profile_id==leg.make_rom_profile().profile_id
        denominators.append(env.reference.components)
    assert not np.allclose(*denominators)


def test_measured_best_budget_and_video_export_consistency():
    if not (OUT/'run_status.json').exists():pytest.skip('requires completed benchmark outputs')
    status=json.loads((OUT/'run_status.json').read_text());assert not status['failures']
    results=pd.read_csv(OUT/'main_results.csv');history=pd.read_csv(OUT/'trial_history.csv')
    for (endpoint,leg,method,seed),h in history.groupby(['endpoint','leg_id','method','seed']):
        h=h.sort_values('trial_index')
        assert h.candidate_id.nunique()==len(h)
        assert h.iloc[0].beta_flex==h.iloc[0].beta_extend==0
        assert np.array_equal(h.best_measured_endpoint.to_numpy(),np.minimum.accumulate(h.measured_endpoint.to_numpy()))
        for k in range(4):
            prefix=h.iloc[:k+1];winner=prefix.loc[prefix.measured_endpoint.idxmin()]
            row=results[(results.endpoint==endpoint)&(results.LEG_ID==leg)&(results.METHOD==method)&(results.seed==seed)&(results.K_additional==k)].iloc[0]
            assert row.selected_measured_J==winner.measured_endpoint
            assert row.best_found_trial==winner.trial_index
            assert row.actual_full_trajectory_evaluations==(1 if method=='REFERENCE' else k+1)
    nohead=results[(results.endpoint=='E2')&~results.LEG_ID.str.startswith('LEG_4')]
    assert nohead.normalized_regret.isna().all()
    assert (nohead.normalized_regret_status=='NO_OPTIMIZATION_HEADROOM').all()
