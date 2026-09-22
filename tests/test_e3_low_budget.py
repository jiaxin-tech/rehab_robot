import numpy as np,pandas as pd,pytest
from lower_limb_sim.e3_low_budget.run import *

@pytest.fixture(scope='module')
def setup():
    d=load_frozen_benchmark_definition();source=build_leg_domain(d.legs[0])
    rows=pd.read_csv(SOURCE/'candidates_with_limits.csv');rows=rows[(rows.model==d.legs[0].leg_id)&(rows.family=='KEY_POSTURE_TIMING')]
    domain=Domain(source,rows,'KEY_POSTURE_TIMING')
    return domain,ReplayStore(d.legs[0].leg_id,'KEY_POSTURE_TIMING'),source,rows

def test_gp_3d_preserves_kernel_and_uses_third_coordinate():
    old=ResidualGaussianProcess();new=GP3D()
    x=np.array([[0,0],[1,0],[0,1]]);y=np.array([1,.7,.9])
    old.fit_arrays(x,y,np.zeros(3));new.fit(np.column_stack((x,np.zeros(3))),y)
    np.testing.assert_allclose(old.predict_beta(.015,.015),new.predict([.5,.5,0]),rtol=1e-12,atol=1e-12)
    assert new.predict([.5,.5,1])!=new.predict([.5,.5,0])

def test_no_truth_prefilter_and_causal_budget(setup):
    domain,store,source,rows=setup
    fake=rows.copy();fake['E2']=999;fake['E3']=-999;fake['comparison_feasible']=False
    other=Domain(source,fake,'KEY_POSTURE_TIMING')
    assert [(p.beta,p.share_shift) for p in domain]==[(p.beta,p.share_shift) for p in other]
    a,_=sequential(domain,store,0.,'MODEL_INFORMED_BO_EI',budget=2)
    b,diag=sequential(domain,store,0.,'MODEL_INFORMED_BO_EI',budget=4)
    assert [r['candidate_id'] for r in a]==[r['candidate_id'] for r in b[:2]]
    assert len(b)==len(set(r['candidate_id'] for r in b))==4
    assert all(x['fit_valid_episode_count']<=x['trial_index']+1 for x in diag)

def test_constraints_consumed_and_scalar_isolation(setup):
    domain,store,_,_=setup
    env=Environment(domain,store,0.,False)
    obs,raw=env.evaluate(domain.reference,1)
    assert obs.valid and obs.identification_payload is None
    violating=next(p for p in domain if p.beta_flex<0 and p.beta_extend<0 and p.share_shift==0)
    obs,raw=env.evaluate(violating,2)
    assert len(env.calls)==2
    if not raw['feasible']:
        assert obs.endpoint_value is None and not obs.valid
    for method in ('PURE_BO_EI','RANDOM','ADAPTIVE_GREEDY'):
        rows,_=sequential(domain,store,.01,method)
        assert len(rows)==4
        assert np.all(np.diff([r['best_E3'] for r in rows])<=0)
