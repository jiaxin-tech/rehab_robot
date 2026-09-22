import numpy as np
import pytest
from lower_limb_sim.e3_candidate_comparison.run import e3,key_posture,BUMP
from lower_limb_sim.five_leg_mujoco_v1 import load_frozen_benchmark_definition,build_leg_domain
from lower_limb_sim.trajectory_sensitivity.study import trajectory

@pytest.fixture(scope='module')
def ref():return build_leg_domain(load_frozen_benchmark_definition().legs[0]).subject_reference

def test_e3_mean_tradeoff_and_scaling():
    assert e3([1,2,3,4],[1,2,3,4])==1
    assert e3([.9,1.05,1,1],[1,1,1,1])==pytest.approx(.9875)
    assert e3([9,10.5,10,10],[10,10,10,10])==pytest.approx(.9875)
    with pytest.raises(ValueError):e3([1]*4,[0]*4)

def test_keypoint_geometry_and_c2(ref):
    assert BUMP(.5)==pytest.approx(1)
    for order in (0,1,2):np.testing.assert_allclose(BUMP([0,1],nu=order),0,atol=1e-12)
    t,_=key_posture(ref,ref.as_v3_mapping(),0,0,.05)
    old,_=trajectory(ref,ref.as_v3_mapping(),0,0,.05)
    for k in ('q','dq','ddq','time_s'):np.testing.assert_allclose(t[k],old[k],atol=1e-12)
    t,m=key_posture(ref,ref.as_v3_mapping(),.03,-.03,0)
    np.testing.assert_array_equal(t['q'][:,0],ref.q[:,0])
    assert np.max(abs(t['q'][:,1]-ref.q[:,1]))>0
    assert m['flex_midpoint_offset_deg']==pytest.approx(np.rad2deg(.03*np.ptp(ref.q[:,1])))
    with pytest.raises(ValueError):key_posture(ref,ref.as_v3_mapping(),1,1,0)
