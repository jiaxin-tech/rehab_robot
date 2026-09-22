import numpy as np
import pytest
from lower_limb_sim.trajectory_sensitivity.study import trajectory
from lower_limb_sim.five_leg_mujoco_v1 import load_frozen_benchmark_definition,build_leg_domain

@pytest.fixture(scope='module')
def ref():
    return build_leg_domain(load_frozen_benchmark_definition().legs[0]).subject_reference


def test_zero_recovers_reference(ref):
    t,_=trajectory(ref,ref.as_v3_mapping(),0,0)
    for name in ('q','dq','ddq','time_s'):
        np.testing.assert_allclose(t[name],getattr(ref,name),atol=1e-14,rtol=1e-14)


def test_retime_preserves_geometry_and_chain_rule(ref):
    base,_=trajectory(ref,ref.as_v3_mapping(),.12,-.12)
    timed,metrics=trajectory(ref,ref.as_v3_mapping(),.12,-.12,.10)
    np.testing.assert_array_equal(base['q'],timed['q'])
    turn=np.flatnonzero(ref.phases=='flexion')[-1]
    for ids in (np.arange(turn+1),np.arange(turn,len(ref.time_s))):
        scale=(timed['time_s'][ids[-1]]-timed['time_s'][ids[0]])/(base['time_s'][ids[-1]]-base['time_s'][ids[0]])
        np.testing.assert_allclose(timed['dq'][ids]*scale,base['dq'][ids],atol=1e-12)
        np.testing.assert_allclose(timed['ddq'][ids]*scale**2,base['ddq'][ids],atol=1e-12)
    assert timed['time_s'][-1]-timed['time_s'][0]==pytest.approx(24)
    assert np.all(np.diff(timed['time_s'])>0)
    assert metrics['ROM_excess_rad']<=2e-5
    for name in ('q','dq','ddq'):
        np.testing.assert_allclose(timed[name][0],timed[name][-1],atol=1e-10)


def test_wide_grid_preserves_rom_and_rejects_fold(ref):
    for a in (-.12,0,.12):
        for b in (-.12,0,.12):
            t,m=trajectory(ref,ref.as_v3_mapping(),a,b,-.10)
            assert m['min_warp_derivative']>0
            assert m['ROM_excess_rad']<=2e-5
            np.testing.assert_allclose(t['q'][:,0],ref.q[:,0],atol=1e-12)
    with pytest.raises(ValueError,match='NONMONOTONE'):
        trajectory(ref,ref.as_v3_mapping(),1,0)
