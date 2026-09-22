import numpy as np
import pytest
from lower_limb_sim.visualization.five_leg_comparison import Comparison,EXPECTED_LEG_IDS
from lower_limb_sim.visualization.e2_overlay import limiting

@pytest.fixture(scope='module')
def comparison():
    c=Comparison()
    yield c
    c.close()


def test_branch_prefix_semantics_and_final_equivalence(comparison):
    for leg in EXPECTED_LEG_IDS:
        for beta in ((0,0),(-.03,.03)):
            e=comparison.episode(leg,beta);r=e.renderer.reference
            assert np.isnan(e.running[0]).all()
            extension=np.flatnonzero(r.phases=='extension')
            assert np.isnan(e.running[extension[0],1::2]).all()
            assert np.isfinite(e.running[extension[1],1::2]).all()
            end_flex=np.flatnonzero(r.phases=='flexion')[-1]
            assert e.finalized[end_flex,0] and not e.finalized[end_flex,1]
            np.testing.assert_allclose(e.running[end_flex:,0],e.normalized[0],atol=1e-12)
            np.testing.assert_allclose(e.running[-1],e.normalized,atol=1e-12)
            assert e.frozen_error<1e-12
            assert np.nanmax(e.running)<2.5
            if beta==(0,0):
                assert e.e2==1 and limiting(e.normalized)=='ALL FOUR (TIE)'


def test_shared_timing_original_q_and_scene_only_palette(comparison):
    time=None
    for leg in EXPECTED_LEG_IDS:
        e=comparison.episode(leg,(-.03,.03));r=e.renderer
        if time is not None:np.testing.assert_array_equal(time,r.reference.time_s)
        time=r.reference.time_s
        before=r.model.geom_rgba.copy();mass=r.model.body_mass.copy()
        for index in (0,180,400):r.scene(e.candidate,index,260,245)
        np.testing.assert_array_equal(r.max_error,[0,0])
        np.testing.assert_array_equal(before,r.model.geom_rgba)
        np.testing.assert_array_equal(mass,r.model.body_mass)


def test_oracle_frozen_source_and_no_claim_change(comparison):
    s=comparison.study
    assert s.common['common_beta']==[0.,0.]
    assert s.summary['E2_PERSONALIZATION_NECESSITY']=='NOT_SUPPORTED'
    assert s.summary['READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON'] is False
    leg=EXPECTED_LEG_IDS[-1];beta=s.oracles[leg]['oracle_beta']
    e=comparison.episode(leg,beta)
    assert e.e2==pytest.approx(s.oracles[leg]['oracle_value'],abs=1e-12)
    assert 100*s.common['maximum_relative_common_regret']==pytest.approx(.07068403701731)
