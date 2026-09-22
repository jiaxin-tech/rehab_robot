import numpy as np
import pytest
from lower_limb_sim.visualization.mujoco_rehab_video import (
    BETAS, VideoRenderer, select, sample_indices, assign_state,
)

@pytest.fixture(scope='module')
def video():
    v=VideoRenderer()
    yield v
    v.close()


def test_selection_rom_and_anchors(video):
    assert len(video.domain)==625
    ref=select(video.domain,BETAS[0])
    for beta in BETAS:
        c=select(video.domain,beta)
        assert (c.beta_flex,c.beta_extend)==beta
        assert c.rom_profile_id==ref.rom_profile_id
        np.testing.assert_allclose(c.trajectory.q.min(axis=0),ref.trajectory.q.min(axis=0),atol=2e-5)
        np.testing.assert_allclose(c.trajectory.q.max(axis=0),ref.trajectory.q.max(axis=0),atol=2e-5)
        assert c.validation['duration_s']==24
        np.testing.assert_array_equal(c.trajectory.q[[0,-1]],ref.trajectory.q[[0,-1]])
    for beta in ((.031,0),(.001,0),(float('nan'),0)):
        with pytest.raises(ValueError): select(video.domain,beta)


def test_frame_plan_and_exact_state(video):
    indices=sample_indices(401,150)
    assert len(indices)==150 and indices[0]==0 and indices[-1]==400
    np.testing.assert_array_equal(indices,sample_indices(401,150))
    for beta in BETAS:
        t=select(video.domain,beta).trajectory
        for index in indices:
            np.testing.assert_array_equal(assign_state(video.model,video.data,t,index),[0,0])


def test_offscreen_does_not_modify_model(video):
    # Includes dynamics, geometry, joint limits and visual arrays after renderer setup.
    names=('body_mass','body_inertia','jnt_range','jnt_stiffness','dof_damping','geom_size','geom_rgba','site_pos')
    before={n:getattr(video.model,n).copy() for n in names}
    c=select(video.domain,BETAS[1])
    first=np.asarray(video.scene(c,0,864,604))
    middle=np.asarray(video.scene(c,170,864,604))
    assert first.max()>0 and not np.array_equal(first,middle)
    for name in names: np.testing.assert_array_equal(before[name],getattr(video.model,name))
