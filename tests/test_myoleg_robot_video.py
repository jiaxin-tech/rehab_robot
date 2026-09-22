"""Focused scene and data contracts; no controller or scientific method changes."""
import json
from pathlib import Path
import numpy as np
import pytest
from lower_limb_sim.visualization.myoleg_robot_scene import (
    MyoLegRobotScene,schematic_ik,ARM_SHOULDER,ARM_LENGTHS,CONNECTOR_OFFSET,
)
from lower_limb_sim.visualization.mujoco_rehab_video import select
from lower_limb_sim.mechanical_endpoints import full_cycle_dual_joint_rms


@pytest.fixture(scope='module')
def scene():
    s=MyoLegRobotScene()
    yield s
    s.close()


def test_native_replay_all_samples_and_fixed_link_ik(scene):
    for beta in ((0.,0.),(-.03,.03),(.03,-.03),(-.03,-.03),(.03,.03)):
        c=select(scene.domain,beta)
        np.testing.assert_array_equal(c.trajectory.q[[0,-1]],scene.domain.reference.trajectory.q[[0,-1]])
        for i in range(len(scene.reference.time_s)):
            cuff,elbow,end=scene.state(c,i)
            np.testing.assert_allclose(np.linalg.norm(elbow-ARM_SHOULDER),ARM_LENGTHS[0],atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(end-elbow),ARM_LENGTHS[1],atol=1e-12)
            np.testing.assert_allclose(end,cuff+CONNECTOR_OFFSET,atol=1e-12)
    np.testing.assert_array_equal(scene.max_q_error,[0.,0.])
    assert scene.max_ik_error<1e-12
    with pytest.raises(ValueError,match='Unreachable'):schematic_ik(np.array([20.,0.,0.]))


def test_scene_render_keeps_muscle_and_mechanical_parameters(scene):
    names=('body_mass','body_inertia','jnt_range','jnt_stiffness','dof_damping',
           'actuator_gainprm','actuator_biasprm','actuator_dynprm','tendon_stiffness',
           'tendon_lengthspring','eq_data','site_pos','geom_rgba')
    before={n:getattr(scene.model,n).copy() for n in names}
    c=scene.domain.reference
    a=scene.render(c,0,875,590);b=scene.render(c,170,875,590)
    assert a.max()>0 and not np.array_equal(a,b)
    for name in names:np.testing.assert_array_equal(before[name],getattr(scene.model,name))


def test_recorded_objectives_are_from_executed_full_trajectories():
    path=Path('outputs/myoleg_bo_demo_run.json')
    if not path.exists():pytest.skip('run optimization-demo first to create its actual data')
    run=json.loads(path.read_text());data=np.load(path.with_suffix('.npz'))
    assert run['method']=='MODEL_INFORMED_BO_EI_TIMESERIES_ID'
    assert len(run['trials'])==run['budget']==4
    best=float('inf')
    for k,row in enumerate(run['trials']):
        t=data[f'trial_{k}_time_s'];tau=data[f'trial_{k}_tau']
        value=full_cycle_dual_joint_rms(tau[:,0],tau[:,1],t)
        assert value==pytest.approx(row['objective'],abs=1e-12)
        best=min(best,value)
        assert best==row['best']
    assert run['relative_improvement_percent']==pytest.approx(100*(run['reference_objective']-best)/run['reference_objective'])
