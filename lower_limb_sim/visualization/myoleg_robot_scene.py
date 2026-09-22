"""Frozen MyoLeg replay with scene-only schematic arm and cuff geometry."""
from pathlib import Path
import numpy as np
import mujoco
from external_simulation.myoleg_reference_trajectory_replay_v1.build_and_replay import (
    SENSITIVITY_MODEL, reset_to_target_state, qpos_address,
)
from external_simulation.myoleg_v2_candidate_domain_design_v1.build_candidate_domain import load_reference_adapter
from personalization.rom_gated_v2.rom import SubjectROMProfile, FROZEN_SUBJECT_ROM_PROFILE_V1
from personalization.rom_gated_v2.reference import SubjectSpecificV3CandidateDomain

MODEL_PATH = SENSITIVITY_MODEL
CAMERA = dict(azimuth=55., elevation=-28., distance=3.65, lookat=[.28,-.20,.96])
ARM_SHOULDER = np.array([.95,-.80,1.04])
ARM_LENGTHS = (.72,.68)
CONNECTOR_OFFSET = np.array([0.,-.15,0.])


def native_domain():
    """Wrap the existing frozen native reference ROM, without choosing new bounds."""
    r=load_reference_adapter()
    lo,hi=r['q'].min(axis=0),r['q'].max(axis=0)
    p=SubjectROMProfile('MYOLEG_NATIVE_REFERENCE_ROM_V1',1,lo[0],hi[0],lo[1],hi[1],
        FROZEN_SUBJECT_ROM_PROFILE_V1,
        'external_simulation_audits/myoleg_knee_rom_compatibility_audit_v1/NATIVE_ROM_REFERENCE_CANDIDATE.csv',frozen=True)
    domain=SubjectSpecificV3CandidateDomain.from_frozen_beta_grid(p)
    np.testing.assert_allclose(domain.reference.trajectory.q,r['q'],atol=1e-14,rtol=0)
    return domain


def schematic_ik(target):
    """Fixed-link analytic position IK. No limits, dynamics or hardware semantics."""
    delta=np.asarray(target)-ARM_SHOULDER
    distance=np.linalg.norm(delta)
    a,b=ARM_LENGTHS
    if not abs(a-b)<distance<a+b:
        raise ValueError(f'Unreachable schematic target: distance={distance:.6f} m')
    axis=delta/distance
    bend=np.array([0.,0.,1.])-axis*axis[2]
    bend/=np.linalg.norm(bend)
    along=(a*a-b*b+distance*distance)/(2*distance)
    elbow=ARM_SHOULDER+along*axis+np.sqrt(a*a-along*along)*bend
    # Recover the endpoint through forward link directions, not copied target.
    first=(elbow-ARM_SHOULDER)/a
    second=(target-elbow)/b
    endpoint=ARM_SHOULDER+a*first+b*second
    return elbow,endpoint


class MyoLegRobotScene:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data=mujoco.MjData(self.model)
        self.domain=native_domain()
        self.reference=self.domain.subject_reference
        self.qadr=[qpos_address(self.model,n) for n in ('hip_flexion_r','knee_angle_r')]
        self.cuff_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'RTB3')
        self.body=int(self.model.site_bodyid[self.cuff_id])
        self.renderer=None
        self.camera=mujoco.MjvCamera()
        for key,value in CAMERA.items():setattr(self.camera,key,value)
        self.options=mujoco.MjvOption()
        self.options.geomgroup[3:]=0
        self.options.sitegroup[:]=0
        self.max_q_error=np.zeros(2)
        self.max_ik_error=0.
        self.model.vis.global_.offwidth=1920
        self.model.vis.global_.offheight=1080

    def state(self,candidate,index):
        t=candidate.trajectory
        reset_to_target_state(self.model,self.data,t.q[index],t.dq[index],t.ddq[index])
        mujoco.mj_forward(self.model,self.data)
        error=np.abs(self.data.qpos[self.qadr]-t.q[index])
        self.max_q_error=np.maximum(self.max_q_error,error)
        cuff=self.data.site_xpos[self.cuff_id].copy()
        target=cuff+CONNECTOR_OFFSET
        elbow,endpoint=schematic_ik(target)
        self.max_ik_error=max(self.max_ik_error,float(np.linalg.norm(endpoint-target)))
        return cuff,elbow,endpoint

    def geom(self,kind,size,pos,rgba,rotation=None):
        scene=self.renderer.scene
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),
            np.eye(3).ravel() if rotation is None else np.asarray(rotation).ravel(),np.asarray(rgba,dtype=np.float32))
        scene.ngeom+=1
        return g

    def capsule(self,a,b,radius,color):
        g=self.geom(mujoco.mjtGeom.mjGEOM_CAPSULE,[0,0,0],[0,0,0],color)
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,np.asarray(a),np.asarray(b))

    def render(self,candidate,index,width=1600,height=850):
        if self.renderer is None or (self.renderer.width,self.renderer.height)!=(width,height):
            self.close()
            self.renderer=mujoco.Renderer(self.model,height,width,max_geom=3000)
        cuff,elbow,end=self.state(candidate,index)
        self.renderer.update_scene(self.data,camera=self.camera,scene_option=self.options)
        scene=self.renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        # Hide environmental decorations from the source model in this viewport only.
        for i in range(scene.ngeom):
            g=scene.geoms[i]
            if g.objtype==mujoco.mjtObj.mjOBJ_GEOM:
                n=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g.objid) or ''
                if self.model.geom_bodyid[g.objid]==0:
                    g.rgba[3]=0
        box=mujoco.mjtGeom.mjGEOM_BOX
        sphere=mujoco.mjtGeom.mjGEOM_SPHERE
        self.geom(box,[1.03,.37,.055],[.05,.12,.78],[.25,.32,.40,1])
        self.geom(box,[.97,.34,.025],[.05,.12,.86],[.64,.72,.75,1])
        for x in (-.7,.8):
            for y in (-.13,.37):
                self.geom(box,[.025,.025,.37],[x,y,.37],[.20,.25,.30,1])
        self.geom(box,[1.5,1.25,.01],[.0,-.2,-.02],[.10,.15,.21,1])
        # Schematic arm, fixed link lengths; no added MjModel bodies or forces.
        self.geom(box,[.16,.16,.055],[.95,-.80,.055],[.16,.22,.30,1])
        self.capsule([.95,-.80,.11],ARM_SHOULDER,.068,[.27,.35,.44,1])
        self.capsule(ARM_SHOULDER,elbow,.048,[.23,.67,.74,1])
        self.capsule(elbow,end,.041,[.34,.77,.80,1])
        for p in (ARM_SHOULDER,elbow,end):self.geom(sphere,[.061]*3,p,[.19,.25,.33,1])
        self.capsule(end,cuff,.018,[.87,.71,.36,1])
        # RTB3 is the existing shank interaction site. Rings use its tibia frame.
        R=self.data.xmat[self.body].reshape(3,3)
        local=self.model.site_pos[self.cuff_id].copy()
        center=local.copy();center[2]=0.
        for offset in (-.026,.026):
            points=[]
            for a in np.linspace(0,2*np.pi,33):
                p=center+np.array([.057*np.cos(a),offset,.057*np.sin(a)])
                points.append(self.data.xpos[self.body]+R@p)
            for a,b in zip(points[:-1],points[1:]):self.capsule(a,b,.009,[.92,.63,.23,1])
        self.geom(box,[.04,.042,.009],cuff,[.94,.71,.32,1],R)
        rgb=self.renderer.render()
        rgb[np.max(rgb,axis=2)==0]=(12,19,29)
        return rgb

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
