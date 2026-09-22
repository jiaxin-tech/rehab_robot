"""Deterministic presentation of frozen V3 samples using MuJoCo offscreen RGB."""
from pathlib import Path
import json
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from lower_limb_sim.five_leg_mujoco_v1.model import load_frozen_benchmark_definition, make_mujoco_model
from lower_limb_sim.five_leg_mujoco_v1.benchmark import build_leg_domain
from lower_limb_sim.five_leg_mujoco_v1.replay import replay_trajectory

BETAS = ((0., 0.), (-.03, .03), (.03, -.03))
NAMES = ('Reference', 'V3 Contrast A', 'V3 Contrast B')
COLORS = ('#63d9da', '#edb46e')
CAMERA = dict(azimuth=90., elevation=0., distance=1.50, lookat=[.30, 0., .19])
FPS = 30


def select(domain, beta):
    for candidate in domain:
        if np.allclose((candidate.beta_flex, candidate.beta_extend), beta, atol=1e-12, rtol=0):
            return candidate
    raise ValueError('beta must be an exact member of the frozen V3 grid')


def sample_indices(count, frames):
    if count < 2 or frames < 2:
        raise ValueError('at least two samples and frames required')
    return np.rint(np.linspace(0, count - 1, frames)).astype(int)


def assign_state(model, data, trajectory, index):
    data.qpos[:] = trajectory.q[index] * (1., -1.)
    data.qvel[:] = trajectory.dq[index] * (1., -1.)
    mujoco.mj_forward(model, data)
    return np.abs(data.qpos * (1., -1.) - trajectory.q[index])


def font(size):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class VideoRenderer:
    def __init__(self, leg_id='LEG_0_NOMINAL', width=1920, height=1080):
        self.definition = load_frozen_benchmark_definition()
        self.leg = next((x for x in self.definition.legs if x.leg_id == leg_id), None)
        if self.leg is None:
            raise ValueError(f'unknown frozen leg: {leg_id}')
        self.domain = build_leg_domain(self.leg)
        self.reference = self.domain.subject_reference
        self.model = make_mujoco_model(self.definition, self.leg)
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        # Render each viewport at its display resolution; framebuffer capacity is visual only.
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.renderer = None
        self.max_error = np.zeros(2)
        self.camera = mujoco.MjvCamera()
        for key, value in CAMERA.items():
            setattr(self.camera, key, value)
        self.validated = set()
        self.traces = {}

    def validate(self, candidate):
        if candidate.candidate_id in self.validated:
            return
        t = candidate.trajectory
        result = replay_trajectory(model=self.model, definition=self.definition, leg=self.leg,
            time_s=self.reference.time_s, q_project=t.q, dq_project=t.dq, ddq_project=t.ddq)
        if not result.valid:
            raise ValueError(result.invalid_reason)
        ref = self.domain.reference.trajectory
        for values, original in ((t.q, ref.q), (t.dq, ref.dq), (t.ddq, ref.ddq)):
            np.testing.assert_allclose(values[[0, -1]], original[[0, -1]], atol=1e-12)
        positions=[]
        scratch=mujoco.MjData(self.model)
        for i in range(len(t.q)):
            assign_state(self.model,scratch,t,i)
            positions.append(scratch.site_xpos[0].copy())
        self.traces[candidate.candidate_id]=np.asarray(positions)
        self.validated.add(candidate.candidate_id)

    def scene(self, candidate, index, width, height):
        self.validate(candidate)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height, width)
        elif (self.renderer.width, self.renderer.height) != (width, height):
            self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height, width)
        self.max_error = np.maximum(self.max_error, assign_state(self.model, self.data, candidate.trajectory, index))
        self.renderer.update_scene(self.data, camera=self.camera)
        scene = self.renderer.scene
        rgba = getattr(self, "palette", ((.22,.28,.35,1), (.22,.78,.80,1), (.92,.61,.29,1)))
        for i in range(scene.ngeom):
            geom = scene.geoms[i]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid < 3:
                geom.rgba[:] = rgba[geom.objid]
        # Joint markers exist only in the rendering scene, never in MjModel.
        for pos in (self.data.xanchor[0], self.data.xanchor[1], self.data.site_xpos[0]):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.018]*3),
                              pos + np.array([0., -.05, 0.]), np.eye(3).ravel(), np.array([.92,.96,1.,1.]))
            scene.ngeom += 1
        tail=self.traces[candidate.candidate_id][max(0,index-22):index+1]
        for a,b in zip(tail[:-1],tail[1:]):
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.32,.49,.62,1.]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0018,a,b)
            scene.ngeom += 1
        rgb = self.renderer.render()
        # MuJoCo's empty background is black; change only empty RGB pixels.
        rgb[np.max(rgb, axis=2) == 0] = (12,19,29)
        return Image.fromarray(rgb)

    def text(self, draw, xy, value, size=28, fill='#dce7f1'):
        draw.text(xy, value, font=font(round(size*self.width/1920)), fill=fill)

    def compose(self, candidates, index, title=None, card=None):
        w,h = self.width,self.height
        s=w/1920
        canvas=Image.new('RGB',(w,h),(12,19,29))
        draw=ImageDraw.Draw(canvas)
        def txt(x,y,value,size=28,fill='#dce7f1'):
            self.text(draw,(round(x*s),round(y*s)),value,size,fill)
        txt(72,40,'MOTION STUDIES  /  01',22,'#63d9da')
        txt(72,86,title or '2-DOF Supine Rehabilitation Simulation',48)
        txt(72,153,'MuJoCo  /  Fixed ROM  /  V3 Coordination',25,'#91a5b8')
        panel_width = (w-round(144*s))//len(candidates)
        # Same camera and aspect for both technical panels; single panel uses same ratio.
        view_w = panel_width-round(24*s)
        view_h = round(view_w*.70)
        if len(candidates)==1:
            view_w,view_h=round(1100*s),round(770*s)
        for k,c in enumerate(candidates):
            beta=(c.beta_flex,c.beta_extend)
            name=NAMES[BETAS.index(beta)] if beta in BETAS else 'V3 trajectory'
            x=round(72*s)+k*panel_width
            y=round(268*s)
            image=self.scene(c,index,view_w,view_h)
            if len(candidates)==1:
                canvas.paste(image,(round(705*s),round(225*s)))
                txt(72,330,name,46)
                txt(72,410,f'beta_flex     {beta[0]:+.3f}',29)
                txt(72,457,f'beta_extend  {beta[1]:+.3f}',29)
                txt(72,555,f't = {self.reference.time_s[index]:05.2f} s',30,'#63d9da')
                txt(72,610,str(self.reference.phases[index]).upper(),22,'#91a5b8')
            else:
                canvas.paste(image,(x,y))
                txt(x/s,222,name,32)
                txt(x/s,270,f'beta_flex {beta[0]:+.3f}  /  beta_extend {beta[1]:+.3f}',23,'#91a5b8')
                hip,knee=np.rad2deg(c.trajectory.q[index])
                txt(x/s,865,f'HIP {hip:5.1f}°     KNEE {knee:5.1f}°',27)
                # Mini plot uses original samples, with a synchronized time cursor.
                px,py=x,round(760*s); pw=panel_width-round(75*s); ph=round(65*s)
                q=np.rad2deg(c.trajectory.q)
                for joint,color in enumerate(COLORS):
                    points=[(px+i*pw/(len(q)-1),py+ph-float(v)/140*ph) for i,v in enumerate(q[:,joint])]
                    draw.line(points,fill=color,width=max(1,round(2*s)))
                cursor=px+index*pw/(len(q)-1)
                draw.line((cursor,py,cursor,py+ph),fill='#ffffff',width=2)
        if len(candidates)>1:
            txt(72,943,f'SIMULATION  {self.reference.time_s[index]:05.2f} s    /    {str(self.reference.phases[index]).upper()}',26,'#63d9da')
        leg=self.leg
        rom=f'Hip {np.rad2deg(leg.hip_min_rad):.0f}–{np.rad2deg(leg.hip_max_rad):.0f}°  /  Knee {np.rad2deg(leg.knee_min_rad):.1f}–{np.rad2deg(leg.knee_max_rad):.1f}°'
        txt(72,1020,'OFFLINE EXACT-STATE REPLAY',20,'#91a5b8')
        txt(1150,1020,rom,22,'#91a5b8')
        if card:
            overlay=Image.new('RGBA',(w,h),(12,19,29,255))
            canvas=Image.alpha_composite(canvas.convert('RGBA'),overlay).convert('RGB')
            draw=ImageDraw.Draw(canvas)
            txt(110,370,'2-DOF Supine Rehabilitation' if card=='intro' else 'Same ROM',64)
            txt(110,455,'Simulation' if card=='intro' else 'Different Hip–Knee Coordination',64)
            txt(110,580,'MuJoCo  •  Fixed ROM  •  V3 Coordination' if card=='intro' else 'Offline MuJoCo Simulation',31,'#63d9da')
        return np.asarray(canvas)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def render_video(mode, output, leg='LEG_0_NOMINAL', beta=(0.,0.), width=1920, height=1080):
    from .encoding import open_video_writer
    if (width,height) not in ((1920,1080),(1280,720)):
        raise ValueError('supported resolutions: 1920x1080 or 1280x720')
    if mode not in ('resume-showcase','side-by-side','single'):
        raise ValueError('unknown mode')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    video=VideoRenderer(leg,width,height)
    writer=None
    try:
        candidates=[select(video.domain,b) for b in BETAS]
        if mode=='resume-showcase':
            plan=[([candidates[0]],0,'intro')]*60
            for c in candidates:
                plan += [([c],int(i),None) for i in sample_indices(len(video.reference.time_s),150)]
            plan += [([candidates[-1]],len(video.reference.time_s)-1,'outro')]*60
        else:
            chosen=candidates[:2] if mode=='side-by-side' else [select(video.domain,beta)]
            plan=[(chosen,int(i),None) for i in sample_indices(len(video.reference.time_s),300)]
        writer=open_video_writer(output,width,height,FPS)
        previews=output.parent/'previews'; previews.mkdir(exist_ok=True)
        for frame,(chosen,index,card) in enumerate(plan):
            rgb=video.compose(chosen,index,title='Reference vs V3 Coordination' if mode=='side-by-side' else None,card=card)
            if not np.isfinite(rgb).all() or not rgb.max():
                raise RuntimeError('invalid frame')
            writer.send(rgb)
            if frame in (0,len(plan)//2,len(plan)-1):
                suffix={0:'start',len(plan)//2:'mid',len(plan)-1:'end'}[frame]
                Image.fromarray(rgb).save(previews/f'{mode}_{suffix}.png')
            if frame%90==0:
                print(f'{mode}: {frame}/{len(plan)} frames',flush=True)
        writer.close(); writer=None
        result=dict(leg_id=leg,rom_deg={k:v for k,v in video.leg.as_table_row().items() if k in ('hip_min_deg','hip_max_deg','knee_min_deg','knee_max_deg')},
            frames=len(plan),fps=FPS,duration_s=len(plan)/FPS,resolution=[width,height],
            simulation_duration_s=video.reference.duration_s,max_render_trajectory_error_rad=video.max_error.tolist(),
            camera=CAMERA,encoder='libx264 / H.264 / yuv420p',offscreen_rendering=True)
        output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return result
    finally:
        if writer is not None: writer.close()
        video.close()
