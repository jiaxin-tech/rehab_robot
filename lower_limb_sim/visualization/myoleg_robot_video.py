"""Presentation compositor for the frozen MyoLeg + schematic robot scene."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .myoleg_robot_scene import MyoLegRobotScene,CAMERA,MODEL_PATH,CONNECTOR_OFFSET
from .mujoco_rehab_video import font,sample_indices,select
from .encoding import open_video_writer
from .optimization_video import run_demo

BG=(12,19,29)
WHITE='#e4edf4';MUTED='#9aaec0';TEAL='#65d9d7';GOLD='#efb860'


class Composer:
    def __init__(self,scene,run=None):
        self.scene=scene;self.run=run

    def text(self,xy,text,size=28,color=WHITE):
        self.draw.text(xy,str(text),font=font(size),fill=color)

    def base(self,title,subtitle):
        self.image=Image.new('RGB',(1920,1080),BG);self.draw=ImageDraw.Draw(self.image)
        self.text((65,28),'REHABILITATION ROBOTICS  /  OFFLINE STUDY',20,TEAL)
        self.text((65,70),title,43)
        self.text((65,134),subtitle,24,MUTED)
        self.draw.line((65,1000,1855,1000),fill='#2b3b4a',width=2)
        self.text((65,1022),'SCHEMATIC ARM  /  VISUAL-ONLY CUFF  /  NO PHYSICAL CONTROL',19,MUTED)
        self.text((1360,1022),'MyoLeg musculoskeletal model',22,TEAL)

    def viewport(self,candidate,index,box):
        x,y,w,h=box
        rgb=self.scene.render(candidate,index,w,h)
        self.image.paste(Image.fromarray(rgb),(x,y))

    def beta(self,c):
        return f'beta_flex {c.beta_flex:+.3f}    beta_extend {c.beta_extend:+.3f}'

    def angles(self,c,index):
        hip,knee=np.rad2deg(c.trajectory.q[index])
        return f'Hip {hip:5.1f}°    Knee {knee:5.1f}°'

    def system(self,c,index,card=None):
        self.base('MyoLeg + Robot Rehabilitation Simulation','Fixed native ROM  /  Prescribed-state replay  /  Shank cuff at RTB3')
        self.viewport(c,index,(370,177,1500,795))
        self.text((65,255),'REFERENCE' if c.beta==(0.,0.) else 'V3 CONTRAST A',30,TEAL)
        self.text((65,323),f'beta_flex    {c.beta_flex:+.3f}',26)
        self.text((65,365),f'beta_extend {c.beta_extend:+.3f}',26)
        hip,knee=np.rad2deg(c.trajectory.q[index])
        self.text((65,456),f'Hip     {hip:5.1f}°',30)
        self.text((65,503),f'Knee  {knee:5.1f}°',30)
        self.text((65,598),f't = {self.scene.reference.time_s[index]:05.2f} s',28,TEAL)
        self.text((65,645),str(self.scene.reference.phases[index]).upper(),21,MUTED)
        self.text((65,915),'Same ROM · Different Hip–Knee Coordination',31)
        if card:
            self.image=Image.new('RGB',(1920,1080),BG);self.draw=ImageDraw.Draw(self.image)
            self.text((110,285),'OFFLINE SYSTEM VISUALIZATION',24,TEAL)
            self.text((110,365),'MyoLeg + Robot' if card=='intro' else 'Same ROM',76)
            self.text((110,465),'Rehabilitation Simulation' if card=='intro' else 'Different Hip–Knee Coordination',65)
            self.text((110,605),'Musculoskeletal model  /  Schematic arm  /  Visual-only cuff',29,MUTED)
            self.text((110,870),'Deterministic trajectory replay • No physical robot control',24,MUTED)
        return np.asarray(self.image)

    def comparison(self,ref,contrast,index):
        self.base('Reference vs V3 Contrast','Same MyoLeg  /  Same ROM and duration  /  Same robot setup and simulation time')
        for x,c,name in ((60,ref,'Reference'),(985,contrast,'V3 contrast trajectory')):
            self.viewport(c,index,(x,290,875,590))
            self.text((x,212),name,34,TEAL)
            self.text((x,263),self.beta(c),23)
            self.text((x,888),self.angles(c,index),27)
        self.text((65,944),f'Simulation time  {self.scene.reference.time_s[index]:05.2f} s  /  {str(self.scene.reference.phases[index]).upper()}',25,TEAL)
        self.text((1000,944),'Coordination comparison; no efficacy claim',23,MUTED)
        return np.asarray(self.image)

    def optimization(self,trial,index,reveal=False):
        rows=self.run['trials'];row=rows[trial];c=select(self.scene.domain,tuple(row['beta']))
        self.base('Offline Model-Informed Bayesian Trajectory Optimization',
            'Full time-series ID → effective gray-box → residual GP → Expected Improvement')
        self.viewport(c,index,(20,260,1130,670))
        self.text((65,215),f'TRIAL {trial}  /  {self.beta(c)}',27,TEAL)
        self.text((65,889),self.angles(c,index)+f'    t = {self.scene.reference.time_s[index]:05.2f} s',25)
        self.text((65,935),'Equivalent-force ID input; not a cuff sensor simulation',21,MUTED)
        self.draw.line((1153,196,1153,977),fill='#2b3b4a',width=2)
        self.text((1200,203),'EXECUTED COORDINATION SPACE',22,TEAL)
        left,top,pw,ph=1278,277,472,260
        self.draw.rectangle((left,top,left+pw,top+ph),outline='#485b6a',width=2)
        def pos(beta):return (left+(beta[0]+.03)/.06*pw,top+(.03-beta[1])/.06*ph)
        for value in (-.03,0.,.03):
            x,_=pos((value,0));_,y=pos((0,value))
            self.draw.line((x,top,x,top+ph),fill='#243544')
            self.draw.line((left,y,left+pw,y),fill='#243544')
            self.text((x-25,top+ph+12),f'{value:+.2f}',19,MUTED)
            self.text((left-76,y-12),f'{value:+.2f}',19,MUTED)
        self.text((1405,584),'beta_flex',21,MUTED)
        self.text((1194,248),'beta_extend',19,MUTED)
        rx,ry=pos((0.,0.));self.draw.rectangle((rx-6,ry-6,rx+6,ry+6),outline=WHITE,width=2)
        for previous in rows[:trial]:
            x,y=pos(previous['beta']);self.draw.ellipse((x-6,y-6,x+6,y+6),fill=TEAL)
        x,y=pos(row['beta']);self.draw.ellipse((x-10,y-10,x+10,y+10),outline=GOLD,width=3)
        # Current scalar objective is revealed only AFTER the complete replay.
        visible=rows[:trial+1] if reveal else rows[:trial]
        best=min((r['objective'] for r in visible),default=None)
        self.text((1200,630),'BEST OBSERVED E0  /  N·m',22,TEAL)
        gx,gy,gw,gh=1280,692,490,120
        self.draw.line((gx,gy,gx,gy+gh,gx+gw,gy+gh),fill='#485b6a',width=2)
        # Fixed nonzero scale is explicitly ticked; no hidden percentage amplification.
        values=[r['objective'] for r in rows]
        lo=np.floor(min(values)*10)/10-.1;hi=np.ceil(max(values)*10)/10+.1
        for val in (lo,hi):
            self.text((1197,gy+(hi-val)/(hi-lo)*gh-10),f'{val:.1f}',19,MUTED)
        pts=[]
        for k,r in enumerate(visible):pts.append((gx+k*gw/3,gy+(hi-r['best'])/(hi-lo)*gh))
        if len(pts)>1:self.draw.line(pts,fill=TEAL,width=3)
        for x,y in pts:self.draw.ellipse((x-4,y-4,x+4,y+4),fill=TEAL)
        for k in range(4):self.text((gx+k*gw/3-20,gy+gh+10),f'Trial {k}',19,MUTED)
        current=f"{row['objective']:.6f} N·m" if reveal else 'pending full trajectory'
        self.text((1200,873),f'J: {current}',25)
        self.text((1200,914),'Best: '+(f'{best:.6f} N·m' if best is not None else 'pending reference'),25)
        if reveal and trial==0:
            self.text((1200,955),f"Reference: {row['objective']:.6f} N·m",23,GOLD)
        if reveal and trial==3:
            self.text((1200,955),f"Reduction: {self.run['relative_improvement_percent']:.3f}% (small)",23,GOLD)
        return np.asarray(self.image)


def render(mode,output,run_path='outputs/myoleg_bo_demo_run.json',rerun=False):
    if mode not in ('system-showcase','reference-vs-selected','optimization-demo'):raise ValueError(mode)
    output=Path(output);run=None
    if mode=='optimization-demo':
        run=run_demo(run_path) if rerun or not Path(run_path).exists() else json.loads(Path(run_path).read_text())
    scene=MyoLegRobotScene();composer=Composer(scene,run)
    reference=scene.domain.reference;contrast=select(scene.domain,(-.03,.03))
    writer=None
    try:
        n=len(scene.reference.time_s)
        if mode=='system-showcase':
            plan=[('system',reference,0,'intro')]*60
            for c in (reference,contrast):plan.extend(('system',c,int(i),None) for i in sample_indices(n,330))
            plan.extend([('system',contrast,n-1,'outro')]*60)
        elif mode=='reference-vs-selected':
            plan=[('comparison',reference,int(i),None) for i in sample_indices(n,360)]
        else:
            plan=[]
            for k in range(4):
                plan.extend(('optimization',k,int(i),False) for i in sample_indices(n,150))
                plan.extend([('optimization',k,n-1,True)]*45)
        writer=open_video_writer(output)
        preview=output.parent/'previews';preview.mkdir(parents=True,exist_ok=True)
        for frame,(kind,c,index,extra) in enumerate(plan):
            if kind=='system':rgb=composer.system(c,index,extra)
            elif kind=='comparison':rgb=composer.comparison(reference,contrast,index)
            else:rgb=composer.optimization(c,index,extra)
            writer.send(rgb)
            if frame in (0,len(plan)//2,len(plan)-1,180):
                Image.fromarray(rgb).save(preview/f'{output.stem}_{frame:04d}.png')
            if frame%120==0:print(f'{mode}: {frame}/{len(plan)}',flush=True)
        writer.close();writer=None
        info=dict(mode=mode,source_model=str(MODEL_PATH),robot_model_available=False,robot_model_type='SCHEMATIC',
            cuff_model='VISUAL_ONLY',camera=CAMERA,resolution=[1920,1080],fps=30,frames=len(plan),duration_s=len(plan)/30,
            max_q_video_source_error_rad=scene.max_q_error.tolist(),max_visualization_ik_error_m=scene.max_ik_error,
            flange_to_RTB3_connector_length_m=float(np.linalg.norm(CONNECTOR_OFFSET)),
            rom_deg=dict(min=np.rad2deg(scene.reference.q.min(0)).tolist(),max=np.rad2deg(scene.reference.q.max(0)).tolist()),
            reference_vs_selected_policy='Explicit V3 contrast; 0.218% demo gain is not established meaningful efficacy',
            offscreen=True,encoder='libx264 H.264 yuv420p')
        output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info,indent=2))
        return info
    finally:
        if writer is not None:writer.close()
        scene.close()
