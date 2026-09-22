"""Read-only E2 presentation modes using frozen data and shared video tools."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from lower_limb_sim.five_leg_mujoco_v1.model import EXPECTED_LEG_IDS
from .mujoco_rehab_video import VideoRenderer,font,sample_indices,CAMERA
from .encoding import open_video_writer
from .e2_overlay import FrozenE2Study,E2Episode,draw_branch_panel,limiting

BG=(12,19,29);TEAL='#65d9d7';WHITE='#e4edf4';MUTED='#9aaec0';GOLD='#efb860'
COLORS=((.22,.78,.80,1),(.46,.68,.93,1),(.78,.60,.92,1),(.92,.66,.31,1),(.86,.47,.55,1))


class Comparison:
    def __init__(self):
        self.study=FrozenE2Study();self.renderers={};self.episodes={}

    def episode(self,leg,beta):
        key=(leg,tuple(beta))
        if key not in self.episodes:
            if leg not in self.renderers:
                r=VideoRenderer(leg);c=COLORS[EXPECTED_LEG_IDS.index(leg)]
                r.palette=((.22,.28,.35,1),c,tuple(.65*v+.25 for v in c[:3])+(1.,))
                self.renderers[leg]=r
                if len(self.renderers)>1:
                    np.testing.assert_array_equal(r.reference.time_s,next(iter(self.renderers.values())).reference.time_s)
                    np.testing.assert_array_equal(r.reference.phases,next(iter(self.renderers.values())).reference.phases)
            self.episodes[key]=E2Episode(self.renderers[leg],beta,self.study)
        return self.episodes[key]

    def text(self,xy,value,size=26,color=WHITE):
        self.draw.text(xy,str(value),font=font(size),fill=color)

    def base(self,title,subtitle):
        self.image=Image.new('RGB',(1920,1080),BG);self.draw=ImageDraw.Draw(self.image)
        self.text((60,25),'FROZEN MECHANICAL BENCHMARK  /  OFFLINE MUJOCO',21,TEAL)
        self.text((60,67),title,45)
        self.text((60,135),subtitle,24,MUTED)
        self.draw.line((60,1000,1860,1000),fill='#2b3b4a',width=2)
        self.text((60,1025),'Each leg normalized to its own beta=(0,0) reference',23,MUTED)

    def scene(self,e,index,box):
        x,y,w,h=box
        self.image.paste(e.renderer.scene(e.candidate,index,w,h),(x,y))

    def explainer(self,e,index):
        self.base('E2 Branch-Balanced Mechanical Endpoint','E0 = aggregate baseline   /   E2 = branch-balanced endpoint')
        self.scene(e,index,(20,245,1120,700))
        self.text((60,205),e.renderer.leg.leg_id,27,TEAL)
        self.text((60,253),f'beta_flex {e.beta[0]:+.3f}    beta_extend {e.beta[1]:+.3f}',25)
        hip,knee=np.rad2deg(e.candidate.trajectory.q[index]);r=e.renderer.reference
        self.text((60,892),f'Hip {hip:.1f}°   Knee {knee:.1f}°   t = {r.time_s[index]:05.2f} s   {str(r.phases[index]).upper()}',27)
        if index==len(r.time_s)-1:
            reference=float(self.study.row(e.renderer.leg.leg_id,(0,0))['E0'])
            self.text((60,947),f'E0: {reference:.6f} → {e.e0:.6f} N·m; branch trade-off remains',25,GOLD)
        else:self.text((60,947),'RMS accumulates over each branch; pending branches stay blank',24,MUTED)
        self.draw.line((1160,200,1160,978),fill='#2b3b4a',width=2)
        draw_branch_panel(self.draw,self.text,e,index,(1220,220,620,700))
        self.text((1220,895),'E2 = max(r_hf, r_he, r_kf, r_ke)',25)
        self.text((1220,945),'No arbitrary weights; dimensionless',22,MUTED)
        return np.asarray(self.image)

    def grid(self,episodes,index):
        beta=episodes[0].beta;r=episodes[0].renderer.reference
        self.base('Five Mechanical Benchmark Legs',f'SAME beta = ({beta[0]:+.3f}, {beta[1]:+.3f})  /  Simulation {r.time_s[index]:05.2f} s  /  {str(r.phases[index]).upper()}')
        boxes=((60,197),(675,197),(1290,197),(365,600),(980,600))
        for e,(x,y) in zip(episodes,boxes):
            self.draw.rounded_rectangle((x-10,y-12,x+565,y+369),radius=12,outline='#324655',width=2)
            self.scene(e,index,(x,y+65,260,245))
            self.text((x,y),e.renderer.leg.leg_id,18,TEAL)
            l=e.renderer.leg
            self.text((x,y+31),f'ROM H [{np.rad2deg(l.hip_min_rad):g}, {np.rad2deg(l.hip_max_rad):g}]°  K [{np.rad2deg(l.knee_min_rad):g}, {np.rad2deg(l.knee_max_rad):g}]°',18,MUTED)
            draw_branch_panel(self.draw,self.text,e,index,(x+276,y+69,265,285),compact=True)
            self.text((x,y+310),f'beta ({beta[0]:+.3f}, {beta[1]:+.3f})',17)
            self.text((x,y+339),str(r.phases[index]).upper(),18,MUTED)
        self.text((60,570),'HF/HE = hip flexion/extension; KF/KE = knee flexion/extension. Bars: 0–2.5; tick=1.',19,MUTED)
        return np.asarray(self.image)

    def oracle_table(self):
        self.base('Common vs Individual Offline Oracle','Frozen 5 × 625 landscape; no online optimizer or patient claim')
        self.text((65,220),'MECHANICAL LEG',24,TEAL);self.text((990,220),'ORACLE BETA',24,TEAL);self.text((1460,220),'ORACLE E2',24,TEAL)
        for k,leg in enumerate(EXPECTED_LEG_IDS):
            row=self.study.oracles[leg];y=286+k*90
            self.text((65,y),leg,29)
            beta=row['oracle_beta'];beta=json.loads(beta) if isinstance(beta,str) else beta
            self.text((990,y),f'({beta[0]:+.4f}, {beta[1]:+.4f})',29)
            self.text((1460,y),f"{row['oracle_value']:.9f}",29)
        b=self.study.common['common_beta']
        self.text((65,804),f'Cohort common beta = ({b[0]:.0f}, {b[1]:.0f})',39,TEAL)
        self.text((65,887),f"Mean common regret = {100*self.study.common['mean_relative_common_regret']:.6f}%",30,GOLD)
        self.text((65,947),'Oracle = offline full-landscape minimum, not an online prediction',23,MUTED)
        return np.asarray(self.image)

    def common_pair(self,left,right,index):
        self.base('Leg 4: Common vs Individual Offline Oracle','Same model, ROM, duration and fixed camera; original V3 sample arrays')
        for x,e,title in ((60,left,'Common trajectory'),(1000,right,'Individual offline oracle')):
            self.scene(e,index,(x,305,850,490))
            self.text((x,213),title,32,TEAL)
            self.text((x,263),f'beta = ({e.beta[0]:+.4f}, {e.beta[1]:+.4f})',26)
            self.text((x,790),f'Frozen full-episode E2 = {e.e2:.9f}',30)
            self.text((x,841),'Limiting: '+limiting(e.normalized),22,MUTED)
            self.text((x,881),'HF / HE / KF / KE: '+ ' / '.join(f'{v:.6f}' for v in e.normalized),18,MUTED)
        row=next(r for r in self.study.common['per_leg'] if r['leg_id']==left.renderer.leg.leg_id)
        self.text((60,945),f"Relative common regret = {100*row['relative_common_regret']:.6f}%",31,GOLD)
        self.text((1240,945),f'Simulation t = {left.renderer.reference.time_s[index]:05.2f} s',25,TEAL)
        return np.asarray(self.image)

    def interpretation(self):
        self.base('What the frozen benchmark supports','Mechanical interaction visibility is not personalization benefit')
        self.text((90,275),'E2 reveals stronger trajectory-dependent mechanical interaction',42)
        self.text((90,380),'BUT',37,GOLD)
        self.text((90,470),'Common trajectory remains near-optimal',51)
        self.text((90,540),'across this five-leg benchmark',45)
        self.text((90,705),'SIMULATED PERSONALIZATION NECESSITY',31,TEAL)
        self.text((90,770),self.study.summary['E2_PERSONALIZATION_NECESSITY'].replace('_',' '),67,GOLD)
        self.text((90,920),f"Mean regret {100*self.study.common['mean_relative_common_regret']:.6f}%  /  No new BO experiments",27,MUTED)
        return np.asarray(self.image)

    def landscapes(self,active):
        self.base('Five Frozen E2 Landscapes','Common color scale across all five legs; values are reference-normalized per leg')
        values=np.array([float(r['E2']) for r in self.study.rows]);lo,hi=values.min(),values.max()
        boxes=((90,212),(710,212),(1330,212),(395,598),(1015,598))
        for k,(leg,(x,y)) in enumerate(zip(EXPECTED_LEG_IDS,boxes)):
            self.text((x-30,y-30),leg,18,TEAL if k==active else MUTED)
            rows=[r for r in self.study.rows if r['leg_id']==leg]
            grid=np.zeros((25,25))
            for row in rows:
                i=round((float(row['beta_flex'])+.03)/.0025);j=round((float(row['beta_extend'])+.03)/.0025)
                grid[24-j,i]=float(row['E2'])
            f=(grid-lo)/(hi-lo)
            rgb=np.zeros((25,25,3));low=np.array([30,65,89]);high=np.array([238,186,97])
            rgb[:]=low+(high-low)*f[:,:,None]
            self.image.paste(Image.fromarray(rgb.astype(np.uint8)).resize((380,280),Image.Resampling.NEAREST),(x,y))
            self.draw.rectangle((x,y,x+380,y+280),outline=TEAL if k==active else '#4b6271',width=3)
            b=self.study.oracles[leg]['oracle_beta'];b=json.loads(b) if isinstance(b,str) else b
            def pos(beta):return x+((beta[0]+.03)/.0025+.5)/25*380,y+((.03-beta[1])/.0025+.5)/25*280
            cx,cy=pos(self.study.common['common_beta']);self.draw.rectangle((cx-8,cy-8,cx+8,cy+8),outline=WHITE,width=2)
            ox,oy=pos(b);self.draw.ellipse((ox-13,oy-13,ox+13,oy+13),outline='#ff6677',width=3)
            self.text((x-5,y+285),'−0.03     beta_flex     +0.03',19,MUTED)
            self.text((x+390,y+5),'+0.03',16,MUTED);self.text((x+390,y+245),'−0.03',16,MUTED)
            self.text((x+390,y+120),'beta',17,MUTED);self.text((x+390,y+145),'extend',17,MUTED)
        self.text((60,956),f'COMMON E2 COLOR SCALE: {lo:.6f} → {hi:.6f}   (dark → light)',23,GOLD)
        self.text((1080,956),'White square: reference/common   Pink ring: oracle',22,MUTED)
        return np.asarray(self.image)

    def close(self):
        for r in self.renderers.values():r.close()


def render_e2(mode,output,leg='LEG_2_KNEE_DOMINANT',beta=(0.,0.)):
    output=Path(output);c=Comparison();writer=None
    try:
        plan=[]
        if mode=='e2-explainer':
            e=c.episode(leg,(-.03,.03));n=len(e.running)
            plan=[('explain',e,int(i)) for i in sample_indices(n,540)]+[('explain',e,n-1)]*90
        elif mode=='five-leg-e2':
            for b in (tuple(beta),(-.03,.03)):
                es=[c.episode(l,b) for l in EXPECTED_LEG_IDS];n=len(es[0].running)
                plan += [('grid',es,int(i)) for i in sample_indices(n,240)]+[('grid',es,n-1)]*90
        elif mode=='e2-common-vs-individual':
            leg=EXPECTED_LEG_IDS[-1];b=c.study.oracles[leg]['oracle_beta'];b=json.loads(b) if isinstance(b,str) else b
            a=c.episode(leg,c.study.common['common_beta']);b=c.episode(leg,b);n=len(a.running)
            plan=[('table',None,0)]*120+[('pair',(a,b),int(i)) for i in sample_indices(n,300)]+[('pair',(a,b),n-1)]*90+[('interpretation',None,0)]*180
        elif mode=='e2-landscapes':plan=[('landscapes',None,i//60) for i in range(300)]
        else:raise ValueError(mode)
        writer=open_video_writer(output)
        previews=output.parent/'previews';previews.mkdir(parents=True,exist_ok=True)
        for frame,(kind,data,index) in enumerate(plan):
            if kind=='explain':rgb=c.explainer(data,index)
            elif kind=='grid':rgb=c.grid(data,index)
            elif kind=='pair':rgb=c.common_pair(*data,index)
            elif kind=='table':rgb=c.oracle_table()
            elif kind=='interpretation':rgb=c.interpretation()
            else:rgb=c.landscapes(index)
            writer.send(rgb)
            if frame in (0,len(plan)//2,len(plan)-1,240):Image.fromarray(rgb).save(previews/f'{output.stem}_{frame:04d}.png')
            if frame%120==0:print(f'{mode}: {frame}/{len(plan)}',flush=True)
        writer.close();writer=None
        result=dict(mode=mode,path=str(output),duration_s=len(plan)/30,frames=len(plan),resolution=[1920,1080],fps=30,
            camera=CAMERA,episodes=[e.record() for e in c.episodes.values()],
            render_source_error_rad={k:r.max_error.tolist() for k,r in c.renderers.items()},
            conclusion=c.study.summary['E2_PERSONALIZATION_NECESSITY'],E2_formula_reimplemented=False,
            frozen_data_modified=False,offscreen=mode!='e2-landscapes',encoder='H.264 yuv420p',
            landscape_E2_range=[min(float(r['E2']) for r in c.study.rows),max(float(r['E2']) for r in c.study.rows)])
        output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return result
    finally:
        if writer is not None:writer.close()
        c.close()
