"""CLI for frozen-model MuJoCo demonstration videos."""
import argparse
from lower_limb_sim.visualization.mujoco_rehab_video import render_video


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['resume-showcase','side-by-side','single','e2-explainer','five-leg-e2','e2-common-vs-individual','e2-landscapes'],required=True)
    p.add_argument('--leg',default=None)
    p.add_argument('--beta-flex',type=float,default=0.)
    p.add_argument('--beta-extend',type=float,default=0.)
    p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    a=p.parse_args()
    if a.mode.startswith('e2-') or a.mode=='five-leg-e2':
        from lower_limb_sim.visualization.five_leg_comparison import render_e2
        if (a.width,a.height)!=(1920,1080):p.error('E2 layouts require 1920x1080')
        render_e2(a.mode,a.output,a.leg or 'LEG_2_KNEE_DOMINANT',(a.beta_flex,a.beta_extend))
    else:
        render_video(a.mode,a.output,a.leg or 'LEG_0_NOMINAL',(a.beta_flex,a.beta_extend),a.width,a.height)

if __name__=='__main__': main()
