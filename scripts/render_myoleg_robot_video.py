"""Render a frozen MyoLeg system or actual offline EI demonstration."""
import argparse
from lower_limb_sim.visualization.myoleg_robot_video import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['system-showcase','reference-vs-selected','optimization-demo'])
    p.add_argument('--output',required=True)
    p.add_argument('--run-data',default='outputs/myoleg_bo_demo_run.json')
    p.add_argument('--rerun-optimization',action='store_true')
    a=p.parse_args();render(a.mode,a.output,a.run_data,a.rerun_optimization)

if __name__=='__main__':main()
