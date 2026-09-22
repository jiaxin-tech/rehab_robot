"""Evaluator-only plots and report from completed frozen runs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/rehab_benchmark_mpl')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import OUT

METHODS=['RANDOM','ADAPTIVE_GREEDY','PURE_BO_EI','MODEL_INFORMED_BO_EI']
LABELS={'RANDOM':'Random (20 seeds)','ADAPTIVE_GREEDY':'Adaptive Greedy','PURE_BO_EI':'Pure BO EI','MODEL_INFORMED_BO_EI':'Model-Informed BO EI','MODEL_INFORMED_BO_LCB':'Model-Informed BO LCB','REFERENCE':'Reference (one trial)'}
COLORS=dict(zip(METHODS,['#8b8b8b','#dd8a22','#4271b7','#278474']))
STYLES=['--',':','-.','-']


def main(output=OUT):
    output=Path(output);d=pd.read_csv(output/'main_results.csv');h=pd.read_csv(output/'trial_history.csv');q=pd.read_csv(output/'prior_quality.csv')
    head=pd.read_csv(output/'headroom.csv');legs=list(head.LEG_ID.unique())
    figdir=output/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
    # Random is averaged within a leg before averaging legs; equal leg weights.
    legmeans=d.groupby(['endpoint','LEG_ID','METHOD','K_additional'],as_index=False).final_regret.mean()
    summaries=legmeans.groupby(['endpoint','METHOD','K_additional']).final_regret.agg(['mean','median']).reset_index()
    summaries.to_csv(output/'cross_leg_summary.csv',index=False)
    random=d[d.METHOD=='RANDOM'].groupby(['endpoint','LEG_ID','K_additional']).final_regret.agg(['mean','median','std','min','max',lambda x:x.quantile(.05),lambda x:x.quantile(.95)]).reset_index()
    random.columns=['endpoint','LEG_ID','K_additional','mean','median','std','min','max','p05','p95'];random.to_csv(output/'random_summary.csv',index=False)
    for endpoint in ('E2','E0'):
        data=d[d.endpoint==endpoint];lm=legmeans[legmeans.endpoint==endpoint]
        prefix=endpoint.lower();unit='dimensionless E2 regret' if endpoint=='E2' else 'E0 regret (N·m)'
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for method,style in zip(METHODS,STYLES):
            a=summaries[(summaries.endpoint==endpoint)&(summaries.METHOD==method)]
            for ax,stat in zip(axes,['mean','median']):ax.plot(a.K_additional,a[stat],style,marker='o',label=LABELS[method],color=COLORS[method],linewidth=2)
        for ax,title in zip(axes,['Mean across equally weighted legs','Median across legs (Random: seed mean per leg)']):
            ax.set(xlabel='Additional full-trajectory trials',ylabel=unit,title=title,xticks=range(4));ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax.set_ylim(bottom=0)
        if endpoint=='E2':
            for ax in axes:ax.set_ylim(0,1.1*float(summaries[(summaries.endpoint==endpoint)]['mean'].max()))
        if endpoint=='E2':axes[1].text(.05,.78,'4/5 legs: reference already oracle\nMedian = 0 is not algorithm equivalence',transform=axes[1].transAxes)
        axes[0].legend(fontsize=8);fig.suptitle(endpoint+(' PRIMARY' if endpoint=='E2' else ' SENSITIVITY ONLY'));fig.tight_layout();fig.savefig(figdir/f'{prefix}_figure1_best_so_far.png');plt.close(fig)
        fig,axes=plt.subplots(1,5,figsize=(17,4),sharex=True)
        for k,(leg,ax) in enumerate(zip(legs,axes)):
            for method,style in zip(METHODS,STYLES):
                a=lm[(lm.LEG_ID==leg)&(lm.METHOD==method)]
                ax.plot(a.K_additional,a.final_regret,style,marker='o',color=COLORS[method],label=LABELS[method])
            ax.set(title=f'Leg {k}',xlabel='Additional trials',xticks=range(4));ax.set_ylim(bottom=0);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
            if endpoint=='E2' and k<4:ax.text(.1,.8,'REFERENCE\nALREADY ORACLE',transform=ax.transAxes,fontsize=9)
        axes[0].set_ylabel(unit);axes[-1].legend(fontsize=7);fig.suptitle(endpoint+' per-leg best-so-far regret');fig.tight_layout();fig.savefig(figdir/f'{prefix}_figure1b_per_leg.png');plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(5);bw=.18
        for j,method in enumerate(METHODS):
            vals=[lm[(lm.LEG_ID==l)&(lm.METHOD==method)&(lm.K_additional==3)].final_regret.iloc[0] for l in legs]
            ax.bar(x+(j-1.5)*bw,vals,bw,label=LABELS[method],color=COLORS[method])
        ax.set(xticks=x,xticklabels=[f'Leg {i}' for i in range(5)],ylabel=unit,title=endpoint+' final measured-best regret | 3 additional trials');ax.legend(fontsize=8)
        if endpoint=='E2':ax.text(.03,.84,'Legs 0–3: no optimization headroom',transform=ax.transAxes)
        fig.tight_layout();fig.savefig(figdir/f'{prefix}_figure2_final_regret.png');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,k in zip(axes,[1,3]):
            for i,leg in enumerate(legs):
                def regret(method):return data[(data.LEG_ID==leg)&(data.METHOD==method)&(data.K_additional==k)].final_regret.iloc[0]
                prior=q[(q.endpoint==endpoint)&(q.leg_id==leg)].iloc[0]
                benefit=regret('PURE_BO_EI')-regret('MODEL_INFORMED_BO_EI')
                ax.scatter(prior.spearman,benefit,s=55);ax.annotate(f'Leg {i}',(prior.spearman,benefit),xytext=(-5 if prior.spearman>.99 else 3,8 if benefit>0 else 8+i*10),ha='right' if prior.spearman>.99 else 'left',textcoords='offset points',fontsize=9)
            ax.axhline(0,color='gray',linewidth=1);ax.set(xlabel='Reference-only prior Spearman',ylabel='Pure EI regret − Model-Informed EI regret',title=f'{k} additional trials; positive = prior helps')
        if endpoint=='E2':
            for ax in axes:ax.set_ylim(-.00006,.0009)
        fig.suptitle(endpoint+' prior quality vs measured-best benefit (evaluator only)');fig.tight_layout();fig.savefig(figdir/f'{prefix}_figure3_prior_quality.png');plt.close(fig)
        fig,axes=plt.subplots(1,5,figsize=(18,4.5))
        for i,(leg,ax) in enumerate(zip(legs,axes)):
            for method,style in zip(METHODS,STYLES):
                a=h[(h.endpoint==endpoint)&(h.leg_id==leg)&(h.method==method)&(h.seed==0)].sort_values('trial_index')
                ax.plot(a.beta_flex,a.beta_extend,style,marker='o',label=LABELS[method],color=COLORS[method],alpha=.8)
                for row in a.itertuples():ax.annotate(str(row.trial_index),(row.beta_flex,row.beta_extend),fontsize=7,color=COLORS[method],xytext=(3,3),textcoords='offset points')
            ax.scatter([0],[0],marker='s',facecolor='none',edgecolor='black',s=90)
            ax.set(xlim=(-.035,.035),ylim=(-.035,.035),xlabel='beta_flex',title=f'Leg {i}',xticks=[-.03,0,.03],yticks=[-.03,0,.03]);ax.set_aspect('equal')
        axes[0].set_ylabel('beta_extend');axes[-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));fig.suptitle(endpoint+' executed beta paths; trial 0=reference; Random seed 0 illustration only');fig.tight_layout();fig.savefig(figdir/f'{prefix}_figure4_beta_paths.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,endpoint in zip(axes,['E2','E0']):
        for method,style in [('MODEL_INFORMED_BO_EI','-'),('MODEL_INFORMED_BO_LCB','--')]:
            a=summaries[(summaries.endpoint==endpoint)&(summaries.METHOD==method)]
            ax.plot(a.K_additional,a['mean'],style,marker='o',label=LABELS[method])
        ax.set(title=endpoint,xlabel='Additional trials',ylabel='Mean regret',xticks=range(4));ax.legend(fontsize=8);ax.set_ylim(bottom=0)
    fig.suptitle('Acquisition sensitivity only: xi=0, kappa=1.5, no tuning');fig.tight_layout();fig.savefig(figdir/'figure5_ei_vs_lcb.png');plt.close(fig)
    # Headroom-normalized primary results are meaningful for Leg 4 only.
    normalized=d[(d.endpoint=='E2')&d.normalized_regret.notna()].groupby(['METHOD','K_additional']).normalized_regret.agg(['mean','median']).reset_index()
    normalized.to_csv(output/'e2_positive_headroom_normalized_summary.csv',index=False)
    print('Generated 11 figures and aggregate tables',flush=True)

if __name__=='__main__':main()
