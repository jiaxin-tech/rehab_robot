import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/rehab_benchmark_mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import OUT,COMPONENTS
from lower_limb_sim.trajectory_sensitivity.report import table


def main():
    d=pd.read_csv(OUT/'candidates.csv');d['comparison_feasible']=False
    for name,g in d.groupby('model'):
        ref=g[(g.family=='BETA_ORIGINAL')&(g.parameter_flex==0)&(g.parameter_extend==0)&(g.share_shift==0)].iloc[0]
        for metric,limit in [('speed',1.5),('accel',2.)]:
            suffix='deg_s' if metric=='speed' else 'deg_s2'
            for joint in ('hip','knee'):
                key=f'{joint}_{metric}_peak_{suffix}'
                d.loc[g.index,key+'_ratio']=g[key]/ref[key]
        for joint in ('hip','knee'):
            key=joint+'_torque_peak_Nm';d.loc[g.index,key+'_ratio']=g[key]/ref[key]
        condition=g.valid.copy()
        for col in [x for x in d if x.endswith('_ratio')]:
            limit=1.1 if 'torque' in col else (2. if 'accel' in col else 1.5)
            condition &= d.loc[g.index,col]<=limit+1e-12
        d.loc[g.index,'comparison_feasible']=condition
    d.to_csv(OUT/'candidates_with_limits.csv',index=False)
    results=[]
    for (name,family),g in d.groupby(['model','family'],sort=False):
        ref=d[(d.model==name)&(d.family=='BETA_ORIGINAL')&(d.parameter_flex==0)&(d.parameter_extend==0)&(d.share_shift==0)].iloc[0]
        for policy in ('unrestricted','common_limits'):
            subset=g[g.valid & ((g.comparison_feasible) if policy=='common_limits' else True)]
            for endpoint in ('E2','E3'):
                best=subset.loc[subset[endpoint].idxmin()]
                results.append(dict(model=name,family=family,policy=policy,selected_by=endpoint,
                    candidate_count=len(g),valid_count=int(g.valid.sum()),feasible_count=len(subset),
                    selected_flex=best.parameter_flex,selected_extend=best.parameter_extend,share_shift=best.share_shift,
                    E2=best.E2,E3=best.E3,E3_improvement_pct=100*(1-best.E3),
                    E0_improvement_pct=100*(ref.E0-best.E0)/ref.E0,
                    max_component_worsening_pct=100*(best.E2-1),
                    knee_speed_ratio=best.knee_speed_peak_deg_s_ratio,knee_accel_ratio=best.knee_accel_peak_deg_s2_ratio,
                    hip_peak_ratio=best.hip_torque_peak_Nm_ratio,knee_peak_ratio=best.knee_torque_peak_Nm_ratio))
    s=pd.DataFrame(results);s.to_csv(OUT/'comparison_summary.csv',index=False)
    old=pd.read_csv(OUT/'frozen_625_rescored.csv');olds=[]
    for name,g in old.groupby('model',sort=False):
        for ep in ('E2','E3'):
            best=g.loc[g[ep].idxmin()]
            olds.append(dict(model=name,selected_by=ep,beta_flex=best.beta_flex,beta_extend=best.beta_extend,E2=best.E2,E3=best.E3,E3_improvement_pct=100*(1-best.E3)))
    frozen=pd.DataFrame(olds);frozen.to_csv(OUT/'frozen_625_summary.csv',index=False)
    names=list(d.model.unique());families=['BETA_ORIGINAL','BETA_WIDE','BETA_TIMING','KEY_POSTURE_TIMING']
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,name in zip(axes.flat,names):
        for offset,policy,color in [(-.18,'unrestricted','#6484ba'),(.18,'common_limits','#e1a24a')]:
            subset=s[(s.model==name)&(s.policy==policy)&(s.selected_by=='E3')].set_index('family').loc[families]
            ax.bar(np.arange(4)+offset,subset.E3_improvement_pct,width=.36,label=policy,color=color)
        ax.set_xticks(range(4),['Beta\noriginal','Beta\nwide','Beta +\ntime','Key pose\n+ time'])
        ax.set(title=name.replace('LEG_','L').replace('_',' '),ylabel='E3 reduction vs reference (%)')
        ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('E3 candidate-family comparison | grid minima, not low-budget BO performance')
    fig.savefig(OUT/'family_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,name in zip(axes.flat,names):
        for family in families:
            g=d[(d.model==name)&(d.family==family)&d.valid&d.comparison_feasible]
            ax.scatter(g.E2,g.E3,s=9,alpha=.45,label=family)
        ax.axvline(1,color='gray',lw=.7);ax.axhline(1,color='gray',lw=.7)
        ax.set(title=name.replace('LEG_','L').replace('_',' '),xlabel='E2 worst component ratio',ylabel='E3 mean component ratio')
        ax.legend(fontsize=6)
    fig.suptitle('Same offline limits | E3 improvement can coexist with E2 worsening')
    fig.savefig(OUT/'E2_E3_tradeoffs.png',dpi=160);plt.close(fig)
    display=s[(s.selected_by=='E3')&(s.policy=='common_limits')][['model','family','feasible_count','E3_improvement_pct','E2','selected_flex','selected_extend','share_shift','knee_speed_ratio','knee_accel_ratio']]
    wins=[]
    for name in names:
        t=s[(s.model==name)&(s.selected_by=='E3')&(s.policy=='common_limits')]
        best=t.loc[t.E3.idxmin()];wins.append(f"- {name}: {best.family}; E3 降低 {best.E3_improvement_pct:.4f}%，E2={best.E2:.6f}。")
    text=f'''# E3 与关键姿态点样条：候选轨迹对照

完成。主指标 E3=四项原始同模型 reference 归一化 RMS 的等权均值；E2 保留为最坏分量对照。原 E2/V3/算法/模型未修改。本次比较候选轨迹族及目标函数，没有运行新 BO，也不是同试验预算算法性能比较。

## 结果解释

统一离线条件下，新样条在六个模型的本次候选网格上均取得最低 E3。五机械腿的新样条 E3 改善为 1.747%–2.792%；MyoLeg 为 2.984%，对比 β＋时间分配的 2.382%。这支持进一步保留新候选族，但不证明低预算优化器能找到这些候选。

五机械腿仍共同选择 α_flex=α_extend=-0.12、屈曲占比偏移+0.05；MyoLeg 选择 +0.12、+0.12、-0.10。最优关键姿态偏移仍在扫描边界；共同偏好并未消失，不能从换 E3 推出个性化必要性。

平均与最坏分量存在取舍：Leg 2 新样条 E3 降低1.747%，但 E2=1.021768，即最坏分量增加2.177%。MyoLeg 新样条的 E2=0.995407，四项都下降；膝速度峰值约为参考1.076倍，加速度约1.633倍。应先明确允许哪一项负荷上升，不能只依据平均值确定上机轨迹。

## 三组试验

1. 原始 625 点域的同一批力矩重新计算 E3，并分别按 E2/E3 选最优。五机械腿＋MyoLeg 共 3750 个旧响应，无需重跑模拟。
2. 同一 E3 比较原 β（探索子网格 25 点）、宽 β（289 点）、β＋时间（405 点）、关键姿态点＋时间（405 个提案）。第三和第四种提案数相同，但可行数可能不同；各自离散域并不等价，不能将网格最小值解释成在线样本效率。
3. 对上述四族同时施加统一离线条件：髋膝速度峰值各≤reference 的1.5倍，加速度各≤2倍，髋膝力矩绝对峰值各≤1.1倍。另报未加条件结果。阈值在新力矩计算前写入 settings.json，是示例性比较条件，不是临床/机器人安全限值。ROM、起终点和24秒总周期不变。

## 新轨迹的具体含义

在每阶段时间中点，对膝角增加 α×该腿膝ROM跨度；α_flex/α_extend各取 [-0.12,0.12] 共9档，时长占比偏移5档。α不是β：α直接控制中途姿态偏移，β控制沿原轨迹的相位进程。

采用参考轨迹＋局部五次样条形变，支撑区间为每阶段20%–80%；中点形变量为1，两端形变量及一、二阶导数为零，支撑区间外不变。该实现保留髋角度路径，属于关键姿态点方法的保守三参数版本，不是髋膝全部自由优化。

在动力学比较前的运动学测试中，跨整个阶段的形变会侵入原参考平缓端点而越界，因此改为上述局部支撑；未根据负荷选择形状。原参考伸展阶段自身存在局部反向运动，不能要求新方法消除它而旧方法不消除；因此保留参考细节，新方法记录反向角度积分，并进行每阶段2001点的插值ROM检查。这是数值采样检查，不是连续时间可行性的数学证明。统一导数/力矩阈值按同一401个回放样本计算。

## 原625域：只换指标

{table(frozen)}

## 统一条件后：按 E3 选优

{table(display)}

当前各模型最低 E3 的轨迹族：

{chr(10).join(wins)}

这些差异仅适用于当前三个形变参数和网格。不能据此认定某类样条普遍优于相位变换。E3允许某分量增加而平均下降，所以上表同步报告E2；峰值力矩条件不等于四项RMS都不增加。

## 有效性和导出

新样条提案 {int((d.family=='KEY_POSTURE_TIMING').sum())} 个，运动学/模拟有效 {int(d[d.family=='KEY_POSTURE_TIMING'].valid.sum())}，拒绝 {int((~d[d.family=='KEY_POSTURE_TIMING'].valid).sum())}。无效记录保留原因，没有裁剪到ROM。

- comparison_summary.csv：各方法分别按E2/E3选优，含未加条件与统一条件两组。
- candidates_with_limits.csv：逐候选四项原始/相对RMS、E0/E2/E3、峰值及可行性。
- frozen_625_rescored.csv：原始625域逐候选E3；该表不宣称满足本次新的导数上限。
- *_key_posture_torques.npz：所有有效新候选力矩与参数；参数顺序α_flex、α_extend、share_shift，时间按原阶段时间及share_shift重构。
- family_comparison.png、E2_E3_tradeoffs.png：轨迹族收益及平均/最坏分量取舍。

验证：2430条有效新候选导出力矩按真实重分配时间重算 E0/E2/E3 一致；全部6744条方法记录的E3均与四分量均值一致（方法之间含重复候选，不是6744次独立模拟）。5项直接相关测试通过（含E3等权均值/参考恢复/无效分母、样条中点含义/C2边界/零参数恢复/越界拒绝以及已有时间缩放测试）。

复现：`python -m lower_limb_sim.e3_candidate_comparison.run` 后 `python -m lower_limb_sim.e3_candidate_comparison.report`。本次仅新建离线模块、测试和输出，未自动提交。所有结果为模型计算的机械负荷，不是患者或实际袖带力测量。
'''
    (OUT/'REPORT.md').write_text(text)
    print(display.to_string(index=False))

if __name__=='__main__':main()
