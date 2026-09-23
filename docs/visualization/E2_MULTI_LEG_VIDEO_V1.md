# E2 Multi-Leg Mechanical Comparison Video V1

四段视频已完成；数据只读，无新 BO 实验、endpoint 调参或自动 commit。冻结结论：**E2_PERSONALIZATION_NECESSITY = NOT_SUPPORTED**。

## E0、E2 与累计显示语义

E0 为完整周期双关节聚合 RMS（N·m），是合法的 aggregate baseline。E2 是四项同腿参考归一化分支 RMS 的最大值（无量纲），没有任意权重。分量顺序：hip flexion、hip extension、knee flexion、knee extension。分母始终为该腿 beta=(0,0) 完整对应分支 RMS；不能把相对 E2 当作跨腿原始扭矩大小。

E2 是分支/episode 级指标，不存在本视频定义的瞬时 E2(t)。播放时只对已到达样本按既有 phase 标签取前缀，调用共享 `time_rms()`，用当前已积累时间长度计算 RMS，再除以完整参考分支 RMS。尚不足两个样本的分支显示 --/PENDING，绝不补零。相关分支最后一个样本到达后标 FINAL；完整 episode 到达后才发布 FINAL E2。累计估计可高于、低于最终值，也不要求单调。

四条柱共用从零开始的 0–2.5 轴，白色刻度=1.0，无独立伸缩；本次所有累计值均在此范围内。金色标出当前已有分量的最大值，最终 limiting branch 单独标注。参考轨迹最终四项全为 1，标 ALL FOUR (TIE)。共同轨迹/oracle 视频在播放中展示的数值明确标为 Frozen full-episode E2，是既有完整结果，不是瞬时计算。

最终值复用 `lower_limb_sim/mechanical_endpoints.py` 的 `branch_rms_components()` 和 `branch_balanced_reference_normalized_rms()`；可视化层未重写 E2 公式。源扭矩由既有 five-leg `replay_trajectory()` 在原始参数、原始 V3 数组上回放，并逐项核对冻结 CSV；没有重生成或覆盖 landscape。

## Frozen legs / ROM

| Leg ID | Hip ROM (deg) | Knee ROM (deg) |
|---|---|---|
| LEG_0_NOMINAL | 29–112 | 18.5–119.5 |
| LEG_1_HEAVY_HIP_STIFF | 35–105 | 20–110 |
| LEG_2_KNEE_DOMINANT | 22–115 | 32–132 |
| LEG_3_NONLINEAR_COUPLED | 30–118 | 20–128 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | 15–96 | 10–102 |

轨迹直接来自每腿 `make_rom_profile()` 和 `build_leg_domain()`，不改 ROM、不放大位移、不重拟合。各腿原始时间/phase 数组相同；以相同 source index 同步播放。原始 episode 24 秒。五腿仅通过 scene-only 材质区分；固定相机 azimuth=90°、elevation=0°、distance=1.5 m、lookat=(0.3,0,0.19) m；网格视口大小、比例和相机完全相同。

## Outputs

所有 MP4：1920×1080、30 fps、H.264/yuv420p；共享 `encoding.py`，无录屏。前三段为 MuJoCo offscreen RGB，地形图是直接由冻结 CSV 绘制的离屏二维图。

| File | Duration / frames | Legs / beta | Endpoint | q error hip/knee |
|---|---|---|---|---|
| outputs/e2_mechanical_endpoint_explainer.mp4 | 21 s / 630 | Leg 2; (-0.03,+0.03) | 1.007024911904 | 0 / 0 rad |
| outputs/five_leg_e2_comparison.mp4 | 22 s / 660 | Legs 0–4; (0,0) then (-0.03,+0.03) | 0.9992936589, 1.0, 1.000254273748, 1.000728196989, 1.000745412256, 1.007024911904 | 0 / 0 rad |
| outputs/e2_common_vs_individual.mp4 | 23 s / 690 | All-five oracle table; Leg 4 replay (0,0) vs (-0.03,+0.0225) | 0.9992936589, 1.0 | 0 / 0 rad |
| outputs/five_leg_e2_landscapes.mp4 | 10 s / 300 | Legs 0–4; all 625 beta pairs per leg | common scale [0.9992936589002313, 1.0159773057335362] | N/A (no trajectory animation) |

讲解版选 Leg 2 的 (-0.03,+0.03)，理由是清楚呈现聚合与分支 trade-off：E0 从 175.786283367 降至 173.915654314 N·m，但 E2=1.007024912，限制分支为 hip extension；候选选择不依赖制造 oracle 多样性。该视频最后保留 3 秒完整结果。五腿版两段各 8 秒回放+3 秒完整结果。共同/oracle 版包含 4 秒列表、10 秒同步回放、3 秒定格、6 秒解释屏。

## Values shown

每腿 reference beta=(0,0)：四项归一化分量均为 1，E2=1。下表为共同 contrast (-0.03,+0.03)：

| Leg | r_hf | r_he | r_kf | r_ke | E2 |
|---|---:|---:|---:|---:|---:|
| LEG_0_NOMINAL | 1.000259092093 | 1.000745412256 | 0.990915924203 | 0.985254648376 | 1.000745412256 |
| LEG_1_HEAVY_HIP_STIFF | 1.000131384365 | 1.000254273748 | 0.989846279745 | 0.986990076084 | 1.000254273748 |
| LEG_2_KNEE_DOMINANT | 1.004043516014 | 1.007024911904 | 0.990123373804 | 0.987130299451 | 1.007024911904 |
| LEG_3_NONLINEAR_COUPLED | 1.000539173149 | 1.000728196989 | 0.990034317894 | 0.987922167037 | 1.000728196989 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | 0.999293658900 | 0.999027935967 | 0.993168278665 | 0.990111549441 | 0.999293658900 |

## Common and individual offline oracle

所有数值来自 `lower_limb_sim/five_leg_mujoco_v1/results_e2_necessity_v1/` 的 `study_summary.json`、`e2_full_landscapes.csv` 与 summary tables，不硬编码 regret。

| Leg | Oracle beta | Oracle E2 | Common E2 | Relative common regret (%) |
|---|---|---:|---:|---:|
| LEG_0_NOMINAL | [0.0, 0.0] | 1.000000000000 | 1.000000000000 | 0.000000000 |
| LEG_1_HEAVY_HIP_STIFF | [0.0, 0.0] | 1.000000000000 | 1.000000000000 | 0.000000000 |
| LEG_2_KNEE_DOMINANT | [0.0, 0.0] | 1.000000000000 | 1.000000000000 | 0.000000000 |
| LEG_3_NONLINEAR_COUPLED | [0.0, 0.0] | 1.000000000000 | 1.000000000000 | 0.000000000 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | [-0.03, 0.0225] | 0.999293658900 | 1.000000000000 | 0.070684037 |

Cohort common beta=[0.0, 0.0]; mean regret=0.014136807%。Leg 4 oracle 分量 (HF, HE, KF, KE) = 0.999293658900, 0.999271498672, 0.993168278665, 0.992524971354。

地形图展示五个原始 25×25 E2 网格，x=beta_flex，y=beta_extend。全图统一色标 [0.9992936589002313, 1.0159773057335362]；没有 per-leg autoscaling。白方框=reference/common，粉色圆环=该冻结表选定 oracle；重合标记同时保留。边界点标记位于相应网格单元中心。

冻结 normalized interaction energy：E0=0.359991272%、E2=5.228441639%；E2 minimum pairwise Spearman=0.8146590115287。228/625 候选对所有腿均处于 0.5% oracle 邻域。这些描述可见 interaction，不等同于个性化收益。共同轨迹仍近最优，最终解释屏明确呈现 NOT SUPPORTED；`READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON=false`。未制作 E2 BO 优化视频。

## Verification

5 项针对性测试通过，包括已有的全部 5×625 frozen E2 helper 一致性覆盖、新增分支前缀/pending/final 语义、同源时间与零角度误差、只改 scene 色彩、不改变模型质量/材质数组、原始 oracle 与结论。所有已动画展示的 leg/candidate，其渲染 q 与原 V3 q 最大差值均为 0 rad；最终 E2 与冻结行差值均为 0。首/中/末预览保存在 `outputs/previews/`，同名 JSON 记录每段视频的腿、beta、四分量、E0/E2、误差与相机。

## Exact commands

在仓库根目录执行。使用当前已配置的临时 Python 环境（MuJoCo 3.6.0、numpy/scipy、Pillow、imageio-ffmpeg）。macOS 需允许 CoreGraphics 离屏上下文；未为本任务修改项目依赖文件。临时环境移除后须用具有同样依赖的 Python 路径替换。

```sh
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_mujoco_rehab_video --mode e2-explainer --leg LEG_2_KNEE_DOMINANT --output outputs/e2_mechanical_endpoint_explainer.mp4
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_mujoco_rehab_video --mode five-leg-e2 --beta-flex 0 --beta-extend 0 --output outputs/five_leg_e2_comparison.mp4
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_mujoco_rehab_video --mode e2-common-vs-individual --output outputs/e2_common_vs_individual.mp4
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_mujoco_rehab_video --mode e2-landscapes --output outputs/five_leg_e2_landscapes.mp4
```

E2_FORMULA_REIMPLEMENTED=false; FROZEN_E2_DATA_MODIFIED=false; FIVE_LEG_PARAMETERS_MODIFIED=false; V3_MODIFIED=false; ROM_MODIFIED=false; PERSONALIZATION_CLAIM_ADDED=false.

## Final MP4 decode checks

- `outputs/e2_mechanical_endpoint_explainer.mp4`: 630 frames, 30 fps, 1619606 bytes, zero black frames, 594 changed frame pairs.
- `outputs/five_leg_e2_comparison.mp4`: 660 frames, 30 fps, 3574643 bytes, zero black frames, 600 changed frame pairs.
- `outputs/e2_common_vs_individual.mp4`: 690 frames, 30 fps, 976925 bytes, zero black frames, 395 changed frame pairs.
- `outputs/five_leg_e2_landscapes.mp4`: 300 frames, 30 fps, 275596 bytes, zero black frames, 168 changed frame pairs.

All four outputs passed codec/resolution/frame-count checks and visual preview inspection. Landscape holds are intentional; active panel highlighting changes sequentially.
