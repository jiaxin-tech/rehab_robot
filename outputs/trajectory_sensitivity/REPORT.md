# 扩大 β 与屈伸时间分配：离线探索结果

已完成两步。共 6 个模型 × 613 个独立参数组合 = 3678 条完整轨迹；有效 3678，拒绝 0。五条冻结机械腿保持原参数/ROM；第六个模型是既有 MyoLeg native-ROM、零控制和零激活条件，不代表第六个真实受试者。

## 主要结论

1. 扩大 β 确实扩大了力矩差异，但没有消除共同偏好。±0.12 下，五条机械腿 E0 全部偏好 (-0.12,+0.12)，仍在边界；MyoLeg 偏好相反的 (+0.12,-0.12)，也在边界。
2. 单独扩大 β，Leg 0–3 的 E2 仍以 reference 为最优；Leg 4 改善约 0.330%，MyoLeg 约 0.402%。所以原范围限制了部分收益，但不是四腿 E2 平局的充分解释。
3. 加入时间分配，E0 下 Leg 0 / MyoLeg 偏好屈曲占比 66.67%，Leg 1–4 偏好 46.67%；出现不同时间分配偏好，但这些网格最优仍都位于时间扫描边界。不能据此宣布已找到连续域最优或证明个性化必要性。
4. MyoLeg 的 E0 改善从旧 ±0.03 下约 0.218%，增至扩大 β 后 0.896%，再增至联合时间分配后 2.125%；对应 E2 改善约 0.112%、0.402%、0.444%。联合最优为 (+0.12,-0.12)，屈曲 16 s / 伸展 8 s，相比原 13.6 s / 10.4 s。
5. 联合最优 MyoLeg 的膝速度峰值为参考的 1.612 倍、膝加速度峰值为 2.756 倍。负荷指标收益并非没有运动代价，不能将这个边界候选直接作为机器人执行方案。

建议保留三维参数化作为后续离线候选，但先明确可接受的速度/加速度和训练节奏，再确定最终搜索范围。当前结果不支持只靠不断扩大 β 来解决问题，也不支持为了产生不同最优而改腿参数。本次止于两步扫描，不自动进入新 BO 实验。

## 参数及比较设计

第一步：β_flex、β_extend 同时扫描 [-0.12,0.12]，间隔 0.015，共 289 点；读取嵌套 ±0.03、±0.06、±0.09、±0.12 子域（25、81、169、289 点）。这是探索网格；原 ±0.03 的 625 点冻结网格没有修改。较宽子域不会删除较窄子域已有候选。

第二步：保留独立的屈曲/伸展协调两个 β，加一个屈曲时间占比参数，共三维。β 间隔 0.03（9×9）；时间占比相对原比例增加 -0.10、-0.05、0、0.05、0.10（5 档），共 405 点。两步共用 81 点，每模型实际只算 613 点。单独比较协调、时间分配、二者联合；没有运行或修改 BO。

原屈曲占比为 0.566667；扫描范围 0.466667–0.666667。总周期固定 24 s；上下各 10 个百分点相当于屈曲时间上下各 2.4 s，伸展时间作相反变化。这是探索范围，非人体许可范围。

β 的意义仍是膝相对固定髋/参考进程的提前/滞后。第一步不变髋轨迹。第二步在保留同一角度路径的前提下重分配两个阶段时间，因此髋和膝相对于物理时间均改变。不是修改肌肉参数，也不是扩大 ROM。

时标变换采用每阶段常数伸缩 a：q_new=q_old，dq_new=dq_old/a，ddq_new=ddq_old/a²。阶段交界原速度、加速度为零，故保持 C2 连续。每候选用实际新时间积分，不能把非均匀采样当作均匀采样。

## 指标

E0 为全周期双关节力矩 RMS（N m）；E2 直接调用已有共享实现，取四个分支/关节 RMS 相对原始同模型 reference 的最大值。所有时间分配候选都用原 (0,0,0) reference 作分母，不为各时间分配重新归一化。

所有最优值均是本次离散网格最小值，非连续域或人体最优。相同参数不等于相同实际角度，因为五条机械腿的 ROM 不同。E0 和 E2 分开解释，不据算法优劣更换 endpoint。

## E0 结果

| model | step | beta_radius | reference | minimum | improvement_pct | beta_flex | beta_extend | share_shift | beta_boundary | share_boundary |
|---|---|---|---|---|---|---|---|---|---|---|
| LEG_0_NOMINAL | step1 | 0.03 | 32.2138 | 32.1811 | 0.101533 | -0.03 | 0.03 | 0 | True | False |
| LEG_0_NOMINAL | step1 | 0.12 | 32.2138 | 32.0707 | 0.444294 | -0.12 | 0.12 | 0 | True | False |
| LEG_0_NOMINAL | step2 | 0.12 | 32.2138 | 31.9222 | 0.905283 | -0.12 | 0.12 | 0.1 | True | True |
| LEG_1_HEAVY_HIP_STIFF | step1 | 0.03 | 98.5221 | 98.5049 | 0.0174378 | -0.03 | 0.03 | 0 | True | False |
| LEG_1_HEAVY_HIP_STIFF | step1 | 0.12 | 98.5221 | 98.4592 | 0.0637617 | -0.12 | 0.12 | 0 | True | False |
| LEG_1_HEAVY_HIP_STIFF | step2 | 0.12 | 98.5221 | 97.6178 | 0.917807 | -0.12 | 0.12 | -0.1 | True | True |
| LEG_2_KNEE_DOMINANT | step1 | 0.03 | 175.786 | 173.916 | 1.06415 | -0.03 | 0.03 | 0 | True | False |
| LEG_2_KNEE_DOMINANT | step1 | 0.12 | 175.786 | 168.705 | 4.02857 | -0.12 | 0.12 | 0 | True | False |
| LEG_2_KNEE_DOMINANT | step2 | 0.12 | 175.786 | 168.458 | 4.1687 | -0.12 | 0.12 | -0.1 | True | True |
| LEG_3_NONLINEAR_COUPLED | step1 | 0.03 | 319.669 | 317.397 | 0.710812 | -0.03 | 0.03 | 0 | True | False |
| LEG_3_NONLINEAR_COUPLED | step1 | 0.12 | 319.669 | 311.312 | 2.61446 | -0.12 | 0.12 | 0 | True | False |
| LEG_3_NONLINEAR_COUPLED | step2 | 0.12 | 319.669 | 310.263 | 2.94247 | -0.12 | 0.12 | -0.1 | True | True |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | step1 | 0.03 | 164.068 | 163.681 | 0.23538 | -0.03 | 0.03 | 0 | True | False |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | step1 | 0.12 | 164.068 | 162.498 | 0.956797 | -0.12 | 0.12 | 0 | True | False |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | step2 | 0.12 | 164.068 | 161.173 | 1.76434 | -0.12 | 0.12 | -0.1 | True | True |
| MYOLEG_NATIVE_P0 | step1 | 0.03 | 37.6419 | 37.5599 | 0.21803 | 0.03 | -0.03 | 0 | True | False |
| MYOLEG_NATIVE_P0 | step1 | 0.12 | 37.6419 | 37.3045 | 0.896304 | 0.12 | -0.12 | 0 | True | False |
| MYOLEG_NATIVE_P0 | step2 | 0.12 | 37.6419 | 36.8419 | 2.12544 | 0.12 | -0.12 | 0.1 | True | True |

## E2 结果

| model | step | beta_radius | reference | minimum | improvement_pct | beta_flex | beta_extend | share_shift | beta_boundary | share_boundary |
|---|---|---|---|---|---|---|---|---|---|---|
| LEG_0_NOMINAL | step1 | 0.03 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_0_NOMINAL | step1 | 0.12 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_0_NOMINAL | step2 | 0.12 | 1 | 0.998792 | 0.120809 | 0 | 0 | 0.1 | False | True |
| LEG_1_HEAVY_HIP_STIFF | step1 | 0.03 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_1_HEAVY_HIP_STIFF | step1 | 0.12 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_1_HEAVY_HIP_STIFF | step2 | 0.12 | 1 | 0.999583 | 0.041732 | 0 | 0.03 | 0.1 | False | True |
| LEG_2_KNEE_DOMINANT | step1 | 0.03 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_2_KNEE_DOMINANT | step1 | 0.12 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_2_KNEE_DOMINANT | step2 | 0.12 | 1 | 0.999741 | 0.0259285 | 0 | 0 | 0.1 | False | True |
| LEG_3_NONLINEAR_COUPLED | step1 | 0.03 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_3_NONLINEAR_COUPLED | step1 | 0.12 | 1 | 1 | 0 | 0 | 0 | 0 | False | False |
| LEG_3_NONLINEAR_COUPLED | step2 | 0.12 | 1 | 0.999941 | 0.00586631 | 0 | 0 | 0.1 | False | True |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | step1 | 0.03 | 1 | 0.999294 | 0.0706341 | -0.03 | 0.03 | 0 | True | False |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | step1 | 0.12 | 1 | 0.9967 | 0.329998 | -0.12 | 0.105 | 0 | True | False |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | step2 | 0.12 | 1 | 0.996691 | 0.330895 | -0.12 | 0.12 | 0.1 | True | True |
| MYOLEG_NATIVE_P0 | step1 | 0.03 | 1 | 0.99888 | 0.112008 | 0.03 | -0.03 | 0 | True | False |
| MYOLEG_NATIVE_P0 | step1 | 0.12 | 1 | 0.99598 | 0.402044 | 0.12 | -0.12 | 0 | True | False |
| MYOLEG_NATIVE_P0 | step2 | 0.12 | 1 | 0.995556 | 0.444448 | 0.12 | -0.12 | 0.1 | True | True |

## 时间分配的独立作用

| model | endpoint | change | improvement_pct | beta_flex | beta_extend | flexion_share | share_shift |
|---|---|---|---|---|---|---|---|
| LEG_0_NOMINAL | E0 | coordination_only | 0.444294 | -0.12 | 0.12 | 0.566667 | 0 |
| LEG_0_NOMINAL | E0 | timing_only | 0.453243 | 0 | 0 | 0.666667 | 0.1 |
| LEG_0_NOMINAL | E0 | coordination_and_timing | 0.905283 | -0.12 | 0.12 | 0.666667 | 0.1 |
| LEG_0_NOMINAL | E2 | coordination_only | 0 | 0 | 0 | 0.566667 | 0 |
| LEG_0_NOMINAL | E2 | timing_only | 0.120809 | 0 | 0 | 0.666667 | 0.1 |
| LEG_0_NOMINAL | E2 | coordination_and_timing | 0.120809 | 0 | 0 | 0.666667 | 0.1 |
| LEG_1_HEAVY_HIP_STIFF | E0 | coordination_only | 0.0637617 | -0.12 | 0.12 | 0.566667 | 0 |
| LEG_1_HEAVY_HIP_STIFF | E0 | timing_only | 0.855388 | 0 | 0 | 0.466667 | -0.1 |
| LEG_1_HEAVY_HIP_STIFF | E0 | coordination_and_timing | 0.917807 | -0.12 | 0.12 | 0.466667 | -0.1 |
| LEG_1_HEAVY_HIP_STIFF | E2 | coordination_only | 0 | 0 | 0 | 0.566667 | 0 |
| LEG_1_HEAVY_HIP_STIFF | E2 | timing_only | 0.0416226 | 0 | 0 | 0.666667 | 0.1 |
| LEG_1_HEAVY_HIP_STIFF | E2 | coordination_and_timing | 0.041732 | 0 | 0.03 | 0.666667 | 0.1 |
| LEG_2_KNEE_DOMINANT | E0 | coordination_only | 4.02857 | -0.12 | 0.12 | 0.566667 | 0 |
| LEG_2_KNEE_DOMINANT | E0 | timing_only | 0.128456 | 0 | 0 | 0.466667 | -0.1 |
| LEG_2_KNEE_DOMINANT | E0 | coordination_and_timing | 4.1687 | -0.12 | 0.12 | 0.466667 | -0.1 |
| LEG_2_KNEE_DOMINANT | E2 | coordination_only | 0 | 0 | 0 | 0.566667 | 0 |
| LEG_2_KNEE_DOMINANT | E2 | timing_only | 0.0259285 | 0 | 0 | 0.666667 | 0.1 |
| LEG_2_KNEE_DOMINANT | E2 | coordination_and_timing | 0.0259285 | 0 | 0 | 0.666667 | 0.1 |
| LEG_3_NONLINEAR_COUPLED | E0 | coordination_only | 2.61446 | -0.12 | 0.12 | 0.566667 | 0 |
| LEG_3_NONLINEAR_COUPLED | E0 | timing_only | 0.319405 | 0 | 0 | 0.466667 | -0.1 |
| LEG_3_NONLINEAR_COUPLED | E0 | coordination_and_timing | 2.94247 | -0.12 | 0.12 | 0.466667 | -0.1 |
| LEG_3_NONLINEAR_COUPLED | E2 | coordination_only | 0 | 0 | 0 | 0.566667 | 0 |
| LEG_3_NONLINEAR_COUPLED | E2 | timing_only | 0.00586631 | 0 | 0 | 0.666667 | 0.1 |
| LEG_3_NONLINEAR_COUPLED | E2 | coordination_and_timing | 0.00586631 | 0 | 0 | 0.666667 | 0.1 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | E0 | coordination_only | 0.956797 | -0.12 | 0.12 | 0.566667 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | E0 | timing_only | 0.799191 | 0 | 0 | 0.466667 | -0.1 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | E0 | coordination_and_timing | 1.76434 | -0.12 | 0.12 | 0.466667 | -0.1 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | E2 | coordination_only | 0.329998 | -0.12 | 0.12 | 0.566667 | 0 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | E2 | timing_only | 0.000927431 | 0 | 0 | 0.666667 | 0.1 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | E2 | coordination_and_timing | 0.330895 | -0.12 | 0.12 | 0.666667 | 0.1 |
| MYOLEG_NATIVE_P0 | E0 | coordination_only | 0.896304 | 0.12 | -0.12 | 0.566667 | 0 |
| MYOLEG_NATIVE_P0 | E0 | timing_only | 1.29522 | 0 | 0 | 0.666667 | 0.1 |
| MYOLEG_NATIVE_P0 | E0 | coordination_and_timing | 2.12544 | 0.12 | -0.12 | 0.666667 | 0.1 |
| MYOLEG_NATIVE_P0 | E2 | coordination_only | 0.402044 | 0.12 | -0.12 | 0.566667 | 0 |
| MYOLEG_NATIVE_P0 | E2 | timing_only | 0 | 0 | 0 | 0.566667 | 0 |
| MYOLEG_NATIVE_P0 | E2 | coordination_and_timing | 0.444448 | 0.12 | -0.12 | 0.666667 | 0.1 |

物理解释：在准静态主导时，延长某阶段通常改变其在全周期 E0 中的时间权重。因此 E0 降低可能主要来自缩短高负荷阶段，不能自动解释为同一姿态力更小。对于 E2，每阶段 RMS 已除以自身阶段时长，纯时长权重效应大体抵消；其变化主要来自速度/加速度相关动力学。

## 运动幅度与可执行性边界

全扫描每模型最大运动导数如下（不是所选候选专属值）：

| model | hip_speed_peak_deg_s | knee_speed_peak_deg_s | hip_accel_peak_deg_s2 | knee_accel_peak_deg_s2 |
|---|---|---|---|---|
| LEG_0_NOMINAL | 28.8438 | 46.459 | 23.8311 | 52.948 |
| LEG_1_HEAVY_HIP_STIFF | 24.3261 | 41.3991 | 20.0985 | 47.1814 |
| LEG_2_KNEE_DOMINANT | 32.3189 | 45.999 | 26.7023 | 52.4238 |
| LEG_3_NONLINEAR_COUPLED | 30.5813 | 49.6789 | 25.2667 | 56.6177 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | 28.1487 | 42.3191 | 23.2568 | 48.2299 |
| MYOLEG_NATIVE_P0 | 28.8841 | 46.5414 | 23.8644 | 53.0419 |

最小相位变换导数：0.587848；最大采样 ROM 超出：1.62359e-06 rad。检查有限值、相位变换单调、ROM 不越界及周期/边界一致性。速度和加速度已记录，但没有已验证的机器人/人体上限，因此本报告不把这些轨迹称为可上机或临床安全。summary.csv 另给最佳候选相对参考的速度/加速度峰值比。

## 文件与复现

- all_candidates.csv：每候选参数、实际时间比例、E0/E2、四项 RMS、峰值力矩、运动导数、有效性。
- summary.csv：每范围每指标的最小值、边界状态、近最优数量和最佳候选运动导数比。
- factor_comparison.csv：协调与时间分配的独立/联合比较。
- *_torques.npz：逐候选 401×2 力矩，parameters 列为 β_flex、β_extend、share_shift；time_s 应由 source_time_s 按对应阶段时长比例重建。
- E0/E2_range_scan.png、E0/E2_timing_scan.png：范围和时间比例效应。

复现：`python -m lower_limb_sim.trajectory_sensitivity.study`，再 `python -m lower_limb_sim.trajectory_sensitivity.report`。

全部 3678 条导出力矩已按实际重分配时间重新计算 E0/E2，与 CSV 一致。三个新增针对性测试验证零参数恢复、时间变换链式法则和固定周期/边界、宽域 ROM/相位单调与折返拒绝。只新增离线代码和结果，没有修改 frozen V3、E2、机械腿参数、原算法、硬件代码和历史结果；没有自动提交。
