# REAL_DUMMY_LEG_TRAJECTORY_VALIDATION_V1

2026-09-26：完成现有门禁下的实验准备与停止记录，**未完成真实机械验证**。

`MOTION_STAGE_NOT_EXECUTED_DUE_TO_EXISTING_SAFETY_GATE`

## 1–4. 物理装配、假腿、连接结构、机器人配置

按任务指定拓扑准备：robot flange → existing connector → cuff / rigid plate / strap assembly → dummy-leg shank。未采用长自由绑带模型。未现场确认安装、假腿 ID、cuff setup ID、robot config ID 或 ROM profile ID；这些字段保持 null，不把用户描述当作已经完成安装检查的证据。没有连接机器人、上电、清错或发出运动命令。

`config/experiment_safety.json` 的 reviewed 和专项审核均为 false，限值未配置；`config/rehab_frame_config.json` 的方向未配置且未审核。`reference_release/reference_release_manifest.json` 明确 `approved_for_first_robot_trial=false`、`robot_execution_status=NO_GO`。离线调用现有安全配置检查即可确定阻塞；未声称完成 live preflight。完整原因与源文件哈希见 [准备清单](results/real_dummy_leg_validation_v1/preparation_manifest.json)。没有修改门禁或安全阈值。

## 5. 轨迹族与实验冻结

研究代码同时包含 KEY_POSTURE_TIMING 和 BETA_TIMING，不能仅由存在代码认定已为本假腿冻结某一族。首选研究族记录为 KEY_POSTURE_TIMING；其 `Domain` 参考参数为 alpha_flex=0、alpha_extend=0、time_share_shift=0。没有将 alpha 改名 beta。

机器人冻结参考为 reference_measured_asymmetric_closed_slow，24 s（flexion 13.6 s、extension 10.4 s），ROM_PROTOCOL_V2：hip [0,120]°、knee [5,145]°，SHA256 `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`。这是发布定义，不是对当前假腿适用 ROM 的确认，也没有证实其与研究族生成参考逐点相同。现有 first-trial 白名单不授权研究对比轨迹。

重复次数没有既有本实验冻结值或操作者 `--repeats N` 输入，保持未设定；静态时长同样待冻结。Contrast A/B 和执行序列未冻结，不凭空挑选可执行候选。未来先按参数空间与运动学定义两个对比、逐条检查并冻结哈希，再采集响应。建议序列规则为轮换三条件排列以平衡次序，具体 N、完整序列及各条件计数必须在测量前固定，不根据结果追加或替换失败试次。

## 6–8. Stage A / B / C 结果

三个阶段均未执行；没有本任务原始 episode、静态均值/漂移、重复性或轨迹效应可报告。历史诊断没有本次假腿与装配身份，不能改标为 Stage A。输出目录已建立，五份 CSV 只有表头，代表零次试验而不是零力或零错误。没有生成 A/B/C 图或演示视频。

## 9. 原始 wrench 语义与分析计划

后续采集复用 RokaeRobotAdapter、RealRobotAcquisition、EpisodeLogger 及现有 preflight/执行链，不新建采集或控制实现。保留原始 Fx/Fy/Fz、Mx/My/Mz、状态、时间戳、关节/TCP、phase、样本有效性、SDK 事件及完整失败 episode；采集时不永久扣 bias。SDK 支持的 query 起止、延迟、最近成功 wrench 与 robot-state 时间必须保留，不支持字段为空并写明原因。

Stage A 描述每轴均值/样本 SD、力与力矩模长分布、预先指定窗口的短期及首尾漂移、缺失/无效率、stale 事件、延迟分布、SDK 错误；多次静态记录分开比较。没有独立方向验证，不声明 VALIDATED_FORCE_DIRECTION；有独立几何记录才可另列 GEOMETRY_DERIVED_PROJECTION。

Stage B 对六轴分别计算完整周期 RMS、绝对峰值以及 FLEXION/EXTENSION RMS，保留相位曲线。比较均值、SD、有意义分母下的 CV、两两曲线相关、相位 RMS 差、峰值变化、时序变化与基线漂移，并导出同条件差值分布。零分母/常量曲线相关未定义时留空并记录原因。

Stage C 在相同装配和 ROM 下比较参考与预声明对比的分支特征效应、条件内 SD、效应/噪声比、曲线距离和跨重复一致性。噪声为零时不伪造有限比值；没有冻结判据则只做描述，不发明 CV 或显著性阈值。不得报告 REAL_E2/REAL_E3，不运行 BO/EI 或个性化。

现有 `measurement_validation.analysis` 可复用数值与特征逻辑，但 `_condition_key` 使用 beta 且不含 time_share_shift，静态接口围绕 PRE/LOAD/POST；尚不能原样用于本任务。后续必须在该包内最小扩展 family-specific identity、完整参数/装配/ROM 分组及静态无外载分析；禁止把 alpha 填入 beta 字段。曲线配准必须保留无效区间，禁止跨长缺口插值。当前包是阻塞准备记录，不声称已完成上述分析器扩展。

## 10. SDK / 时序异常

历史 [2026-08-13 长测](diagnostics/wrench_hardware_validation_20260813T110502Z.md) 记录 SDK 263 与原生阻塞；[2026-08-14 A/B 诊断](diagnostics/state_wrench_timing_comparison_20260814T093551709145Z.md) 中并发 wrench 测试在 169.610/900 s 因 RT worker hung 终止，SDK 263 共 3 次，当前状态年龄最大 724.138 ms，READY_FOR_FIRST_MOTION_TEST=false。这不是本次新观测。生产采集仍为线程实现，诊断进程隔离不能直接视为生产路径已修复。

## 11–13. 限制、decision-value readiness、下一实验

没有真实假腿证据，Q1–Q4 均不能得到支持；decision-value 实验尚未就绪。COMPLETE_WITH_LIMITATIONS 仅指按任务明确的安全阻塞分支完成准备和停止，不表示机械验证成功。DUMMY_LEG_USED=false 表示本次未开展物理试验，不否认现场已有假腿；WRENCH_SDK_BLOCKING_OBSERVED=true 专指上述历史证据。

下一步先解决既有采集阻塞并形成独立审核记录，然后在已确认固定假腿/装配、机器人静止、无人为加载、无人体参与的条件下执行 **Stage A 静态基线**，提前填写身份、时长、采样配置和停止规则。通过现有协议的进入条件后，才冻结并开展 Stage B 同参考重复；Stage C 还需对比轨迹的独立运动授权。不能以本报告解除任何门禁。每次失败保留并计数，不自动重试。全部测量后再决定下一 decision-value 协议，禁止直接继续个性化。

```text
REAL_DUMMY_LEG_TRAJECTORY_VALIDATION_V1 = COMPLETE_WITH_LIMITATIONS
HANDHELD_FORCE_GAUGE_USED = false
HUMAN_SUBJECT_USED = false
DUMMY_LEG_USED = false
STATIC_BASELINE_EXECUTED = false
STATIC_WRENCH_STABILITY = INSUFFICIENT
REFERENCE_REPEATABILITY_EXECUTED = false
SAME_TRAJECTORY_REPEATABILITY = NOT_EXECUTED
TRAJECTORY_SENSITIVITY_EXECUTED = false
TRAJECTORY_SENSITIVITY = NOT_EXECUTED
ABSOLUTE_FORCE_CALIBRATION_VALIDATED = false
TASK_FORCE_DIRECTION_VALIDATED = false
WRENCH_SDK_BLOCKING_OBSERVED = true
DECISION_VALUE_EXPERIMENT_READY = false
PERSONALIZATION_EXECUTED = false
BO_EXECUTED = false
ROBOT_CONTROL_CODE_MODIFIED = false
SAFETY_THRESHOLDS_MODIFIED = false
```
