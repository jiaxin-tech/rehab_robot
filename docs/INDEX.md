# 论文分支文档索引

更新时间：2026-11-28。当前分支：`paper-personalized-load`。

**11月28日当前：操作者报告drive4使能后发生反关节，6064到141185，低于名义下限159870。** 本轮最新6041=0x0231已确认退出运行使能，操作者报告两套设备停稳、小腿已支撑；按[越界记录](lab/KNEE_LIMIT_BREACH_2026-11-28.md)保持停用并核对本机固件行为，不再使能验证、带腿示教或运行轨迹。当前607D未建立有效防护。此前静态回读见[限位记录](KNEE_EXTENSION_LIMIT_WORKFLOW.md)，原始0x4466事件见[记录](lab/LEG_ERROR_4466_2026-11-28.md)，下方保留研究计划。

这个索引是`docs/`的主入口。先看“当前现场”，再按研究或代码任务进入对应文档。带日期的实验记录和`results/manual_lab/`原始文件是证据，保留原路径和原状态，不把历史成功或失败改写成当前结论。

## 当前现场状态

| 事项 | 当前结论 | 下一步 |
|---|---|---|
| 机械臂与假腿连接 | 髋基座固定，髋膝可转动，机械臂刚性接近膝小腿；操作者已确认原厂拖动可顺畅上抬和回落 | 按[工具/TCP与负载入口](ROKAE_TOOL_SETUP_WORKFLOW.md)查看实际工具定义 |
| 连接后静止采集 | 2026-11-22 18:31完成；501条受力有效，就绪后无失效，0条运动命令 | 不重复采集，见[静止数据复核](lab/CONNECTED_STATIC_REVIEW_2026-11-22.md) |
| 10厘米/10秒参考 | 已生成五次平滑标量参考，`ready_for_execution=false` | 仅作幅度/节奏参考，不能直接送入`run`，见[微幅参考流程](MICRO_LIFT_10S_WORKFLOW.md) |
| 自动带腿轨迹 | 尚无经过审核的完整六维可执行路径；11月28日越界后暂停 | 先完成越界证据和厂家核对；[带腿六维示教](ACTIVE_LEG_TEACHING_WORKFLOW.md)仅保留为恢复计划 |
| 假腿采集 | 2026-11-22一轮在219帧后刷新位超时，仍未证明已修复 | 保留失败记录，见[假腿采集失败复核](lab/LEG_CAPTURE_FAILURE_2026-11-22.md) |
| 90秒空载曲线 | 已有完整空载执行记录，但受力覆盖未通过；固定朝向不适用于刚性带腿回放 | 仅作为空载基线，见[90秒空载说明](CONTINUOUS_90S_WORKFLOW.md) |

## 历史流程入口（当前暂停动态操作）

当前只维持停用和支撑，整理已有证据及厂家核对；以下入口供查阅，不能顺次执行运动、录制或强制开闸。

1. [后续实验操作总手册](NEXT_LAB_OPERATIONS.md)：现场总入口和回传原则。
2. [珞石工具/TCP与负载入口](ROKAE_TOOL_SETUP_WORKFLOW.md)：查看工具、工件和负载；当前不要启动自动辨识。
3. [连接后静止数据复核](lab/CONNECTED_STATIC_REVIEW_2026-11-22.md)：了解已完成的采集质量和限制。
4. [主动假腿带腿示教](ACTIVE_LEG_TEACHING_WORKFLOW.md)：当前暂停；恢复前仅阅读所需条件。录制器只读不代表可以在边界未验收时拖动。
5. [10秒上抬10厘米参考](MICRO_LIFT_10S_WORKFLOW.md)：离线参考和当前阻断条件。
6. [带腿参考与联合采集计划](REFERENCE_AND_COMBINED_CAPTURE_PLAN.md)：R0/R1/R2层级、假腿复测和同步试验顺序。

## 论文与算法

- [当前论文与实验工作流](CURRENT_WORKFLOW.md)：研究问题、个性化定义、实验假设和截止日期。
- [论文架构](PAPER_ARCHITECTURE.md)：论文章节、变量和证据边界。
- [受力驱动设计审核](FORCE_DRIVEN_DESIGN_REVIEW.md)：受力目标、个性化方法和不可宣称的结论。
- [TCP受力实验指南](TCP_FORCE_EXPERIMENT.md)：正式受力候选、重复和评价流程。
- [代码地图](CODE_MAP.md)：模块职责和主要入口。
- [代码审计](PAPER_CODE_AUDIT.md)：实现边界、测试和当前风险。

## 真机与安全边界

- [真机操作手册](REAL_ROBOT_RUNBOOK.md)：机器人连接、准备、执行和清理。
- [刚性连接/带载诊断](LOADED_DIAGNOSTIC_WORKFLOW.md)：连接建档和受力诊断；不授权任意空载曲线回放。
- [膝伸直端限制流程](KNEE_EXTENSION_LIMIT_WORKFLOW.md)：drive4软限位、反馈和只读核对。
- [假腿采集失败记录](lab/LEG_CAPTURE_FAILURE_2026-11-22.md)：ADS采集问题与已保留证据。
- [实验记录模板](templates/LAB_RECORD_TEMPLATE.md)、[刚性连接记录模板](templates/RIGID_SHANK_SETUP_RECORD.md)、[带腿示教记录模板](templates/ACTIVE_LEG_TEACHING_RECORD.md)。

## 实验结果索引

[结果状态表](RESULTS_INDEX.md)列出当前分支中所有主要`results/manual_lab/`会话、离线设计目录和历史复核。结果按原始目录保留；失败记录同样是实验边界证据。

## 历史与离线参考

以下文档用于复核历史方案，不能作为当前带腿自动运行入口：

- [空载示教轨迹步骤](TAUGHT_TRAJECTORY_WORKFLOW.md)：90秒空载流程，仅适用于空载审核。
- [连续90秒设计说明](CONTINUOUS_90S_WORKFLOW.md)：90秒几何与时序设计，固定朝向。
- [70秒与425秒历史记录](TAUGHT_70S_REVIEW.md)：历史节奏和回传复核。
- [空载复核目录](lab/EMPTY_70S_REVIEW_2026-11-15.md)、[90秒复核](lab/EMPTY_90S_REVIEW_2026-11-22.md)、[起步失败复核](lab/CONTINUOUS_90S_FAILED_REVIEW_2026-11-16.md)。
- [分支整合记录](BRANCH_INTEGRATION_2026-11-14.md)：来源分支和整合边界。

缓存和测试临时目录（`.cache/`、`.pytest_cache/`、`test_artifacts/`）由`.gitignore`排除，不属于论文证据。不要使用`git clean`删除现场或结果文件。
