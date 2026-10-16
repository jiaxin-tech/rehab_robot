# 首次空载机器人试验发布流程

更新：2026-10-15。本文描述离线软件链路；测试里的审批和现场证据均为合成数据，不是现场放行记录。

## 已实现的链路

1. `python -m scripts.prepare_robot_trial_release` 生成 `approved=false` 草稿。必须明确传入 `--wrench-hz`，并绑定参考轨迹、安全配置、坐标系、起点、力采集配置、机器人身份、关键代码和有效期。
2. 对草稿的 `bundle_sha256` 作现场评审。记录需有 `scope=FIRST_ROBOT_TRIAL_EMPTY_LOAD`、`decision=approved`、与草稿一致的 `reviewer`、带时区的 `reviewed_at` 和四类 `records`：`controller_safety`、`acquisition_stability`、`wrench_semantics`、`setup_and_stop`。每类记录有 `passed=true`、非空证据文件路径与 SHA-256。路径相对于评审文件目录。
3. 同一命令加 `--approve --input-draft ... --site-review ... --confirmation "I CONFIRM REVIEWED FIRST ROBOT TRIAL RELEASE"`，并保留原草稿的身份、审核人、操作者、有效期、频率和碰撞来源参数。写出前执行完整校验，失败不发布批准文件。评审文件可在发布目录以外，路径以绝对路径绑定。
4. `python -m scripts.run_rehab_experiment --mode execute` 使用 `--release-approval` 载入批准文件；仍需既有 `--enable-motion`、操作者确认、IP、起点、配置、力频率和碰撞来源参数。用各命令的 `--help` 查看完整参数。
5. 在构造机器人适配器之前，核对文件、代码、内存配置、轨迹、频率及事件时效参数。预检仅接受经过完整验证的发布对象，普通 `approved=true` 字典不构成放行。
6. 采集健康后再次检查实际身份（包括 SDK 版本）、碰撞来源和事件时效配置，再执行实时预检。离线预检只检查静态条件，不虚构现场状态已验证。
7. 执行器在启动前重新核对发布文件和证据；在等待和每次命令分发前检查有效期。发布过期进入已有统一停止路径，不再补发后续目标。文件哈希不在每个运动目标的实时热路径中反复读取。

`reference_release` 中的冻结 `NO_GO` 不被改写；单次空载试验许可由独立发布文件表达。修改已绑定配置、代码或证据后需重新生成及评审。哈希绑定内容，不认证签字人身份，也不能证明现场证据内容真实。

## 离线验证范围

`tests/test_robot_release_workflow.py` 覆盖草稿 CLI、审批 CLI、执行 CLI、真实预检、真实执行器、日志及停止。仅机器人、实时采集与调度时间由测试对象替代，没有连接硬件。正常流程使用完整冻结 slow 轨迹，不缩减轨迹或跳过内容审核。

拒绝场景包括：未批准草稿、缺频率、范围错误、证据缺失/未通过、配置或代码变化、过期、审核人/身份不符、运行频率和事件时效不符、内存力配置被换、普通字典冒充批准对象、预检后对象/文件变化、附着后证据变化，以及运行中到期。代码中的参数检查均保留，不能将测试数据用作现场配置。

复现：

```powershell
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini tests/test_robot_release_workflow.py tests/test_robot_trial_release.py tests/test_execution_preflight.py tests/test_run_rehab_experiment.py tests/test_robot_trajectory_executor.py -q
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q
```

## 当前现场进展

离线发布流程可以验证；尚没有本次改动后的真机运行证据。默认安全配置和坐标系中的未填写项仍需实测与评审。厂家事件回调的初始有效状态、时效、断连/恢复，以及状态与力并发持续采集仍需现场确认。订阅回调不等于收到有效安全状态；厂家未保证初始事件或心跳时，软件可能保持未知或到期阻断。

本次工作没有签发真实批准记录，没有连接、上电或运动。软件链路通过不代表现场条件已经满足。
