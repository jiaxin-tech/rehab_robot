# WRENCH_STATIONARY_RUNNER_INTEGRATION_V1

2026-09-26。已完成 stationary runner 接线及离线验证。**LIVE_STATIONARY 软件入口已实现；本次没有连接机器人，也没有授权任何实机试验。R9 保持 OPEN，reference release 保持 NO_GO。**

## 1. 架构与改动

新增 [runner](scripts/run_wrench_stationary_validation.py)，直接使用 `RealRobotAcquisition`、`WrenchProcessProvider`、`EpisodeLogger` 和既有 health。没有另建采集框架，没有导入 trajectory executor。

```text
versioned request + case + mode
  -> validate request
  -> LIVE only: authorization/request/source/operator/previous A binding
  -> runner owns parent state/RT session
  -> RealRobotAcquisition(manage_connection=False)
       A: diagnostic_state_only=True
       B: production WrenchProcessProvider -> spawn-owned wrench SDK session
  -> stop acquisition -> existing-session final state -> parent disconnect
  -> persisted evidence / summary / optional A-vs-B comparison
```

本轮文件范围：

- `scripts/run_wrench_stationary_validation.py`：runner、授权校验、窄观察 facade、证据导出与比较。
- `scripts/prepare_wrench_stationary_validation.py`：增加 schema version、request UUID、明确采样配置与待填现场字段。原 `live_runner_ready=false` 表示生成的**请求尚未获准执行**，不是本报告的软件入口状态。
- `collection/wrench_process.py`：仅增加可选 `connection_audit` 的 child CONNECT_START/COMPLETE 事件及顺序验证；stationary B 启用该选项。进程所有权、查询、ACK、health、清理算法保持既有实现。
- `tests/test_wrench_stationary_runner.py`：离线 runner、故障、授权和比较测试。

没有修改本轮之外的 control/motion、安全配置、限值或 reference release。没有自动 commit。工作树包含此前任务的未提交文件，本报告不把它们重新计为本轮实现。

## 2. A / B 与默认离线模式

两种 case 的请求时长固定 900 s，顺序固定 A→B。

| case | 路径 | wrench 含义 |
|---|---|---|
| A | 生产 acquisition + state/RT | `DISABLED_FOR_STATIONARY_A`，N/A；aggregate health 仍为 false，不伪造双流健康 |
| B | 相同 state/RT + 生产隔离 provider | 真正 spawn 子进程；offline 仅将 child session 换为既有故障 stub |

默认 `OFFLINE_DRY_RUN`。短测独立记录 `offline_wall_duration_s`，不改变 request 的 900 s；即使短测完全正常，`completed_900s=false`。`offline_dry_run_pass` 只表示短测协议/清理结果，不代表 900 s 验证。

请求明确记录 wrench 50 Hz、state poll 250 Hz、alignment 50 Hz、supervisor 100 Hz、RT 请求间隔 8 ms。后三者属于协议配置，不是安全阈值；不会自动替换为历史 20 Hz。实际到达率和 latency 从记录中计算，不能以请求频率证明实际频率。

## 3. 硬授权边界

`LIVE_STATIONARY` 在构造 hardware adapter 前要求：

1. request schema=1、UUID、A/B 定义、900 s、显式 rates、NO_GO/no-retry 均合法。
2. 独立授权 artifact 的 scope=`LIVE_STATIONARY_NO_MOTION`，绑定完整 request canonical SHA256、case 和当前七份执行源码 SHA256。
3. reviewer/operator、现场 setup、厂家 session 依据、stop/recovery 记录非空；有效期未过；现场人员、恢复闭环、parent lifecycle 审查为 true。
4. 操作者给出精确 `authorization_id:operator` 确认。单独选择 mode 不足以放行。
5. 完整 WrenchBudgets、parent connect/stop/end-state/disconnect/logger budgets、supervisor interval criterion 均已明确；拒绝 `NOT_SAFETY_THRESHOLDS` 用于实机。
6. connection、robot model/serial、controller/SDK version、允许观察的 power 状态明确。连接后核对实际 parent 身份、idle/power；B child 独立核对同一身份。
7. B 另需同一 request/source 的 LIVE A summary：完整 900 s、无失败，且其 SHA256 被 B 授权记录引用。A 失败不会自动启动 B。

授权 ID 在 `results/wrench_stationary_validation_v2/authorization_claims/` 独占登记，失败也不删除；不能自动重用该 artifact 重试。A 和 B 是分别授权的 session，不在 runner 中自动重连。

这是本地审核记录和完整性绑定，**不是数字签名或外部身份认证服务**。填写 true 或 provenance 字符串不构成现场审核证据；真实记录由独立审核者负责。motion release 不参与该诊断授权，也不会被修改。

## 4. Session、时钟和证据

parent 使用现有 adapter 的 observation-only connect/state/stop/disconnect 路径；未创建 motion controller。child 独占 wrench SDK session。run UUID、parent session UUID、child session UUID、PID、配置、源码和 expected/observed identity 分别保存。

输出位于 `results/wrench_stationary_validation_v2/<run_id>/`，每次创建新目录：

| 文件 | 含义 |
|---|---|
| protocol_snapshot.json | 完整请求、模式、case、预算及来源、授权、源码哈希、身份配置 |
| rt_state_events.jsonl / rt_state.csv | production state consumer 观察到的独立 sequence、host receive/delivery、状态/TCP/joints、validity、间隔和跳号 |
| wrench_query_events.jsonl | provider 原有持久 intent/ACK/pre-call boundary/completion/error/pending/lifecycle 审计 |
| wrench_samples.csv | 从成功审计事件导出原始六轴向量、sample midpoint、query/session ID、parent receive |
| health_timeline.jsonl | 每次 supervisor tick、loop interval、state/wrench age、skew、query/progress age、health/latch、首次检测和发布时刻 |
| process_lifecycle.jsonl | parent connect/stop/end-state/disconnect；保留 child 原事件 origin/ID 的 lifecycle 索引 |
| cleanup_outcome.json | parent、child、logger、各审计 writer 的独立收尾结果 |
| end_state.json | 末态、来源、查询结束时间或 null+reason |
| case_summary.json | 固定目标/实际时长、计数、pending、263 等 error histogram、分布、清理和失败 |
| sha256_index.json | 顶层输出文件 SHA256，排除索引自身 |
| episode/ | 既有 EpisodeLogger 原始五文件，格式未改 |

设备时间缺失时保持 null。`rt_host_receive_interval_s` 是 host 接收间隔，不能称为设备源周期；parent state 路径没有 IPC，`ipc_latency_s=null`。RT consumer 观察到的 sequence gap 明确计数并使 trial 失败；该 runner 不声称缓存采样能恢复每一原生 RT frame。

query started 计数表示持久 start intent；PRE_CALL_BOUNDARY 不证明已经进入 SDK 指令。pending 没有虚构 completion/end。瞬时崩溃或磁盘故障仍可能导致 completion unknown；原始残缺记录保留，解析缺口使结果失败。

sidecar 复用既有 bounded asynchronous durable writer；磁盘错误/背压/超时使 trial 失败。不能保证失效磁盘上仍能写出完整 summary；writer 无法退出时保持 audit-unconfirmed，不能把缓存当作完整持久记录。

## 5. 停止、末态与 parent 限制

runner 请求 acquisition 停止，复用 provider 的正常退出/强杀区分。仅在 stop 已确认且原 parent session 仍有效时，通过该 session 的 operationState 查询末态，再 disconnect。stop 失败不会再次尝试 disconnect；parent 调用超时后也不并发补查或重连。

parent native 调用在独立 daemon lifecycle 线程中等待指定预算。**超时只表示未确认，不是 native cancellation。它不能保证在 parent native 持有 GIL 时按时获得调度。**不创建新的 RT 进程或扩大控制架构。出现 `UNRESOLVED_PARENT_CALL_DESIGN_REVIEW_REQUIRED` 时保留未决状态，交独立设计/现场处置；不能循环重试。

child 强杀保留 `FORCED_WORKER_TERMINATION=true`、`GRACEFUL_SDK_DISCONNECT=false`、controller recovery=null。offline 正常退出只确认 stub disconnect，SDK graceful 字段为 null。parent 仍是正在运行的宿主进程，不伪造其 exitcode=0。

末态不可得保持 null+reason，并计入失败；即使没有观察到非 idle，也不能据此改为 PASS。RT frame 的 operation_state 可能是缓存值，不能证明整个试验 operationState 连续保持 idle。真实 vendor session 可靠性、parent native 阻塞、SDK graceful 故障清理仍未获硬件验证。

## 6. 判据与比较

保留 SDK 263、hung、state failure、producer exit、transport/logger failure、末态缺失和 cleanup failure；无 restart/retry/replacement。早停保留 `requested_duration_s=900`、`completed_900s=false`。

已有 WrenchBudgets 用于执行对应故障监督；新增 runner 支持显式 `max_supervisor_interval_s`、`max_rt_receive_interval_s`、`max_delivery_latency_s`。不默认选安全数值。offline budgets 全部标 `NOT_SAFETY_THRESHOLDS`，不复制到现场配置。RT/delivery 条件未定义时不判 PASS。

summary 的 timing/reliability acceptance 当前保守保留 UNDEFINED，供后续独立审核；没有自动宣称完整科学验收通过。数据时长/结构完整与 timing/reliability 获准是不同结论。

`compare(A,B)` 核对 request/source/mode 一致，比较 host receive interval、delivery latency、state age、supervisor interval、失败和 cleanup。短测或早停标 INCOMPLETE；缺设备时间不能比较真实设备源周期。差异不证明因果。

## 7. 离线结果与产物

**74 passed，19.06 s**。这是选定离线回归集，不是全仓测试或实机验证。

最终离线回归包括 runner、production process、acquisition 和 EpisodeLogger 四组。结果见 [JUnit](results/wrench_stationary_runner_integration_v1/final_verified/tests.xml) 和 [机器汇总](results/wrench_stationary_runner_integration_v1/final_verified/verification_summary.json)。覆盖正常 A/B、固定 900 s 请求、delayed 263、never-returning/pending、强杀非 graceful、末态 null、state/telemetry/stop failure、残缺日志、授权缺失/过期/错绑/缺预算/B 缺 A、无 control dependency，以及此前 GIL/native-like 与 IPC 故障回归。

实际 CLI 默认离线模式也已验证：

- [A](results/wrench_stationary_validation_v2/698fdc1e-a936-4d95-b77b-7bfae7993ba9/case_summary.json)：正常短测，900 s 未完成。
- [B](results/wrench_stationary_validation_v2/2662b14e-09bb-495b-8a73-598a84575d8b/case_summary.json)：生产 spawn provider + offline stub，正常短测，900 s 未完成。
- [比较](results/wrench_stationary_validation_v2/2662b14e-09bb-495b-8a73-598a84575d8b/ab_comparison.json)：INCOMPLETE / UNDEFINED / NOT_ESTABLISHED。
- 先前 run_01/run_02/run_03、故障注入和早期 CLI 输出均保留；最终结论仅引用上面的匹配源码结果。

## 8. 未来命令形状与剩余前提

先独立完成实际身份/SDK/controller/setup、厂家 multi-session/recovery 依据、现场人员及停止职责、预算和 source/binary/protocol 审核。当前准备请求仍有 null，不能直接 live。补齐并冻结 request 后，再生成引用其 canonical SHA256 和 `source_hashes()` 的审核 artifact。

以下只是未来命令形状，本次未执行：

```powershell
python -B -m scripts.run_wrench_stationary_validation --request <reviewed-request.json> --case A --mode LIVE_STATIONARY --authorization <reviewed-A-authorization.json> --operator-ack "<A-authorization-id>:<operator>"

python -B -m scripts.run_wrench_stationary_validation --request <same-reviewed-request.json> --case B --mode LIVE_STATIONARY --authorization <reviewed-B-authorization.json> --operator-ack "<B-authorization-id>:<operator>" --prior-a-summary <accepted-A-case_summary.json>
```

B 需要 A 完成后单独审核的授权，不能将上一条命令成功启动视为 A 已通过。发生强杀或未决 session 时先按厂家认可的独立恢复程序处理。没有允许运动、上电、模式修改、清故障或人为诱发故障的入口。

## 9. 最终状态

```text
WRENCH_STATIONARY_RUNNER_INTEGRATION_V1 = COMPLETE_WITH_LIMITATIONS
CASE_A_OFFLINE_DRY_RUN = PASS
CASE_B_OFFLINE_DRY_RUN = PASS
USES_PRODUCTION_WRENCH_PROVIDER = true
MOTION_API_REACHABLE = false
LIVE_AUTHORIZATION_REQUIRED = true
STATIONARY_LIVE_ENTRY_READY = true
R9_CLOSED = false
REFERENCE_RELEASE = NO_GO
ROBOT_CONNECTED = false
ROBOT_MOTION_EXECUTED = false
SAFETY_THRESHOLDS_CHANGED = false
CONTROL_CODE_MODIFIED = false
```

STATIONARY_LIVE_ENTRY_READY 仅指带授权校验的**软件入口**与离线证据结构就绪，不表示当前请求、现场或硬件已获批准，也不证明 parent-native 硬实时监督。完整实机 A/B、未定义科学标准、厂家证据和 R9 closure 仍需后续独立任务。
