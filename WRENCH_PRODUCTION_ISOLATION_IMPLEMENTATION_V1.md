# WRENCH_PRODUCTION_ISOLATION_IMPLEMENTATION_V1

2026-09-26。**完成最小 wrench 进程隔离候选及离线验证，48 项测试通过；R9 未关闭，NO_GO 未改变。** stationary runner 当前完成请求准备工具，尚未具备可直接执行实机 A/B 的入口。

## 1. 已实现架构与改动范围

```text
RealRobotAcquisition（单一入口，parent）
  state/RT adapter + alignment + raw EpisodeLogger
  wrench IPC consumer + nonblocking supervisor polling
    -> spawn child（独占 wrench session）
       -> WrenchSession -> existing Windows get_end_wrench conversion
       -> getEndTorque
    <- ordered acknowledged audit mailbox + separate latest-result slot
```

- [collection/wrench_process.py](collection/wrench_process.py)：进程 owner、固定容量 IPC、协议校验、fsync ACK、health、failure latch、关闭协议。
- [collection/real_robot_acquisition.py](collection/real_robot_acquisition.py)：显式 provider 注入；parent 原生 wrench fallback 已移除；保留 state/alignment/raw frame 与现有 health 接口。
- [hardware/wrench_session.py](hardware/wrench_session.py)：仅 child 导入的只读独立 SDK session；复用现有 wrapper 的 wrench 转换和 adapter frame。不启动 child RT，不调用 power/mode/fault/motion API。
- [collection/wrench_test_provider.py](collection/wrench_test_provider.py)：仅 offline 分支导入的故障 stub。
- [scripts/prepare_wrench_stationary_validation.py](scripts/prepare_wrench_stationary_validation.py)：900 s A/B 准备工具，不导入 live SDK、不提供 live 执行开关。
- [tests/test_wrench_production_process.py](tests/test_wrench_production_process.py)：spawn 故障与集成测试；既有 acquisition 测试改用明确的 IPC consumer double 保留生命周期回归，其进程隔离结论以新 OS-process 测试为准。

`control/`、`safety/`、安全配置和 release 未修改；EpisodeLogger 原始五文件格式未修改。没有自动 commit。

## 2. Session 所有权与启动条件

child 接收 JSON 配置和 IPC primitives，不接收 parent adapter/SDK。live 分支在 child 内加载 SDK、创建 session，并逐项核对 robot model、serial、controller version、SDK version。连接参数和 parent adapter 配置需一致；parent 实际 metadata 也必须与 expected identity 一致。child 只读核对 idle 与明确允许的 power 状态，不自行改变机器人状态。

构造 `WrenchBudgets` 必须提供全部正有限预算及 provenance，没有生产安全数值默认值。offline 预算必须标 `NOT_SAFETY_THRESHOLDS`；该标签不能用于 live provider。provenance 字符串不是自动审批机制，实机配置仍需人工审核和独立授权。

**调用兼容性变化：** 旧 `RealRobotAcquisition(adapter, logger).start()` 现在在连接前拒绝，要求显式 isolated provider；不会静默继续旧线程查询。现有未接入 provider 的采集/执行 CLI 因而不会获得实机执行能力。本轮没有修改控制入口来接线或绕过门禁。`wrench_hz` 保持原默认 50 Hz，必须与 provider 显式 rate 一致。

## 3. Query 生命周期和时间含义

每个 child session 单调 query_id；事件包括 PROCESS_START/READY、QUERY_START_INTENT、QUERY_NATIVE_ENTER、QUERY_SUCCESS/ERROR、DISCONNECT_START/COMPLETE、PROCESS_STOPPED。parent 另记 QUERY_PENDING、QUERY_HUNG、FAILURE_LATCH、DURABLE_ACK、PROCESS_EXIT、STOP_REQUEST、FORCED_WORKER_TERMINATION、CLEANUP_OUTCOME。

事件带 schema version、run/session ID、PID、event_id、query_id 和 host monotonic timestamp。child 与 supervisor 各有单调 event_id；唯一键是 `(origin, session_id, event_id)`，child 事件没有 origin 字段时默认为 child，不能跨 origin 混作单一序列。

start intent 经 parent writer 写入、flush/fsync 后才 ACK；child 未获 ACK 不能调用查询。completion 同样先持久化 ACK 后才能发下一请求，SDK error 终止本轮查询，不自动重试。

dispatch、start intent、pre-call boundary、实际 wrapper 调用 start/end、parent receive、log-write、durable ACK 分开记录。`QUERY_NATIVE_ENTER` 的 `entry_semantics=PRE_CALL_BOUNDARY`，不是对 SDK 内部指令执行的证明；pending 查询的 `entry_unconfirmed=true` 保守保留。未返回则 completion=null；强杀不伪造 QUERY_ERROR 或 end timestamp。成功 frame 中的 wrapper query 时间、原始 sample midpoint 和发布时刻保留，不以 IPC 接收时间替换。

fsync 完成时刻通过后续 DURABLE_ACK 记录；原行的 log_write_ns 不冒充 fsync 完成时间。若恰在 native 返回与事件发布之间崩溃，该次 completion 仍可能未知，必须保留 pending/telemetry failure，不能宣称 crash-atomic 的绝对完整性。

## 4. IPC 与日志

使用固定容量 shared-byte slot（含发布版本和 SHA256 完整性校验），不使用可能在强杀后遗留 feeder/pipe 锁的 multiprocessing Queue。audit 是 stop-and-wait mailbox，未 ACK 禁止覆盖；latest-result slot 独立、允许覆盖。parent 的权威样本只来自完成审计确认的事件，latest slot 不是有效性捷径。

检查事件序号缺口、重复/重新发布、乱序 query/lifecycle、PID/run/session 错配、时钟异常、容量溢出和未 ACK 覆盖；出现异常锁存 failure。读取 slot 的版本和内容作为同一快照返回，避免正常发布竞争被误判重复。

磁盘写入由单个 bounded-queue daemon writer 执行；parent 不等待其 fsync。日志超时/失败独立进入 health；不会 ACK 未持久化 intent。writer 卡住时不创建替代 writer，也不无限 join；可能仍存活的 writer 会使 audit cleanup 不确认。这是显式失败，不是完整日志已保存的保证。

in-memory events 仅保留最多 1024 条诊断缓存；完整 audit 以 JSONL 为准。发生磁盘故障时，内存状态及测试导出的 cleanup/gap 记录说明哪些证据未持久化，不能把降级缓存宣称为完整审计。原始 episode CSV 格式保持不变。

查询按照显式 intended rate 的 start-to-start 调度，start intent 保存 scheduled_dispatch_ns/dispatch_lateness_ns；durable ACK 有成本，**请求 50 Hz 不证明实际实现 50 Hz**。实机必须检查实际到达率和时序，不允许为了达到表面通过而修改请求率。

## 5. Health 与故障锁存

provider 分开提供 LATEST_GOOD_SAMPLE_VALID、CURRENT_QUERY_STATE、LAST_ERROR、QUERY_IN_PROGRESS、QUERY_START_TIME、LAST_COMPLETED_QUERY_TIME、WRENCH_SAMPLE_AGE、WORKER_HEARTBEAT_AGE、PROCESS_ALIVE、STREAM_HEALTH、FAILURE_LATCH。

last-good frame 的向量转为 tuple，保留历史样本有效性。SDK error 不更新时间；freshness/progress/query timeout、crash、传输或日志故障锁存本 trial failure。之后即使收到成功返回也不能清除 failure。heartbeat 表示实际 lifecycle/query 进展，没有在 stuck query 期间继续宣称健康的独立 heartbeat 线程。缺首包使用 parent startup 时钟判断。

AcquisitionHealth 新增 wrench_process_alive、wrench_stream_healthy、wrench_query_state、wrench_failure_latch；`wrench_thread_alive` 仍是实际 IPC consumer 线程存活，含义未改。`wrench_valid=true` 可以与 `valid=false` 共存，表示历史样本有效但当前流不可用。state age/skew 失效同样锁存，不以好 wrench 覆盖坏 RT。

consumer 和 `latest_health` 都可调用非阻塞 poll；poll 的宿主锁只尝试获取、不等待。这样 raw CSV consumer 若卡在日志 I/O，另一健康检查入口仍可推进监督。该设计不保证 Windows 硬实时，也不隔离 parent 自身 native RT 调用。

## 6. 清理、强杀与恢复

stop 请求后在显式预算内继续处理 ACK/已有事件，等待正常退出；必要时 terminate，再至多一次 kill，并有界 join。pending 查询不补完成记录。强杀后不再消费 child IPC；独立记录 process exit、forced、graceful SDK disconnect、controller recovery。

- 离线 fake 正常清理：OFFLINE_DISCONNECT_COMPLETED=true，GRACEFUL_SDK_DISCONNECT=null，不假装调用过 SDK。
- 强杀：FORCED_WORKER_TERMINATION=true，GRACEFUL_SDK_DISCONNECT=false，CONTROLLER_SESSION_RECOVERY=null。
- SDK disconnect 自身阻塞：由同一 child 进程边界覆盖；若需强杀则清理失败。
- 没有自动 restart/reconnect/new session。provider 实例不能重启；新实例不代表获得现场新连接授权。

parent state/RT session 的 stop/disconnect 仍沿用既有实现，**不保证有界**。本轮没有测试真实 parent RT native 堵塞，也未扩大成多进程控制架构。若后续证据表明它阻塞监督，必须 DESIGN_REVIEW_REQUIRED；不能用本轮 child 结果盖过它。

## 7. 离线验证结果

最终命令：

```powershell
$env:WRENCH_TEST_OUTPUT = 'results/wrench_production_isolation_v1/final_verified'
python -B -m pytest tests/test_wrench_production_process.py tests/test_real_robot_acquisition.py tests/test_episode_logger.py -q --junitxml=results/wrench_production_isolation_v1/final_verified/test_case_summary.xml
```

**48 passed，13.41 s**。这是选定离线回归集，不是全仓测试结论。

| 验证项 | 结果与范围 |
|---|---|
| 正常协议 | start 的 durable ACK 早于 query start；成功 frame 与审计对账；正常 fake 清理 |
| 10 s 阻塞、never-returning | parent/模拟 RT 推进；query budget 触发 hung；pending 无 completion；强杀不冒充 graceful |
| delayed 263 | 一条 ERROR completion，code 保留，旧有效样本不使流健康；没有下一查询重试 |
| crash | exitcode=73 被观察，pending 无伪造 completion |
| stale last-good / later success | 旧样本有效与流不健康共存；后续成功不解除锁存 |
| shutdown | startup、ACK、query、completion、disconnect 阶段覆盖 |
| transport | overflow、gap、duplicate/republication、乱序 query/cleanup、lost ACK、PID/run 错配 |
| 日志 | backpressure/write failure、bounded queue overflow 明确失败 |
| clock/startup | 首 heartbeat 缺失、非法事件/监督时钟 |
| acquisition | parent wrench 方法设置为抛 AssertionError 的禁用桩，真实 child provider 集成测试不调用它；保留 raw CSV |
| state freshness | 过期 state 使流不健康并锁存 |
| stationary prepare | duration=900、明确50 Hz、未授权/无 live entry，拒绝非法 rate |

早期 run_01/run_04 等失败输出保留，没有把失败试次删除。修正包括：清理覆盖 primary ACK 错误、pre-call 与实际阻塞测试边界、发布版本读取竞争、旧 fake provider 的显式 mode。首次测试复用官方 reference fixture 时还遇到已有 `reference_cycle_closure_audit.csv` checksum mismatch；没有修改 release 来使测试通过。正式 health 拒绝测试改用明确的未批准 fake 请求，不依赖该轨迹 bundle。

### GIL / native-like

Windows child 使用 `ctypes.PyDLL('kernel32.dll').Sleep`，PyDLL 调用持有 GIL；不是 `time.sleep` 替代测试。10 s 调用在测试 budget 越界后被中止；parent supervisor 和模拟 RT tick 继续，query pending，health failure 可见，强杀完成。因此 NATIVE_LIKE_GIL_BLOCK_TEST=PASS **仅限此离线 foreign-call stub**；不证明 ROKAE SDK 本身的 GIL、网络或控制器行为。

### preflight / executor

未修改两份控制源码。使用历史 sample_valid=true、aggregate valid=false 的 fake health，真实 `evaluate_execution_preflight` 返回 `acquisition_streams_unhealthy` 并由 require_allowed 拒绝；真实 executor `_runtime_reasons` 保留 sdk_error。未调用 execute() 或任何 motion API。该证据证明健康信息拒绝路径，不证明实际物理停止时延。

## 8. 产物

- [最终测试 JSON 汇总](results/wrench_production_isolation_v1/final_verified/test_case_summary.json)
- [JUnit 汇总](results/wrench_production_isolation_v1/final_verified/test_case_summary.xml)
- [源码哈希](results/wrench_production_isolation_v1/final_verified/source_hashes.json)
- 每个 OS-process 场景目录中的 query_events.jsonl、process_lifecycle.json、health_timeline.json、cleanup_outcome.json、gap_drop_diagnostics.json；协议单元故障还由对应 JUnit case 保留结果。
- [stationary 请求准备](results/wrench_production_isolation_v1/stationary_preparation/stationary_request.json)

测试预算仅用于离线故障注入，provenance=NOT_SAFETY_THRESHOLDS。未覆盖历史 W1 任何文件。

## 9. Stationary runner 边界

按任务允许的“implement or prepare”分支，已实现准备工具：

```powershell
python -B -m scripts.prepare_wrench_stationary_validation --output-dir <new-directory> --intended-wrench-hz 50
```

请求冻结 A→B、各900 s、明确 intended rate；预算未审核时保持 null。A-only 的 acquisition health 始终包含 DISABLED_FOR_STATIONARY_A，不能通过现有运动 preflight。**STATIONARY_RUNNER_READY=false**：尚缺完整 live A/B 调度、父 session 有界结束状态/清理、现场授权及预算审核。没有提供一个表面可运行却缺失这些证据的硬件命令。

下一步可独立审核本 diff，再完成 stationary runner 接线与现场 protocol；随后才授权实机静止验证。不得把本轮 test-only 参数用于现场，也不得因 child 隔离通过就解除 NO_GO。

## 10. Q1–Q10 更新

| 问题 | 状态 | 证据与限制 |
|---|---|---|
| Q1 wrench block 是否阻碍 RT/state | PARTIALLY_SUPPORTED | child GIL stub 下模拟 parent RT 推进；真实 RT/session 未验证 |
| Q2 supervisor 有界决策 | SUPPORTED_OFFLINE | 测试预算内检测/锁存；无硬实时或真实 parent-native 保证 |
| Q3 不等返回发现失鲜 | SUPPORTED_OFFLINE | never-returning、query age、stale last-good |
| Q4 查询/错误完整记录 | PARTIALLY_SUPPORTED | durable intent/ACK、唯一 completion、gap检测；崩溃瞬间或磁盘失效仍可能 completion未知 |
| Q5 停止/隔离不破坏其他采集 | PARTIALLY_SUPPORTED | child强杀、consumer集成/日志测试；真实控制器和 parent RT 不在覆盖内 |
| Q6 强杀后保证 | PARTIALLY_SUPPORTED | OS exit和IPC quarantine；SDK/session恢复无保证 |
| Q7 故障 graceful SDK disconnect | NOT_APPLICABLE_OFFLINE | fake正常退出不构成SDK证据，强杀明确false |
| Q8 下一session所需处置 | NOT_PROVEN | 禁止自动重连已实现；厂家支持的现场恢复程序仍缺 |
| Q9 完整900 s并发可信RT | NOT_PROVEN | 没有执行实机A/B，也没有以短fake测试替代 |
| Q10 保留失败证据 | PARTIALLY_SUPPORTED | 错误/缺口/锁存/未完成记录；磁盘故障不能声称绝对无丢失 |

## 11. 最终状态

```text
WRENCH_PRODUCTION_ISOLATION_IMPLEMENTATION_V1 = COMPLETE_WITH_LIMITATIONS
PRODUCTION_WRENCH_NATIVE_PROCESS_ISOLATION = IMPLEMENTED
PARENT_NATIVE_WRENCH_CALLS = 0
LAST_GOOD_AND_STREAM_HEALTH_SEPARATED = true
QUERY_LIFECYCLE_AUDIT = COMPLETE
FAILURE_LATCH = IMPLEMENTED
FORCED_TERMINATION_DISTINCT_FROM_GRACEFUL = true
OFFLINE_BLOCKING_TEST = PASS
OFFLINE_NEVER_RETURNING_TEST = PASS
OFFLINE_263_TEST = PASS
OFFLINE_TRANSPORT_FAILURE_TEST = PASS
OFFLINE_SHUTDOWN_TEST = PASS
NATIVE_LIKE_GIL_BLOCK_TEST = PASS
EXISTING_PREFLIGHT_UNHEALTHY_REJECTION = PASS
STATIONARY_RUNNER_READY = false
R9_CLOSED = false
REFERENCE_RELEASE = NO_GO
ROBOT_CONNECTED = false
ROBOT_MOTION_EXECUTED = false
CONTROL_CODE_MODIFIED = false
SAFETY_THRESHOLDS_MODIFIED = false
```

IMPLEMENTED/COMPLETE 指软件候选和生命周期协议已接入，不意味着所有故障下证据绝对完整或硬件已获验证。当前原始 wrapper、motion/control、安全值和发布均保持不变。停止于离线实现与测试。
