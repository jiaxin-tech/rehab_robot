# WRENCH_NATIVE_BLOCKING_EVIDENCE_CLOSURE_PLAN_V1

2026-09-26。本文为工程与验证计划，未实施、未连接机器人。**R9 保持 OPEN；推荐评估独立 OS 进程隔离 wrench native 调用，但进程隔离不等于 SDK/控制器可靠性已经成立。**

## 1. 当前 production 故障模型

实际调用链：

```text
RealRobotAcquisition._wrench_loop
  -> RokaeRobotAdapter.read_internal_wrench
  -> hardware/windows/rokae_xcore.py:get_end_wrench
  -> forceControl().getEndTorque
```

`RealRobotAcquisition.start` 在生产宿主进程创建 state、wrench、alignment 三个 daemon 线程；adapter 启动的 RT 接收也在该进程。各线程使用同一 adapter/SDK session。acquisition 的 `_state_lock`、`_wrench_lock` 保护缓存；Windows wrapper 的 `_state_sdk_lock` 与 `_wrench_sdk_lock` 分离，不能据此证明 native 调用或 GIL 隔离。

Windows wrapper 在获取 wrench SDK lock **之前**记录 query start，native 返回后记录 query end，再调用 `_check_ec`。因此该计时包含等待 lock 的时间，不能无条件称为纯 native 时间。成功返回包含原始六轴 wrench、joint torque、query 起止及 host midpoint；没有机器人设备时间。adapter 仅在 wrapper 返回后递增 sample sequence，记录 publish time，并形成 `RobotWrenchFrame`；它不是包括失败请求在内的 query 序号。

| 调用结果 | 当前实际行为 | 证据缺口 |
|---|---|---|
| A 正常返回 | wrapper 检查 SDK code 和数据完整性；adapter 生成 frame；wrench loop 更新缓存、清除 `_wrench_error` 并写原始 wrench CSV | 成功样本不代表设备同步、绝对标定或长期可靠性 |
| B 长延迟后 263 | native 返回后 `_check_ec` 抛异常；wrapper 已计算的起止时间未随成功结果返回；loop 只写 `_wrench_error`，保留旧缓存，之后仍继续循环查询 | 缺完整逐失败 query 事件；旧有效 frame + alive 线程可能使 `latest_health.valid` 仍为 true；并非立即锁存失败 |
| C 超过响应预算仍不返回 | 无 completion、无该次错误返回；loop 无法观察 stop event；旧缓存时间不变，年龄应随宿主时钟增长 | 若宿主调度受 native/GIL 影响，年龄计算与监督决策也可能无法及时运行；没有 native cancel |

`latest_health` 计算 `now - sample midpoint` 与 state/wrench timestamp 差。它检查 frame validity 和线程存活，不比较 freshness 限值；已有有效缓存时 `_wrench_error` 不一定出现在 invalid_reason。正式 preflight 和 `RokaeMotionExecutor._runtime_reasons` 另行检查 age/skew 及 health，但这些 Python 检查的及时执行依赖调度。

`stop` 设置 event 并按现有 join timeout 等待线程；仍 alive 则记录 `acquisition_threads_did_not_stop`、signal logger failure，并拒绝在 native 可能仍活动时 SDK disconnect。正常路径先 stop_state_stream，再由 connection owner disconnect。超时 join 不会取消 native 调用；不能把拒绝 disconnect 记录成清理成功。

## 2. 历史实机序列重建

来源为 2026-08-14 Test A/B 原始 CSV、events 与 summary，**没有重跑**。

- Test A：RT-only，计划 900 s，完成 900.0082691 s；无 hung/crash，RT 正常退出和 graceful disconnect 确认。时间接受标准仍有 UNDEFINED，不能叫全部 PASS。
- Test B：20 Hz wrench + RT，计划 900 s，169.6096996 s 提前结束；2701 条 completed query，2698 成功、3 次连续未恢复 263。

下面相对时间统一以 **第一条 wrench call_start_ns=3198175313203100 为零**；不是整个测试的起点。

| query | start（s） | end（s） | 结果 |
|---|---:|---:|---|
| 2698 | 134.8497155 | 134.8510666 | 最后成功样本 |
| 2699 | 134.9003201 | 144.9008516 | 263，延迟 10000.5315 ms |
| 2700 | 144.9011838 | 154.9023654 | 263，延迟 10001.1816 ms |
| 2701 | 154.9024909 | 164.9036260 | 263，延迟 10001.1351 ms |

事件时钟（同一 host monotonic ns）：wrench hung 首次为 3198310922641900，第二次为 3198340969433100；RT hung 为 3198344929815200。**首次 hung 在第一个 263 返回之前已经被观察到**，不能按文字示例把全部 hung 排在全部错误之后。W1 对 wrench hung 计数后继续观察，在 RT hung 时抛 `RuntimeError:RT worker hung` 并终止。

最大 wrench age=34765.5455 ms；最大当前 state age=724.1383 ms。两个 worker 正常退出等待约 5 s 后均未退出，随后强杀成功，exitcode=-15、graceful_disconnect_confirmed=false；RT 强杀耗时 13.2995 ms，wrench 为 7.8304 ms。这些是观测值，不是拟议超时标准。最终 robot operation state=null。

已完成查询日志终止于 2701；不能由此断言之后绝无新查询尝试，更不能给没有 completion 的调用补一个完成时间。原型开始心跳不是完整持久化的 query-start 日志。

native blocking/263 与 RT hung **co-occurred**；跨 session、控制器、网络或 SDK 内部因果机制 **not established**。SDK 错误文本提及超时/可能网络问题，不足以诊断网络为唯一根因。operation_state_stability FAIL 来自末态缺失，不证明发生过非 idle 转移。

## 3. Production 与 diagnostic prototype 对照

| 项目 | 当前 production | diagnostic process prototype | 复用边界 |
|---|---|---|---|
| process isolation | 同宿主多个线程 | spawn 的独立 wrench OS process | 复用进程边界，不能声称控制器层隔离 |
| SDK session owner | production adapter 所在宿主 | child 自建并独占其 SDK session；RT 另有 session | 不传递/共享 SDK 对象；需要厂家多 session 约束 |
| heartbeat | thread alive、缓存年龄 | query 前后 heartbeat，starting/connecting 分开判定 | 进程 alive 不是 progress；无首个 heartbeat 也需 startup watchdog |
| query timestamps | 成功路径起止/midpoint；失败未完整导出 | completed event 有 start/end/error | 新方案必须持久化 started 事件并保留 pending |
| error output | `_wrench_error`，旧缓存可保留 | 独立 error/result event 与计数 | 原始 code/message 必须保留，不能只复用最新值 |
| latest transport | 锁保护本地 frame | maxsize=1 最新结果队列；有 last-success cache | 最新快照可以覆盖，不可覆盖审计事件 |
| hang detection | freshness 在后续 gate 检查 | heartbeat age/startup thresholds | prototype 默认值不是正式生产安全值 |
| forced termination | 无法安全杀线程 | terminate/kill 后 join，记录结果 | 强杀只证明进程结束；IPC 也可能损坏 |
| graceful disconnect | 无活动 producer 后清理 | child finally 调 disconnect 并发确认 | native 不返回则 finally 不保证执行 |
| RT coexistence | 同进程/session | 独立 RT 与 wrench session；短试部分支持，长 B 失败 | 不能直接把原型当生产已验证实现 |
| cleanup guarantee | 阻塞时拒绝 disconnect，显式失败 | bounded wait 后强杀，forced 标识 | 不复制自动重启；不把正常退出与控制器恢复合并 |

原型还有两个不可直接复制的语义：`poll.wrench_valid` 在存在 last-success 且 stale 未判 true 时可能为真，即使最新查询失败；`stale_age_ms=None` 不表示满足正式 freshness 要求。事件队列虽有 drop counter，但允许 drop 不能满足完整证据要求；队列 `join_thread`、父进程 drain 和 child shutdown 的阻塞风险也需离线覆盖。

### 证据等级

| 等级 | 当前结论 |
|---|---|
| PROVEN（源码限定） | 同进程线程实现；getEndTorque 无 timeout/cancel 参数；成功/错误传播顺序；runtime age 检查存在 |
| OBSERVED（历史配置限定） | native 长阻塞、263、旧样本年龄增长、诊断 supervisor 检测 hung、RT 共现 hung、强杀而非 graceful 清理 |
| NOT_PROVEN | production supervisor 在 native 阻塞时有界运行；完整并发 900 s 的可信 RT/wrench；进程隔离即可保护 controller/session；全部失败事件不丢失 |
| UNKNOWN | SDK 对 GIL 的确切行为、内部锁/网络超时机制、跨 session RT 故障原因、强杀后 controller session 的生命周期 |

## 4. R9 closure questions 与所需证据

“待证明”不等于 SUPPORTED；以下没有因写出计划而关闭任何问题。

| Q | 当前回答 | 后续关闭证据 |
|---|---|---|
| Q1 blocked getEndTorque 会否阻碍 RT/state？ | 生产隔离 NOT_SUPPORTED；跨 session 机制 UNKNOWN | native-like GIL 阻塞注入下 RT/state 继续推进；真实并发全程 RT 指标及厂家 session 说明 |
| Q2 supervisor 能否有界 fail closed？ | NOT_PROVEN | 独立时钟的检测/决策/发布时间及最坏观测延迟，满足预先审核预算；不是仅测平均频率 |
| Q3 不等 query 返回能否发现失鲜？ | 原型 OBSERVED，production 有界性未证 | pending query + last-good age 独立监督；never-returning 注入不依赖 completion |
| Q4 completed queries/errors 完整记录？ | production 有缺口 | starts/completions/error codes 逐 ID 对账；未完成 end=null；事件丢失明确失败 |
| Q5 能否隔离/停止而不损坏其他采集？ | prototype 部分 OBSERVED | 强杀、IPC 损坏及 logger 背压下父监督可退出，其他采集证据可读、无共享资源误释放 |
| Q6 强杀能保证什么？ | 仅已观察 OS process 结束 | PID/exit/join、IPC 隔离、日志状态明确；controller 状态仍 UNKNOWN，不能推断恢复 |
| Q7 故障下能 graceful disconnect？ | 历史否 | native 返回/可取消且 disconnect 调用成功确认的真实证据；没有证据继续 NOT_PROVEN/FAIL |
| Q8 不能 graceful 时如何开展下一 session？ | 缺已审核恢复程序 | 厂家支持的人工处置/检查步骤、操作者签核、独立新 session 的单独授权，保留原失败 |
| Q9 全时长 RT 可信？ | B 未完成，NOT_PROVEN | production-intended A/B 各 900 s、完整数据/末态/清理、冻结标准下审核 |
| Q10 所有故障证据可保留？ | production 不足 | 不清空 last-error、不替换 stale、不重启；持久事件、pending 查询和失败 trial 全部可追溯 |

厂家询证至少包括：当前 SDK 0.7.0/controller 3.2.1 组合的多只读 session 支持范围；getEndTorque 的 GIL/线程与并发约束、263 的定义和超时实现；是否有正式取消接口；同控制器不同 session 的相互影响；进程异常退出后的 session 清理及人工恢复程序。保存版本限定答复；不推测更换版本即可修复。

## 5. 最小生产隔离方案（仅提案）

保留 `RealRobotAcquisition` 为唯一生产采集入口，复用 adapter、原始 frame、EpisodeLogger 和现有 control/safety 接口。最小候选是：**只把 wrench native 查询及其 SDK session 放入专用 spawn OS process**；宿主保留 state/RT 接口、alignment、监督及日志。宿主 wrench consumer 只消费 IPC，不进入 native wrench 方法，不因 child 锁、queue drain 或 disconnect 阻塞监督循环。

```text
parent: RealRobotAcquisition / existing state-RT / supervisor / logger
  | immutable configuration + bounded protocol IPC
  v
child: exclusively owned read-only SDK session -> getEndTorque
  | query-start / completion / error / lifecycle
  v
parent: durable audit + latest-good cache + latched stream health
```

这隔离 wrench 的进程/GIL 堵塞，**不解决 RT 自己 native 阻塞或共享控制器故障**。若离线 native-like 测试或静止 B 表明宿主 RT native 会阻塞监督，则本候选验收失败，停止并升级设计审查：评估 RT session 也放独立进程、parent 完全不调用 native。不能在实现任务内偷偷扩大控制架构，也不能把尚未证明的最低方案称作充分方案。

### IPC 与所有权

1. child 只接收纯配置、run/session UUID 和 IPC；在自身创建/销毁 SDK，不 pickle adapter/SDK，不复用 parent session。父子必须绑定 robot identity、SDK/controller 版本与实际连接参数；不得自动改 mode、power、fault。
2. 分开 **latest-result transport** 与 **ordered audit events**。latest cache 允许覆盖；audit 具有单调 event_id/query_id、run UUID、PID，不能静默丢失。非阻塞收发和有限每轮 drain；溢出、序号断裂、协议错误锁存失败，保留缺口数量与范围。
3. query-start 必须在 native 进入之前由父进程持久记录并 ACK，child 未获 ACK 不得查询；ACK 等待也必须有界并可停止。记录 dispatch/start-intent、child native-enter 和父接收时间的不同语义，不声称它们完全相同。无法保全进入事件时标 `entry_unconfirmed`，不可伪造精确 native 开始。可用预调用同步发布槽缩小边界，但必须测试强杀后的完整性。
4. child 返回后发一次 completion（success 或 error），含 query_id、真实起止、code/message、raw wrench 或空值；父端记录 receive/durable time。completion 未 ACK 前不发下一查询；已返回但事件无法保全时，标 `completion_unknown/telemetry_failure` 并使试验失败，不能保证物理上不存在的绝对 crash-atomic 记录。
5. last-good 独立缓存。任何错误都不刷新 last-good 时间；任何读取/传输都不能把旧样本 timestamp 更新成 now。IPC 延迟单列，host midpoint 沿用现有定义并明确不是设备时间。
6. 子进程 progress heartbeat 在 lifecycle 与 query 前后更新；可辅助 liveness，但不能用独立心跳线程持续发包来掩盖 stuck query。parent 自身 watchdog 和 child progress、query age 分开。未知/倒退/不一致 clock 状态使 timing invalid；Windows 同机 perf_counter_ns 的跨进程可比性需先离线验证。

### 文件范围与最小改动

| 拟议范围 | 需要的改动 | 不应顺带修改 |
|---|---|---|
| `collection/` 新的 wrench process transport/provider | 提取原型可复用的 spawn、事件、监督、关闭逻辑；production 不直接依赖 diagnostic 脚本 | 不新造整套采集/控制框架 |
| `collection/real_robot_acquisition.py` | 注入/接入 process provider，保留单一入口；consumer、health、生命周期、latched failure；明确 Test A wrench-disabled 模式只限诊断 | 不改变 servo、轨迹执行或安全值 |
| `hardware/rokae_adapter.py`、Windows wrapper（若必要） | 仅观察型结构化异常/时间/SDK code 保存和 child session 所有权支持；复用原查询逻辑 | 不改 motion API、power/mode/fault、标定或力变换 |
| `collection/episode_logger.py` 或独立 episode sidecar | 增加版本化 query/lifecycle/health 事件；保留现有原始五文件兼容性、bounded failure signal | 不把日志 schema 破坏性覆盖 |
| stationary runner + tests | 使用同一 production provider；读取冻结诊断预算；生成完整证据 | 不改 release manifest 或自动读写安全审核位 |

兼容现有控制接口：现有 `wrench_thread_alive` 如保留，应仍表示 IPC consumer 线程的真实存活，新增 `PROCESS_ALIVE` 描述 child；不得把两者混用。`health.valid=false` 和 explicit reason 将 child failure 传给现有 preflight/executor，二者本次后续实现任务也保持代码不变；用 fake 证明它们确实拒绝。如果现有接口不足以无歧义表达，停止并提出小范围接口评审，不编造 alive 语义。

## 6. Health 语义合同

| 字段 | 定义与禁止行为 |
|---|---|
| LATEST_GOOD_SAMPLE | 最近一次成功的 immutable 原始 frame，sample_valid 是历史记录属性，不等于 stream_healthy；无样本为空 |
| CURRENT_QUERY_STATE | NOT_STARTED / STARTING / IDLE / START_INTENT / IN_PROGRESS / SUCCEEDED / ERROR / HUNG / EXITED / STOPPING；锁存 failure 独立于瞬时状态 |
| LAST_ERROR | 最近错误 code/message/query_id/time；即使后来成功也不得删审计历史或自动解除本次 failure latch |
| QUERY_IN_PROGRESS | 已开始且尚无匹配 completion；stop/强杀不把它转换成成功完成 |
| QUERY_START_TIME | 带语义标签的 host monotonic 开始记录；intent 与 confirmed-native-enter 分列 |
| LAST_COMPLETED_QUERY_TIME | 最近成功或失败的真实返回时间；不能用 heartbeat/exit time 代替 |
| WRENCH_SAMPLE_AGE | parent_now 减 last-good 的原始 sample midpoint；无样本/时钟异常为空并 unhealthy；不归零、不 clip 负值掩盖异常 |
| WORKER_HEARTBEAT_AGE | parent_now 减最近有效 child progress 时间；无 heartbeat 未初始化，不能以 alive=True 判健康 |
| PROCESS_ALIVE | OS process 存活事实；附 PID、exitcode、generation；不是 query progress |
| STREAM_HEALTH / FAILURE_LATCH | 样本、进程、query、时序、transport 和 logger 都满足已定义要求才可健康；最新 error/hung/exit 立即锁存；不能由 last-good 覆盖 |

健康聚合用保守 AND，不用“存在 last-good 就健康”。未定义要求明确 `UNDEFINED_REQUIREMENT`，不默认 PASS。错误后的记录继续保留，但本次试验不得恢复运行或自行重连；新试验需要独立授权和新 run ID。

## 7. Fail-closed 逻辑及待审核预算

这里规定事件意义，不指定新数值。实施 fake 测试可注入测试专用时间界限，必须标注非安全配置。

| 条件 | 仍可用的证据 | 未来执行模式所需行为（本任务无运动） | 必须记录 |
|---|---|---|---|
| wrench age 超限 | last-good 及 age、pending | health invalid；现有停止链必须触发，不能继续使用旧样本 | 原始样本 ID、age、阈值来源、首次检测与发布时刻 |
| query 时间超预算 | start/pending、last-good | 无需等 SDK 返回即可 fail closed | query_id、elapsed、completion=null、decision latency |
| heartbeat 超预算/首包缺失 | parent tick、PID、startup epoch | startup 与运行分开判定；失去监督即禁止继续 | last heartbeat、parent-observed age、state |
| skew 超限 | 原始 state/wrench 时间 | fail closed，不插值遮盖 | 两端时间、skew、时钟来源 |
| child exit/crash | OS exitcode、最后事件 | 立即 failure latch，不重启 | PID、exit/观察时间、未完成 ID |
| SDK error（包括263） | completion/error、此前 good | 本轮失败并停止新查询，不因旧缓存继续 | code/message、完整起止、exception provenance |
| RT/state freshness 丢失 | 最近 state、RT sequence、supervisor tick | 同样 fail closed；不能只盯 wrench | state age、sequence、native/IPC 时序 |
| 日志/IPC 丢失或阻塞 | 缺口计数、failure signal | 证据链不可靠即失败，不能通过放大 queue 隐藏 | drop/gap、日志故障、未持久化范围 |

实际物理停止是否成功必须将来在单独授权运动任务验证；stationary 测试只能证明 health/决策/接口传播及时，不能证明制动距离或机器人已停。

需要冻结：wrench/state age、skew、query 响应预算、startup/progress watchdog、supervisor decision budget、logger/IPC deadline、normal cleanup/termination budget，以及 RT interval/退化/错误可靠性判据。前三类现有安全字段可引用已审核值；新诊断预算需要单独版本化协议，不能偷偷写入 `experiment_safety.json`。依据应来自控制/监测误差预算、SDK 合同、系统调度与故障处置需求。缺依据统一 **THRESHOLD_NOT_YET_JUSTIFIED**；历史 10 s、750 ms、5 s 等不得自动升级为标准。

## 8. Cleanup / recovery 四种独立状态

| 状态 | 证明要求 | 不能推断 |
|---|---|---|
| NORMAL CLEANUP | 禁止新请求、pending 已处理、事件对账、日志落盘、IPC 关闭、所有进程 join 结果 | 仅 exitcode=0 不证明 SDK 已断开 |
| GRACEFUL SDK DISCONNECT | 所有权进程内 disconnect start/end、SDK result、确认事件持久化；分别记录 RT/wrench session | finally 代码存在不等于运行过；无 ACK 不默认成功 |
| FORCED WORKER TERMINATION | request_stop、超时、terminate/kill 原因和时刻、OS exit/join、IPC 隔离状态 | 不等于 graceful disconnect 或 controller/session recovery |
| CONTROLLER/SESSION RECOVERY | 厂家支持且现场审核的人工检查/处置记录；后续独立 session 授权、当前状态与身份验证 | 后续连接成功不追溯证明前次正常 disconnect |

关闭协议：锁存 trial failure → 停止发新请求 → 非阻塞收集已有事件 → 在审核预算内请求正常退出 → 若仍阻塞，按预声明诊断授权执行有界强杀并记 failed cleanup → 隔离可能损坏的 child IPC（不无限 drain/join）→ 独立完成其他证据收尾。若强杀也不能确认结束，保留 unresolved process，交现场处置，不循环强杀/自动重启。disconnect 本身也可能阻塞，必须处于受监督所有权边界中。

native 不返回时 **不能保证 graceful disconnect**。若发生强杀，normal/graceful gates 保留 FAIL/NOT_PROVEN，不能改成 warning。可单独支持“故障被隔离”的软件子结论；完整 R9 不能仅靠该子结论关闭。若业务需接受“隔离但无法 graceful”的模式，必须另立版本化故障运行政策、厂家处置依据及批准流程，不在本任务放宽现有 cleanup gate。

## 9. 确定性离线 fault-test 计划

只使用 fake/stub，无 SDK live 初始化、无网络。通过显式 barrier/event 控制进入、返回、ACK 与失败阶段；使用 parent 独立 monotonic trace。定时测试 bounds 是测试配置，不是安全标准。

| 场景 | 必须证明 |
|---|---|
| 10 s blocking call | query 开始可见，尚未返回无 completion；parent/模拟 RT 连续推进；到测试预算时锁存失败；即使稍后成功也不解除 |
| never-returning | 无 completion/end；独立监督检测并按测试预算收尾；强杀记录非 graceful；无 orphan |
| delayed SDK 263 | 一条精确 completion/error；保留 last-good 但 stream invalid；不自动发下一请求 |
| child crash | OS exit 检测，pending 不补完成；故障日志与未完成 ID 保留 |
| stale last-good | sample 本身可 valid，流必须 invalid；消费或 IPC 到达不得更新时间 |
| supervisor response | 从 injected event/预算越界到 health latch/既有 gate 可见的全部时刻；不靠平均 Hz 判通过 |
| shutdown | startup、query、ACK、completion、disconnect、queue flush 各阶段 stop；可返回正常路径与不能返回路径分开 |
| transport/logger 故障 | event overflow、丢序、duplicate/out-of-order、ACK 丢失、writer 卡住、强杀损坏 IPC；不无限等待、不静默漏记 |
| clock/startup | 未收到首 heartbeat、时间倒退/跨进程不一致、PID/run 错配；不得健康 |
| integration | fake acquisition health 经现有 preflight/executor 拒绝，确认未发任何 motion API；控制源码不改 |

`time.sleep` fake 会释放 GIL，只能证明协议行为。需补一个 **native-like 持有 GIL 的阻塞 stub**（例如明确验证调用语义的本地测试扩展），在 wrench child 内执行，以证明 parent 不依赖 child Python 调度。若环境无法提供此测试，标 NOT_PROVEN，不把 sleep 测试当 GIL 证据。即使通过，也不证明真实 SDK 多 session 或控制器可靠性。

## 10. 未来 stationary 验证协议

### 进入条件与固定项

仅在独立授权的静止诊断任务执行。先完成上述离线实现与验收，freeze implementation commit/dirty diff、源与二进制哈希、SDK/controller/机器人身份、OS/Python、网络/接口、tool/payload、采样参数、判据和现场操作者/急停/中止职责。所有连接必须考虑 SDK connect/disconnect 的 session 副作用；检查现有 stationary 许可，不把用户规划任务当硬件授权。

不改变供电/模式、不清故障、不触发运动、无人体/手动加载。进入状态必须符合预声明 stationary 协议，不能把历史 off/idle 当今天状态。缺现场信息、未审核停止预算或前次 session 未闭环则不启动。若仅科学性能比较标准未定义，可在已有安全诊断授权下输出描述性数据，但相关 gate 保持 UNDEFINED、R9 不关闭。

### TEST A / TEST B

- **A：RT/state only，900 s。** 使用 production-intended 的同一 RT/state 路径与监督/日志。wrench 明确 `DISABLED_FOR_STATIONARY_A`，不将其伪报健康；此模式不可用于运动或正式双流采集。
- **B：RT/state + production-intended wrench，900 s。** 使用将进入 `RealRobotAcquisition` 的相同 provider、session、IPC、health、logger 和 shutdown 代码，不以旧 diagnostic 原型代替。
- 既有 W1 参数为 RT 8 ms、supervisor 100 Hz、wrench 20 Hz；它们是历史 protocol settings，不是安全阈值。production acquisition 当前默认 wrench 50 Hz、state poll 250 Hz、alignment 50 Hz，必须明确不等同 RT source rate。A/B 前冻结实际 intended 配置。若保留 intended 50 Hz，20 Hz 对照不能关闭 50 Hz 可靠性问题，需额外独立 900 s 条件；若选择20 Hz用于production，必须显式评审用途，不得仅为通过试验静默降频。
- 首组顺序预声明 A→B；追加重复数/顺序在看新数据前冻结，不由结果自适应。A 失败则不自行进入 B；B 失败保留原 trial，禁止替换、重试或重连。安全中止允许早停，但 `completed_900s=false`，不得改目标时长使它 PASS。

### 必录字段和产物

所有流用 run/session UUID、PID、sequence/event/query ID 对齐；保存真实 clock source/host vs device 的区别。

| 流 | 必要内容 |
|---|---|
| RT/state | source/receive timestamp（无设备源时间须标注）、publish/delivery、sequence、状态/TCP/joints、有效性、state age、source interval、IPC latency |
| wrench | start intent/ACK/native-enter证据、completion/error、原始 Fx/Fy/Fz/Mx/My/Mz、sample midpoint、parent receive、last-good、query pending/elapsed |
| supervisor | 每轮 tick、loop interval/deadline、state/wrench age、skew、query/progress age、health latch/reason、检测与发布时刻 |
| lifecycle | process/session identity、heartbeat、connect/stop/disconnect 起止与 SDK 结果、PID exitcode、normal/forced outcome、IPC/log failure |
| end-state | 每个 session 的最后已知状态和新鲜度、最终 robot operation state 的查询时间/结果/来源；不可取得则 null + reason |

输出完整 raw episodes 与 query_events、health_timeline、lifecycle、case_summary、A/B comparison、协议快照和 SHA256 索引。汇总每一 planned/attempted/completed query、缺口、未完成查询、SDK codes、age/skew/latency 分布、RT持续性、supervisor决策延迟、900 s 完成情况、末态和各 session cleanup。host timestamps 不能冒称物理同步。

末态检查只能使用预审的仍可用 session/有界查询路径；若已全部强杀不得为补末态悄悄新建 session。拿不到末态保留失败，下一次独立授权诊断再获取当前状态。

### 判定

结构性 PASS 可依据精确不变量：无伪造 completion、错误原样保留、无 silent retry、序号对账完整、正常退出确有确认。时序/可靠性 PASS 还需预先冻结数值判据与足够完整记录。错误、hung、crash、丢日志、末态缺失或 cleanup 失败保留各自 FAIL；缺阈值保留 UNDEFINED。A 不启用 wrench 的项明确 N/A（历史 schema 若用 UNDEFINED 必须注明 intentionally off），不得计入 wrench 支持证据。

不因一次 A/B 成功声称所有故障概率为零；结论限定到配置、时长、重复数和测试范围。无真实 native 故障发生时，只能支持该次正常可靠性，不能伪称验证了真实故障 graceful disconnect。不得为产生故障擅自断网、拔线、改变机器人状态。

## 11. 关闭 R9 所需证据包

1. 可审阅的最小实现 diff、production 路径绑定及 source/config hashes，证明实际 native wrench 在独立进程，不是仅测试原型。
2. Q1–Q10 逐项 evidence matrix：raw source、配置、test ID、结论、限制；UNKNOWN/NOT_PROVEN 不自动提升 SUPPORTED。
3. 确定性离线 fault 注入及 GIL/native-like 对照，含延迟/死锁/丢事件/清理失败，证明软件检测与隔离。
4. 实际 intended 参数下完整 900 s A/B，时序依据、完整结束状态、原始失败记录与每个 session cleanup；只读硬件稳定性结论不外推到运动安全。
5. 厂家多 session/263/取消/强杀后处置依据；若真正硬件根因未解释，至少需要明确适用范围与故障控制证据，不能标 SDK 内因已修复。
6. 独立人工审核 closure decision。强杀隔离通过但 graceful 失败时，只能记录部分关闭子项；现有完整 R9 仍 OPEN，除非未来独立正式政策变更明确接受并验证替代处置。当前计划不修改该政策。

**R9 不是全部 30 项配置/发布消息的共同原因。关闭 R9 不自动 GO，R1–R8 仍需各自证据与审核；补全 R1–R8 也不能使未解决的 R9 可接受。** reference loader 当前仍固定要求 NO_GO，本计划不提供自动解除逻辑。

## 12. Copy-ready implementation task

```text
TASK: WRENCH_PRODUCTION_ISOLATION_IMPLEMENTATION_V1

Goal:
Implement only the minimum wrench-process acquisition candidate specified in
WRENCH_NATIVE_BLOCKING_EVIDENCE_CLOSURE_PLAN_V1.md, sections 5-9.

Scope:
- Keep RealRobotAcquisition as the only production acquisition entry.
- Add a spawn-owned read-only wrench SDK session/provider in collection/.
- Parent supervision/IPC consumer must never call native wrench methods.
- Reuse adapter/raw frames/EpisodeLogger; minimally add structured query,
  error, lifecycle and health evidence without breaking existing raw logs.
- Separate immutable last-good sample validity from current stream health.
- Implement query IDs, durable start acknowledgement, pending/completion
  distinction, error latch, bounded IPC and explicit telemetry-loss failure.
- Preserve actual consumer-thread liveness and separately expose child health.
- Implement bounded reviewed-budget shutdown semantics; forced termination
  must never report graceful SDK disconnect or controller recovery.
- Add a stationary A/B runner using these same production components;
  live operation must require an explicit separately authorized invocation.
- Run deterministic offline tests only, including 10 s, never-returning,
  delayed 263, crash, stale cache, IPC/logger failures, shutdown, and a
  native-like GIL-holding stub if supported. Mark untested claims NOT_PROVEN.
- Exercise existing preflight/executor refusal with fakes without editing
  control code or issuing motion APIs. Never enable A-only mode for motion.

Constraints:
No real robot/SDK connection, no hardware tests, no control/motion changes,
no safety threshold/release changes, no automatic restart/reconnect or commit.
Test-only budgets are not safety thresholds. No silent production-rate change.
If RT-native blocking prevents independent supervision or current interfaces
cannot represent health honestly, stop and report the required design review;
do not expand architecture or claim the minimum candidate is sufficient.

Deliver:
Implementation diff, offline tests and outputs, schema/ownership documentation,
Q1-Q10 evidence updates, and explicit remaining stationary validation blockers.
R9 remains OPEN; NO_GO remains unchanged. Stop after offline implementation.
```

## 13. Copy-ready stationary hardware task

```text
TASK: WRENCH_STATIONARY_RELIABILITY_VALIDATION_V2

Goal:
Evaluate the reviewed production-intended acquisition implementation using
the protocol in WRENCH_NATIVE_BLOCKING_EVIDENCE_CLOSURE_PLAN_V1.md, section 10.

Entry requirements:
Implementation/offline results reviewed; actual robot/session/setup identified;
existing stationary access permission satisfied; operator and stop procedure
present; intended rates/order/counts and diagnostic budgets frozen; no unresolved
previous-session recovery. This task must be separately explicitly authorized.
If a prerequisite is missing, prepare records and stop before connection.

Execution:
TEST A: production-intended RT/state only, 900 s.
TEST B: same RT/state + production-intended isolated wrench path, 900 s.
Freeze A/B order and any repetitions before examining outcomes.
Resolve historical 20 Hz versus production-default 50 Hz scope explicitly;
a 20 Hz run cannot validate intended 50 Hz use. Never silently lower rates.
Record all RT/query/heartbeat/health/decision/lifecycle clocks and sequences,
raw six-axis wrench, errors, stale/skew, pending calls, and each cleanup result.
Preserve 263 and all failed trials. Abort under reviewed failure criteria;
an early abort does not satisfy 900 s. No hidden retry or replacement trial.
Evaluate final operation state and graceful disconnect separately from exit.
After forced termination, no new session without separately reviewed procedure.

Constraints:
No robot motion, power/mode changes, fault clearing, manual loading, automatic
restart/reconnect, induced hardware fault, safety threshold changes, automatic
release update or commit. Use the production-intended path, not a substitute
prototype. Undefined formal criteria stay UNDEFINED, not arbitrary PASS.

Deliver:
Immutable raw episodes/event logs, source/config hashes, A/B summaries and
comparison, error/age/skew/latency/RT/supervisor evidence, end-state and session
cleanup outcomes, Q1-Q10 matrix and scoped R9 closure recommendation.
Do not automatically close R9 or change NO_GO. Stop after stationary analysis.
```

## 14. 最终状态

```text
WRENCH_NATIVE_BLOCKING_EVIDENCE_CLOSURE_PLAN_V1 = COMPLETE
CURRENT_PRODUCTION_NATIVE_ISOLATION = NOT_SUPPORTED
PROCESS_ISOLATION_RECOMMENDED = true
OFFLINE_FAULT_TESTS_POSSIBLE = true
STATIONARY_VALIDATION_REQUIRED = true
ROBOT_MOTION_REQUIRED = false
CURRENT_R9_STATUS = OPEN
R9_CLOSED_BY_THIS_TASK = false
SAFETY_THRESHOLDS_CHANGED = false
REFERENCE_RELEASE_CHANGED = false
ROBOT_CONTROL_CODE_CHANGED = false
ROBOT_CONNECTED = false
ROBOT_MOTION_EXECUTED = false
```

COMPLETE 仅表示本计划完成。硬件验证、实现和未知阈值没有在本任务完成。

## 15. 直接依据

- [根阻塞分析](SAFETY_GATE_ROOT_BLOCKER_ANALYSIS_V1.md)
- [生产采集](collection/real_robot_acquisition.py)、[adapter](hardware/rokae_adapter.py)、[Windows wrapper](hardware/windows/rokae_xcore.py)
- [进程隔离原型](scripts/wrench_process_isolation.py)、[W1 runner/gates](scripts/audit_state_wrench_timing.py)
- [preflight](control/execution_preflight.py)、[executor](control/robot_trajectory_executor.py)
- [Test A summary](diagnostics/state_wrench_timing_test_a_20260814T091310505282Z_summary.json)
- [Test B summary](diagnostics/state_wrench_timing_test_b_20260814T092935649185Z_summary.json)、[wrench CSV](diagnostics/state_wrench_timing_test_b_20260814T092935649185Z_wrench.csv)、[events](diagnostics/state_wrench_timing_test_b_20260814T092935649185Z_events.json)

核验范围：直接阅读上述现行源码，复核 A/B summary、B 最后成功与三次错误的原始时间；未执行诊断入口、未加载 live SDK。
