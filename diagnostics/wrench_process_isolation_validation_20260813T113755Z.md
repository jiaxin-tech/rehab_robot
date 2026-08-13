# ROKAE wrench process-isolation validation

本报告只覆盖 powered-IDLE、automatic、stationary 条件下的严格只读诊断。没有调用运动、轨迹、power、mode、reset、clear fault、collision/safety/speed configuration，也没有运行 50 Hz。

## A. Architecture

最终选择 Option B；Option A 已被实测否定为最终方案。

```text
Supervisor process (no SDK object)
├── 100 Hz diagnostic/safety-side loop
├── bounded latest-state IPC readers
├── wrench age / stale calculation
├── RT age / stale calculation
└── worker heartbeat / hung decisions

RT process — Session A, exclusively owned
├── connectToRobot(robot_ip, local_rt_ip)
├── start/update/get RT state
├── publish pure-data latest RT snapshot + heartbeat
└── stopReceiveRobotState + disconnectFromRobot

Wrench process — Session B, exclusively owned
├── connectToRobot(robot_ip), no local RT IP
├── getEndTorque(world)
├── publish pure-data latest wrench snapshot + heartbeat
└── disconnectFromRobot
```

所有进程均使用 Windows `spawn`。SDK/robot/force-control 对象不经过 pickle、IPC 或跨进程继承。IPC 只传纯字典，wrench/state 通道为 bounded latest-snapshot，heartbeat 也是有界通道；supervisor 只执行 non-blocking drain。

本地审计结论：RT 的 `startReceiveRobotState/updateRobotState/getStateData` 数据路径不依赖 `getEndTorque()`，但当前 production wrapper 把二者放在同一个 native robot object/session 中。本地没有绕开 SDK 的 raw UDP listener 实现。README、CHANGELOG、`.pyi` 与 examples 均未承诺 multi-session。

Option A 的 30 秒验证虽然证明两条 session 可以共存，但 SDK RT producer 与 supervisor 同进程时，supervisor 只有 21.03 Hz；因此最终采用 Option B。没有修改 production `hardware/`、`control/`、`safety/` 或 trajectory 代码。

## B. Offline fault injection

离线九场景全部通过，`PROCESS_ISOLATION_OFFLINE_PASS = true`。

| Scenario | Parent Hz | P99 period | Stale | Hung | Result |
|---|---:|---:|---:|---:|---|
| normal 1 ms | 100.000 | 10.106 ms | startup 可见 | false | PASS |
| slow 40 ms | 100.000 | 10.895 ms | startup 可见 | false | PASS |
| stale 500 ms | 100.000 | 10.182 ms | true | false | PASS |
| long block 10 s | 99.771 | 10.156 ms | true | true | PASS；5.375 ms terminate |
| permanent block | 100.000 | 10.067 ms | true | true | PASS；4.281 ms terminate |
| synthetic SDK 263 | 100.000 | 10.067 ms | true | false | PASS；263 可见 |
| worker exception | 97.168 | 17.574 ms | true | false | PASS |
| worker crash | 100.000 | 10.038 ms | true | false | PASS；death 可见 |
| IPC saturation | 98.484 | 15.578 ms | true | false | PASS |

IPC saturation 中 worker sequence 达 583，而 parent 只有 79 ticks；没有无界 FIFO backlog，也没有 producer/consumer 因队列容量永久阻塞。强杀始终单独记录为 `graceful_disconnect=false`。

## C. Fresh SDK session

上一轮 263 storm 后的第一个 fresh process 成功连接、读取 `idle/on/automatic` 并正常 disconnect。

一次 prototype 启动门槛误判导致 child 被强杀后，新的 fresh process 再次正常连接/读取/disconnect。该误判源于把 Windows spawn/SDK import 的 `starting` 阶段错误套用 750 ms query hung 阈值；已修正为 startup/connecting 15 s 与 query 750 ms 两级阈值，该次无效 run 不作为 native-block 证据。

Option-B 自然 wrench block 被强杀后，最终 fresh PID 20432 在 428.137 ms 内再次完成连接、状态读取和正常 disconnect。

`FRESH_SDK_RECONNECT_PASS = true`。

## D. Multi-session

本地 vendor 文档未给出合同级保证，但实机验证支持当前组合：SDK 0.7.0、controller 3.2.1、XMC12-R1300-W7G3B4 可以同时存在两个只读 session。

30 秒 overlap 内：

- RT source period mean/P99/max：8.004/8.114/11.309 ms；
- RT sequence 增加 3,733；
- wrench sequence 增加 572；
- 未出现 SDK error；
- 先正常断开 RT Session A 后，wrench Session B 在 3 秒内继续增加 60 帧；
- 随后 Session B 正常 disconnect；两边 graceful disconnect 均确认。

因此 `MULTI_SESSION_SUPPORTED = true`，含义仅限本控制器/SDK/严格只读观察条件，不泛化为 vendor 对任意 session/命令组合的保证。正常“先断开 B 再观察 A”没有完成，因为 P3.5 中 B 自然阻塞后被强杀；不过强杀 B 后 A 明确继续运行。

## E. Live 20 Hz short validation

Option B 请求运行 180 秒，但在约 68 秒时自然出现 wrench worker hung，因此按协议提前终止并保存结果。

在故障前共观察到 wrench sequence 1,318。全 overlap：

- supervisor rate：99.384 Hz；
- main period mean/P95/P99/max：10.062/10.790/11.022/48.931 ms；
- RT sequence 增加 8,481；
- RT source period mean/P99/max：8.020/8.408/15.980 ms；
- wrench age max：753.130 ms；
- heartbeat age max：1,130.284 ms；
- 没有等到 SDK 263 返回，因为 supervisor 在 hung threshold 到达后终止了 worker。

这是自然故障，不是人为网络或机器人故障注入。

## F. Native block

从 wrench age 超过 150 ms 到 750 ms hung 判定的 0.600 s 窗口内：

- supervisor：99.927 Hz，P99/max period 10.786/10.815 ms；
- wrench sequence：保持 1,318，不再增长；
- wrench age：152.656 → 753.130 ms；
- stale：61/61 ticks；
- heartbeat age：最高 753.107 ms，hung 独立触发；
- RT sequence：增加 80；
- RT source period P99/max：8.154/8.163 ms。

因此被阻塞的 wrench process 没有冻结 supervisor，也没有冻结 RT source/RT process。wrench child 在 3.299 ms 内 terminate，未使用更强的 kill；其 SDK graceful disconnect 必须记录为 false。

终止 wrench child 后继续观察 3 秒：supervisor 99.984 Hz，RT sequence 增加 374，RT source period mean/P99/max 为 7.998/8.177/8.195 ms；RT process 随后正常 disconnect。

但是当前 RT process 内部仍是“SDK RT thread → process main loop → IPC”的两级发布。其 IPC age 存在明显长尾：全 overlap P99/max 为 201.162/367.933 ms；终止 wrench 后 300 ticks 中有 97 ticks 超过 50 ms diagnostic threshold。因此：

- `RT_SURVIVES_WRENCH_BLOCK = true` 指 RT source 与 RT acquisition process 持续推进；
- 它不表示当前 RT IPC snapshot 总能满足 freshness；
- 完整 live architecture 只能判 PARTIAL。

## G. Cleanup / reconnect

- Wrench blocked child：terminate 成功，3.299 ms，exit code -15，graceful SDK disconnect false。
- RT child：正常退出，109.063 ms，graceful SDK disconnect true。
- Host residual check：Python process 0；Python-owned local RT UDP endpoint 0；Python-owned TCP connection to robot 0。
- 强杀后 fresh SDK process：连接、状态读取、正常 disconnect 全部成功。
- 正常操作下 graceful disconnect：fresh sanity 与 multi-session 两侧均成功。
- native block 中的原 wrench session：graceful disconnect 不可能确认，答案为 false；只证明后续全新 session 可连接。

## H. Decision

十二个问题的回答：

1. RT 不需要与 wrench 共用同一个 SDK session；实机已证明 Session A/Session B 可分离。RT 仍需要自己的 SDK session，本地没有独立 raw UDP 实现。
2. 当前 controller/SDK 在本次严格只读条件下支持 simultaneous sessions；vendor 文档未提供通用合同保证。
3. 最终所有权：RT process 独占 Session A；wrench process 独占 Session B；supervisor 不拥有 SDK。
4. Blocked wrench process 不能冻结 supervisor；实测 block 窗口 99.927 Hz。
5. Blocked wrench process 没有冻结 RT source/process；RT sequence 在窗口内增加 80。但当前 IPC freshness 不合格。
6. Supervisor 可以独立检测 stale wrench；61/61 block-window ticks 为 stale。
7. Supervisor 可以独立检测 hung worker；heartbeat age 超过 750 ms 后触发。
8. Hung worker 可终止；实机 3.299 ms，离线 permanent block 4.281 ms。
9. Worker 强杀后可以建立 fresh SDK connection；最终检查 PASS。
10. 正常操作可以 graceful disconnect；fresh 与 multi-session 均证实。
11. Native block 中被强杀的 wrench session 不能 graceful disconnect，必须记录 false。
12. Process-isolated wrench 作为未来异步、时间戳传感器架构具有可行性，但目前仅 PARTIAL；RT IPC freshness 路径仍需重设计和重新只读验证。

最终字段：

```text
PROCESS_ISOLATION_OFFLINE_PASS = true
FRESH_SDK_RECONNECT_PASS = true
MULTI_SESSION_SUPPORTED = true
LIVE_PROCESS_ISOLATION_PASS = partial
SUPERVISOR_SURVIVES_WRENCH_BLOCK = true
RT_SURVIVES_WRENCH_BLOCK = true
HUNG_WORKER_TERMINATABLE = true
FRESH_RECONNECT_AFTER_WORKER_FAILURE = true
WRENCH_ASYNC_ARCHITECTURE_VIABLE = partial
READY_FOR_FIRST_MOTION_TEST = false
```

```text
thread architecture: rejected
process architecture: PARTIAL
```

下一步（本轮不执行）：把 RT process 改成单线程直接拥有 `startReceiveRobotState/updateRobotState/getStateData → IPC publish` 的路径，去掉同一 RT process 内部 thread 与 publisher 的 GIL 竞争；完成离线 jitter/stale 测试后，仅重做 3–5 分钟只读 Option-B 验证。P1 safety observability 的 SDK 259 仍未解决，继续阻止任何运动测试。
