# 阶段 4.5B：变化延迟因果缓存与样本匹配审计

本审计只覆盖软件生成的虚拟状态流和二维 wrench 流。它验证的是变化延迟、
抖动、长尾、dropout 和 stale freeze 条件下的时间戳处理、历史缓存和
fail-closed 样本匹配，不是对真实患者、真实传感器或真实机器人控制周期的
验证。

人体模型继续严格使用：

```text
theta_shank = q_hip - q_knee
```

缓存、匹配和延迟跟踪始终保存独立的 `q_hip`、`q_knee`，没有把模型改成
`q_hip + q_knee`。

## 1. 三类时间戳及正延迟定义

阶段 4.5B 区分：

```text
state_timestamp_s
wrench_arrival_timestamp_s
wrench_sample_timestamp_s
```

- `state_timestamp_s`：q、dq、ddq 所代表的状态源时刻；
- `wrench_arrival_timestamp_s`：wrench 已经可供算法读取的到达时刻；
- `wrench_sample_timestamp_s`：wrench 实际代表的物理采样时刻。虚拟数据
  中它模拟可靠设备时间戳；
- `wrench_timestamp_s`：为兼容既有数据保留的到达时刻字段，不能把它
  静默解释成 sample timestamp。

正延迟定义为：

```text
delay =
    wrench_arrival_timestamp_s
    - wrench_sample_timestamp_s
```

所有时间戳使用同一个轨迹局部单调仿真时钟，单位为秒，不是 wall-clock。
可靠 sample timestamp 存在时，匹配器优先使用它；缺失时才使用：

```text
target_state_timestamp_s =
    wrench_arrival_timestamp_s - estimated_delay_s
```

生成器保存的 `true_delay_s` 只用于最终软件评价。延迟跟踪器的观测白名单
不得包含 `true_delay_s`、`wrench_age_s`、场景名、subject id 或其他真值
提示。

## 2. 严格因果的状态历史缓存

`StateHistoryBuffer` 默认只保留最近 `2.0 s` 状态，时间戳必须有限且严格
递增。新样本到达后，早于当前时刻减去缓存长度的状态会被裁剪。

查询支持：

- `nearest`：选择已缓存的最近状态；等距时稳定地选择较早样本；
- `linear_interpolation`：仅在两个已经缓存的状态之间插值。

两种查询都不向缓存未来外推。目标早于缓存起点返回
`state_history_expired`，目标晚于缓存最新状态返回
`state_history_future_query`，无可用 bracket 返回
`state_history_no_bracket`。线性插值左右状态间隔不得超过 `20 ms`；
nearest 的绝对时间误差以及匹配器最终接受的状态误差也分别受门限约束，
其中匹配器的状态匹配误差门限为 `5 ms`。

“已经在 buffer 中”本身仍不足以证明因果。调用端必须只把到达时刻以前的
状态 append 到缓存。匹配器进一步拒绝：

- 缓存最新状态晚于 wrench 到达时刻；
- 匹配状态时间晚于 wrench 到达时刻；
- wrench 到达时间晚于当前处理时刻；
- 目标状态时间晚于 wrench 到达时刻。

因此所谓 `causal` 是由数据可用时刻、缓存内容和查询门控共同保证，而不是
仅靠方法名称保证。

## 3. 四种对齐方法的用途边界

阶段 4.5B 比较四种方法，它们不是四种都可用于在线控制：

| 方法 | 作用 | 信息边界 | 不能声称 |
|---|---|---|---|
| `row_index_alignment` | 同行状态与力直接配对的负面对照 | 忽略真实 sample timestamp 和变化延迟 | 不能作为有效延迟补偿或在线匹配 |
| `global_fixed_delay` | 用一个全局固定延迟校正整段记录 | 适合固定延迟离线基线，无法跟踪抖动、分段变化和漂移 | 不能代表逐样本变化延迟，也不能把全记录选择结果当作实时可用 |
| `causal_history_latest` | 将当前状态与当前已经到达的最新 wrench 直接配对 | 不读取当前时刻之后的数据，但仍把旧力配给当前运动 | 不能称为延迟补偿，也不能等价于可靠 sample timestamp 匹配 |
| `causal_buffered_matching` | 优先用可靠 sample timestamp，否则用最新估计延迟，并在状态缓存中匹配 | 只查询已经到达的历史状态；所有拒绝原因显式保存 | 只是首选的软件因果路径，尚未经过真实实时线程、时钟或硬件验证 |

`row_index_alignment` 和 `global_fixed_delay` 是比较基线。
`causal_history_latest` 用来隔离“只限制未来信息、但仍直接配对最新已到达
wrench”的效果。
`causal_buffered_matching` 才组合了 sample timestamp 优先级、历史状态
查询、age、stale、dropout 和时间误差门控。四种结果必须分别报告，不能
用一个离线方法的较低 RMSE 替代另一个方法的因果性证明。

## 4. 变化延迟与跟踪器边界

虚拟场景包括固定 16 ms、分段延迟、渐变漂移、低/中抖动、双峰分布、
50–105 ms 长尾、100–250 ms stale freeze、5% dropout 和组合场景。
固定随机种子只保证软件实验可复现，不代表真实系统具有相同分布。

`WindowedDelayTracker` 的集中配置为：

```text
window duration       = 2.0 s
update interval       = 0.5 s
search range          = -50 ... 80 ms
search step           = 1 ms
smoothing alpha       = 0.5
max change per update = 8 ms
minimum samples       = 25
```

1 ms 是候选网格分辨率，不是亚毫秒测量精度。非整数延迟只能落在邻近网格
候选；噪声、采样率、窗口长度和激励不足还会引入额外误差。搜索命中
`-50 ms` 或 `80 ms` 边界时必须标记 `search_boundary_hit`，不能把边界
点当作精确真值。

平滑和每次最多 8 ms 的变化限制可以抑制跳变，但也会增加跟踪突变延迟的
滞后；它们不是机器人安全滤波器。

## 5. 低激励和低置信度时保持

延迟不能在近静止窗口中可靠辨识。跟踪器依据 dq、ddq 计算 excitation
score；低于门限时：

```text
delay_update_valid = false
delay_update_reason = insufficient_excitation
delay_value_held = true
```

算法保持上一延迟值，不调用评分器，也不伪造新估计。有效样本不足、评分
不可用、有限候选不足、搜索曲线过平、低置信度或边界低置信度时同样
fail closed 并保持旧值。保持意味着“本窗口没有足够证据更新”，不意味着
旧延迟仍然正确。

tracker 只接受 `train` 或明确标记的 `online` 行；validation 和 test
不能进入窗口更新或候选选择。

## 6. dropout、冻结、长尾和长缺口拒绝

以下情况不进入有效匹配：

- Fx/Fz 非有限或 wrench 标记无效：
  `wrench_dropout_or_non_finite_force`；
- stale/frozen wrench：`stale_or_frozen_wrench`；
- 可靠 sample timestamp 已重复出现：
  `duplicate_wrench_sample_timestamp`；
- wrench age 超过 `100 ms`：`wrench_age_limit_exceeded`；
- 状态缓存过期、无 bracket、状态间隔超过 `20 ms` 或匹配误差超过
  `5 ms`；
- 到达、目标或匹配状态使用了未来信息。

stale freeze 中即使 Fx/Fz 数值有限，也不能把重复旧值当成新鲜测量。
dropout 不能用未来 wrench 回填；长缺口不能跨越门限插值。100–250 ms
冻结远大于 20 ms 状态插值门限，必须保持无效。长尾延迟超过 100 ms age
门限时也必须拒绝，而不是为了提高有效率放宽或删除审计标记。

长尾和抖动可能造成唯一样本按 sample timestamp 看起来“乱序到达”。只要
该 timestamp 从未成功匹配、样本未超龄且所需状态仍在缓存中，就允许它
迟到后匹配；不能把所有倒序一概当作 freeze。真正 freeze 依靠 stale 标记
和重复 sample timestamp 门控拒绝。

所有无效结果保留明确 reason，q、dq、ddq 不以零值伪装有效状态。

## 7. `offline_only` 与因果回放不是同一个结论

阶段 4.5A 的 `offline_only` 会使用目标时刻两侧样本，可能读取未来数据，
只能用于离线校正、参数辨识和参照分析，绝不能进入实时控制。

阶段 4.5B 的 causal buffer、matcher 和 tracker 在离线测试中按时间顺序
回放，并限制只读取当前时刻已经可用的数据。这可以验证算法信息边界，但
仍然不等于真实在线运行，因为当前实验没有验证：

- ROKAE SDK 与 wrench 设备时钟是否同源或正确同步；
- 实时线程调度、查询耗时、缓存锁竞争和 deadline miss；
- 丢包重连、进程暂停、设备重启和时间戳回绕；
- wrench 坐标系、符号、零偏、带宽和滤波群延迟；
- 真实患者动力学、主动肌力、束缚带滑移和机器人安全门控。

因此即使 `causal_buffered_matching` 在虚拟回放中优于其他方法，也不能
直接接入真实控制，更不能据此设定真实机器人安全阈值。

## 8. 虚拟数据和泄漏边界

变化延迟、抖动、freeze、dropout 和可靠 sample timestamp 都是软件合成。
`wrench_sample_timestamp_s` 在本阶段代表“虚拟设备可观测时间戳”，而
`true_delay_s` 只属于生成和最终评价侧。窗口评分输入经过白名单化并清除
DataFrame attrs；test/validation、subject id、真值延迟和场景标签不应
进入在线估计。

当前主实验始终优先使用可靠的虚拟 `wrench_sample_timestamp_s`，所以
`causal_buffered_matching` 接近零的力矩 RMSE 证明的是“可靠 sample
timestamp + 历史状态缓存”的配对能力，并不证明没有 sample timestamp 时
的 delay fallback 同样精确。fallback 使用窗口估计值，已通过独立单元测试；
真实 ROKAE 若没有硬件采样时间戳，仍需专门验证查询时刻代理、设备时钟、
调度抖动和跟踪误差，不能直接套用主实验的 RMSE。

本阶段结果只能表述为：

> 在当前二维虚拟下肢模型和软件时间轴上，因果历史缓存能够拒绝未来状态、
> 长缺口、dropout 与 stale wrench，并可在有足够激励时跟踪变化延迟。

不能表述为真实患者参数估计、真实机器人在线性能、临床有效性、舒适性或
安全验证。阶段 4.5B 没有修改真实机器人控制、采集、安全或硬件代码。

## 审计检查表

- [x] 保持 `theta_shank = q_hip - q_knee`；
- [x] 状态、wrench 到达和 wrench sample 时间戳含义分开；
- [x] 状态缓存有限、严格递增并裁剪过期历史；
- [x] nearest/linear 查询均不外推未来；
- [x] 四种方法的信息边界分别说明；
- [x] 低激励、低置信度和样本不足时保持旧值；
- [x] 延迟候选只报告 1 ms 网格精度；
- [x] dropout、stale、长缺口和超龄样本 fail closed；
- [x] `offline_only` 不可用于在线控制；
- [x] 虚拟数据与真实机器人验证边界明确。
