# 碰撞状态替代接口与验证入口

日期：2026-10-19。本轮仅修改离线工具并执行假SDK测试，没有连接机器人。

## 采用的接口

采用项目已有的collision_source=events模式，通过厂家setEventWatcher(Event.safety, callback, ec)接收collided布尔值，避开queryEventInfo(Event.safety, ec)主动查询路径。实现位于hardware/windows/rokae_xcore.py和hardware/safety_events.py，适配器、实验入口及执行器已有对应接入。

接口存在不代表这台控制器已经返回有效数据。此前现场注册成功但未收到初始事件，因此不能声称本轮已经消除了现场阻断。SDK没有在本地接口说明中保证初始快照或周期心跳；事件静默不能当作无碰撞。当前缓存会保持未知、过期和碰撞锁存的阻断行为，不把powerState、operationState、历史日志或普通外力阈值当作同等碰撞观测源。

## 新增工具

scripts/diagnose_collision_events.py连接后使用上述事件模式，周期读取本地回调缓存并保存JSON。它不调用主动安全查询，不选择实时控制模式，不上电，不复位，不改变碰撞保护配置，不发送运动目标。连接、状态订阅和断开仍是实际SDK会话操作，不宣称完全无控制器副作用。

记录包括：设备身份、状态序号、碰撞布尔值和锁存、接收时间、年龄、过期原因、异常及断连后失效状态。它采样的是缓存，不是完整原始回调日志，可能跳过两次采样之间的中间事件。碰撞锁存由已有监视器保留。

在实验电脑仓库目录使用（以下0.2秒仅为诊断观测值，不是新运动预算或签认）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.diagnose_collision_events --robot-ip 192.168.50.103 --local-ip 192.168.50.209 --duration-s 10 --max-age-s 0.2 --output diagnostics/collision_events_probe_new.json
```

使用实验电脑实际Python路径和网卡IP；输出文件必须不存在。此命令本轮没有执行。不要为了刷新事件触发真实碰撞，也不要直接运行包含保护开关或复位的厂家完整示例。

## 如何解读

- observed_event_count=0、initial_state_unknown：未取得状态，不能用于启动运动。
- observed_clear=true但final_status_valid=false：曾收到无碰撞消息，后来失效，不能当成持续有效。
- observed_collision=true：已收到碰撞证据；后续false不会自动清掉锁存。
- 最终状态有效且断连确认：仅说明该诊断窗口末端的观测有效，不证明整个132秒轨迹或训练期间持续有效。
- 退出码0只表示末端状态有效、未观测到碰撞且会话清理无报告错误；所有结果的execution_authorized始终为false。

## 接入发布的限制

现有正式实验入口支持--collision-source events和显式的--collision-event-max-age-s。但现场批准的空载259豁免发布绑定query来源，不能直接拿来运行events模式。需要先取得真实数据并确认是否能满足所需观察时长，再按既有发布流程绑定相同来源与参数。未修改默认来源、已签认配置或豁免范围。

如果事件始终无初始状态，或仅在状态变化时通知而无法满足当前新鲜度契约，该接口仍不能承担当前门禁；需要另一个经过验证的当前安全状态输入，不能默认false或通过重订阅清除故障。

## 验证

82项相关测试通过：新增诊断、既有安全事件生命周期、机器人适配器、实验入口及轨迹执行器。覆盖无初始状态、正常事件、过期、碰撞锁存、断连、注册失败、参数错误和防覆盖。未做硬件验证。
