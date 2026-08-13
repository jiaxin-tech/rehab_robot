# ROKAE safety-state read-only audit

- Timestamp UTC: `2026-08-13T10:17:44.654380+00:00`
- Robot: `XMC12-R1300-W7G3B4` / controller `3.2.1`
- SDK: `0.7.0`
- Robot IP / local RT IP: `192.168.50.103` / `192.168.50.209`
- Result: **BLOCKED WITH EVIDENCE**

No power, mode, recovery, reset, watcher-registration, collision configuration, register write, or motion API was called.

| Safety information | SDK API | Read-only evidence | IDLE callable | Result | Error |
|---|---|---|---:|---|---|
| robot/controller identity | `robotInfo` | high: bundled stub says query | yes | {"python_type": "Info", "id": "0f98879e-e936-47a2-9c0d-f7b748a7eb86", "type": "XMC12-R1300-W7G3B4", "version": "3.2.1", "joint_num": 6} |  |
| operation state | `operationState` | high: bundled stub says query current running state | yes | {"enum_type": "OperationState", "name": "idle", "value": 0, "string": "OperationState.idle"} |  |
| power / E-stop / safety-door state | `powerState` | high: bundled stub documents power/E-stop/safety-door getter | yes | {"enum_type": "PowerState", "name": "on", "value": 0, "string": "PowerState.on"} |  |
| manual/automatic mode | `operateMode` | high: bundled stub says query current operation mode | yes | {"enum_type": "OperateMode", "name": "automatic", "value": 1, "string": "OperateMode.automatic"} |  |
| collision event state | `queryEventInfo(safety)` | high: bundled stub explicitly documents active event query | no | unsupported/error | xCoreSDK queryEventInfo(safety) failed (259): 参数错误,参数类型或个数错误 |
| recent warning/error evidence | `queryControllerLog` | high: bundled stub and vendor example explicitly document log query | yes | [{"python_type": "LogInfo", "id": 32001, "timestamp": "2026-08-12 16:15:46", "content": "上电失败, 具体原因:上电过程中中断上电(松开使能把手) ", "repair": "确认操作方式、检查伺服、清除伺服报警后再进行上电"}, {"python_type": "LogInfo", "id": 10012, "timestamp": "2026-08-12 16:13:41", "content": "上电条件检查失败,不满足上电条件，原因：控制器未启动完成，不执行上电操作", "repair": "1.切换到正确模式; 2.恢复急停状态;3.切换机器人至位置模式;4.检查伺服故障;5.等待控制器启动完成"}, {"python_type": "LogInfo", "id": 10011, "timestamp": "2026-08-08 15:30:25", "content": "切换至自动模式失败", "repair": "1.停止运动; 2.恢复急停状态; 3.关闭拖动"}, {"python_type": "LogInfo", "id": 10011, "timestamp": "2026-08-08 15:30:23", "content": "切换至自动模式失败", "repair": "1.停止运动; 2.恢复急停状态; 3.关闭拖动"}, {"python_type": "LogInfo", "id": 10012, "timestamp": "2026-08-08 15:30:19", "content": "上电条件检查失败,不满足上电条件，原因：未开启或未支持无示教器模式，不执行上电操作.s", "repair": "1.切换到正确模式; 2.恢复急停状态;3.切换机器人至位置模式;4.检查伺服故障;5.等待控制器启动完成"}, {"python_type": "LogInfo", "id": 10012, "timestamp": "2026-08-08 15:29:58", "content": "上电条件检查失败,不满足上电条件，原因：未开启或未支持无示教器模式，不执行上电操作.s", "repair": "1.切换到正确模式; 2.恢复急停状态;3.切换机器人至位置模式;4.检查伺服故障;5.等待控制器启动完成"}, {"python_type": "LogInfo", "id": 10014, "timestamp": "2026-08-08 15:29:42", "content": "开启拖动功能失败！上电失败！", "repair": "切换到手动模式，保持电机下电状态，位置模式，重新尝试；或者重启后尝试。"}, {"python_type": "LogInfo", "id": 13013, "timestamp": "2026-08-07 16:59:11", "content": "急停触发", "repair": "手动恢复急停"}, {"python_type": "LogInfo", "id": 10013, "timestamp": "2026-08-07 16:58:23", "content": "下电失败", "repair": "停止机器人运行"}, {"python_type": "LogInfo", "id": 13013, "timestamp": "2026-08-07 16:42:39", "content": "急停触发", "repair": "手动恢复急停"}] |  |

## Collision query finding

The bundled stub declares `queryEventInfo(eventType: Event, ec: dict) -> dict`. The tested arguments were exactly `Event.safety` (Python `Event`, numeric value 1) and a Python `dict`; the documented result key is `EventInfoKey.Safety.Collided == 'collided'`.

The documented call form still returns SDK error 259 on this SDK/controller/model. No alternate enum/string/integer form was guessed and no watcher was registered.

## Pre-motion observation

`powerState` is the verified read-only E-stop/safety-door signal and `operationState` is the verified IDLE signal. No verified current collision/protective-stop getter is available, so they are insufficient as the complete pre-motion gate.
