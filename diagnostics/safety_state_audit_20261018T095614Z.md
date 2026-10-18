# ROKAE safety-state read-only audit

- Timestamp UTC: `2026-10-18T09:56:14.756247+00:00`
- Robot: `XMC12-R1300-W7G3B4` / controller `3.2.1`
- SDK: `0.7.0`
- Robot IP / local RT IP: `192.168.50.103` / `192.168.50.209`
- Result: **BLOCKED WITH EVIDENCE**

No power, mode, recovery, reset, watcher-registration, collision configuration, register write, or motion API was called.

| Safety information | SDK API | Read-only evidence | IDLE callable | Result | Error |
|---|---|---|---:|---|---|
| robot/controller identity | `robotInfo` | high: bundled stub says query | yes | {"python_type": "Info", "id": "0f98879e-e936-47a2-9c0d-f7b748a7eb86", "type": "XMC12-R1300-W7G3B4", "version": "3.2.1", "joint_num": 6} |  |
| operation state | `operationState` | high: bundled stub says query current running state | yes | {"enum_type": "OperationState", "name": "idle", "value": 0, "string": "OperationState.idle"} |  |
| power / E-stop / safety-door state | `powerState` | high: bundled stub documents power/E-stop/safety-door getter | yes | {"enum_type": "PowerState", "name": "off", "value": 1, "string": "PowerState.off"} |  |
| manual/automatic mode | `operateMode` | high: bundled stub says query current operation mode | yes | {"enum_type": "OperateMode", "name": "automatic", "value": 1, "string": "OperateMode.automatic"} |  |
| collision event state | `queryEventInfo(safety)` | high: bundled stub explicitly documents active event query | no | unsupported/error | xCoreSDK queryEventInfo(safety) failed (259): 参数错误,参数类型或个数错误 |
| recent warning/error evidence | `queryControllerLog` | high: bundled stub and vendor example explicitly document log query | yes | [{"python_type": "LogInfo", "id": 35100, "timestamp": "2026-10-17 17:36:16", "content": "轴(Axis 1, Axis 2, Axis 3, Axis 4, Axis 5, Axis 6)驱动器直流母线欠压 (0x3220)", "repair": "1.检查机器人供电电压是否正常; 2.检查驱动器电容是否正常; 3.更换伺服驱动器"}, {"python_type": "LogInfo", "id": 10012, "timestamp": "2026-10-17 17:09:42", "content": "上电条件检查失败,不满足上电条件，原因：未开启或未支持无示教器模式，不执行上电操作.s", "repair": "1.切换到正确模式; 2.恢复急停状态;3.切换机器人至位置模式;4.检查伺服故障;5.等待控制器启动完成"}, {"python_type": "LogInfo", "id": 10014, "timestamp": "2026-10-17 17:09:36", "content": "开启拖动功能失败！急停模式无法打开拖动", "repair": "切换到手动模式，保持电机下电状态，位置模式，重新尝试；或者重启后尝试。"}, {"python_type": "LogInfo", "id": 10012, "timestamp": "2026-10-17 17:09:33", "content": "上电条件检查失败,不满足上电条件，原因：未开启或未支持无示教器模式，不执行上电操作.s", "repair": "1.切换到正确模式; 2.恢复急停状态;3.切换机器人至位置模式;4.检查伺服故障;5.等待控制器启动完成"}, {"python_type": "LogInfo", "id": 13013, "timestamp": "2026-10-17 17:09:10", "content": "急停触发", "repair": "手动恢复急停"}, {"python_type": "LogInfo", "id": 41447, "timestamp": "2026-10-17 17:06:56", "content": "开始拖动功能执行失败，力矩传感器与模型偏差较大：Axis2Error between sensor and dynamics model:58.980959,sensor_torque:-134.279980,torque command:-75.299022", "repair": "1.检查当前工具(Tool)设置是否与实际一致，设定的工具质量质心是否合理; 2.查看监控窗口，检查机器人坐标系、位姿是否与实际一致; 3.确认当前机型、RD参数与实际一致; 4.可尝试回到机械零位，对传感器清零后再拖动; 6.确认开启拖动时机器人没有受到外力; 7.其他详见《拖动故障排查手册》"}, {"python_type": "LogInfo", "id": 41447, "timestamp": "2026-10-17 17:06:55", "content": "开始拖动功能执行失败，力矩传感器与模型偏差较大：Axis2Error between sensor and dynamics model:59.172947,sensor_torque:-134.469993,torque command:-75.297046", "repair": "1.检查当前工具(Tool)设置是否与实际一致，设定的工具质量质心是否合理; 2.查看监控窗口，检查机器人坐标系、位姿是否与实际一致; 3.确认当前机型、RD参数与实际一致; 4.可尝试回到机械零位，对传感器清零后再拖动; 6.确认开启拖动时机器人没有受到外力; 7.其他详见《拖动故障排查手册》"}, {"python_type": "LogInfo", "id": 41447, "timestamp": "2026-10-17 17:06:54", "content": "开始拖动功能执行失败，力矩传感器与模型偏差较大：Axis2Error between sensor and dynamics model:58.030747,sensor_torque:-133.329917,torque command:-75.299170", "repair": "1.检查当前工具(Tool)设置是否与实际一致，设定的工具质量质心是否合理; 2.查看监控窗口，检查机器人坐标系、位姿是否与实际一致; 3.确认当前机型、RD参数与实际一致; 4.可尝试回到机械零位，对传感器清零后再拖动; 6.确认开启拖动时机器人没有受到外力; 7.其他详见《拖动故障排查手册》"}, {"python_type": "LogInfo", "id": 41447, "timestamp": "2026-10-17 17:06:27", "content": "开始拖动功能执行失败，力矩传感器与模型偏差较大：Axis2Error between sensor and dynamics model:41.500628,sensor_torque:-116.798819,torque command:-75.298191", "repair": "1.检查当前工具(Tool)设置是否与实际一致，设定的工具质量质心是否合理; 2.查看监控窗口，检查机器人坐标系、位姿是否与实际一致; 3.确认当前机型、RD参数与实际一致; 4.可尝试回到机械零位，对传感器清零后再拖动; 6.确认开启拖动时机器人没有受到外力; 7.其他详见《拖动故障排查手册》"}, {"python_type": "LogInfo", "id": 41447, "timestamp": "2026-10-17 17:06:24", "content": "开始拖动功能执行失败，力矩传感器与模型偏差较大：Axis2Error between sensor and dynamics model:58.982160,sensor_torque:-134.279980,torque command:-75.297820", "repair": "1.检查当前工具(Tool)设置是否与实际一致，设定的工具质量质心是否合理; 2.查看监控窗口，检查机器人坐标系、位姿是否与实际一致; 3.确认当前机型、RD参数与实际一致; 4.可尝试回到机械零位，对传感器清零后再拖动; 6.确认开启拖动时机器人没有受到外力; 7.其他详见《拖动故障排查手册》"}] |  |

## Collision query finding

The bundled stub declares `queryEventInfo(eventType: Event, ec: dict) -> dict`. The tested arguments were exactly `Event.safety` (Python `Event`, numeric value 1) and a Python `dict`; the documented result key is `EventInfoKey.Safety.Collided == 'collided'`.

The documented call form still returns SDK error 259 on this SDK/controller/model. No alternate enum/string/integer form was guessed and no watcher was registered.

## Pre-motion observation

`powerState` is the verified read-only E-stop/safety-door signal and `operationState` is the verified IDLE signal. No verified current collision/protective-stop getter is available, so they are insufficient as the complete pre-motion gate.
