# 空载参考轨迹预检记录（2026-10-18）

## 本次物理范围

- 机器人：ROKAE XMC12-R1300-W7G3B4
- 机器人序列号：`0f98879e-e936-47a2-9c0d-f7b748a7eb86`
- 当前末端：法兰夹具保留，未安装假腿，无人体受试者
- 控制器 SDK 负载读回：质量 `1.085 kg`，质心 `[0, 0, 0.100] m`，惯量 `[0, 0, 0] kg·m²`
- 本次未发送上电、模式切换、使能或运动命令

## 只读状态

- SDK 连接成功，控制器版本 `3.2.1`，SDK `0.7.0`
- 状态：`IDLE`
- `operator_powered=false`
- 六轴软限位可读且与此前记录一致
- SDK 碰撞主动查询仍返回错误 259，不能证明当前“无碰撞”

## 当前起点

- 起点文件：`../results/empty_load_preflight_20261018_173808_265/anchor.json`
- 起点 ID：`empty_load_fixture_20261018_173808_265`
- TCP Base 位置：`[0.7378495541, -0.0431018139, 0.4358238542] m`
- 起点仅由只读状态流取得，尚未人工评审，因此 `reviewed=false`

## 原始 24 秒参考轨迹

- 文件：`../reference_release/reference_measured_asymmetric_closed_slow.csv`
- SHA-256：`f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881`
- 时长：`24 s`
- 当前起点下 TCP 峰值速度：约 `0.165419 m/s`
- 当前起点下 TCP 峰值加速度：约 `0.116580 m/s²`
- 发布清单仍明确记录 `approved_for_first_robot_trial=false`、`robot_execution_status=NO_GO`
- 结论：不得直接下发。

## 临时等几何降速副本

- 文件：`../results/reference_commissioning/formal_reference_temp_20261018/commissioning_reference.csv`
- SHA-256：`892da699b9afd0f57b8e77a5fa1c9b8da6069599515f7e2c24fec6b064b3b945`
- 时长：`132.335441 s`
- 当前起点下 TCP 峰值速度：`0.030000 m/s`
- 当前起点下 TCP 峰值加速度：约 `0.003834 m/s²`
- Base 工作空间包络：X `[0.211153, 0.738269] m`，Y `-0.043102 m`，Z `[0.434531, 0.529231] m`
- 包络位于已登记工作空间 X `[0, 1.000] m`、Y `[-0.150, 1.000] m`、Z `[0.100, 1.200] m` 内
- 预览：`../results/empty_load_preflight_20261018_173808_265/preview_retimed/`
- 几何、闭合、有限值和离线轨迹检查通过，但该副本仍为 `PREPARED_NOT_AUTHORIZED`，现有正式执行器也未绑定该 SHA-256。

## 结论

本次没有开始运动。阻断原因不是网络：机器人可连接。当前阻断为机器人未上电、起点未评审、碰撞状态不可观测、空载夹具负载身份尚未由控制器/HMI活动项核对，以及降速副本尚无有效空载发布批准。不得用原始 24 秒 `NO_GO` 轨迹或修改标志绕过这些条件。

## 待签认包

- 签认请求：`../results/empty_load_preflight_20261018_173808_265/review_request.json`
- 请求文件 SHA-256：`c38dd96f4cad8d6c00a0941dc8a90e8454f8d2940dc97e4ace0d472733d64676`
- 空载安全草稿：`../config/empty_load_commissioning_safety_20261018.json`
- 空载采集草稿：`../config/empty_load_live_observation_20261018.json`
- 签认前两份配置均保持 `reviewed=false`，不会形成运动授权。
