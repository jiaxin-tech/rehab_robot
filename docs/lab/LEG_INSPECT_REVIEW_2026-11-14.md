# 17:03连续映像复核

回传提交`42ad8edc`，目录`results/manual_lab/20261114_170304_inspect`，运行源码为`3d59ed56`。全部输出、配置和源码哈希匹配；20帧的原始SHA、独立字节解码及汇总均已复核。原始五个文件未改，派生结果保存为同目录`review.json`。

## 观察结果

20次均返回1366字节，20帧各不相同。31个字节位置累计变化455次，toggle变化9次。每轴位置计数和力矩值均有变化；首读开始至末帧接收为155.477ms。

| 轴 | 20次状态字 | 20次603F错误码 | 3C1A力矩范围/N·m |
|---|---|---|---:|
| 1 | 0x0218 | 0x0022 | -12.23068至-11.86450 |
| 2 | 0x0231 | 0x0000 | -4.76044至-4.39426 |
| 3 | 0x0231 | 0x0000 | -1.03143至-0.79341 |
| 4 | 0x0231 | 0x0000 | 4.99847至5.19684 |

单次查询耗时0.8714至8.9851ms，中位数1.0169ms。这是约0.16秒的诊断观察，不是500Hz采集验收，也不能由toggle计算确切设备丢帧。

## 结论及边界

可以排除这段观察中“每次都返回同一幅静态映像”；未发现保存后解码、布尔位或十六进制格式化错误。按配置的第1轴字段，故障位和非零603F持续存在，并非孤立的一帧。

仍不能排除状态字段锁存、读取了错误对象，或当前在线布局与所用偏移不一致。原始映像在更新并不单独证明每个字段都来自预期物理驱动器；也不能仅凭TwinCAT主界面无错误就解释这些对象值。故障具体原因尚未确定，不能自动复位或放宽守卫。

旧电脑报告的`CurrentConfig.xml` SHA为`c5d10c97c6d4c131a3ff6871a509bc71a7a559dfff585a44dff70f89a2f5934c`，与10月16日历史记录相同。当前报告209425字节、`TcBootProject`、418个元素。文件原文未回传，通用提取器没有识别出设备/PDO结构；空列表不等于无设备。该文件哈希也不证明当前运行系统一定使用同一映射。

## 当前最有区分力的检查

以下在线对照已由操作者完成并反馈：6041为`0x0218`，603F为`0x0022`，与程序读取一致。证据记为`OPERATOR_REPORTED_MATCH`，见同目录数据中的`online_object_confirmation.json`；确认范围仅是轴1这两个对象，不扩大为四轴完整映射、力矩单位或故障原因已核实。当前未发现这两个值的程序解码问题，驱动厂家/型号及0x0022具体含义仍未知。

在TwinCAT的I/O设备树中选择与第1轴连接的具体驱动器，进入CoE-Online，只查看`0x6041:00`（Statusword）和`0x603F:00`（Error code），同时记录设备名称/厂家/型号。十进制或十六进制均可，保留实际显示的格式。

若同一对象与程序值不同，继续查在线对象和偏移；若相同，依据该厂家型号解释错误，而不是用TwinCAT全局状态代替驱动器诊断。全程不切换运行状态、不激活工程、不写对象、不复位、不使能或运动。

若找不到该页面，可以只复制下述两个已知配置文件原文供离线解析。此举只读文件，不证明在线映射；不存在的项目文件照实记录，不搜索其他目录：

```powershell
$legFiles = 'results/manual_lab/' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_twincat_files'
New-Item -ItemType Directory -Path $legFiles -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath 'C:\TwinCAT\3.1\Boot\CurrentConfig.xml' -Destination (Join-Path $legFiles 'CurrentConfig.xml') -ErrorAction Stop
$legProject = 'C:\Users\liumai\Documents\TcXaeShell\TwinCAT 机器腿\TwinCAT 机器腿\TwinCAT 机器腿.tsproj'
if (Test-Path -LiteralPath $legProject -PathType Leaf) {
    Copy-Item -LiteralPath $legProject -Destination (Join-Path $legFiles 'leg_project.tsproj') -ErrorAction Stop
}
Get-ChildItem -LiteralPath $legFiles -File | Get-FileHash -Algorithm SHA256
Write-Host "本次文件目录：$legFiles"
```

核对文件及暂存区后，只`git add -- $legFiles`，沿[原回传流程](../MANUAL_LAB_COMMANDS_2026-11-14.md)提交至`paper-personalized-load`。不要更改原配置文件或把复制的文件重新激活。

## 对比旧代码与继续诊断

`5d0d3f59`（2026年10月16日）新增的`collection/twincat_leg.py::capture`已经检查axis1/4的fault、非零603F、使能预期和力矩，`capture_leg_segment.py`委托同一采集器。后续到cleanup和`c2003d26`均保留这些条件，今天的正式采集报错不是新加了这项判断。

旧`6373f95c`中的`inspect_twincat_mapping_readonly.py::inspect`只检查读取长度、解码及变化，不按fault/error中断。它能完成仅表示读取/解码完成。10月16日和17日历史真实报告末帧轴1均为`0x0231 / fault=false / 0x0000`，所以过去记录本身也与今天不同；旧报告没有保存新版全部逐帧原始证据，不能同等强度复核其全部200读。

当前新增`inspect --live --duration-s 10`对应独立诊断用途：继续保存带有当前故障字段的十秒读数，同时明确标记`DIAGNOSTIC_ONLY`，不能进入正常采集/训练或授权运动。原probe/capture守卫没有更改。具体命令见[本地操作清单](../MANUAL_LAB_COMMANDS_2026-11-14.md)。

该扩展后完整离线核心集合525项通过，耗时87.34秒，无失败、错误或跳过；[验证报告](../validation/PAPER_BRANCH_TESTS_2026-11-14_DIAGNOSTIC10S.xml)保留为独立新记录。新增模拟检查覆盖十秒边界、帧数上限、故障字段持续记录、无效时长拒绝及诊断目录不能授权采集，未连接真实硬件。
