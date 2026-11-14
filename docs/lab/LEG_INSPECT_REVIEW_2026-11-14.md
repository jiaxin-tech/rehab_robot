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
