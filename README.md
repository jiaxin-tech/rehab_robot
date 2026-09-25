# 仰卧位髋膝康复研究

研究机器人通过小腿束带等效牵引点执行被动髋膝康复时，固定活动范围内的轨迹协调是否值得个体化，以及有限试验预算下如何利用测量和力学模型选择轨迹。

仓库包含离线算法、仿真与实验记录，以及独立的 ROKAE 诊断和门控执行代码。模型约定为 `theta_shank = q_hip - q_knee`；正式机器人参考 ROM 为髋 0–120°、膝 5–145°，个体化研究使用单独冻结的 subject-specific ROM。

## 从这里开始

| 想做什么 | 阅读入口 |
|---|---|
| 安装环境、运行测试 | [上手指南](docs/GETTING_STARTED.md) |
| 理解课题、当前结论和指标 | [研究主线](docs/RESEARCH.md) |
| 找算法、脚本、测试和数据流 | [代码地图](docs/CODE_MAP.md) |
| 配置 MyoLeg 资产、运行新算法比较 | [MyoLeg 入门](docs/GETTING_STARTED.md#myoleg-独立开发实验)、[研究与论文计划](docs/research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md) |
| 查看真机诊断、采集和执行契约 | [机器人指南](docs/ROBOT_OPERATIONS.md) |
| 区分数据、冻结产物、SDK 和缓存 | [产物管理](docs/ARTIFACTS.md) |
| 查阅所有专题、历史报告与视频 | [文档索引](docs/README.md) |

## 当前状态

| 部分 | 已有证据与边界 |
|---|---|
| 冻结 V3/E2 | 二维 625 点域；当前五腿仿真不支持有意义的协调个体化必要性，不能外推到真实患者。 |
| V2 模型引导搜索 | 已接通五参数时序辨识、E0/E2、残差 GP 与 EI，并有冻结五腿算法比较；默认总预算 4 含参考。 |
| 独立 E3 探索 | 三参数轨迹族；36/36 个模型/轨迹族/约束组合中，BO EI 与 Adaptive Greedy 最终已执行合格最佳 E3 相同。 |
| 新 MyoLeg 开发实验 | 已完成 native＋24 development、两轨迹族、七算法比较。主 K=4 下关键姿态族的 Physics Greedy、残差 Greedy、MI-EI 均改善 E3 约 2.97%；工程默认选较简单的 Physics Greedy，尚未证明个体化必要性。[实际报告](outputs/myoleg_benchmark_v1/development_20260923_r2/REPORT.md) |
| 真实测量与机器人 | 离线分析接口已有；生产采集仍用线程，原生并发阻塞尚未解决。真机运动保持 **NO-GO**，在线个体化未接入。 |
| CONTROLLED_ACTUATION_V3 | 固定辅助峰值的 27 候选 × 3 development 主体已完成。出现两种 oracle，但最佳公共辅助方案最大相对 regret 仅 0.003589%，暂不扩展或确认；V1 30-seed 仍暂停。[协议、结果与复现](docs/research/MYOLEG_CONTROLLED_ACTUATION_V3.md) |
| Resistance interaction V1 | 6 个 controlled development profile 已完成 null/positive pilot；null gate 0%，4 个 divergent profile 中 3 个有实用 regret，随后完成 4 方法固定预算比较。仅支持软件机制证据，尚未进入确认集或真实患者建模。[执行计划与边界](docs/research/MYOLEG_RESEARCH_EXECUTION_PLAN_V3.md) |

指标定义、报告先后和限制见[研究主线](docs/RESEARCH.md)。软件测试通过不代表真机运动放行。

## 快速验证

本机已配置 `.venv`（Python 3.12.14 x64）。尚未安装时先按[上手指南](docs/GETTING_STARTED.md)创建环境。

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q
```

当前核心回归为 **404 passed**，包含新增 MyoLeg benchmark、报告完整性与并发缓存检查。它覆盖当前离线研究、采集、fake SDK 和诊断代码；不连接机器人，也不是历史全仓套件。环境版本与实际验证记录见[上手指南](docs/GETTING_STARTED.md)。

新 MyoLeg 实验通过 `python -m lower_limb_sim.myoleg_benchmark.prepare_assets` 准备官方 MyoSuite 2.12.2 资产，再由可移植加载器在内存中定位资产；冻结 XML 和既有实验结果保持原样。旧视频 renderer 仍使用 XML 中的原机器绝对路径，不能据此认为旧渲染入口也已完成移植。

## 目录布局

| 目录 | 内容 |
|---|---|
| `docs/` | 当前指南、研究专题、历史报告、视频索引 |
| `personalization/` | 观测/辨识接口、候选域、GP、选择器与低预算搜索 |
| `lower_limb_sim/` | 下肢力学、轨迹、五参数估计、V3/E2、独立 E3 与新 MyoLeg 开发实验 |
| `measurement_validation/` | 真实测量有效性、重复性和轨迹敏感性分析 |
| `hardware/` | ROKAE 适配器、Windows 运行 SDK 和厂商原始发行包 |
| `collection/`、`control/`、`safety/` | 采集与日志、参考轨迹与执行、审核配置及门控 |
| `config/`、`utils/` | 协议和配置模板、时钟/日志/来源记录等公共工具 |
| `scripts/`、`tests/` | 操作/诊断入口、回归测试；部分研究测试位于各包内 |
| `external_simulation/` | MyoLeg 外部仿真及逐阶段审计生成器 |
| `outputs/`、`external_simulation_audits/` | 当前 E3 等结果、外部仿真冻结证据包 |
| `diagnostics/`、`reference_release/` | 真机诊断原始记录、按 SHA 固定的机器人参考发布包 |

部分结果与算法同目录，保留原路径供测试和 provenance 解析；详见[代码地图](docs/CODE_MAP.md)与[产物管理](docs/ARTIFACTS.md)。根目录的 `bone_return_3_leg.csv` 是冻结源数据；`CURRENT_ARCHITECTURE.md`、`REAL_ROBOT_EXPERIMENT.md` 是按原路径及内容校验的历史输入，因此保留原位。根目录其余专题文档已归入 `docs/`，参见[整理记录](docs/REPOSITORY_ORGANIZATION.md)。
