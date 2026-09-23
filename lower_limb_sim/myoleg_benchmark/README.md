# MyoLeg 开发阶段算法实验

这是独立版本的 E3 实验入口：在固定 ROM、24 s 周期的 MyoLeg 轨迹上，比较七种方法在少量试验后推荐的轨迹质量。它保留历史 V3/E2 与 E3 产物，重新执行请求到的候选，不用历史候选真值表筛选或推荐。

研究问题、选优规则和 10 月底实验冻结计划见 [研究与论文计划](../../docs/research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md)。环境安装见 [入门指南](../../docs/GETTING_STARTED.md)。

2026-09-23 已完成首份完整开发比较，见 [结果与当前建议](../../outputs/myoleg_benchmark_v1/README.md)。主K=4下，关键姿态族的三种物理方法均改善E3约2.97%，当前工程默认选Physics Greedy；尚未证明EI增量收益或个体化必要性。

## 运行

在仓库根目录使用已安装锁定依赖的 Python 环境。首次准备模型资产：

```powershell
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.prepare_assets
```

此命令下载并校验官方 MyoSuite 2.12.2 wheel（约 91 MB），只提取冻结 XML 需要的 22 项资产及许可证至 `.cache/myosuite-assets/`，不安装 MyoSuite 依赖。离线时可通过 `--wheel <本地wheel路径>` 提供同一校验和的文件。模型加载器在内存中定位资产，原 XML 不变；已有资产也可用 `MYOSUITE_ASSET_ROOT` 指向 `myo_sim` 目录。

先运行 native 单模型：

```powershell
$env:PYTHONUTF8='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.run --subjects native --workers 3 --output-dir outputs/myoleg_benchmark_v1/native_new
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.report --output-dir outputs/myoleg_benchmark_v1/native_new
```

完整开发基线（24 个 development 主体加 native）：

```powershell
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.run --subjects native development --workers 3 --output-dir outputs/myoleg_benchmark_v1/development_new
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.report --output-dir outputs/myoleg_benchmark_v1/development_new
```

输出目录必须尚不存在，避免覆盖已有实验。默认同时比较两种轨迹族和七种方法：无噪时 Random 使用 5 个种子，确定性方法各一次；native 上的 1%/3% 合成力矩噪声各使用 10 个配对种子。报告 K=1/2/4/8，其中 K=4 是主预算，首次参考也计入预算。此配置是开发基线，不等同于论文计划中的完整 30 种子稳健性实验。

正式扩展配置使用 `--random-seeds 30 --noise-seeds 30 --noise-subjects all`；预定 E2 约束敏感性使用 `--tier 0` 或 `--tier 0.02`，各自另建输出目录。用 `--help` 查看筛选主体、轨迹族、算法和预算的选项。运行器只接受 native 与原 24 个 development 身份，拒绝 sealed 主体。

单独复现当前开发默认及参考对照（无噪K=4，不代替七算法公平比较）：

```powershell
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.run --subjects native development --families KEY_POSTURE_TIMING --methods REFERENCE PHYSICS_GREEDY --budgets 4 --random-seeds 1 --noise-levels --workers 3
```

## 实验定义与信息边界

| 项目 | 含义 |
|---|---|
| 轨迹族 | `BETA_TIMING`、`KEY_POSTURE_TIMING`；三维参数包含屈/伸形状与时间分配 |
| 算法 | Reference、Random、3D Space Filling、Physics Greedy、Residual GP Mean Greedy、Pure BO EI、Model-Informed BO EI |
| 主端点 | E3：髋/膝 × 屈/伸四个 RMS 相对同主体原参考的比值均值；越低越好，参考为 1 |
| 负荷约束 | E2（上述四比值最大值）≤1.01，关节峰值比≤1.10；这些是仿真比较限值 |
| 运动学预筛 | 速度峰值≤参考1.5倍、加速度峰值≤2倍；无真值负荷预筛 |
| 有效超限试验 | 消耗预算并用于模型拟合，不能成为最终推荐 |
| 推荐与评分 | 算法按已执行、观测合格的数据推荐；独立评估器随后计算推荐的真实 E3 与合格性 |
| 主比较损失 | 合格推荐用真实 E3；不合格用 `max(1,E3)`；失败/未完成/无推荐用1，失败另报。此惩罚不是实际执行的参考回退 |
| 噪声 | 对完整力矩采样加噪；相同主体/族/候选/种子跨算法共用实现，灰箱与端点来自同一带噪时序 |
| 模拟器含义 | 指定关节状态下的逆动力学所需广义驱动力矩；尚非前向闭环跟踪、真实束带力或临床疗效 |

五参数灰箱只拟合已执行轨迹。选点器不接收模拟器、缓存、候选全景、主体生成参数或 oracle。GP 使用固定三维 Matérn-5/2；噪声不确定度是对角近似，未建模参考分母共享造成的相关性。

```mermaid
flowchart LR
    K[固定 ROM 与运动学候选] --> L[算法选下一条轨迹]
    L -->|只请求一个候选| S[MyoLeg 私有模拟器与缓存]
    S --> N[配对力矩噪声]
    N --> O[已执行观测与合格标记]
    O --> L
    O --> R[算法给出最终推荐]
    R --> E[独立评估器核对推荐真值]
    E --> P[主体等权报告与来源清单]
```

评估器的真实 E3、真实合格性不回传给选点器；两条轨迹族及 native/development 分别汇总。

Windows 加载时保留 XML/delta 原始与 LF 规范化校验和；编译模型指纹可能因网格产生的浮点末位差异而不同。加载器保留两种指纹，并独立验证公开参考的 19 个物理量、约束数量和肌肉身份，在声明的容差内等价才运行；不把数值等价写成位级一致。

## 代码与产物

| 文件 | 职责 |
|---|---|
| `prepare_assets.py` / `portable_model.py` | 官方资产准备与冻结模型的可移植加载 |
| `simulation.py` | 主体范围、冻结源校验、私有模拟器和内容寻址缓存 |
| `experiment.py` | 运动学候选、配对观测、七算法、因果预算前缀和最终推荐 |
| `run.py` | 先保存协议，再执行任务；推荐后的独立评分和完成清单 |
| `report.py` | 只读实验结果生成汇总、主体配对区间与图表 |

每次运行保存 `protocol.json`、`results.csv`、`trial_history.csv`、`identification_diagnostics.json`、`simulation_provenance.json`。`results.csv` 同时保留推荐与参考的四个原始 RMS（Nm）。全部任务处理完后生成 `completion.json`；必须其中 `completed=true` 才算完整。基础设施异常立即写入 `job_errors.json`，其他可完成任务继续保存，不用伪造的力学结果填补缺失任务。

报告先检查完成状态、协议校验和、全部主体/方法/种子/预算身份集合，再生成 `REPORT.md`、三份汇总/比较表、两张图和 `analysis_provenance.json`。后者记录输入及报告源码校验和。先对每主体的种子求均值，再对主体等权；native 单列，不能把种子重复当作更多主体。配对区间使用1e-12数值容差避免把浮点舍入误当优势；这不是有实际意义的效果阈值。

`elapsed_s` 是完整最长预算运行的墙钟时间，含缓存/模拟成本；各预算前缀沿用同一值，不能用于算法速度排名。缓存位于 `.cache/myoleg-benchmark-v1/`，只减少重复模拟，不给算法额外观测。并发缓存通过同目录硬链接原子发布，支持本机 NTFS；目标已存在时验证内容一致后复用，不覆盖正在读取的文件。不支持硬链接的文件系统会明确报基础设施错误。原始端点、超限和失败记录始终保留。

测试已纳入核心回归：

```powershell
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q
```

首次开发比较的结果登记和可支持结论见 [研究与论文计划的当前状态](../../docs/research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md#1-当前交付与证据层次)。只有实验完成且主体、算法和种子覆盖完整后，才讨论开发阶段优选；区间不支持唯一优势时保留并列。
