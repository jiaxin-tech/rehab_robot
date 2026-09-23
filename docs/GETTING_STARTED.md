# 安装与核心回归

[返回文档索引](README.md)

所有命令均从仓库根目录运行。已有本地环境时，可跳过安装段，直接运行下方核心回归命令。

## 安装与离线回归

推荐 CPython 3.12 x64，与仓内 xCoreSDK 的 `cp312-win_amd64` 扩展一致。本机已在项目内配置 Python 3.12.14 和 `.venv`；可直接使用 `.venv\Scripts\python.exe`，无需激活环境或修改系统 PATH。`.tools/`、`.venv/`、`.cache/` 均为忽略的本地产物。

Windows 上使用已有 `uv` 重建同一环境（从仓库根目录运行）：

```powershell
$uvExe = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uvExe) { $uvExe = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
& $uvExe python install 3.12.14 --install-dir .tools/python --no-bin --no-registry --cache-dir .cache/uv
& $uvExe venv .venv --python .tools/python/cpython-3.12.14-windows-x86_64-none/python.exe --cache-dir .cache/uv
& $uvExe pip install --python .venv/Scripts/python.exe -r requirements-win-py312.lock.txt pip --cache-dir .cache/uv
```

若已自行安装 Python 3.12，可用 `py -3.12 -m venv .venv`，再用 `.venv\Scripts\python.exe -m pip install -r requirements-win-py312.lock.txt`。`requirements.txt` 声明通用依赖范围，`requirements-win-py312.lock.txt` 固定本次 Windows/Python 3.12 的直接及间接依赖；其中 MuJoCo 3.6.0 与既有仿真记录一致。NumPy 下界为 2.0，因为当前端点计算使用 `np.trapezoid`；`imageio-ffmpeg` 提供视频编码所需程序。

当前核心回归使用显式文件集合，避免将真机连接或历史大规模实验混入默认命令：

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -c pytest-core.ini -q --junitxml=.cache/core-regression.xml
```

[pytest-core.ini](../pytest-core.ini) 覆盖 V3/ROM、E0/E2 时序辨识与 EI、E3、新 MyoLeg benchmark、测量分析、日志、轨迹预检/执行器 fake、离线进程诊断及 Windows SDK 加载/fake 测试；不包含 `test_rokae_hardware_integration.py`。从仓库根目录运行，因为 E3 测试读取已跟踪的相对路径 CSV/NPZ。完整历史套件另用 `python -m pytest`，不能把核心回归结果称为全仓通过。

2026-09-23 本机环境验证：Windows x64、CPython 3.12.14，使用上述锁定依赖，`pip check` 通过。环境配置后的原核心回归为 `333 passed in 62.70 s`，文档整理后复跑为 333 项通过、耗时 64.30 s。本轮新增 MyoLeg 实验、报告与缓存检查后，完整当前核心回归为 **404 passed in 99.72 s**，无跳过。新 JUnit 记录位于本地 `.cache/myoleg-core-regression.xml`，原记录在 `.cache/core-regression.xml`。核心结果验证离线算法与 fake/诊断代码，不代表真机运动验证或历史全仓回归。

若受限环境无法读取系统临时目录，给 pytest 加上 `--basetemp .cache/pytest-temp-<本次唯一名称> -o cache_dir=.cache/pytest-cache-<本次唯一名称>`，使用仓库内新建的独立目录。不要把已有研究数据目录传给 `--basetemp`，pytest 会清理该目录。

冻结产物按原始字节校验 SHA。[.gitattributes](../.gitattributes) 对核心依赖的冻结文件分别保留原始 LF 或 CRLF，避免 Windows `core.autocrlf` 改写字节后触发校验失败；此次仅还原原始换行，未修改冻结校验值或重生成实验结果。

核心回归不需要安装 MyoSuite 软件包。使用实际 MyoLeg 模型的集成检查与新 benchmark 需要下节的官方模型资产；缺资产时，相关资产集成测试会跳过。Windows 不提供 `resource` 时，回放模块的可选峰值内存统计返回 `null`，不会阻断 E3 导入或伪造内存值。这个兼容修复改变了历史 replay builder 的源码，原阶段针对源码校验值的测试需结合对应历史版本复现；本次不重写其冻结记录。

## MyoLeg 独立开发实验

先准备模型所需的官方 MyoSuite 2.12.2 资产。首次下载约 91 MB，仅提取冻结模型引用的 22 个 mesh/texture 资产，不安装 MyoSuite/Gym 运行环境；后续运行复用本机缓存：

```powershell
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.prepare_assets
```

已有官方 wheel 时，可追加 `--wheel D:\path\myosuite-2.12.2-py3-none-any.whl` 离线提取。命令校验固定版本 wheel 与提取内容，默认资产目录为 `.cache/myosuite-assets/2.12.2/myosuite/simhive/myo_sim/`。直接使用加载 API 时也可通过 `asset_root` 或 `MYOSUITE_ASSET_ROOT` 指向含 `scene/` 和 `meshes/` 的目录；运行 benchmark 推荐先执行上述准备命令。

先用 native 做小规模运行，再对该次完成产物生成报告。以下命令关闭噪声实验，比较一个轨迹族的三种方法；输出目录使用时间戳，避免覆盖已有运行：

```powershell
$myolegRun = Join-Path 'outputs/myoleg_benchmark_v1' ('native_smoke_' + (Get-Date -Format 'yyyyMMddTHHmmss'))
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.run --subjects native --families BETA_TIMING --methods REFERENCE PHYSICS_GREEDY MODEL_INFORMED_BO_EI --budgets 1 4 --random-seeds 1 --noise-levels --output-dir $myolegRun
.\.venv\Scripts\python.exe -m lower_limb_sim.myoleg_benchmark.report --output-dir $myolegRun
```

完整本轮开发设置使用 `python -m lower_limb_sim.myoleg_benchmark.run --subjects native development --workers 3`，默认比较两候选族和七算法，报告 K=1/2/4/8；无噪 Random 为 5 个种子，native 另做 1%/3% 噪声各 10 个配对种子。可用 `--help` 查看范围与默认值。未指定 `--output-dir` 时结果写入新的 `outputs/myoleg_benchmark_v1/<UTC时间>/`；完成后再用 `report --output-dir <该次目录>` 汇总。`completion.json` 表示本次运行完成，`protocol.json`、`results.csv`、`trial_history.csv` 和诊断文件描述实际执行；不能把部分输出视为最终算法排名。

新运行器仅允许 native 和冻结 development 主体，不读取 sealed held-out。其可移植加载器只在内存中重定位资产，原冻结 XML 和主体 delta 保持不变。Windows 换行差异仅在 LF 内容严格匹配原冻结校验值时接受；跨平台编译末位差异会明确记录，并通过公开参考轨迹的多项物理分量验证，不宣称 compiled model 逐字节相同。

旧 `myoleg_robot_scene.py` 及其视频 renderer 仍直接加载含原机器绝对路径的 XML，未接入新加载器。因此新 benchmark 运行成功不代表旧渲染命令也能直接运行。实验设计、证据边界及后续安排见[MyoLeg 研究与论文计划](research/MYOLEG_RESEARCH_AND_PAPER_PLAN.md)。

## 历史记录与机器人 SDK

旧文档中的 `667 passed, 5 skipped` 是 reference freeze 阶段的 macOS 历史记录，不代表当前代码或 Windows 环境的回归结果。

平台边界：

- macOS/Linux：主项目可 import，真实 SDK 测试自动 skip。
- Windows + Python 3.12 x64、无机器人：可运行 SDK import/fake 测试。
- Windows + SDK + 机器人：只有设置 `ROKAE_HARDWARE_TEST=1` 和 `ROKAE_TEST_IP` 后，才运行观察型 connection integration test；厂商 session 副作用仍需人工监督。

`xCoreSDK_python` 不是 pip 依赖。仓库实际运行副本在 `hardware/windows/xcoresdk/`，要求 Windows x64 + CPython 3.12；普通离线研究入口不会创建硬件会话。核心集合中的 SDK 测试只加载扩展并使用 fake robot，不连接真实机器人。
