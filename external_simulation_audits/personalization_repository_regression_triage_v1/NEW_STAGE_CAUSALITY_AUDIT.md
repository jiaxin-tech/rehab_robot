# New-stage causality audit

## Verdict

```text
CAUSED_BY_NEW_STAGE: 0
NOT_CAUSED_BY_NEW_STAGE: 452
UNRESOLVED: 0
```

这是针对本次 full suite 的 269 failures 和 183 errors 的因果判定，不是对所有潜在未来风险的保证。

## Counterfactual baseline

在不改当前工作树的前提下，对同一 HEAD 创建 detached 临时 worktree。该基线不包含：

- `personalization/`
- `tests/test_physics_informed_sequential_personalization_v1.py`
- `MEASUREMENT_DRIVEN_PERSONALIZATION_ALGORITHM_V1.md`
- `requirements.txt` 中未提交的 `scikit-learn>=1.4`
- 其他当前未跟踪测试或用户工作文件

两边使用相同的 Windows、Python site-packages、Git system config 和 `core.autocrlf=true`。

| suite | tests | passed | skipped | failed | errors |
|---|---:|---:|---:|---:|---:|
| 当前工作区 | 1919 | 1466 | 1 | 269 | 183 |
| 干净 HEAD | 1895 | 1442 | 1 | 269 | 183 |

不仅汇总相同：452 个异常 node ID 集合完全相同，且每个节点的 `failure`/`error` 类型完全相同。当前工作区多出的 24 项全部通过，其中本阶段测试 18 项、此前未跟踪 timing-audit 测试 6 项。

## Import-time 与命名空间审计

静态搜索和 import smoke 未发现新包执行以下行为：

- 修改 `os.environ` 或 `sys.path`
- 设置全局 `random.seed` / `numpy.random.seed`
- 导入硬件、控制、安全、采集或 socket/network 接口
- 导入 `sklearn`
- 与现有顶层包发生命名冲突

`personalization.benchmarks.run_equal_budget` 显式选择 Matplotlib `Agg` backend，但 full suite 的新测试只导入 benchmark metrics，不导入 runner；该副作用没有进入旧测试收集链路。

## 依赖审计

```text
Python       3.12.10
NumPy        2.5.2
SciPy        1.18.0
Pandas       3.0.5
scikit-learn 1.9.0
pip check    No broken requirements found
```

新增 requirements 行是 `scikit-learn>=1.4`。实际安装时 NumPy 与 SciPy 已满足，没有因该命令升级；新算法使用自有 NumPy residual GP，并不导入 sklearn。旧 `lower_limb_sim/test_equal_budget_model_informed_bo_baseline.py` 在当前环境通过。

观察到的 dependency-related failure 是独立的 Pandas 3.0 行为变化：`to_numpy()` 返回只读视图，旧代码执行原地运算。这个问题在不含新阶段文件的 baseline 中逐项存在，不能归因于个性化实现或 sklearn requirements 行。

## 独立验证

```text
tests/test_physics_informed_sequential_personalization_v1.py: 18 passed
old BO + static validation logging + new personalization: 53 passed
personalization import smoke: passed
```

默认 benchmark 以 seeds 0–4、K=4、原敏感性配置在内存重算：

```text
primary rows: 1440
sensitivity rows: 144
logical benchmark_summary JSON equal: true
LF-normalized serialized bytes equal: true
stored normalized SHA-256: 16327ed2c30af8eec3145d1976d06cc42f955c32b305b3e9020f91beeeef3ac7
rerun serialized SHA-256: 16327ed2c30af8eec3145d1976d06cc42f955c32b305b3e9020f91beeeef3ac7
```

没有调 benchmark 参数，没有重写结果，也没有进行 PINN 训练。

## 集成边界

新阶段已达到 `MODULE_LEVEL_VALIDATED_WITH_REPOSITORY_CAUSALITY_CLEARED`：模块测试、旧 BO 兼容、导入隔离和 benchmark 确定性均成立，而且没有造成现有 full-suite 异常。

但仓库仍有 452 个可复现的既有/跨平台异常，因此总体状态不能升级为 `FULLY_INTEGRATED`。
