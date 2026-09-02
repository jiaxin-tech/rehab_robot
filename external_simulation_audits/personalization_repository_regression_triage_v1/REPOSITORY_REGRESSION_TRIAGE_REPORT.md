# PHYSICS_INFORMED_PERSONALIZATION_REPOSITORY_REGRESSION_TRIAGE_V1

## 结论

`FULL_SUITE_FAILURE_ROOT_CAUSE_IDENTIFIED`

当前 269 个 failures 与 183 个 errors 可由 5 类根因完整解释。决定性的因果证据是：在同一台 Windows 主机、同一 Python 环境、同一 Git 配置下，从 `HEAD=3ea00104e8cd90f68146cd8db3c1ee8b5ef69c29` 创建不含任何未提交个性化文件的 detached 临时 worktree，仍得到完全相同的 269 failures、183 errors，而且 452 个异常测试节点及其 failure/error 类型逐项相同。因此，本阶段没有造成任何已观测到的 full-suite regression。

当前算法状态保持为：

`PHYSICS_INFORMED_SEQUENTIAL_PERSONALIZATION_V1_IMPLEMENTED_WITH_LIMITATIONS`

不得称为 `FULLY_INTEGRATED`。原因不是算法定向验证失败，而是仓库级跨平台、环境和数据前置条件尚未修复。

## 冻结的诊断范围

- 当前分支：`physics`
- HEAD：`3ea00104e8cd90f68146cd8db3c1ee8b5ef69c29`
- 合并父提交：Windows 父 `4284da5cf23a6d97f22fd3987e3534a49814fc2d`；Mac 父 `cc218f31176997e63406a5837f708d14dc1f045d`
- 当前工作区 full suite（重复两次）：`1919 tests = 1466 passed + 1 skipped + 269 failed + 183 errors`
- 干净 HEAD 基线：`1895 tests = 1442 passed + 1 skipped + 269 failed + 183 errors`
- 当前工作区多出的 24 个通过项：新个性化测试 18 项，以及此前已存在的未跟踪 timing-audit 测试 6 项
- 未修改冻结 artifact、expected SHA、旧 checksum、旧测试或算法参数；未标记 skip/xfail；未自动提交

## 全部异常分类

| FAILURE_CLASS | failed | errors | total | affected modules | 与新阶段有关？ |
|---|---:|---:|---:|---:|---|
| `FROZEN_BYTE_IDENTITY_AND_CHECKPOINT_CASCADE` | 234 | 137 | 371 | 62 | 否；干净 HEAD 完全复现 |
| `MISSING_IGNORED_LOCAL_INPUTS` | 8 | 41 | 49 | 8 | 否；文件不受 Git 跟踪且基线同样缺失 |
| `PANDAS3_READ_ONLY_NUMPY_VIEW` | 21 | 5 | 26 | 3 | 否；旧代码与 Pandas 3.0 Copy-on-Write 的兼容问题 |
| `CROSS_PLATFORM_PATH_SERIALIZATION` | 5 | 0 | 5 | 4 | 否；Windows 分隔符和冻结的 Mac 绝对路径 |
| `WINDOWS_DEFAULT_ENCODING_CP936` | 1 | 0 | 1 | 1 | 否；无显式 encoding 的旧测试读取 UTF-8 源码 |
| **合计** | **269** | **183** | **452** |  |  |

完整的聚类字段、首个样例和受影响模块见 `FULL_SUITE_FAILURE_CLASSIFICATION.csv`。

## 根因证据

### 1. 冻结 SHA 与检查点级联：371

系统 Git 配置为 `core.autocrlf=true`，仓库没有 `.gitattributes`，代表性受影响文件的属性均为 `text: unspecified, eol: unspecified`。`git ls-files --eol` 对 JSON、CSV、XML 和被 hash 固定的 Python builder 均报告 `i/lf w/crlf attr/`。

首个失败的最小例子：

```text
file: external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json
expected SHA-256: de27c80d3ca93cd299c016ccb5d80032a8af417a2d06b91e2a01e5f0b2680f9e
working-tree SHA-256: 1db5037c59abc656dad268867c733eb92794a9ea5962516898a098ca8d342ca8
Git blob SHA-256: de27c80d3ca93cd299c016ccb5d80032a8af417a2d06b91e2a01e5f0b2680f9e
Git blob ID: 889997afe8ba9971d05e6a5c67e13193aebdf8e6
working bytes / blob bytes: 10286 / 10014
working CRLF / blob LF: 272 / 272
LF-normalized working bytes == Git blob bytes: true
BOM: absent in both
final newline: present in both
```

对 6 个不同类型和大小的文件做了同样的 byte-level 检查；每个文件都满足：大小差正好等于 CRLF 数量，去除 CR 后与 Git blob 逐字节相同。直接 SHA 失败随后触发 release bundle、admissible-region、active-reference、provenance、tree hash 和多级 checkpoint 的 fail-closed 级联。详见 `FROZEN_SHA_BYTE_LEVEL_AUDIT.md`。

### 2. 缺失的 ignored 本地输入：49

5 个实际路径族在当前 Windows checkout 不存在，且在 HEAD、两个合并父提交中均未被跟踪；`.gitignore` 明确忽略它们：

- `lower_limb_sim/data/reference_trajectories/processed/reference_selected_cycle.csv`
- `lower_limb_sim/data/reference_candidates/reference_execution_versions.csv`
- `lower_limb_sim/data/reference_candidates/reference_local_identification_dataset.csv`
- `external_simulation/data/myoleg_v3_development_truth_landscape_v1/shards/MYOLEG_VP_001.npz`
- `external_simulation/data/myoleg_objective_and_musculoskeletal_heterogeneity_decision_audit_v1/development_replay_subset.npz`

这些是工作区/数据供给差异，不是提交内容差异。没有可访问的历史 Mac 物理工作区，因此不能假设 Mac 上是否仍保有这些 untracked 数据。

### 3. Pandas 3.0 只读 NumPy view：26

环境为 Python 3.12.10、NumPy 2.5.2、SciPy 1.18.0、Pandas 3.0.5。Pandas 3.0 Copy-on-Write 始终开启；`DataFrame.loc[..., ...].to_numpy(dtype=float)` 在当前环境返回 `writeable=False`。旧代码随后执行原地除法并报：

```text
ValueError: output array is read-only
```

最小测试节点：

```text
lower_limb_sim/test_decision_relevant_global_model_reliability.py::test_distance_to_support_uses_formal_grid_step_units
```

该问题在不含新阶段代码的干净基线中逐项复现。

### 4. 路径序列化：5

4 项将 `Path` 直接转成字符串，Windows 得到反斜杠，而测试/冻结清单要求 POSIX `/`；例如实际 `control\active_rom_gate.py:1`，期望 `control/active_rom_gate.py:1`。另 1 项冻结 metadata 保存了 `/Users/fengjiaxin/.../build_and_audit.py` 的 Mac 绝对路径，在 Windows 无法与当前仓库路径相等。

### 5. Windows 默认编码：1

Windows 默认编码是 CP936。旧测试对 UTF-8 Python 源文件调用未指定 encoding 的 `Path.read_text()`；CP936 错误解码了 Unicode en-dash，使合法 UTF-8 f-string 被 `ast.parse` 误报为单右花括号。显式 `encoding="utf-8"` 后同一源码可正常解析。

## 分支与 worktree 判断

`WORKTREE_DIVERGENCE_STATUS`：

```text
TRACKED_FROZEN_GIT_BLOBS_EQUAL_BETWEEN_HEAD_AND_MAC_PARENT;
WINDOWS_PHYSICAL_CHECKOUT_DIFFERS_FROM_CANONICAL_GIT_BLOBS_BY_CRLF;
MAC_PHYSICAL_WORKTREE_UNVERIFIED;
IGNORED_LOCAL_INPUTS_ABSENT_ON_WINDOWS_AND_CROSS_MACHINE_STATE_UNVERIFIED
```

6 个代表性冻结/被固定文件在 HEAD 和 Mac 父提交的 Git blob ID 完全相同。不能据此声称两台机器的物理 worktree 字节相同：当前 Windows 工作树已证明不相同，Mac 物理工作树不可访问。语义内容在 LF 规范化后相同，但 frozen SHA 的契约是 byte identity，因此当前 Windows checkout 不满足该契约。

## 新阶段与依赖因果性

- 干净 HEAD 的 452 个异常节点集合与当前工作区完全相同；failure/error 类型差异数为 0。
- 新包无 `sklearn`、机器人硬件、控制、socket、`os.environ`、`sys.path` 或全局 random seed 修改。
- `scikit-learn>=1.4` 只存在于当前未提交的 `requirements.txt` diff；实际安装为 scikit-learn 1.9.0。
- 安装记录显示 NumPy、SciPy 已满足要求；当前 `pip check` 为 `No broken requirements found`。Pandas 3.0.5 的旧代码兼容问题与 sklearn 不构成因果链。
- 旧 BO、静态日志和新个性化测试合计 `53 passed`；import smoke 通过。
- 新阶段默认 benchmark 未调参内存重算，逻辑 JSON 与现有摘要完全相同，LF 规范化的序列化字节也相同。

结论：`CAUSED_BY_NEW_STAGE=0`，`UNRESOLVED=0`（针对这 452 个异常节点）。

## 独立算法复验

```text
targeted personalization tests: 18 passed
old BO + static logging + personalization: 53 passed
import smoke: passed
benchmark primary rows: 1440
benchmark sensitivity rows: 144
benchmark logical JSON equal: true
benchmark LF-normalized serialized bytes equal: true
K: 4
PINN_NOT_JUSTIFIED: true
PINN_TRAINING: 0
```

现有 `benchmark_summary.json` 的物理 Windows 字节为 CRLF；内存重算器产生的 LF 规范序列与其去 CR 后完全相同。这是确定性复现，不是重新冻结。

## Q1–Q8

**Q1. 多少类根因解释全部 269 failures 和 183 errors？** 5 类，精确合计 452 个异常节点。

**Q2. frozen SHA mismatch 是 CRLF/LF 还是其他字节差异？** 对 6 个代表性直接根文件，明确且仅为 CRLF/LF；无 BOM 差异，无末尾换行差异，LF 规范化后逐字节等于 Git blob。其余大量 SHA/checkpoint 异常是这些直接不匹配的级联。

**Q3. Windows 与 Mac worktree 是否 scientifically byte-identical？** 不能这样声明。Git 中 HEAD 与 Mac 父提交的代表性 frozen blobs 相同；当前 Windows 物理 checkout 与这些 blob 不同；Mac 物理 worktree 和 untracked 数据状态不可验证。

**Q4. 个性化阶段是否造成 full-suite regression？** 没有已观测到的回归。干净 HEAD 基线的所有 452 个异常节点与当前工作区逐项相同。

**Q5. scikit-learn 依赖变更是否影响旧测试？** 未发现。旧 BO 测试通过，基线在相同已安装环境下复现全部异常；新个性化包也不导入 sklearn。当前环境仍有独立的 Pandas 3.0 兼容问题。

**Q6. K=4 benchmark 能否不变复现？** 能。默认种子和敏感性配置均未改变，摘要逻辑内容与 LF 规范化序列化字节均相同。

**Q7. 最小安全修复是什么？** 分阶段处理：先建立不改 Git blob 的显式 EOL/byte contract 并在全新 Windows/Mac checkout 验证；再恢复有权威来源和 SHA 的 ignored 输入；以历史已验证环境约束处理 Pandas 3；最后分别修复相对 POSIX 路径和显式 UTF-8 读取。不得更新 expected SHA 来匹配 CRLF。详见 `SAFE_REMEDIATION_RECOMMENDATION.md`。

**Q8. 当前实现是 repository-integrated 还是仅 module-level validated？** 仅 `MODULE_LEVEL_VALIDATED_WITH_REPOSITORY_CAUSALITY_CLEARED`。算法本身通过且未造成回归，但仓库 full suite 尚非绿色，因此不能称为 fully integrated。

## 临时 worktree 清理说明

Git worktree 注册已移除，checkout 内容已清理。Windows 拒绝访问由 pytest 创建的一个残留 `.pytest_cache` 目录，因此 `C:\Users\liumai\AppData\Local\Temp\rehab_robot_triage_baseline_3ea0010\.pytest_cache` 仍存在；它不再是 Git worktree，也不在仓库内。
