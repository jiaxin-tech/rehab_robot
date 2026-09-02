# Safe remediation recommendation

## 当前阶段

`DIAGNOSIS ONLY`。本次没有实施任何修复。

不得用以下方式获得绿色测试：更新 expected SHA 以接受 CRLF、重生成旧 protocol、改写冻结 artifact、放松 byte-identity 测试、删除测试、skip 或 xfail。

## 最小安全顺序

### 1. 单独建立跨平台 byte-contract 迁移

在独立提交/阶段中设计 `.gitattributes`，目标是让 hash-pinned 文件在 Windows 与 macOS checkout 时都保持 canonical Git blob bytes。候选策略是：

- 对普通文本明确 `eol=lf`。
- 对需要逐字节固定的证据目录或精确文件清单使用不会发生 checkout 转换的属性策略；必须覆盖被 SHA 固定的 Python builder，而不只是 CSV/JSON。
- 在应用前扫描仓库中所有已有 CRLF blob、二进制文件及 intentional EOL，避免 `text=auto` 误分类。
- 不执行未经审计的 `git add --renormalize`，不重写历史 blob，不更新 expected SHA。
- 用全新 Windows 与 Mac worktree 验证：working-tree SHA 等于既有 expected SHA，且 `git diff` 为空。

仅修改 `core.autocrlf` 是机器本地规避，不足以构成仓库契约；仅添加 `.gitattributes` 但不验证 fresh checkout 也不足够。

### 2. 恢复缺失输入的可审计供给链

从原始权威来源恢复 5 个 ignored CSV/NPZ 路径族，并逐项验证已有 manifest SHA。不要重新生成或编造它们。随后选择明确的仓库策略：

- 小文件可跟踪；大文件可使用 LFS/content-addressed storage；或
- 提供有版本、固定 SHA、fail-closed 的只读获取流程。

历史 Mac worktree 中可能存在的 untracked 文件必须先盘点和 hash，不得假设与 Windows 或 manifest 相同。

### 3. 固化已验证的 Python 数据栈

当前 `pandas>=2.0` 允许安装 Pandas 3.0，而旧代码依赖可写 `to_numpy()` 结果。安全优先级：

1. 找回此前 full-suite 0-failed 环境的 lock/pip-freeze；
2. 在独立 clean worktree 验证该环境；
3. 再采用兼容上限/lockfile，或在非冻结代码中显式请求副本；
4. 若源码本身被 tree/source SHA 固定，代码修改必须作为显式版本迁移，不能同步更新旧期望值掩盖变化。

不要为了本报告随机 downgrade/upgrade。`scikit-learn>=1.4` 也应进入同一份经过验证的约束集合，而不是单独漂移。

### 4. 修复路径与编码的非科学兼容层

- 新生成的清单路径统一存 repository-relative POSIX path（`Path.as_posix()`），不要存机器绝对路径。
- 已冻结的 `/Users/...` metadata 保留原件；需要新 schema/version 的迁移 artifact 来表达相对路径，不要静默改旧证据。
- 所有源码/JSON/CSV 文本读取显式声明 `encoding="utf-8"`。

这些修改仍需独立测试提交，因为部分相关 Python 文件本身也被 source/tree hash 固定。

### 5. 集成门槛

按以下矩阵全部通过后，才重新评估 `FULLY_INTEGRATED`：

```text
fresh Windows checkout + locked environment + authoritative data: full pytest 0 failed
fresh macOS checkout + same lock/data manifests: full pytest 0 failed
all historical expected SHA unchanged
Git working trees clean after checkout and test
18 personalization tests pass
K=4 benchmark logical summary and canonical serialized SHA unchanged
PINN_TRAINING remains 0
```

## 风险最小的即时动作

当前最安全的即时动作是保留本诊断、不要触碰冻结 bytes，并把 EOL contract 作为下一项独立 repository migration。它能处理最大的一类（371 项），同时避免把平台 checkout 差异误当成新的科学证据。
