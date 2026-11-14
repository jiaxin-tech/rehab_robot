# 仓内版本化离线算法

当前入口为 `experiment_algorithm.v2`，版本名 `REHAB_OFFLINE_V2`。它从2026-09-29独立研究快照移入仓库，保留六参数公共刚体先验、残差GP、Greedy/EI公式和原有研究阈值，移除外部个人目录和机器盘符的导入。来源与原始SHA保存在 `v2/UPSTREAM_PROVENANCE.json`。原快照、旧结果和现场配置不被替换。

`v2/SOURCE_LOCK.json` 明确按LF规范化绑定源码内容，解决跨机器行尾表示问题；内容变化仍会失败。`binding.current_binding()` 先核对该静态版本锁，再为本机实际字节生成原训练边界继续检查的SHA。它不会自动接受代码内容漂移，也不会刷新历史现场配置的期望值。修改本版本科学实现后，应审查差异并发布新的来源锁，不应关闭检查。

从已有候选目录生成一份新的离线训练模板：

```powershell
.\.venv\Scripts\python.exe -m scripts.create_offline_training_template catalog.json --setup-id <同一机械工况编号> --setup-sha256 <安装定义SHA256> --output new_offline_template.json
```

目录JSON含 `reference_candidate_id` 和 `candidates`。每个候选保持既有 candidate_id、path、sha256、beta_flex、beta_extend、share_shift 字段。模板只用于离线研究，不批准真实运动，不填写任何现场 reviewed 标记。

多轮接口支持两种范围：

- `fit_only` 用已校验测量拟合模型，可容纳按独立实验方案执行的轨迹。原策略动作只作诊断，action为null，没有完整试验预算或历史策略复现主张。
- `strict_policy` 必须提供完整按序attempts账本，包括失败尝试。程序使用同一个Policy实例逐前缀拟合、恢复probe角色和停止状态，核对每次实际candidate及role确由前缀策略提出。失败消耗一次预算并终止本版本；停止后追加、历史动作不匹配、重复身份、超预算均失败。输出的算法proposal仍不是机器人运动许可。

严格模式不是事后把人工选择轨迹改称算法选择；若历史不符合原策略，就使用fit-only并如实报告实验方案。新的现场多轮执行仍需另外接入尝试账本和审核后的候选运动路径。

真实测量的峰值与RMS必须来自完整原始采样率的endpoint_source。401行模型输入无法排除采样点之间的短峰，只能用于拟合。所有回合使用一致端点方法及同一工况第一轮的完整实测参考分母。

公式中的六参数是有效机械模型参数；约束概率、辨识门槛与停止收益均为研究设置，尚非校准的临床安全或舒适度保证。软件正确运行也不证明个性化必要性、EI优越性或患者舒适度改善。
