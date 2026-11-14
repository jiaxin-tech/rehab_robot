# 假腿多轮测量离线回放

本入口把同一安装与机械工况下、已经完成上游校验的测量 CSV，按实验顺序组成现有六参数残差 GP Greedy 训练器可接受的 history。第一轮必须是参考轨迹；所有轮次的 E3、E2、峰值比都相对第一轮实测参考计算。它不连接设备，不执行下一条轨迹，也不修改现场配置或任何 reviewed 标记。

接口支持两种明确范围。`fit_only` 用已校验的完整测量轮次拟合模型，允许这些轨迹由独立实验方案人工选取；输出 policy_state_replayed=false、completed_rounds_only=true、total_attempt_budget_accounted=false，原策略动作仅保存在 diagnostic_action，action为null。

`strict_policy` 要求完整按序的attempts账本，并声明attempt_ledger_complete=true。程序从空前缀开始，用同一个Policy逐轮拟合和提出动作，核对历史candidate及role，恢复probe计数和停止状态。失败尝试消耗一次预算并立即停止本版本；不匹配策略、重复、超预算或停止后追加都会失败。成功回放输出policy_state_replayed=true、total_attempt_budget_accounted=true，并保留完整transcript。其action是离线算法proposal，仍不批准运动，也不替代真实采集与候选执行链。

## 输入

当前推荐使用仓内`REHAB_OFFLINE_V2`，无需个人C盘算法目录。先通过`python -m scripts.create_offline_training_template catalog.json --setup-id <工况编号> --setup-sha256 <安装定义SHA256> --output new_template.json`生成新模板。程序校验版本的静态SOURCE_LOCK，再绑定当前本机文件字节；不会修改旧现场配置、旧expected SHA或任何reviewed标记。

模板保留空history，字段沿用`scripts.train_robot_leg_round`：algorithm_root、algorithm_sha256、algorithm_dependencies、geometry_m、candidates、setup_id、setup_sha256、reference_candidate_id、reference_sha256，另含algorithm_version和来源锁SHA。历史模板也可以显式使用，但不匹配原绑定的源码必须先解决版本问题。

模板中的算法和依赖路径、候选路径相对**模板文件所在目录**解析；manifest 的模板路径与测量路径相对**manifest 所在目录**解析。绝对路径也支持。错误的历史机器路径会直接失败，不自动猜测替换。每个 SHA-256 必须与实际文件的字节一致。

manifest 格式如下；示例哈希需替换为对应文件的真实 SHA-256：

```json
{
  "schema_version": 1,
  "data_classification": "MEASURED_DATA",
  "training_template": {"path": "template.json", "sha256": "<模板文件SHA256>"},
  "rounds": [
    {
      "episode_id": "profile_a_round_01",
      "candidate_id": "<参考候选ID>",
      "measured_path": "round_01/measured.csv",
      "measured_sha256": "<测量CSV的SHA256>",
      "status": "CAPTURE_ONLY_COMPLETE",
      "data_classification": "MEASURED_DATA",
      "setup_id": "<与模板相同>",
      "setup_sha256": "<与模板相同>",
      "geometry_m": [0.42, 0.30],
      "reference_candidate_id": "<与模板相同>",
      "reference_sha256": "<与模板相同>"
    }
  ]
}
```

后续轮次按实际顺序追加，保持同一 setup、geometry、reference 身份。改变机械工况后另建参考和 manifest，不跨工况混入历史。`status` 允许 `CAPTURE_ONLY_COMPLETE`、`ROUND_COMPLETE`、`VALIDATED_COMPLETE`，表示输入方声明上游测量校验已完成；本工具不会替代标定、同步和现场有效性判断。合成接口演示的 manifest 及每轮必须统一写 `SYNTHETIC_INTERFACE_DEMO`。

上述schema_version=1的rounds默认fit_only。严格模式改为schema_version=2、replay_mode=strict_policy、attempt_ledger_complete=true，并把rounds改为attempts。每条尝试增加从1连续的attempt_index、由上一前缀实际提出的role（如REFERENCE、INFORMATION_PROBE、SEARCH）及outcome。COMPLETED记录保留测量字段；FAILED记录只保留身份、role、attempt_index和非空failure_reason，不提供测量或endpoints。不得把失败从账本删除后续跑，不得把人工选择的轨迹事后改称算法选择。历史不符合策略时使用fit-only如实建模。

测量CSV必须含time_s、phase、髋膝q/dq/ddq、tau_hip_nm、tau_knee_nm，全部有限；phase必须完整依次含flexion和extension。候选之间保持相同起点时间、总周期、样本数及phase标签，允许不同屈伸时间分配，所以不要求所有候选的完整timegrid等于参考。每轮测量只与它实际候选的timegrid核对：有command_time_s时用该列核对完整命令网格，没有时time_s本身须相同。禁止用计划角度替代实测角度。

MEASURED_DATA每个完整轮次还必须提供`endpoint_source={"path":"该轮measured_metrics.json","sha256":"<实际SHA256>"}`。该文件由新版aligned_measurement产生，包含完整原始采样率endpoint_trace、原CSV/dispatch/轨迹及分析/标定定义哈希；编排器保存measured.csv后追加measured_sha256，回放必须精确匹配本轮，不能配用另一回合的端点。回放会重新积分完整trace、重算峰值并核对声明数值和运动phase边界。两分支共享精确turn边界，同一工况所有轮次使用相同方法、analysis/calibration定义及首轮完整参考分母。原始采样数可因轮次不同而不同，不按计数要求相等。

401行模型CSV只用于拟合，可能遗漏采样点之间的短峰，不能据此验证真实峰值约束。只有明确标记为SYNTHETIC_INTERFACE_DEMO的接口数据允许用稀疏CSV计算演示端点，并在输出中标为SPARSE_INTERFACE_ONLY。

当前 learner 不接受重复 candidate；同轨重复性校准应单独处理。不同 episode 不能冒用同一候选当作新的搜索点。无效、缺字段、截断、身份或哈希不一致的输入会整体失败，不静默跳过。

## 运行

从仓库根目录运行，仅准备请求：

```powershell
.\.venv\Scripts\python.exe -m scripts.replay_robot_leg_history manifest.json --output-dir outputs/offline_history/new_run
```

显式增加 `--train` 才在隔离子进程运行原训练器：

```powershell
.\.venv\Scripts\python.exe -m scripts.replay_robot_leg_history manifest.json --output-dir outputs/offline_history/new_training_run --train
```

输出目录必须不存在。输出包括training_request.json、history_audit.json、replay_status.json；训练时另有training.log和model_update.json。固定参考指标与每轮原始分支RMS/峰值可在history_audit.json核对。严格模式另在模型输出保存逐前缀transcript、policy_roles、attempted/completed/failed计数和剩余预算；fit-only只保留诊断动作。所有模式均不构成运动许可。

本机历史现场配置中build_and_replay.py的期望SHA与当前已提交版本不一致，因此直接复制后会被拒绝。新仓内版本通过明确的SOURCE_LOCK和来源记录解决当前训练依赖，不改写旧现场期望值。当前合成测试固定使用仓内版本，不再依赖外部个人目录；真实数据使用哪一版本仍需依据实验记录确定，不能关闭哈希检查。
