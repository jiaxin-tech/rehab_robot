"""Generate the offline-only P2 Revision V2 design analysis.

The runner reads frozen root-cause and convergence artifacts.  It does not run
P2, construct a formal personalization result, or import robot-side packages.
Historical virtual truth is used only for post-hoc counterfactual labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rehab_robot_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import OFFLINE_PERSONALIZATION_SEARCH_BOUNDS
from .decision_relevant_global_model_reliability import (
    GRID_HIP_STEP_DEG,
    GRID_KNEE_STEP_DEG,
    GRID_PHASE_STEP,
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_ID,
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    sha256_file,
    validate_active_reference_file,
)
from .mechanical_objective import (
    MECHANICAL_OBJECTIVE_VERSION,
    OBJECTIVE_EQUIVALENCE_TOLERANCE,
)
from .p2_revision_v2_design import (
    DESIGN_PROTOCOL_ID,
    DESIGN_STATUS,
    EXPLORATION_STOPPING_CANDIDATE_ID,
    LOCAL_PROTOCOL_ID,
    OFFLINE_METHOD_STATUS,
    P2_V2_IMPLEMENTATION_STATUS,
    build_exploration_stopping_counterfactual,
    build_exploration_value_components,
    build_global_vs_local_error_distribution,
    build_local_guard_counterfactual,
    build_local_pair_stratum_summary,
    build_local_uncertainty_candidates,
    build_retrospective_local_pair_errors,
    build_subject_specificity_gap,
    design_recommendation,
    local_decision_validation_protocol,
)
from .parameter_estimator import PARAMETER_NAMES
from .research_decision_guarded_sequential_personalization import (
    apply_research_decision_guard,
    policy_definitions,
    rank_exploration_frontier,
    select_exploit_candidate,
)
from .run_p2_revision_root_cause_audit import (
    DEFAULT_OUTPUT_DIRECTORY as ROOT_CAUSE_ARTIFACT_DIRECTORY,
    ESTIMATOR_SOURCE,
    GENERATOR_SOURCE,
    MECHANICAL_OBJECTIVE_SOURCE,
    POLICY_ARTIFACT_DIRECTORY,
    POLICY_CORE_SOURCE,
)
from .run_sequential_personalization_convergence_stopping_audit import (
    DEFAULT_OUTPUT_DIRECTORY as CONVERGENCE_ARTIFACT_DIRECTORY,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_revision_v2_design.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_revision_v2_design_analysis_v1"
)

CSV_FILENAMES = (
    "retrospective_local_decision_pair_errors.csv",
    "global_vs_local_pair_error_distribution.csv",
    "local_pair_stratum_summary.csv",
    "local_uncertainty_candidates.csv",
    "local_guard_counterfactual_detail.csv",
    "local_guard_counterfactual_summary.csv",
    "exploration_value_components.csv",
    "exploration_stopping_counterfactual.csv",
    "knee_stiff_exploration_design_audit.csv",
    "matched_late_exploration_design_audit.csv",
    "subject_specificity_gap.csv",
    "P2_REVISION_V2_DESIGN_DECISION_MATRIX.csv",
)
JSON_FILENAMES = (
    "LOCAL_DECISION_VALIDATION_PROTOCOL_V1.json",
    "P2_REVISION_V2_RECOMMENDATION.json",
)
REPORT_FILENAMES = (
    "P2_REVISION_V2_DESIGN_REPORT.md",
    "P2_REVISION_V2_IMPLEMENTATION_PROMPT.md",
    "DATA_LEAKAGE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "global_vs_local_pair_error_distribution.png",
    "local_guard_counterfactual_summary.png",
    "exploration_value_components.png",
    "exploration_stopping_counterfactual.png",
    "subject_specificity_gap.png",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    return completed.stdout.rstrip("\n")


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_error_distribution(table: pd.DataFrame, path: Path) -> None:
    labels = ["global\nidentification", "local\ndecision"]
    x = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for column, label, marker in (
        ("p95_e_delta_J", "P95", "o"),
        ("p99_e_delta_J", "P99", "s"),
        ("max_e_delta_J", "max", "^"),
    ):
        axis.plot(x, table[column], marker=marker, linewidth=2, label=label)
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set(ylabel="absolute decision error", title="Global vs retrospective local pair errors")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    _save(figure, path)


def _plot_guard(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.set_index("guard_id").loc[
        [
            "G0_CURRENT_GLOBAL_MAX",
            "G1_LOCAL_MAX_CANDIDATE",
            "G2_LOCAL_P95_CANDIDATE",
            "G2_LOCAL_P99_CANDIDATE",
        ]
    ]
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    width = 0.27
    for offset, column, label in (
        (-width, "missed_improvement_candidate_count", "missed improvement"),
        (0.0, "false_improvement_candidate_count", "false improvement"),
        (width, "conservative_stop_round_count", "conservative stop"),
    ):
        axis.bar(x + offset, ordered[column], width, label=label)
    axis.set_xticks(x, ["G0 global", "G1 local max", "G2 local P95", "G2 local P99"])
    axis.set(ylabel="retrospective count", title="Guard candidates: leave-one-case-out counterfactual")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(figure, path)


def _plot_exploration(components: pd.DataFrame, path: Path) -> None:
    totals = [
        int(components["MODEL_VALUE"].sum()),
        int(components["SUPPORT_VALUE"].sum()),
        int(components["DECISION_VALUE"].sum()),
    ]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(["MODEL_VALUE", "SUPPORT_VALUE", "DECISION_VALUE"], totals)
    axis.bar_label(bars)
    axis.set(ylabel="explore rounds", title="Value dimensions are not interchangeable")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_stopping(stopping: pd.DataFrame, path: Path) -> None:
    totals = stopping.groupby("stopping_candidate_id", sort=False)[
        "executed_trials_avoided"
    ].sum()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(["S0 current", "S1 first zero", "S2 two zeros"], totals.to_numpy())
    axis.bar_label(bars)
    axis.set(ylabel="historical trials avoided", title="Disabled exploration stopping candidates")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_specificity(table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(table["subject_id"], table["J_truth_regret"])
    axis.axhline(
        OBJECTIVE_EQUIVALENCE_TOLERANCE,
        color="red",
        linestyle="--",
        label="unchanged 0.005 equivalence",
    )
    axis.bar_label(bars, fmt="%.4f")
    axis.set(ylabel="post-hoc truth regret", title="Truth optimum vs P2-selected optimum")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(figure, path)


def _design_matrix(recommendation: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": "local_uncertainty_in_next_version",
                "recommendation": recommendation["local_uncertainty_next_version"],
                "freeze_now": False,
                "current_P2_modified": False,
            },
            {
                "question": "decision_value_aware_exploration_in_next_version",
                "recommendation": recommendation[
                    "decision_value_aware_exploration_next_version"
                ],
                "freeze_now": False,
                "current_P2_modified": False,
            },
            {
                "question": "modify_objective",
                "recommendation": "NO",
                "freeze_now": False,
                "current_P2_modified": False,
            },
            {
                "question": "expand_generator_search_direction",
                "recommendation": "NO_NOT_JUSTIFIED_BY_CURRENT_EVIDENCE",
                "freeze_now": False,
                "current_P2_modified": False,
            },
            {
                "question": "implement_P2_V2_research_prototype",
                "recommendation": P2_V2_IMPLEMENTATION_STATUS,
                "freeze_now": False,
                "current_P2_modified": False,
            },
        ]
    )


def _design_report(
    distribution: pd.DataFrame,
    strata: pd.DataFrame,
    candidates: pd.DataFrame,
    guard: pd.DataFrame,
    exploration: pd.DataFrame,
    stopping: pd.DataFrame,
    specificity: pd.DataFrame,
    recommendation: Mapping[str, Any],
) -> str:
    global_row = distribution.iloc[0]
    local_row = distribution.iloc[1]
    support_counts = strata.groupby("model_supported")["pair_count"].sum()
    observed_steps = sorted(
        {
            (str(row.changed_coordinate), float(row.absolute_coordinate_step))
            for row in strata.itertuples(index=False)
        }
    )
    pooled = candidates.loc[candidates["calibration_scope"].eq("POOLED_RETROSPECTIVE")].set_index("candidate_id")
    guard_rows = guard.set_index("guard_id")
    p95 = guard_rows.loc["G2_LOCAL_P95_CANDIDATE"]
    knee = exploration.loc[exploration["subject_id"].eq("knee_stiff")]
    matched_late = exploration.loc[
        exploration["subject_id"].isin(("baseline", "hip_stiff", "heavy_leg"))
        & exploration["iteration"].between(7, 13)
    ]
    stop_totals = stopping.groupby("stopping_candidate_id", sort=False)[
        [
            "executed_trials_avoided",
            "later_exploit_trials_in_frozen_history",
            "later_accepted_best_changes_in_frozen_history",
        ]
    ].sum()
    knee_gap = specificity.set_index("subject_id").loc["knee_stiff"]
    return f"""# P2 Revision V2 Design Report

## 结论与边界

- 协议：`{DESIGN_PROTOCOL_ID}`。
- 当前状态保持：`{DESIGN_STATUS}`、`{OFFLINE_METHOD_STATUS}`。
- 本任务只读取既有 convergence/root-cause 产物；没有执行 P2、formal personalization、人体或机器人实验。
- active reference、ROM、`theta_shank = q_hip - q_knee`、五参数模型、mechanical objective、generator bounds、0.005 equivalence tolerance 与 90% support gate 均未修改。
- 历史 truth 只标注反事实结果，不是 policy、拟合、正式 calibration 或 stopping feature 输入。

## Part A — `{LOCAL_PROTOCOL_ID}`

### Local decision pair

一个 local decision pair 必须由“当前 formal candidate 与一个候选”组成：两点都在既有 generator admissible parameter space；只允许 hip、knee、phase 中恰好一个坐标变化；有符号变化量必须等于该轮既有 trust-region step（1、1/2 或 1/4 层级）；禁止 clipping。定义完全处于 generator 参数空间，不使用欧氏物理距离，也没有发明任何物理距离阈值。

### Global 与 local 样本不是同一尺度

| pair class | n | mean | P95 | P99 | max | Pearson(pred,actual) | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| current global identification pair | {int(global_row['pair_instance_count'])} | {global_row['mean_e_delta_J']:.9g} | {global_row['p95_e_delta_J']:.9g} | {global_row['p99_e_delta_J']:.9g} | {global_row['max_e_delta_J']:.9g} | {global_row['pearson_delta_pred_vs_actual']:.6f} | {global_row['spearman_delta_pred_vs_actual']:.6f} |
| retrospective local decision pair | {int(local_row['pair_instance_count'])} | {local_row['mean_e_delta_J']:.9g} | {local_row['p95_e_delta_J']:.9g} | {local_row['p99_e_delta_J']:.9g} | {local_row['max_e_delta_J']:.9g} | {local_row['pearson_delta_pred_vs_actual']:.6f} | {local_row['spearman_delta_pred_vs_actual']:.6f} |

global 的 61 个实例是 identification excitation pair；local 的 341 个实例是历史 decision opportunities。后者更接近待决策尺度，但它是 post-hoc retrospective 样本，不是 designated validation，因此不能直接冻结阈值。

局部样本中 supported={int(support_counts.get(True, 0))}、unsupported={int(support_counts.get(False, 0))}；观察到的坐标步长只有 `{observed_steps}`，即当前历史只覆盖 initial trust step，没有 1/2 与 1/4 层级。pooled 候选保守地保留两种 support status，但未来 designated protocol 必须按 support status、坐标和全部既有 trust levels 预声明分层；不能从这 341 个样本推断跨层级不变性。

仅生成三个数值候选：local max `{pooled.loc['LOCAL_MAX_UNCERTAINTY_CANDIDATE', 'candidate_uncertainty_bound']:.12g}`、local P95 `{pooled.loc['LOCAL_P95_UNCERTAINTY_CANDIDATE', 'candidate_uncertainty_bound']:.12g}`、local P99 `{pooled.loc['LOCAL_P99_UNCERTAINTY_CANDIDATE', 'candidate_uncertainty_bound']:.12g}`。三者均为 `threshold_frozen=false`。

### Historical counterfactual（不修改真实 policy）

| guard | exploit candidates | missed improvements | false improvements | conservative-stop rounds |
|---|---:|---:|---:|---:|
| G0 current global replay | {int(guard_rows.loc['G0_CURRENT_GLOBAL_MAX','would_exploit_candidate_count'])} | {int(guard_rows.loc['G0_CURRENT_GLOBAL_MAX','missed_improvement_candidate_count'])} | {int(guard_rows.loc['G0_CURRENT_GLOBAL_MAX','false_improvement_candidate_count'])} | {int(guard_rows.loc['G0_CURRENT_GLOBAL_MAX','conservative_stop_round_count'])} |
| G1 local max candidate | {int(guard_rows.loc['G1_LOCAL_MAX_CANDIDATE','would_exploit_candidate_count'])} | {int(guard_rows.loc['G1_LOCAL_MAX_CANDIDATE','missed_improvement_candidate_count'])} | {int(guard_rows.loc['G1_LOCAL_MAX_CANDIDATE','false_improvement_candidate_count'])} | {int(guard_rows.loc['G1_LOCAL_MAX_CANDIDATE','conservative_stop_round_count'])} |
| G2 local P95 candidate | {int(p95['would_exploit_candidate_count'])} | {int(p95['missed_improvement_candidate_count'])} | {int(p95['false_improvement_candidate_count'])} | {int(p95['conservative_stop_round_count'])} |
| G2 local P99 candidate | {int(guard_rows.loc['G2_LOCAL_P99_CANDIDATE','would_exploit_candidate_count'])} | {int(guard_rows.loc['G2_LOCAL_P99_CANDIDATE','missed_improvement_candidate_count'])} | {int(guard_rows.loc['G2_LOCAL_P99_CANDIDATE','false_improvement_candidate_count'])} | {int(guard_rows.loc['G2_LOCAL_P99_CANDIDATE','conservative_stop_round_count'])} |

为了降低直接复用同一 case 的偏差，G1/G2 反事实对每个 case 使用 leave-one-case-out 候选 bound。P95 相对 G0 少 4 个 missed、false 不变为 0、conservative stop 少 4 轮；但这仍是同一历史研究体系内的回放，不能把“未观察到 false”解释为安全证明。local uncertainty 架构值得进入下一版设计，具体 max/P95/P99 均不应现在冻结；下一步需要预声明且与拟合、adaptation outcome、final held-out test 分离的 designated local validation。

## Part B — `{EXPLORATION_STOPPING_CANDIDATE_ID}`

三个价值严格分开：`MODEL_VALUE` 是参数、prediction map 或 validation uncertainty 的可观察变化；`SUPPORT_VALUE` 是新支持点；`DECISION_VALUE` 是新增 exploit eligibility 或在既有 0.005 规则下 best trajectory 改变。support 不是 decision value。

32 次 explore 中：MODEL_VALUE={int(exploration['MODEL_VALUE'].sum())}，SUPPORT_VALUE={int(exploration['SUPPORT_VALUE'].sum())}，DECISION_VALUE={int(exploration['DECISION_VALUE'].sum())}，纯 support 且 exact-zero decision/model value={int(exploration['exact_zero_decision_value_round'].sum())}。

- `knee_stiff` 的 {len(knee)} 次 explore 都增加 support，但参数、prediction map、validation uncertainty、exploit eligibility 和 best trajectory 均没有改变，所以 8 次都没有 decision value。
- baseline / hip_stiff / heavy_leg 的 Trial 7–13 共 {len(matched_late)} 次：仍然只是扩大 support；新增 exploit=0、best change=0、MODEL_VALUE=0。
- 禁用候选 S1 在历史上会少执行 {int(stop_totals.loc['S1_STOP_AFTER_FIRST_EXACT_ZERO_DECISION_VALUE_EXPLORE','executed_trials_avoided'])} 次，之后原历史 exploit={int(stop_totals.loc['S1_STOP_AFTER_FIRST_EXACT_ZERO_DECISION_VALUE_EXPLORE','later_exploit_trials_in_frozen_history'])}、accepted best change={int(stop_totals.loc['S1_STOP_AFTER_FIRST_EXACT_ZERO_DECISION_VALUE_EXPLORE','later_accepted_best_changes_in_frozen_history'])}。
- 禁用候选 S2 会少执行 {int(stop_totals.loc['S2_STOP_AFTER_TWO_CONSECUTIVE_EXACT_ZERO_DECISION_VALUE_EXPLORES','executed_trials_avoided'])} 次，之后原历史 exploit={int(stop_totals.loc['S2_STOP_AFTER_TWO_CONSECUTIVE_EXACT_ZERO_DECISION_VALUE_EXPLORES','later_exploit_trials_in_frozen_history'])}。

因此 decision-value-aware exploration 值得作为下一版 research candidate；S1/S2 只是结构候选，`candidate_enabled=false`、`threshold_frozen=false`，本任务不选择 stopping rule。

## Part C — Subject specificity

`subject_specificity_gap.csv` 分开记录 A objective、B generator direction、C search policy、D guard：

- baseline、hip_stiff、heavy_leg 的 P2 alpha 与 truth optimum 不同，但 truth regret 分别为 {specificity.set_index('subject_id').loc['baseline','J_truth_regret']:.6f}、{specificity.set_index('subject_id').loc['hip_stiff','J_truth_regret']:.6f}、{specificity.set_index('subject_id').loc['heavy_leg','J_truth_regret']:.6f}，都在不变的 0.005 equivalence 内。
- knee_stiff regret={knee_gap['J_truth_regret']:.6f}，是唯一超出 0.005 的 meaningful gap；在 P2 selected 点，历史一步最优 truth ΔJ={knee_gap['best_one_step_truth_delta_J_at_P2_selected_posthoc']:.9f}，单步改善幅度仍小于 0.005。主要成因是 local stepwise acceptance 无法累积多个 sub-threshold knee moves；不是 local uncertainty bound 单独造成。
- 当前 generator 已包含所有四个 observed truth optima，因此现有证据不支持扩大 search direction。全部 optimum 触及 knee=-5 边界只说明边界外未知，需要另行科学审查，不能据此扩界。
- objective 仍产生 subject-dependent complete optima；本任务禁止且不建议修改 objective。

## Part D — Revision recommendation

1. local uncertainty 值得进入下一版本吗？**值得进入架构设计，但不能冻结任何候选阈值。**
2. decision-value-aware exploration 值得进入下一版本吗？**值得，以 disabled research candidate 先实现并复核。**
3. 是否需要修改 objective？**不需要。**
4. 是否需要扩大 generator direction？**当前证据不支持。**
5. 最小修改集合：增加可审计的 local uncertainty provider（正式冻结前 G0 保持默认）；分开记录 MODEL/SUPPORT/DECISION；增加默认关闭的 decision-value stopping candidate；保留 reference、ROM、模型、objective、bounds、0.005 与 90% gate。

是否值得实现 P2 V2：`{recommendation['implementation_status']}`。实现必须是下一项单独授权的 research-prototype 任务，不能把本报告直接当作 formal policy freeze。
"""


def _implementation_prompt() -> str:
    return f"""# 下一步实现 Prompt：P2 V2 Research Prototype

前提：仅在导师/实验负责人审核并冻结 `{LOCAL_PROTOCOL_ID}` 的数据划分方案、候选选择原则和 `{EXPLORATION_STOPPING_CANDIDATE_ID}` 的研究开关后执行。

请实现一个 **research-only、default-off、可与当前 P2 完全对照** 的 P2 V2 prototype：

1. 不修改 active reference SHA、ROM_PROTOCOL_V2、`theta_shank = q_hip - q_knee`、五参数模型、mechanical objective、generator bounds、0.005 equivalence tolerance、90% support gate，且不触碰 hardware/control/collection/safety。
2. 保留 current G0 作为默认行为；新增 uncertainty-provider 接口，但不得把本次 retrospective max/P95/P99 直接写成正式阈值。
3. 建立预声明的 designated local-decision validation split：只包含既有 generator admissible points 与既有 trust-region 单坐标 pair；与 model fitting、adaptation executed outcomes、final held-out test 完全隔离。
4. 只在独立 local validation 上重新报告 n、distribution、P95/P99/max、correlation；候选必须带 provenance，未审核时 fail closed 回到 G0。
5. 对每次 explore 持久化 information gain、support increase、parameter change、prediction-map change、validation-uncertainty change、exploit-eligibility change、best-trajectory change，并分别输出 MODEL_VALUE、SUPPORT_VALUE、DECISION_VALUE。
6. 将 stopping candidate 实现成 disabled shadow evaluation；不允许 support alone 触发继续，也不允许 truth 成为 stopping feature。先复现 S0/S1/S2 counterfactual，再由人工审核是否启用。
7. 输出 current P2 与 prototype 的 paired offline comparison；truth 只用于最后的 post-hoc missed/false/regret 标签，不得用于 proposal、guard、fitting 或 threshold tuning。
8. 新增测试证明默认 P2 bit-for-bit/row-for-row 不变、所有冻结项哈希不变、candidate 未审核不能启用、无机器人连接、无 formal personalization 声明。

完成后状态仍应为 `RESEARCH_ONLY`、`NOT_HUMAN_READY`、`NOT_ROBOT_MOTION_APPROVED`；除非另有独立证据与人工冻结，不得宣称 P2 V2 formalized。
"""


def _leakage_report() -> str:
    return f"""# Data Leakage Audit

- Current P2 was not run or modified in `{DESIGN_PROTOCOL_ID}`.
- The 341 local pairs are reconstructed historical decision opportunities; they are not designated validation.
- Their errors use post-hoc virtual truth. They may support design comparison, not threshold freezing.
- G1/G2 use leave-one-case-out bounds, so each evaluated case is excluded from its bound calculation. This reduces direct case reuse but does not create an independent study population.
- Held-out final-test data were not loaded.
- Truth was not used for model fitting, candidate proposal, current guard execution, exploration feature construction, or stopping input.
- `subject_specificity_gap.csv` uses truth only to classify post-hoc regret and possible cause categories.
- No human threshold, formal personalization, safety evidence, or robot approval was created.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    validate_active_reference_file()
    if ACTIVE_REFERENCE_ID != "reference_measured_asymmetric_closed_slow":
        raise RuntimeError("active reference identifier changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")

    root_metadata_path = ROOT_CAUSE_ARTIFACT_DIRECTORY / "metadata.json"
    convergence_metadata_path = CONVERGENCE_ARTIFACT_DIRECTORY / "metadata.json"
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    convergence_metadata = json.loads(
        convergence_metadata_path.read_text(encoding="utf-8")
    )
    previous_policy = json.loads(
        (POLICY_ARTIFACT_DIRECTORY / "policy_definition.json").read_text(
            encoding="utf-8"
        )
    )
    if previous_policy != policy_definitions():
        raise RuntimeError("current P2 policy definitions differ from frozen artifact")
    if root_metadata["policy_core_source_sha256"] != _sha256(POLICY_CORE_SOURCE):
        raise RuntimeError("P2 source differs from root-cause audit provenance")
    if (
        convergence_metadata["core_policy_source_sha256"]
        != root_metadata["policy_core_source_sha256"]
    ):
        raise RuntimeError("root-cause and convergence P2 source hashes disagree")

    source_paths = {
        "root_counterfactual": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "counterfactual_guard_comparison.csv",
        "root_global_provenance": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "current_guard_uncertainty_provenance.csv",
        "root_exploration": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "exploration_value_decomposition.csv",
        "root_truth_summary": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "truth_landscape_summary.csv",
        "convergence_best_history": CONVERGENCE_ARTIFACT_DIRECTORY
        / "best_trajectory_stability.csv",
        "convergence_executed_history": CONVERGENCE_ARTIFACT_DIRECTORY
        / "boundary_chasing_audit.csv",
        "convergence_natural_stopping": CONVERGENCE_ARTIFACT_DIRECTORY
        / "natural_stopping_summary.csv",
    }
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
    protected_diff_before = _git_output(
        "diff", "--", "hardware", "control", "collection", "safety"
    )
    frozen_hashes_before = {
        "active_reference": sha256_file(ACTIVE_REFERENCE_PATH),
        "policy_core": _sha256(POLICY_CORE_SOURCE),
        "mechanical_objective": _sha256(MECHANICAL_OBJECTIVE_SOURCE),
        "generator": _sha256(GENERATOR_SOURCE),
        "estimator": _sha256(ESTIMATOR_SOURCE),
    }

    root_counterfactual = pd.read_csv(source_paths["root_counterfactual"])
    global_provenance = pd.read_csv(source_paths["root_global_provenance"])
    exploration_source = pd.read_csv(source_paths["root_exploration"])
    truth_summary = pd.read_csv(source_paths["root_truth_summary"])
    best_history = pd.read_csv(source_paths["convergence_best_history"])
    executed_history = pd.read_csv(source_paths["convergence_executed_history"])
    natural_stopping = pd.read_csv(source_paths["convergence_natural_stopping"])

    protocol = local_decision_validation_protocol()
    local_pairs = build_retrospective_local_pair_errors(
        root_counterfactual, best_history
    )
    distribution = build_global_vs_local_error_distribution(
        global_provenance, local_pairs
    )
    strata = build_local_pair_stratum_summary(local_pairs)
    candidates = build_local_uncertainty_candidates(local_pairs)
    guard_detail, guard_summary = build_local_guard_counterfactual(
        local_pairs, candidates
    )
    exploration = build_exploration_value_components(exploration_source)
    stopping = build_exploration_stopping_counterfactual(
        exploration, executed_history, natural_stopping
    )
    knee = exploration.loc[exploration["subject_id"].eq("knee_stiff")].copy()
    matched_late = exploration.loc[
        exploration["subject_id"].isin(("baseline", "hip_stiff", "heavy_leg"))
        & exploration["iteration"].between(7, 13)
    ].copy()
    specificity = build_subject_specificity_gap(truth_summary, local_pairs)
    recommendation = design_recommendation(
        guard_summary, stopping, specificity
    )
    decision_matrix = _design_matrix(recommendation)

    tables = {
        "retrospective_local_decision_pair_errors.csv": local_pairs,
        "global_vs_local_pair_error_distribution.csv": distribution,
        "local_pair_stratum_summary.csv": strata,
        "local_uncertainty_candidates.csv": candidates,
        "local_guard_counterfactual_detail.csv": guard_detail,
        "local_guard_counterfactual_summary.csv": guard_summary,
        "exploration_value_components.csv": exploration,
        "exploration_stopping_counterfactual.csv": stopping,
        "knee_stiff_exploration_design_audit.csv": knee,
        "matched_late_exploration_design_audit.csv": matched_late,
        "subject_specificity_gap.csv": specificity,
        "P2_REVISION_V2_DESIGN_DECISION_MATRIX.csv": decision_matrix,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)
    _write_json(output / JSON_FILENAMES[0], protocol)
    _write_json(output / JSON_FILENAMES[1], recommendation)
    (output / REPORT_FILENAMES[0]).write_text(
        _design_report(
            distribution,
            strata,
            candidates,
            guard_summary,
            exploration,
            stopping,
            specificity,
            recommendation,
        ),
        encoding="utf-8",
    )
    (output / REPORT_FILENAMES[1]).write_text(
        _implementation_prompt(), encoding="utf-8"
    )
    (output / REPORT_FILENAMES[2]).write_text(
        _leakage_report(), encoding="utf-8"
    )

    _plot_error_distribution(
        distribution, output / "global_vs_local_pair_error_distribution.png"
    )
    _plot_guard(guard_summary, output / "local_guard_counterfactual_summary.png")
    _plot_exploration(exploration, output / "exploration_value_components.png")
    _plot_stopping(stopping, output / "exploration_stopping_counterfactual.png")
    _plot_specificity(specificity, output / "subject_specificity_gap.png")

    generated = (*CSV_FILENAMES, *JSON_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    missing = [name for name in generated if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")

    source_hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
    frozen_hashes_after = {
        "active_reference": sha256_file(ACTIVE_REFERENCE_PATH),
        "policy_core": _sha256(POLICY_CORE_SOURCE),
        "mechanical_objective": _sha256(MECHANICAL_OBJECTIVE_SOURCE),
        "generator": _sha256(GENERATOR_SOURCE),
        "estimator": _sha256(ESTIMATOR_SOURCE),
    }
    protected_diff_after = _git_output(
        "diff", "--", "hardware", "control", "collection", "safety"
    )
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("input artifact changed during design analysis")
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("frozen research source changed during design analysis")
    if protected_diff_before != protected_diff_after:
        raise RuntimeError("protected package diff changed during design analysis")

    output_hashes = {name: _sha256(output / name) for name in generated}
    guard_index = guard_summary.set_index("guard_id")
    stopping_index = stopping.groupby("stopping_candidate_id", sort=False)[
        [
            "executed_trials_avoided",
            "later_exploit_trials_in_frozen_history",
            "later_accepted_best_changes_in_frozen_history",
        ]
    ].sum()
    metadata = {
        "protocol_id": DESIGN_PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "checkpoint_boundary_note": (
            "P2_and_prior_audit_sources_artifacts_are_untracked_after_HEAD_0ae022c;"
            "this_design_did_not_stage_or_commit"
        ),
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "active_reference_id": ACTIVE_REFERENCE_ID,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "active_reference_sha256_observed": sha256_file(ACTIVE_REFERENCE_PATH),
        "five_parameter_names": list(PARAMETER_NAMES),
        "mechanical_objective_version": MECHANICAL_OBJECTIVE_VERSION,
        "algorithm_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "support_coverage_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "generator_bounds": OFFLINE_PERSONALIZATION_SEARCH_BOUNDS,
        "formal_grid_steps": {
            "hip_deg": GRID_HIP_STEP_DEG,
            "knee_deg": GRID_KNEE_STEP_DEG,
            "phase": GRID_PHASE_STEP,
        },
        "local_decision_validation_protocol": LOCAL_PROTOCOL_ID,
        "local_pair_instance_count": len(local_pairs),
        "global_pair_instance_count": len(global_provenance),
        "local_pair_is_designated_validation": False,
        "local_candidate_threshold_frozen": False,
        "current_research_guard_modified": False,
        "current_P2_executed": False,
        "current_P2_behavior_modified": False,
        "formal_personalization_executed": False,
        "counterfactual_trajectory_executed": False,
        "exploration_stopping_protocol": EXPLORATION_STOPPING_CANDIDATE_ID,
        "explore_trial_count": len(exploration),
        "model_value_trial_count": int(exploration["MODEL_VALUE"].sum()),
        "support_value_trial_count": int(exploration["SUPPORT_VALUE"].sum()),
        "decision_value_trial_count": int(exploration["DECISION_VALUE"].sum()),
        "support_only_exact_zero_decision_trial_count": int(
            exploration["exact_zero_decision_value_round"].sum()
        ),
        "knee_stiff_explore_count": len(knee),
        "matched_trial_7_to_13_count": len(matched_late),
        "guard_counterfactual_summary": {
            guard: {
                "would_exploit_candidate_count": int(row["would_exploit_candidate_count"]),
                "missed_improvement_candidate_count": int(row["missed_improvement_candidate_count"]),
                "false_improvement_candidate_count": int(row["false_improvement_candidate_count"]),
                "conservative_stop_round_count": int(row["conservative_stop_round_count"]),
            }
            for guard, row in guard_index.iterrows()
        },
        "stopping_counterfactual_summary": {
            candidate: {
                column: int(value)
                for column, value in row.items()
            }
            for candidate, row in stopping_index.iterrows()
        },
        "truth_role": "POST_HOC_COUNTERFACTUAL_AND_SPECIFICITY_GAP_ONLY",
        "truth_used_to_modify_policy": False,
        "truth_used_for_model_fitting": False,
        "truth_used_for_candidate_proposal": False,
        "truth_used_as_stopping_feature": False,
        "heldout_final_test_used": False,
        "mechanical_objective_modified": False,
        "generator_modified": False,
        "five_parameter_model_modified": False,
        "active_reference_modified": False,
        "rom_protocol_modified": False,
        "design_recommendation": recommendation,
        "design_status": DESIGN_STATUS,
        "offline_method_status": OFFLINE_METHOD_STATUS,
        "research_status": "RESEARCH_ONLY",
        "not_human_ready": True,
        "not_robot_motion_approved": True,
        "real_robot_connected": False,
        "human_threshold_created": False,
        "protected_package_diff_unchanged": protected_diff_before == protected_diff_after,
        "protected_package_git_diff_empty": protected_diff_after == "",
        "protected_package_git_diff": protected_diff_after,
        "policy_definition_source_sha256": _sha256(
            POLICY_ARTIFACT_DIRECTORY / "policy_definition.json"
        ),
        "decision_guard_source_sha256": _text_sha256(
            inspect.getsource(apply_research_decision_guard)
        ),
        "exploit_selector_source_sha256": _text_sha256(
            inspect.getsource(select_exploit_candidate)
        ),
        "exploration_ranker_source_sha256": _text_sha256(
            inspect.getsource(rank_exploration_frontier)
        ),
        "policy_core_source_sha256": _sha256(POLICY_CORE_SOURCE),
        "mechanical_objective_source_sha256": _sha256(MECHANICAL_OBJECTIVE_SOURCE),
        "generator_source_sha256": _sha256(GENERATOR_SOURCE),
        "estimator_source_sha256": _sha256(ESTIMATOR_SOURCE),
        "design_core_source_sha256": _sha256(CORE_SOURCE_PATH),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "source_input_sha256": source_hashes_after,
        "root_cause_metadata_sha256": _sha256(root_metadata_path),
        "convergence_metadata_sha256": _sha256(convergence_metadata_path),
        "runtime_seconds": time.perf_counter() - started,
        "output_sha256": output_hashes,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_directory)
    print(f"protocol={metadata['protocol_id']}")
    print(f"output={arguments.output_directory}")
    print(f"design_status={metadata['design_status']}")
    print(f"offline_method_status={metadata['offline_method_status']}")
    print("current_P2_modified=false")
    print("robot_connected=false")
    print(f"runtime_seconds={metadata['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
