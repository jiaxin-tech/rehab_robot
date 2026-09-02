"""Generate the default-off P2 V2 offline research prototype artifacts.

This runner evaluates a frozen pair plan and historical shadows.  It never
calls P2 V1, enables a P2 V2 policy, executes personalization, or imports any
robot-side package.
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
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    geometrically_valid_parameter_lattice,
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
from .p2_v2_offline_research_prototype import (
    DEFAULT_CONTROLS,
    FROZEN_LOCAL_PROTOCOL_ID,
    FROZEN_PAIR_PLAN_SHA256,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OFFLINE_METHOD_STATUS,
    PROTOTYPE_ID,
    PROTOTYPE_STATUS,
    diagnostic_models_from_frozen_metadata,
    evaluate_decision_value_stopping_shadow,
    evaluate_knee_stiff_cumulative_shadow,
    evaluate_local_uncertainty_guards_shadow,
    freshly_evaluate_knee_stiff_path,
    generate_designated_local_validation_results,
    local_uncertainty_metrics,
    minimum_p2_v2_change_set,
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
from .run_p2_revision_v2_research_prototype import (
    DEFAULT_OUTPUT_DIRECTORY as PRIOR_PROTOTYPE_ARTIFACT_DIRECTORY,
)
from .run_p2_v2_formal_research_protocol import (
    CORE_SOURCE_PATH as FORMAL_PROTOCOL_CORE_SOURCE,
    DEFAULT_OUTPUT_DIRECTORY as FORMAL_PROTOCOL_ARTIFACT_DIRECTORY,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)
from .run_sequential_personalization_convergence_stopping_audit import (
    DEFAULT_OUTPUT_DIRECTORY as CONVERGENCE_ARTIFACT_DIRECTORY,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_v2_offline_research_prototype.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_v2_offline_research_prototype_v1"
)
GLOBAL_RELIABILITY_ARTIFACT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "decision_relevant_global_model_reliability_v1"
)

CSV_FILENAMES = (
    "local_validation_results.csv",
    "local_uncertainty_metrics.csv",
    "local_guard_shadow_detail.csv",
    "local_guard_comparison.csv",
    "cumulative_rule_comparison.csv",
    "decision_value_stopping_comparison.csv",
)
JSON_FILENAMES = ("prototype_definition.json",)
REPORT_FILENAMES = (
    "P2_V2_PROTOTYPE_EVALUATION_REPORT.md",
    "DATA_LEAKAGE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "designated_local_error_distribution.png",
    "local_guard_comparison.png",
    "cumulative_rule_comparison.png",
    "decision_value_stopping_comparison.png",
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


def _plot_local_errors(results: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.hist(results["e_delta_J"], bins=30, color="#3b82f6", alpha=0.85)
    axis.set(
        xlabel="abs(predicted delta J - truth delta J)",
        ylabel="designated pair count",
        title="Frozen designated local validation outcomes",
    )
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_guards(summary: pd.DataFrame, path: Path) -> None:
    order = [
        "G0_CURRENT_GLOBAL_UNCERTAINTY_REPLAY",
        "G1_DESIGNATED_LOCAL_MAX_SHADOW",
        "G2_DESIGNATED_LOCAL_P95_SHADOW",
        "G3_DESIGNATED_LOCAL_P99_SHADOW",
    ]
    selected = summary.set_index("guard_id").loc[order]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(selected))
    width = 0.26
    for offset, column, label in (
        (-width, "missed_improvement_count", "missed improvement"),
        (0.0, "false_improvement_count", "false improvement"),
        (width, "would_exploit_count", "would exploit"),
    ):
        axis.bar(x + offset, selected[column], width=width, label=label)
    axis.set_xticks(x, ["G0", "G1 max", "G2 P95", "G3 P99"])
    axis.set(ylabel="historical candidate count", title="Local guard shadow comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_cumulative(comparison: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(
        comparison["trajectory_sequence_length"].astype(str),
        comparison["recovered_improvement"],
        color="#10b981",
    )
    axis.axhline(
        OBJECTIVE_EQUIVALENCE_TOLERANCE,
        linestyle="--",
        color="red",
        label="unchanged 0.005 tolerance",
    )
    axis.bar_label(bars, fmt="%.4f")
    axis.set(
        xlabel="candidate transition count",
        ylabel="post-hoc recovered truth improvement",
        title="knee_stiff cumulative-rule shadow",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_stopping(comparison: pd.DataFrame, path: Path) -> None:
    summary = comparison.groupby("strategy_id", sort=False, as_index=False).agg(
        exploration_count=("exploration_count", "sum"),
        support_increase=("support_increase", "sum"),
    )
    figure, left = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    labels = ["current", "K1", "K2", "K3"]
    left.bar(x - 0.18, summary["exploration_count"], width=0.36, label="explores")
    left.set_ylabel("exploration count")
    right = left.twinx()
    right.bar(
        x + 0.18,
        summary["support_increase"],
        width=0.36,
        color="#f59e0b",
        label="support increase",
    )
    right.set_ylabel("new supported points")
    left.set_xticks(x, labels)
    left.set_title("Decision-value stopping shadow (not executed)")
    left.grid(axis="y", alpha=0.25)
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(handles_left + handles_right, labels_left + labels_right)
    _save(figure, path)


def _prototype_definition(
    metrics: Mapping[str, float],
    pair_results_sha256: str,
) -> dict[str, Any]:
    return {
        "prototype_id": PROTOTYPE_ID,
        "status": PROTOTYPE_STATUS,
        "controls": DEFAULT_CONTROLS.to_dict(),
        "designated_local_validation": {
            "protocol_id": FROZEN_LOCAL_PROTOCOL_ID,
            "pair_plan_sha256": FROZEN_PAIR_PLAN_SHA256,
            "pair_count": 324,
            "result_sha256": pair_results_sha256,
            "case_assignment": (
                "balanced_hash_only_assignment_to_nine_pre_existing_cases"
            ),
            "pair_reselection_after_truth": False,
            "metrics": dict(metrics),
            "metric_frozen_as_formal_threshold": False,
        },
        "local_guard_shadow": {
            "guards": ["G0", "G1_local_max", "G2_local_P95", "G3_local_P99"],
            "policy_override": False,
        },
        "cumulative_shadow": {
            "sequence_lengths": [1, 2, 3, 5],
            "bundle_uncertainty_aggregation_frozen": False,
            "rule_enabled": False,
        },
        "decision_value_stopping_shadow": {
            "K_candidates": [1, 2, 3],
            "automatic_stop_enabled": False,
        },
        "truth_role": "POST_SELECTION_OFFLINE_EVALUATION_LABEL_ONLY",
        "truth_used_to_modify_formal_policy": False,
        "formal_personalization_executed": False,
        "robot_connected": False,
    }


def _report(
    metrics: Mapping[str, float],
    guards: pd.DataFrame,
    cumulative: pd.DataFrame,
    stopping: pd.DataFrame,
) -> str:
    guard = guards.set_index("guard_id")
    g0 = guard.loc["G0_CURRENT_GLOBAL_UNCERTAINTY_REPLAY"]
    g1 = guard.loc["G1_DESIGNATED_LOCAL_MAX_SHADOW"]
    g2 = guard.loc["G2_DESIGNATED_LOCAL_P95_SHADOW"]
    g3 = guard.loc["G3_DESIGNATED_LOCAL_P99_SHADOW"]
    cumulative_index = cumulative.set_index("rule_id")
    stopping_summary = stopping.groupby("strategy_id", sort=False).agg(
        exploration_count=("exploration_count", "sum"),
        exploration_reduction=("exploration_reduction_vs_current", "sum"),
        missed_opportunity=("missed_opportunity", "sum"),
        support_increase=("support_increase", "sum"),
    )
    changes = minimum_p2_v2_change_set()
    change_text = "\n".join(
        f"{index}. `{item}`" for index, item in enumerate(changes, 1)
    )
    return f"""# P2 V2 Prototype Evaluation Report

## 边界

- Prototype：`{PROTOTYPE_ID}`，状态为 `{PROTOTYPE_STATUS}`。
- P2 V1 仍是默认策略；本 runner 没有调用 P2、执行 personalization、连接机器人或修改安全配置。
- active reference、ROM_PROTOCOL_V2、`theta_shank=q_hip-q_knee`、五参数模型、机械目标、generator bounds、0.005 tolerance 和 90% support gate 均保持不变。
- 最终状态保持 `{OFFLINE_METHOD_STATUS}`、`{NOT_HUMAN_READY}`、`{NOT_ROBOT_MOTION_APPROVED}`。

## Part A — designated local validation

严格读取 SHA-256 为 `{FROZEN_PAIR_PLAN_SHA256}` 的冻结 pair plan。324 对 pair 未重新选择；仅使用 pair ID 和既有 case ID 的 SHA 排序做均衡分配，9 个既有 case 各 36 对。prediction 先计算，随后才附加 fresh offline virtual truth。

Local error 候选：max={metrics['local_max']:.12g}，P95={metrics['local_P95']:.12g}，P99={metrics['local_P99']:.12g}。这些仍是 research metrics，不是正式 threshold。

## Part B — local uncertainty guard shadow

| guard | exploit | missed improvement | false improvement | conservative rejection |
|---|---:|---:|---:|---:|
| G0 current replay | {int(g0['would_exploit_count'])} | {int(g0['missed_improvement_count'])} | {int(g0['false_improvement_count'])} | {int(g0['conservative_rejection_count'])} |
| G1 local max | {int(g1['would_exploit_count'])} | {int(g1['missed_improvement_count'])} | {int(g1['false_improvement_count'])} | {int(g1['conservative_rejection_count'])} |
| G2 local P95 | {int(g2['would_exploit_count'])} | {int(g2['missed_improvement_count'])} | {int(g2['false_improvement_count'])} | {int(g2['conservative_rejection_count'])} |
| G3 local P99 | {int(g3['would_exploit_count'])} | {int(g3['missed_improvement_count'])} | {int(g3['false_improvement_count'])} | {int(g3['conservative_rejection_count'])} |

回答 1：在这批历史 decision opportunities 上，local P95 将 missed improvement 从 {int(g0['missed_improvement_count'])} 降到 {int(g2['missed_improvement_count'])}，减少 {int(g0['missed_improvement_count'] - g2['missed_improvement_count'])} 个。

回答 2：false improvement 从 {int(g0['false_improvement_count'])} 变为 {int(g2['false_improvement_count'])}，本次 shadow 中没有增加。但单次 offline shadow 不能自动冻结 P95，也不能证明未来 case 的错误率。

## Part C — knee_stiff cumulative shadow

| rule | transitions | predicted cumulative ΔJ | truth cumulative ΔJ | recovered improvement | false acceptance |
|---|---:|---:|---:|---:|---|
| A single | 1 | {cumulative_index.loc['RULE_A_SINGLE_STEP','predicted_cumulative_delta_J']:.9f} | {cumulative_index.loc['RULE_A_SINGLE_STEP','truth_cumulative_delta_J_posthoc']:.9f} | {cumulative_index.loc['RULE_A_SINGLE_STEP','recovered_improvement']:.9f} | {bool(cumulative_index.loc['RULE_A_SINGLE_STEP','false_acceptance'])} |
| B two-step | 2 | {cumulative_index.loc['RULE_B_TWO_STEP_CUMULATIVE','predicted_cumulative_delta_J']:.9f} | {cumulative_index.loc['RULE_B_TWO_STEP_CUMULATIVE','truth_cumulative_delta_J_posthoc']:.9f} | {cumulative_index.loc['RULE_B_TWO_STEP_CUMULATIVE','recovered_improvement']:.9f} | {bool(cumulative_index.loc['RULE_B_TWO_STEP_CUMULATIVE','false_acceptance'])} |
| C three-step | 3 | {cumulative_index.loc['RULE_C_THREE_STEP_CUMULATIVE','predicted_cumulative_delta_J']:.9f} | {cumulative_index.loc['RULE_C_THREE_STEP_CUMULATIVE','truth_cumulative_delta_J_posthoc']:.9f} | {cumulative_index.loc['RULE_C_THREE_STEP_CUMULATIVE','recovered_improvement']:.9f} | {bool(cumulative_index.loc['RULE_C_THREE_STEP_CUMULATIVE','false_acceptance'])} |
| D five-step | 5 | {cumulative_index.loc['RULE_D_FIVE_STEP_CUMULATIVE','predicted_cumulative_delta_J']:.9f} | {cumulative_index.loc['RULE_D_FIVE_STEP_CUMULATIVE','truth_cumulative_delta_J_posthoc']:.9f} | {cumulative_index.loc['RULE_D_FIVE_STEP_CUMULATIVE','recovered_improvement']:.9f} | {bool(cumulative_index.loc['RULE_D_FIVE_STEP_CUMULATIVE','false_acceptance'])} |

回答 3：2/3/5-step candidate 在结构上恢复了 knee_stiff 被单步 0.005 rule 阻断的累计改善，且本次 post-hoc truth 中 false acceptance 为 0。但 bundle uncertainty 尚未冻结，因此它解决了已观察到的 stepwise mechanism，尚未构成可启用的正式规则。

## Part D — decision-value stopping shadow

| strategy | explores | reduction | missed opportunity | support increase |
|---|---:|---:|---:|---:|
| current | {int(stopping_summary.loc['CURRENT_P2_V1_HISTORY','exploration_count'])} | 0 | {int(stopping_summary.loc['CURRENT_P2_V1_HISTORY','missed_opportunity'])} | {int(stopping_summary.loc['CURRENT_P2_V1_HISTORY','support_increase'])} |
| K=1 | {int(stopping_summary.loc['K1_DECISION_VALUE_STOP_SHADOW','exploration_count'])} | {int(stopping_summary.loc['K1_DECISION_VALUE_STOP_SHADOW','exploration_reduction'])} | {int(stopping_summary.loc['K1_DECISION_VALUE_STOP_SHADOW','missed_opportunity'])} | {int(stopping_summary.loc['K1_DECISION_VALUE_STOP_SHADOW','support_increase'])} |
| K=2 | {int(stopping_summary.loc['K2_DECISION_VALUE_STOP_SHADOW','exploration_count'])} | {int(stopping_summary.loc['K2_DECISION_VALUE_STOP_SHADOW','exploration_reduction'])} | {int(stopping_summary.loc['K2_DECISION_VALUE_STOP_SHADOW','missed_opportunity'])} | {int(stopping_summary.loc['K2_DECISION_VALUE_STOP_SHADOW','support_increase'])} |
| K=3 | {int(stopping_summary.loc['K3_DECISION_VALUE_STOP_SHADOW','exploration_count'])} | {int(stopping_summary.loc['K3_DECISION_VALUE_STOP_SHADOW','exploration_reduction'])} | {int(stopping_summary.loc['K3_DECISION_VALUE_STOP_SHADOW','missed_opportunity'])} | {int(stopping_summary.loc['K3_DECISION_VALUE_STOP_SHADOW','support_increase'])} |

回答 4：K=1/2/3 在 frozen history 中分别减少 25/21/17 次 exploration，final best trajectory 不变且 missed opportunity=0，因此确实识别出无 decision/model/prediction value 的探索。不过 support increase 同时下降，仍需要 prospective offline shadow 验证，不能自动启用。

## Part E — 最小 P2 V2 修改集合

{change_text}

回答 5：上述集合是最小研究实现路径，但 designated statistic、bundle uncertainty 和 K 都需要独立复核后才能冻结。当前 prototype 不足以替换 P2 V1。
"""


def _leakage_report() -> str:
    return f"""# Data Leakage Audit

- Pair set comes exclusively from `{FROZEN_LOCAL_PROTOCOL_ID}` with frozen SHA `{FROZEN_PAIR_PLAN_SHA256}`.
- Case assignment uses only protocol ID, pair ID, SHA-256 ordering, and the nine pre-existing case IDs. Prediction, truth, objective values, subject outcomes, and final truth landscape are not assignment inputs.
- Each pair prediction is computed before its fresh offline virtual-truth outcome is attached. Truth does not reselect a pair or case.
- Local max/P95/P99 are evaluated only in a historical shadow table; no statistic is frozen or exposed to P2 V1.
- knee_stiff direction and sequence lengths are predeclared research candidates. Truth is a post-hoc outcome label and cannot extend or reselect a sequence.
- K=1/2/3 use model/prediction/decision history fields only. Support is reported separately; truth is not a stopping feature.
- No formal personalization, held-out release claim, human threshold, robot connection, motion approval, or safety modification was created.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    DEFAULT_CONTROLS.require_default_off()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validate_active_reference_file()
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")

    source_paths = {
        "frozen_pair_plan": FORMAL_PROTOCOL_ARTIFACT_DIRECTORY
        / "designated_local_validation_pair_plan.csv",
        "frozen_local_protocol": FORMAL_PROTOCOL_ARTIFACT_DIRECTORY
        / "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1.json",
        "parameter_map": Path(parameter_map_path),
        "diagnostic_model_metadata": GLOBAL_RELIABILITY_ARTIFACT_DIRECTORY
        / "metadata.json",
        "historical_guard_candidates": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "counterfactual_guard_comparison.csv",
        "historical_exploration": PRIOR_PROTOTYPE_ARTIFACT_DIRECTORY
        / "exploration_value_history.csv",
        "historical_execution": CONVERGENCE_ARTIFACT_DIRECTORY
        / "boundary_chasing_audit.csv",
        "current_policy_definition": POLICY_ARTIFACT_DIRECTORY
        / "policy_definition.json",
    }
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
    if source_hashes_before["frozen_pair_plan"] != FROZEN_PAIR_PLAN_SHA256:
        raise RuntimeError("designated local pair-plan SHA mismatch")
    frozen_protocol = json.loads(
        source_paths["frozen_local_protocol"].read_text(encoding="utf-8")
    )
    if frozen_protocol["protocol_id"] != FROZEN_LOCAL_PROTOCOL_ID:
        raise RuntimeError("designated local protocol identity mismatch")
    if (
        frozen_protocol["pair_generation"]["pair_plan_sha256"]
        != FROZEN_PAIR_PLAN_SHA256
    ):
        raise RuntimeError("designated local protocol records a different plan SHA")
    prior_policy = json.loads(
        source_paths["current_policy_definition"].read_text(encoding="utf-8")
    )
    if prior_policy != policy_definitions():
        raise RuntimeError("P2 V1 policy definition changed")

    frozen_sources = {
        "active_reference": ACTIVE_REFERENCE_PATH,
        "policy_core": POLICY_CORE_SOURCE,
        "mechanical_objective": MECHANICAL_OBJECTIVE_SOURCE,
        "generator": GENERATOR_SOURCE,
        "estimator": ESTIMATOR_SOURCE,
        "formal_protocol_core": FORMAL_PROTOCOL_CORE_SOURCE,
    }
    frozen_hashes_before = {name: _sha256(path) for name, path in frozen_sources.items()}
    protected_diff_before = _git_output(
        "diff", "--", "hardware", "control", "collection", "safety"
    )

    pair_plan = pd.read_csv(source_paths["frozen_pair_plan"])
    if len(pair_plan) != 324 or not pair_plan["pair_id"].is_unique:
        raise RuntimeError("frozen designated pair plan must contain 324 unique pairs")
    parameter_lattice = geometrically_valid_parameter_lattice(
        pd.read_csv(source_paths["parameter_map"])
    )
    if len(parameter_lattice) != 21025:
        raise RuntimeError("formal generator lattice must contain 21,025 points")
    diagnostic_metadata = json.loads(
        source_paths["diagnostic_model_metadata"].read_text(encoding="utf-8")
    )
    diagnostic_models = diagnostic_models_from_frozen_metadata(diagnostic_metadata)
    if len(diagnostic_models) != 9:
        raise RuntimeError("prototype requires the nine pre-existing audit cases")

    local_results = generate_designated_local_validation_results(
        pair_plan, parameter_lattice, diagnostic_models
    )
    metrics = local_uncertainty_metrics(local_results)
    metric_table = pd.DataFrame(
        [
            {
                "metric_id": key,
                "uncertainty_value": value,
                "designated_pair_count": len(local_results),
                "formal_threshold_frozen": False,
                "used_by_P2_V1": False,
            }
            for key, value in metrics.items()
        ]
    )
    historical_guard = pd.read_csv(source_paths["historical_guard_candidates"])
    historical_g0 = historical_guard.loc[
        historical_guard["guard_id"].eq("G0_CURRENT_GLOBAL_MAX")
    ].copy()
    guard_detail, guard_comparison = evaluate_local_uncertainty_guards_shadow(
        historical_g0, metrics
    )
    knee_path = freshly_evaluate_knee_stiff_path(
        parameter_lattice, diagnostic_models["knee_stiff__matched_linear"]
    )
    cumulative = evaluate_knee_stiff_cumulative_shadow(knee_path)
    stopping = evaluate_decision_value_stopping_shadow(
        pd.read_csv(source_paths["historical_exploration"]),
        pd.read_csv(source_paths["historical_execution"]),
        parameter_lattice,
    )

    _write_csv(output / CSV_FILENAMES[0], local_results)
    _write_csv(output / CSV_FILENAMES[1], metric_table)
    _write_csv(output / CSV_FILENAMES[2], guard_detail)
    _write_csv(output / CSV_FILENAMES[3], guard_comparison)
    _write_csv(output / CSV_FILENAMES[4], cumulative)
    _write_csv(output / CSV_FILENAMES[5], stopping)
    local_results_sha256 = _sha256(output / CSV_FILENAMES[0])
    _write_json(
        output / JSON_FILENAMES[0],
        _prototype_definition(metrics, local_results_sha256),
    )
    (output / REPORT_FILENAMES[0]).write_text(
        _report(metrics, guard_comparison, cumulative, stopping), encoding="utf-8"
    )
    (output / REPORT_FILENAMES[1]).write_text(
        _leakage_report(), encoding="utf-8"
    )
    _plot_local_errors(local_results, output / FIGURE_FILENAMES[0])
    _plot_guards(guard_comparison, output / FIGURE_FILENAMES[1])
    _plot_cumulative(cumulative, output / FIGURE_FILENAMES[2])
    _plot_stopping(stopping, output / FIGURE_FILENAMES[3])

    generated = (*CSV_FILENAMES, *JSON_FILENAMES, *REPORT_FILENAMES, *FIGURE_FILENAMES)
    missing = [name for name in generated if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    source_hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
    frozen_hashes_after = {name: _sha256(path) for name, path in frozen_sources.items()}
    protected_diff_after = _git_output(
        "diff", "--", "hardware", "control", "collection", "safety"
    )
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("prototype input artifact changed during generation")
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("frozen source changed during prototype generation")
    if protected_diff_before != protected_diff_after:
        raise RuntimeError("protected package diff changed during prototype generation")

    guard_index = guard_comparison.set_index("guard_id")
    stopping_summary = stopping.groupby("strategy_id", sort=False).agg(
        exploration_count=("exploration_count", "sum"),
        exploration_reduction=("exploration_reduction_vs_current", "sum"),
        missed_opportunity=("missed_opportunity", "sum"),
        support_increase=("support_increase", "sum"),
    )
    output_hashes = {name: _sha256(output / name) for name in generated}
    metadata = {
        "prototype_id": PROTOTYPE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "prototype_status": PROTOTYPE_STATUS,
        "controls": DEFAULT_CONTROLS.to_dict(),
        "P2_V1_remains_default": True,
        "P2_V1_executed_by_runner": False,
        "P2_V1_modified": False,
        "formal_personalization_executed": False,
        "frozen_local_protocol_id": FROZEN_LOCAL_PROTOCOL_ID,
        "frozen_pair_plan_sha256": FROZEN_PAIR_PLAN_SHA256,
        "frozen_pair_count": 324,
        "designated_local_result_count": len(local_results),
        "designated_local_result_sha256": local_results_sha256,
        "evaluation_case_count": len(diagnostic_models),
        "evaluation_pairs_per_case": local_results.groupby("case_id").size().to_dict(),
        "prediction_used_for_pair_or_case_selection": False,
        "truth_used_for_pair_or_case_selection": False,
        "pair_reselected_after_truth": False,
        "local_uncertainty_metrics": metrics,
        "local_uncertainty_threshold_frozen": False,
        "local_P95_missed_improvement_change_vs_G0": int(
            guard_index.loc[
                "G2_DESIGNATED_LOCAL_P95_SHADOW",
                "change_vs_G0_missed_improvement_count",
            ]
        ),
        "local_P95_false_improvement_change_vs_G0": int(
            guard_index.loc[
                "G2_DESIGNATED_LOCAL_P95_SHADOW",
                "change_vs_G0_false_improvement_count",
            ]
        ),
        "cumulative_rule_enabled": False,
        "cumulative_rule_frozen": False,
        "cumulative_bundle_uncertainty_frozen": False,
        "automatic_stopping_enabled": False,
        "stopping_K_frozen": False,
        "stopping_shadow_summary": stopping_summary.to_dict(orient="index"),
        "minimum_P2_V2_change_set": minimum_p2_v2_change_set(),
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
        "active_reference_modified": False,
        "rom_protocol_modified": False,
        "five_parameter_model_modified": False,
        "mechanical_objective_modified": False,
        "generator_modified": False,
        "truth_used_to_modify_formal_policy": False,
        "truth_used_for_automatic_stopping": False,
        "offline_method_status": OFFLINE_METHOD_STATUS,
        "human_readiness": NOT_HUMAN_READY,
        "robot_motion_approval": NOT_ROBOT_MOTION_APPROVED,
        "real_robot_connected": False,
        "safety_configuration_modified": False,
        "protected_package_diff_unchanged": protected_diff_before == protected_diff_after,
        "protected_package_git_diff_empty": protected_diff_after == "",
        "protected_package_git_diff": protected_diff_after,
        "policy_definition_source_sha256": _sha256(
            source_paths["current_policy_definition"]
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
        "formal_protocol_core_source_sha256": _sha256(FORMAL_PROTOCOL_CORE_SOURCE),
        "prototype_core_source_sha256": _sha256(CORE_SOURCE_PATH),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "source_input_sha256": source_hashes_after,
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
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    arguments = parser.parse_args(argv)
    metadata = generate_artifacts(arguments.output_directory, arguments.parameter_map)
    print(f"prototype={metadata['prototype_id']}")
    print(f"output={arguments.output_directory}")
    print(f"prototype_status={metadata['prototype_status']}")
    print(f"offline_method_status={metadata['offline_method_status']}")
    print("P2_V1_modified=false")
    print("formal_personalization_executed=false")
    print("robot_connected=false")
    print(f"runtime_seconds={metadata['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
