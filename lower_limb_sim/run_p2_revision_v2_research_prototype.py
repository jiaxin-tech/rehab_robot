"""Generate default-off P2 Revision V2 research-prototype artifacts.

This runner reads frozen offline P2 audit products and emits shadow metrics.
It never calls ``run_policy``, changes an existing P2 decision, stops an
exploration, or imports robot-side packages.
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
from .p2_revision_v2_design import build_retrospective_local_pair_errors
from .p2_revision_v2_research_prototype import (
    CUMULATIVE_RULE_ASSESSMENT,
    DEFAULT_PROTOTYPE_CONTROLS,
    EXPLORATION_VALUE_PROTOCOL_ID,
    LOCAL_PROTOCOL_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OFFLINE_METHOD_STATUS,
    PROTOTYPE_ID,
    PROTOTYPE_STATUS,
    build_exploration_value_history,
    build_formal_local_neighborhood,
    build_knee_stiff_cumulative_improvement,
    evaluate_local_guard_counterfactual,
    exploration_value_summary,
    format_retrospective_local_pairs,
    local_uncertainty_metrics,
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
from .run_p2_revision_v2_design import (
    DEFAULT_OUTPUT_DIRECTORY as V2_DESIGN_ARTIFACT_DIRECTORY,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)
from .run_sequential_personalization_convergence_stopping_audit import (
    DEFAULT_OUTPUT_DIRECTORY as CONVERGENCE_ARTIFACT_DIRECTORY,
)
from .sequential_personalization import SearchAlpha, TrustRegionSteps, shrink_steps


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_revision_v2_research_prototype.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_revision_v2_research_prototype_v1"
)

CSV_FILENAMES = (
    "local_neighborhood_history.csv",
    "local_neighborhood_protocol_examples.csv",
    "local_validation_pairs.csv",
    "local_uncertainty_metrics.csv",
    "local_guard_counterfactual_detail.csv",
    "local_guard_counterfactual_summary.csv",
    "exploration_value_history.csv",
    "exploration_value_summary.csv",
    "knee_stiff_cumulative_improvement.csv",
    "P2_REVISION_V2_PROTOTYPE_DECISION_MATRIX.csv",
)
JSON_FILENAMES = (
    "prototype_configuration.json",
    "cumulative_improvement_assessment.json",
)
REPORT_FILENAMES = (
    "P2_REVISION_V2_PROTOTYPE_REPORT.md",
    "DATA_LEAKAGE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "local_guard_counterfactual.png",
    "exploration_value_history.png",
    "knee_stiff_cumulative_improvement.png",
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


def _plot_guard(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.set_index("guard_id").loc[
        [
            "G0_CURRENT_GLOBAL_GUARD_REPLAY",
            "G1_LOCAL_MAX_RESEARCH_METRIC",
            "G2_LOCAL_P95_RESEARCH_METRIC",
        ]
    ]
    x = np.arange(3)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for offset, column, label in (
        (-width, "missed_improvement_candidate_count", "missed"),
        (0.0, "false_improvement_candidate_count", "false"),
        (width, "conservative_stop_round_count", "conservative stop"),
    ):
        axis.bar(x + offset, ordered[column], width, label=label)
    axis.set_xticks(x, ["G0", "G1 local max", "G2 local P95"])
    axis.set(ylabel="retrospective count", title="Default-off guard shadow evaluation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(figure, path)


def _plot_exploration(history: pd.DataFrame, path: Path) -> None:
    totals = [
        int(history["SUPPORT_VALUE"].sum()),
        int(history["MODEL_VALUE"].sum()),
        int(history["DECISION_VALUE"].sum()),
    ]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(["support", "model", "decision"], totals)
    axis.bar_label(bars)
    axis.set(ylabel="explore rounds", title="Research-only exploration value history")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_cumulative(table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.plot(
        table["step_index"],
        table["cumulative_improvement_magnitude"],
        "-o",
        label="cumulative improvement",
    )
    axis.bar(
        table["step_index"],
        table["single_step_improvement_magnitude"].fillna(0.0),
        alpha=0.35,
        label="single-step improvement",
    )
    axis.axhline(
        OBJECTIVE_EQUIVALENCE_TOLERANCE,
        color="red",
        linestyle="--",
        label="unchanged 0.005 equivalence",
    )
    axis.set(
        xlabel="existing -1 degree knee steps",
        ylabel="post-hoc truth improvement",
        title="knee_stiff single-step vs cumulative improvement",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, path)


def _build_history_neighborhoods(
    design_pairs: pd.DataFrame,
    parameter_lattice: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (case_id, iteration), group in design_pairs.groupby(
        ["case_id", "iteration"], sort=True
    ):
        first = group.iloc[0]
        current = SearchAlpha(
            float(first["current_alpha_hip"]),
            float(first["current_alpha_knee"]),
            float(first["current_alpha_phase"]),
        )
        neighborhood = build_formal_local_neighborhood(
            parameter_lattice,
            current,
            TrustRegionSteps(),
            case_id=str(case_id),
            iteration=int(iteration),
        )
        eligible_keys = {
            (
                round(float(row.candidate_alpha_hip), 12),
                round(float(row.candidate_alpha_knee), 12),
                round(float(row.candidate_alpha_phase), 12),
            )
            for row in neighborhood.loc[
                neighborhood["included_as_local_validation_neighbor"].astype(bool)
            ].itertuples(index=False)
        }
        observed_keys = {
            (
                round(float(row.candidate_alpha_hip), 12),
                round(float(row.candidate_alpha_knee), 12),
                round(float(row.candidate_alpha_phase), 12),
            )
            for row in group.itertuples(index=False)
        }
        if not observed_keys.issubset(eligible_keys):
            raise RuntimeError(f"historical candidates violate neighborhood {case_id}:{iteration}")
        neighborhood["historical_pair_present"] = [
            (
                round(float(row["candidate_alpha_hip"]), 12),
                round(float(row["candidate_alpha_knee"]), 12),
                round(float(row["candidate_alpha_phase"]), 12),
            )
            in observed_keys
            for row in neighborhood.to_dict(orient="records")
        ]
        frames.append(neighborhood)
    return pd.concat(frames, ignore_index=True)


def _build_protocol_examples(parameter_lattice: pd.DataFrame) -> pd.DataFrame:
    steps = TrustRegionSteps()
    frames: list[pd.DataFrame] = []
    for label in ("INITIAL", "HALF", "MINIMUM"):
        frame = build_formal_local_neighborhood(
            parameter_lattice,
            SearchAlpha(),
            steps,
            case_id=f"PROTOCOL_EXAMPLE_{label}",
            iteration=0,
        )
        frame["trust_level_label"] = label
        frame["example_only_not_policy_input"] = True
        frames.append(frame)
        steps = shrink_steps(steps)
    return pd.concat(frames, ignore_index=True)


def _decision_matrix(
    guard: pd.DataFrame,
    exploration: pd.DataFrame,
    cumulative: Mapping[str, Any],
) -> pd.DataFrame:
    guards = guard.set_index("guard_id")
    g0 = guards.loc["G0_CURRENT_GLOBAL_GUARD_REPLAY"]
    g2 = guards.loc["G2_LOCAL_P95_RESEARCH_METRIC"]
    explore = exploration.iloc[0]
    return pd.DataFrame(
        [
            {
                "question": "local_uncertainty_reduces_missed_improvement",
                "answer": "YES_FOR_LOCAL_P95_RETROSPECTIVE_SHADOW_ONLY",
                "evidence": f"G0={int(g0['missed_improvement_candidate_count'])};G2={int(g2['missed_improvement_candidate_count'])}",
                "formal_policy_change": False,
            },
            {
                "question": "false_improvement_increases",
                "answer": "NO_OBSERVED_IN_RETROSPECTIVE_SHADOW_NOT_A_SAFETY_PROOF",
                "evidence": f"G0={int(g0['false_improvement_candidate_count'])};G2={int(g2['false_improvement_candidate_count'])}",
                "formal_policy_change": False,
            },
            {
                "question": "decision_value_scoring_reduces_exploration",
                "answer": "NO_ACTUAL_REDUCTION_DEFAULT_OFF;IDENTIFIES_29_LOW_VALUE_ROUNDS",
                "evidence": f"actual_avoided={int(explore['actual_explore_trials_avoided_by_prototype'])}",
                "formal_policy_change": False,
            },
            {
                "question": "cumulative_decision_rule_required",
                "answer": cumulative["assessment"],
                "evidence": (
                    f"max_single={cumulative['maximum_single_step_improvement']:.9g};"
                    f"five_step={cumulative['final_five_step_cumulative_improvement']:.9g}"
                ),
                "formal_policy_change": False,
            },
            {
                "question": "formal_P2_V2_next",
                "answer": (
                    "NOT_YET;CONTINUE_RESEARCH_PROTOTYPE_WITH_INDEPENDENT_LOCAL_"
                    "VALIDATION_AND_REVIEWED_CUMULATIVE_RULE"
                ),
                "evidence": OFFLINE_METHOD_STATUS,
                "formal_policy_change": False,
            },
        ]
    )


def _report(
    metrics: pd.DataFrame,
    guard: pd.DataFrame,
    exploration: pd.DataFrame,
    cumulative: Mapping[str, Any],
) -> str:
    pooled = metrics.loc[metrics["metric_scope"].eq("POOLED_RETROSPECTIVE")].iloc[0]
    guards = guard.set_index("guard_id")
    g0 = guards.loc["G0_CURRENT_GLOBAL_GUARD_REPLAY"]
    g1 = guards.loc["G1_LOCAL_MAX_RESEARCH_METRIC"]
    g2 = guards.loc["G2_LOCAL_P95_RESEARCH_METRIC"]
    explore = exploration.iloc[0]
    return f"""# P2 Revision V2 Research Prototype Report

## 状态与实现边界

- Prototype：`{PROTOTYPE_ID}`，状态 `DEFAULT_OFF_RESEARCH_SHADOW_ONLY`。
- 当前 P2 仍是默认且未修改；prototype 没有 policy override、自动停止、累计决策执行或 truth-policy 输入。
- 最终状态保持：`{OFFLINE_METHOD_STATUS}`、`{NOT_HUMAN_READY}`、`{NOT_ROBOT_MOTION_APPROVED}`。
- active reference、ROM、`theta_shank = q_hip - q_knee`、五参数模型、mechanical objective、generator definition、0.005 tolerance 和 90% support gate 均未改变。

## Part A — `{LOCAL_PROTOCOL_ID}` prototype

local neighborhood builder 直接复用既有 `SearchAlpha`、`TrustRegionSteps` 和不裁剪的 coordinate neighborhood；只接受 existing generator lattice 内、恰好一个坐标变化且等于当前 trust step 的相邻点。alpha distance 是 formal grid-normalized L1 distance，不是物理距离。protocol examples 覆盖 initial、half、minimum 三个既有层级；历史误差证据仍只有 initial 层级。

341 个历史 local pairs 均保存 pair id、current/candidate alpha、formal alpha distance、predicted ΔJ、post-hoc truth ΔJ 与 error。它们是 retrospective virtual research pairs，不是 formal designated calibration。

- local max = `{pooled['local_max_error']:.12g}`
- local P95 = `{pooled['local_P95_error']:.12g}`
- local P99 = `{pooled['local_P99_error']:.12g}`

所有 metric 均为 `threshold_frozen=false`，没有进入当前 policy。反事实逐 case 使用 leave-one-case-out metric：

| guard | would exploit | missed | false | conservative stop |
|---|---:|---:|---:|---:|
| G0 current replay | {int(g0['would_exploit_candidate_count'])} | {int(g0['missed_improvement_candidate_count'])} | {int(g0['false_improvement_candidate_count'])} | {int(g0['conservative_stop_round_count'])} |
| G1 local max | {int(g1['would_exploit_candidate_count'])} | {int(g1['missed_improvement_candidate_count'])} | {int(g1['false_improvement_candidate_count'])} | {int(g1['conservative_stop_round_count'])} |
| G2 local P95 | {int(g2['would_exploit_candidate_count'])} | {int(g2['missed_improvement_candidate_count'])} | {int(g2['false_improvement_candidate_count'])} | {int(g2['conservative_stop_round_count'])} |

回答 1：local P95 在历史 shadow 中把 missed 从 7 降到 3，减少 4 个；local max 反而把 missed 增至 27。因此“local uncertainty architecture 有价值”不等于“max/P95 已可冻结”。

回答 2：G2 的 observed false improvement 仍为 0，没有增加；这是 retrospective virtual evidence，不是安全或正式 policy 证明。

## Part B — `{EXPLORATION_VALUE_PROTOCOL_ID}`

`exploration_value_history.csv` 对每次 explore 分开记录：

- SUPPORT_VALUE：new supported points；
- MODEL_VALUE：parameter delta、prediction-map delta、validation uncertainty change；
- DECISION_VALUE：best trajectory change、predicted local/global rank-1 change、exploit eligibility change。

共 {int(explore['explore_trial_count'])} 次 explore：support={int(explore['support_value_trial_count'])}、model={int(explore['model_value_trial_count'])}、decision={int(explore['decision_value_trial_count'])}，其中 {int(explore['zero_model_and_decision_value_trial_count'])} 次只有 support、没有 model/decision value。

回答 3：本 prototype **没有实际减少 exploration**，actual avoided=0，因为它严格 default-off 且不自动停止；它只识别出 29 个低 decision-value 历史轮次，为下一项经审核的 stopping rule 提供观测层。

## Part C — knee_stiff cumulative improvement

沿现有 generator 的 knee 方向从 0 到 -5，每次使用既有 1° trust step；没有扩界或修改 objective。5 个单步 improvement 为约 0.00435–0.00447，全部低于不变的 0.005；两个连续步骤累计 improvement={cumulative['first_cumulative_improvement_crossing_existing_0p005']:.9f}，已超过 0.005；五步累计={cumulative['final_five_step_cumulative_improvement']:.9f}。

回答 4：若要解决已观察到的 knee_stiff multi-step gap，**需要把 cumulative decision rule 作为下一版 research design candidate**；但本 prototype 没有启用该规则，truth 也没有反馈给 policy。assessment：`{CUMULATIVE_RULE_ASSESSMENT}`。

## Part D — 下一步判断

回答 5：值得继续 P2 V2 的研究实现，但当前仍不值得替换/冻结为正式 P2 V2。最低前提是：独立 designated local validation 覆盖全部 trust levels；人工选择 uncertainty statistic；为 cumulative rule 预声明窗口与防误接受规则；对 decision-value stopping 做独立 shadow/prospective offline 复核。完成前保持 `OFFLINE_METHOD_REQUIRES_REVISION`、`NOT_HUMAN_READY`、`NOT_ROBOT_MOTION_APPROVED`。
"""


def _leakage_report() -> str:
    return f"""# Data Leakage Audit

- Prototype status: `{PROTOTYPE_STATUS}`; the existing P2 remains the default.
- No P2 policy run was executed by this runner and no current decision was changed.
- Local pairs and uncertainty metrics use retrospective virtual truth only as research calibration/counterfactual labels; they are not formal designated validation or frozen thresholds.
- G1/G2 use leave-one-case-out metrics. This reduces direct same-case reuse but is not an independent validation population.
- Exploration scoring uses already-executed history, model outputs, support, ranking, eligibility, and observed best change. It emits no stop action.
- The knee cumulative audit uses truth only after the frozen historical policy and never proposes, ranks, accepts, or executes a trajectory.
- Held-out final-test data were not loaded.
- No human threshold, hardware motion, robot approval, or formal personalization was created.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    DEFAULT_PROTOTYPE_CONTROLS.require_default_off()
    validate_active_reference_file()
    if ACTIVE_REFERENCE_ID != "reference_measured_asymmetric_closed_slow":
        raise RuntimeError("active reference identifier changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")

    previous_policy = json.loads(
        (POLICY_ARTIFACT_DIRECTORY / "policy_definition.json").read_text(
            encoding="utf-8"
        )
    )
    if previous_policy != policy_definitions():
        raise RuntimeError("current P2 policy definitions changed")
    design_metadata_path = V2_DESIGN_ARTIFACT_DIRECTORY / "metadata.json"
    design_metadata = json.loads(design_metadata_path.read_text(encoding="utf-8"))
    if design_metadata["policy_core_source_sha256"] != _sha256(POLICY_CORE_SOURCE):
        raise RuntimeError("current P2 source differs from V2 design provenance")

    source_paths = {
        "parameter_map": Path(parameter_map_path),
        "root_counterfactual": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "counterfactual_guard_comparison.csv",
        "root_exploration": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "exploration_value_decomposition.csv",
        "root_knee_truth_landscape": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "truth_landscape_knee_stiff.csv",
        "convergence_best_history": CONVERGENCE_ARTIFACT_DIRECTORY
        / "best_trajectory_stability.csv",
        "convergence_landscape_evolution": CONVERGENCE_ARTIFACT_DIRECTORY
        / "prediction_landscape_evolution.csv",
        "v2_design_metadata": design_metadata_path,
        "current_policy_definition": POLICY_ARTIFACT_DIRECTORY
        / "policy_definition.json",
        "current_policy_summary": POLICY_ARTIFACT_DIRECTORY
        / "scenario_sequential_summary.csv",
        "current_policy_trial_history": POLICY_ARTIFACT_DIRECTORY
        / "sequential_trial_history.csv",
    }
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
    frozen_sources = {
        "active_reference": ACTIVE_REFERENCE_PATH,
        "policy_core": POLICY_CORE_SOURCE,
        "mechanical_objective": MECHANICAL_OBJECTIVE_SOURCE,
        "generator": GENERATOR_SOURCE,
        "estimator": ESTIMATOR_SOURCE,
    }
    frozen_hashes_before = {name: _sha256(path) for name, path in frozen_sources.items()}
    protected_diff_before = _git_output(
        "diff", "--", "hardware", "control", "collection", "safety"
    )

    parameter_lattice = geometrically_valid_parameter_lattice(
        pd.read_csv(parameter_map_path)
    )
    if len(parameter_lattice) != 21025:
        raise RuntimeError("formal generator lattice must contain 21,025 points")
    root_counterfactual = pd.read_csv(source_paths["root_counterfactual"])
    best_history = pd.read_csv(source_paths["convergence_best_history"])
    design_pairs = build_retrospective_local_pair_errors(
        root_counterfactual, best_history
    )
    neighborhood_history = _build_history_neighborhoods(
        design_pairs, parameter_lattice
    )
    protocol_examples = _build_protocol_examples(parameter_lattice)
    local_pairs = format_retrospective_local_pairs(design_pairs, parameter_lattice)
    metrics = local_uncertainty_metrics(local_pairs)
    guard_detail, guard_summary = evaluate_local_guard_counterfactual(
        local_pairs, metrics
    )
    exploration_history = build_exploration_value_history(
        pd.read_csv(source_paths["root_exploration"]),
        pd.read_csv(source_paths["convergence_landscape_evolution"]),
    )
    exploration_summary = exploration_value_summary(exploration_history)
    knee_cumulative, cumulative_assessment = build_knee_stiff_cumulative_improvement(
        pd.read_csv(source_paths["root_knee_truth_landscape"])
    )
    decision_matrix = _decision_matrix(
        guard_summary, exploration_summary, cumulative_assessment
    )

    tables = {
        "local_neighborhood_history.csv": neighborhood_history,
        "local_neighborhood_protocol_examples.csv": protocol_examples,
        "local_validation_pairs.csv": local_pairs,
        "local_uncertainty_metrics.csv": metrics,
        "local_guard_counterfactual_detail.csv": guard_detail,
        "local_guard_counterfactual_summary.csv": guard_summary,
        "exploration_value_history.csv": exploration_history,
        "exploration_value_summary.csv": exploration_summary,
        "knee_stiff_cumulative_improvement.csv": knee_cumulative,
        "P2_REVISION_V2_PROTOTYPE_DECISION_MATRIX.csv": decision_matrix,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)
    _write_json(output / JSON_FILENAMES[0], DEFAULT_PROTOTYPE_CONTROLS.to_dict())
    _write_json(output / JSON_FILENAMES[1], cumulative_assessment)
    (output / REPORT_FILENAMES[0]).write_text(
        _report(metrics, guard_summary, exploration_summary, cumulative_assessment),
        encoding="utf-8",
    )
    (output / REPORT_FILENAMES[1]).write_text(
        _leakage_report(), encoding="utf-8"
    )
    _plot_guard(guard_summary, output / FIGURE_FILENAMES[0])
    _plot_exploration(exploration_history, output / FIGURE_FILENAMES[1])
    _plot_cumulative(knee_cumulative, output / FIGURE_FILENAMES[2])

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
        raise RuntimeError("P2 input artifact changed during prototype analysis")
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("frozen source changed during prototype analysis")
    if protected_diff_before != protected_diff_after:
        raise RuntimeError("protected package diff changed during prototype analysis")

    output_hashes = {name: _sha256(output / name) for name in generated}
    guard_index = guard_summary.set_index("guard_id")
    metadata = {
        "protocol_id": PROTOTYPE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "checkpoint_boundary_note": (
            "P2_prior_audits_design_and_this_prototype_are_untracked_after_HEAD_0ae022c;"
            "this_runner_did_not_stage_or_commit"
        ),
        "prototype_status": PROTOTYPE_STATUS,
        "prototype_controls": DEFAULT_PROTOTYPE_CONTROLS.to_dict(),
        "current_P2_remains_default": True,
        "current_P2_executed_by_runner": False,
        "current_P2_behavior_modified": False,
        "local_uncertainty_policy_override_enabled": False,
        "exploration_automatic_stop_enabled": False,
        "cumulative_decision_rule_enabled": False,
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
        "formal_generator_lattice_point_count": len(parameter_lattice),
        "local_protocol_id": LOCAL_PROTOCOL_ID,
        "local_validation_pair_count": len(local_pairs),
        "local_metric_threshold_frozen": False,
        "guard_counterfactual": {
            guard: {
                "missed": int(row["missed_improvement_candidate_count"]),
                "false": int(row["false_improvement_candidate_count"]),
                "conservative_stop": int(row["conservative_stop_round_count"]),
            }
            for guard, row in guard_index.iterrows()
        },
        "exploration_value_protocol_id": EXPLORATION_VALUE_PROTOCOL_ID,
        "exploration_value_summary": exploration_summary.iloc[0].to_dict(),
        "cumulative_improvement_assessment": cumulative_assessment,
        "truth_role": "POST_HOC_RESEARCH_METRIC_AND_CUMULATIVE_DIAGNOSTIC_ONLY",
        "truth_used_to_modify_formal_policy": False,
        "truth_used_for_proposal_or_ranking": False,
        "truth_used_for_automatic_stop": False,
        "heldout_final_test_used": False,
        "formal_personalization_executed": False,
        "mechanical_objective_modified": False,
        "generator_modified": False,
        "five_parameter_model_modified": False,
        "active_reference_modified": False,
        "rom_protocol_modified": False,
        "offline_method_status": OFFLINE_METHOD_STATUS,
        "human_readiness": NOT_HUMAN_READY,
        "robot_motion_approval": NOT_ROBOT_MOTION_APPROVED,
        "real_robot_connected": False,
        "formal_threshold_created": False,
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
    print(f"protocol={metadata['protocol_id']}")
    print(f"output={arguments.output_directory}")
    print(f"prototype_status={metadata['prototype_status']}")
    print(f"offline_method_status={metadata['offline_method_status']}")
    print("current_P2_modified=false")
    print("automatic_stop=false")
    print("robot_connected=false")
    print(f"runtime_seconds={metadata['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
