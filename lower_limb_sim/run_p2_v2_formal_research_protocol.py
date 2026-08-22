"""Generate formal P2 V2 research-protocol design artifacts.

The runner freezes research plans, not a policy.  It does not run or replace
P2 V1, execute personalization, or connect to robot-side packages.
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
from .p2_v2_formal_research_protocol import (
    CUMULATIVE_RULE_ID,
    DESIGN_STATUS,
    FORMAL_DESIGN_ID,
    LOCAL_PROTOCOL_ID,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    OFFLINE_METHOD_STATUS,
    STOPPING_RULE_ID,
    build_cumulative_rule_candidate_matrix,
    build_decision_value_stopping_shadow,
    build_designated_local_validation_pair_plan,
    compare_global_and_designated_local_validation,
    compare_single_and_cumulative_rules,
    cumulative_decision_rule_protocol,
    decision_value_exploration_stopping_protocol,
    designated_local_validation_protocol,
    enumerate_designated_local_pair_universe,
    minimum_p2_v2_revision_set,
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
    DEFAULT_OUTPUT_DIRECTORY as PROTOTYPE_ARTIFACT_DIRECTORY,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)
from .run_sequential_personalization_convergence_stopping_audit import (
    DEFAULT_OUTPUT_DIRECTORY as CONVERGENCE_ARTIFACT_DIRECTORY,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CORE_SOURCE_PATH = MODULE_DIR / "p2_v2_formal_research_protocol.py"
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_v2_formal_research_protocol_v1"
)

CSV_FILENAMES = (
    "designated_local_validation_pair_plan.csv",
    "designated_local_validation_strata.csv",
    "global_vs_designated_local_validation_design.csv",
    "cumulative_decision_rule_candidate_matrix.csv",
    "single_vs_cumulative_rule_comparison.csv",
    "decision_value_stopping_shadow_detail.csv",
    "decision_value_stopping_shadow_summary.csv",
    "P2_V2_FORMAL_DESIGN_DECISION_MATRIX.csv",
)
JSON_FILENAMES = (
    "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1.json",
    "CUMULATIVE_DECISION_RULE_V1.json",
    "DECISION_VALUE_EXPLORATION_STOPPING_V1.json",
)
REPORT_FILENAMES = (
    "LOCAL_VALIDATION_PROTOCOL_REPORT.md",
    "P2_V2_FORMAL_DESIGN_REPORT.md",
    "DATA_LEAKAGE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "designated_local_validation_plan.png",
    "cumulative_rule_candidates.png",
    "decision_value_stopping_shadow.png",
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


def _plot_local_plan(strata: pd.DataFrame, path: Path) -> None:
    totals = strata.groupby(["coordinate", "trust_level"])[
        "planned_pair_count"
    ].sum().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    totals.loc[:, ["INITIAL", "HALF", "MINIMUM"]].plot.bar(ax=axis)
    axis.set(
        xlabel="generator coordinate",
        ylabel="pre-registered pair count",
        title="Designated local validation plan (outcomes pending)",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="trust level")
    _save(figure, path)


def _plot_cumulative(matrix: pd.DataFrame, path: Path) -> None:
    selected = matrix.drop_duplicates("maximum_accumulation_steps_candidate")
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(
        selected["maximum_accumulation_steps_candidate"].astype(str),
        selected["posthoc_knee_stiff_cumulative_improvement"],
    )
    axis.axhline(
        OBJECTIVE_EQUIVALENCE_TOLERANCE,
        color="red",
        linestyle="--",
        label="unchanged 0.005",
    )
    axis.bar_label(bars, fmt="%.4f")
    axis.set(
        xlabel="maximum accumulation steps candidate",
        ylabel="post-hoc cumulative improvement",
        title="Cumulative rule design candidates (disabled)",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _plot_stopping(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(
        summary["consecutive_zero_value_candidate"].astype(str),
        summary["historical_trials_potentially_avoided"],
    )
    axis.bar_label(bars)
    axis.set(
        xlabel="consecutive zero-value explores candidate K",
        ylabel="historical trials potentially avoided",
        title="Decision-value stopping shadow (never executed)",
    )
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _decision_matrix(
    comparison: pd.DataFrame,
    cumulative_comparison: pd.DataFrame,
    stopping: pd.DataFrame,
) -> pd.DataFrame:
    local = comparison.set_index("validation_class").loc[
        "DESIGNATED_LOCAL_PAIR_PLAN"
    ]
    cumulative = cumulative_comparison.set_index("rule").loc[
        "RULE_B_MULTI_STEP_CANDIDATE"
    ]
    k2 = stopping.set_index("consecutive_zero_value_candidate").loc[2]
    return pd.DataFrame(
        [
            {
                "question": "local_validation_in_formal_P2_V2",
                "recommendation": (
                    "YES_AS_MANDATORY_PRE_REGISTERED_EVIDENCE_LAYER_AFTER_"
                    "INDEPENDENT_OUTCOMES_AND_REVIEW"
                ),
                "evidence": f"planned_pairs={int(local['pair_instance_count'])};outcomes_pending",
                "P2_V1_modified": False,
            },
            {
                "question": "cumulative_improvement_solves_stepwise_problem",
                "recommendation": (
                    "YES_MECHANISTICALLY_AS_DISABLED_RESEARCH_CANDIDATE_WITH_"
                    "DIRECTION_AND_UNCERTAINTY_GUARDS"
                ),
                "evidence": f"candidate_windows_over_0p005={int(cumulative['moves_exceeding_existing_0p005'])}",
                "P2_V1_modified": False,
            },
            {
                "question": "decision_value_stopping_reduces_meaningless_exploration",
                "recommendation": (
                    "PROMISING_IN_HISTORICAL_SHADOW_NOT_YET_PROSPECTIVELY_VALIDATED"
                ),
                "evidence": f"K2_potentially_avoided={int(k2['historical_trials_potentially_avoided'])}",
                "P2_V1_modified": False,
            },
            {
                "question": "formal_P2_V2_ready",
                "recommendation": "NO_OFFLINE_METHOD_REQUIRES_REVISION",
                "evidence": OFFLINE_METHOD_STATUS,
                "P2_V1_modified": False,
            },
        ]
    )


def _local_report(
    protocol: Mapping[str, Any],
    comparison: pd.DataFrame,
) -> str:
    table = comparison.set_index("validation_class")
    global_row = table.loc["CURRENT_GLOBAL_IDENTIFICATION_PAIR"]
    local_row = table.loc["DESIGNATED_LOCAL_PAIR_PLAN"]
    return f"""# Local Validation Protocol Report

## Protocol

`{LOCAL_PROTOCOL_ID}` defines an independent, pre-registered local validation plan. Candidate pairs are drawn only from the existing geometry-valid generator lattice. A pair changes exactly one of hip/knee/phase by one existing initial, half, or minimum trust step; clipping and bounds expansion are prohibited.

The complete geometry/trust pair universe contains {protocol['pair_generation']['universe_pair_count']} pairs. The pilot plan contains {protocol['pair_generation']['planned_pair_count']} pairs: coordinate × trust-level × lower/interior/upper strata, with 12 deterministic SHA-selected pairs per stratum. Pilot sample count still requires power/reviewer approval and is not a decision threshold.

Pair plan SHA-256:

```text
{protocol['pair_generation']['pair_plan_sha256']}
```

This file must be frozen before predictions or independent truth outcomes are evaluated. The final truth landscape is forbidden as a selection source. Current `predicted_delta_J`, `truth_delta_J`, and `e_delta_J` fields are intentionally blank.

## Global vs designated local

| class | pairs | alpha mappable | trust levels | outcomes | P95 | max |
|---|---:|---|---|---|---:|---:|
| current global identification validation | {int(global_row['pair_instance_count'])} | no | none | available | {global_row['P95_e_delta_J']:.9g} | {global_row['max_e_delta_J']:.9g} |
| designated local plan | {int(local_row['pair_instance_count'])} | yes | initial/half/minimum | pending | pending | pending |

The current global pair measures differences between identification excitations and cannot be mapped to generator alpha or a trust step. The designated plan measures exactly the future local decision relationship, but it has no result yet. Therefore it should become a mandatory research evidence layer for a future P2 V2, while no local uncertainty statistic or threshold can be selected in this task.

## Required outcome attachment

After plan freeze, a predeclared model checkpoint records predicted ΔJ and a newly generated independent designated offline evaluation records truth ΔJ. Outcomes must match the frozen pair-ID set exactly; error is `abs(predicted_delta_J - truth_delta_J)`. The data cannot be used for fitting, adaptation update, held-out final test, P2 V1, or an unreviewed P2 V2 guard.
"""


def _formal_report(
    protocol: Mapping[str, Any],
    cumulative_comparison: pd.DataFrame,
    stopping: pd.DataFrame,
) -> str:
    rules = cumulative_comparison.set_index("rule")
    rule_a = rules.loc["RULE_A_SINGLE_STEP"]
    rule_b = rules.loc["RULE_B_MULTI_STEP_CANDIDATE"]
    stopping_index = stopping.set_index("consecutive_zero_value_candidate")
    minimal = minimum_p2_v2_revision_set()
    minimal_text = "\n".join(f"{index}. `{item}`" for index, item in enumerate(minimal, 1))
    return f"""# P2 V2 Formal Research Design Report

## Boundary

- Design ID: `{FORMAL_DESIGN_ID}`; this is a formal research protocol design, not a frozen policy.
- P2 V1 remains unchanged and default. No formal personalization or robot connection was executed.
- active reference, ROM_PROTOCOL_V2, five-parameter model, mechanical objective, generator bounds, 0.005 tolerance, and 90% support gate remain unchanged.
- Status remains `{OFFLINE_METHOD_STATUS}`, `{NOT_HUMAN_READY}`, `{NOT_ROBOT_MOTION_APPROVED}`.

## Part A — Designated local validation

`{LOCAL_PROTOCOL_ID}` should enter a future formal P2 V2 as a mandatory evidence layer, but only after its {protocol['pair_generation']['planned_pair_count']}-pair plan receives independent outcomes and reviewer approval. Pair selection is geometry/trust/hash only; final truth landscape selection is forbidden. The present protocol does not choose max/P95/P99 or create a guard threshold.

## Part B — `{CUMULATIVE_RULE_ID}`

Rule A evaluates each step separately. In knee_stiff evidence, its maximum single-step improvement is {rule_a['maximum_observed_improvement']:.9f}, so 0 of {int(rule_a['evaluated_move_count'])} steps exceed the unchanged 0.005 criterion.

Rule B evaluates a same-coordinate, same-sign bundle. Candidate windows are 2, 3, and 5 steps; all three observed cumulative windows exceed 0.005, with maximum {rule_b['maximum_observed_improvement']:.9f}. This directly addresses the stepwise mechanism but is not enabled.

To avoid accumulating the wrong direction, a shadow bundle must be selected before truth, remain inside existing geometry/support at every intermediate point, keep one signed coordinate direction, reject predicted sign/ranking reversal, fix the model checkpoint within the bundle, and pass a bundle uncertainty constraint. Uncertainty aggregation candidates are worst-case sum, newly estimated block-P95, and RSS only after residual independence is demonstrated. Maximum steps and aggregation method remain unfrozen.

## Part C — `{STOPPING_RULE_ID}`

Each explore separates SUPPORT (coverage), MODEL (parameter), PREDICTION (map), and DECISION (ranking/best; exploit eligibility supplemental). Support alone is not a reason to continue.

Historical shadow results:

| consecutive zero-value candidate K | potentially avoided | later exploits | later accepted best changes |
|---:|---:|---:|---:|
| 1 | {int(stopping_index.loc[1,'historical_trials_potentially_avoided'])} | {int(stopping_index.loc[1,'later_exploit_trials_in_frozen_history'])} | {int(stopping_index.loc[1,'later_accepted_best_changes_in_frozen_history'])} |
| 2 | {int(stopping_index.loc[2,'historical_trials_potentially_avoided'])} | {int(stopping_index.loc[2,'later_exploit_trials_in_frozen_history'])} | {int(stopping_index.loc[2,'later_accepted_best_changes_in_frozen_history'])} |
| 3 | {int(stopping_index.loc[3,'historical_trials_potentially_avoided'])} | {int(stopping_index.loc[3,'later_exploit_trials_in_frozen_history'])} | {int(stopping_index.loc[3,'later_accepted_best_changes_in_frozen_history'])} |

These candidates could reduce low-value exploration in the frozen history, but no automatic stop occurred and K is not frozen. Prospective offline shadow validation and reviewed change-detection tolerance are still required.

## Final recommendation

1. **Local validation should enter formal P2 V2**, but only after independent outcomes, sample/power review, and uncertainty-statistic review.
2. **Cumulative improvement addresses the observed stepwise problem mechanistically**, provided direction/path/uncertainty constraints are retained; it is not yet a formal rule.
3. **Decision-value stopping is promising**, with historical potential reductions of 25/21/17 trials for K=1/2/3 and no later exploit in this history; it has not yet demonstrated prospective validity.
4. Minimal P2 V2 revision set:

{minimal_text}

P2 V2 is not ready to replace P2 V1. Final state remains `{OFFLINE_METHOD_STATUS}`, `{NOT_HUMAN_READY}`, `{NOT_ROBOT_MOTION_APPROVED}`.
"""


def _leakage_report() -> str:
    return f"""# Data Leakage Audit

- `{LOCAL_PROTOCOL_ID}` pair enumeration and selection use only generator alpha, geometry validity, existing trust steps, location strata, protocol ID, and SHA-256.
- No prediction, truth, final truth landscape, subject label, P2 outcome, or objective value participates in pair selection.
- Pair-plan outcome fields are blank and the plan SHA is recorded before future outcome attachment.
- The future outcome attachment requires an exact pair-ID match and cannot reselect pairs.
- Knee cumulative evidence is post-hoc rule-design evidence only and is not fed into P2 V1 or a live P2 V2.
- Stopping analysis uses frozen exploration history in shadow mode; no automatic stop or policy modification occurred.
- No held-out final test, human threshold, formal personalization, hardware motion, or robot approval was created.
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
    validate_active_reference_file()
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("0.005 equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("90 percent support gate changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank definition changed")
    prior_policy = json.loads(
        (POLICY_ARTIFACT_DIRECTORY / "policy_definition.json").read_text(
            encoding="utf-8"
        )
    )
    if prior_policy != policy_definitions():
        raise RuntimeError("P2 V1 policy definition changed")

    source_paths = {
        "parameter_map": Path(parameter_map_path),
        "global_validation_provenance": ROOT_CAUSE_ARTIFACT_DIRECTORY
        / "current_guard_uncertainty_provenance.csv",
        "prototype_exploration_history": PROTOTYPE_ARTIFACT_DIRECTORY
        / "exploration_value_history.csv",
        "prototype_knee_cumulative": PROTOTYPE_ARTIFACT_DIRECTORY
        / "knee_stiff_cumulative_improvement.csv",
        "prototype_metadata": PROTOTYPE_ARTIFACT_DIRECTORY / "metadata.json",
        "convergence_executed_history": CONVERGENCE_ARTIFACT_DIRECTORY
        / "boundary_chasing_audit.csv",
        "convergence_natural_stopping": CONVERGENCE_ARTIFACT_DIRECTORY
        / "natural_stopping_summary.csv",
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
    universe = enumerate_designated_local_pair_universe(parameter_lattice)
    pair_plan, strata = build_designated_local_validation_pair_plan(
        parameter_lattice
    )
    _write_csv(output / CSV_FILENAMES[0], pair_plan)
    pair_plan_sha256 = _sha256(output / CSV_FILENAMES[0])
    local_protocol = designated_local_validation_protocol(
        pair_plan_sha256=pair_plan_sha256,
        planned_pair_count=len(pair_plan),
        universe_pair_count=len(universe),
    )
    global_comparison = compare_global_and_designated_local_validation(
        pd.read_csv(source_paths["global_validation_provenance"]), pair_plan
    )
    cumulative_history = pd.read_csv(source_paths["prototype_knee_cumulative"])
    cumulative_matrix = build_cumulative_rule_candidate_matrix(
        cumulative_history
    )
    cumulative_comparison = compare_single_and_cumulative_rules(
        cumulative_history
    )
    stopping_detail, stopping_summary = build_decision_value_stopping_shadow(
        pd.read_csv(source_paths["prototype_exploration_history"]),
        pd.read_csv(source_paths["convergence_executed_history"]),
        pd.read_csv(source_paths["convergence_natural_stopping"]),
    )
    cumulative_protocol = cumulative_decision_rule_protocol()
    stopping_protocol = decision_value_exploration_stopping_protocol()
    decision_matrix = _decision_matrix(
        global_comparison, cumulative_comparison, stopping_summary
    )

    tables = {
        "designated_local_validation_strata.csv": strata,
        "global_vs_designated_local_validation_design.csv": global_comparison,
        "cumulative_decision_rule_candidate_matrix.csv": cumulative_matrix,
        "single_vs_cumulative_rule_comparison.csv": cumulative_comparison,
        "decision_value_stopping_shadow_detail.csv": stopping_detail,
        "decision_value_stopping_shadow_summary.csv": stopping_summary,
        "P2_V2_FORMAL_DESIGN_DECISION_MATRIX.csv": decision_matrix,
    }
    for filename, table in tables.items():
        _write_csv(output / filename, table)
    _write_json(output / JSON_FILENAMES[0], local_protocol)
    _write_json(output / JSON_FILENAMES[1], cumulative_protocol)
    _write_json(output / JSON_FILENAMES[2], stopping_protocol)
    (output / REPORT_FILENAMES[0]).write_text(
        _local_report(local_protocol, global_comparison), encoding="utf-8"
    )
    (output / REPORT_FILENAMES[1]).write_text(
        _formal_report(local_protocol, cumulative_comparison, stopping_summary),
        encoding="utf-8",
    )
    (output / REPORT_FILENAMES[2]).write_text(
        _leakage_report(), encoding="utf-8"
    )
    _plot_local_plan(strata, output / FIGURE_FILENAMES[0])
    _plot_cumulative(cumulative_matrix, output / FIGURE_FILENAMES[1])
    _plot_stopping(stopping_summary, output / FIGURE_FILENAMES[2])

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
        raise RuntimeError("input artifact changed during protocol design")
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("frozen source changed during protocol design")
    if protected_diff_before != protected_diff_after:
        raise RuntimeError("protected package diff changed during protocol design")

    output_hashes = {name: _sha256(output / name) for name in generated}
    metadata = {
        "protocol_id": FORMAL_DESIGN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "checkpoint_boundary_note": (
            "P2_V1_audits_prototype_and_this_design_are_untracked_after_HEAD_0ae022c;"
            "this_runner_did_not_stage_or_commit"
        ),
        "design_status": DESIGN_STATUS,
        "current_P2_V1_remains_default": True,
        "current_P2_V1_executed_by_runner": False,
        "current_P2_V1_modified": False,
        "formal_personalization_executed": False,
        "local_protocol_id": LOCAL_PROTOCOL_ID,
        "designated_local_pair_universe_count": len(universe),
        "designated_local_pair_plan_count": len(pair_plan),
        "designated_local_pair_plan_sha256": pair_plan_sha256,
        "designated_local_outcomes_available": False,
        "final_truth_landscape_used_for_pair_selection": False,
        "local_uncertainty_threshold_frozen": False,
        "cumulative_rule_id": CUMULATIVE_RULE_ID,
        "cumulative_rule_enabled": False,
        "cumulative_max_steps_frozen": False,
        "cumulative_uncertainty_aggregation_frozen": False,
        "stopping_rule_id": STOPPING_RULE_ID,
        "automatic_stopping_enabled": False,
        "stopping_candidate_frozen": False,
        "minimal_P2_V2_revision_set": minimum_p2_v2_revision_set(),
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
        "mechanical_objective_modified": False,
        "generator_modified": False,
        "five_parameter_model_modified": False,
        "active_reference_modified": False,
        "rom_protocol_modified": False,
        "truth_used_to_modify_formal_policy": False,
        "truth_used_for_designated_pair_selection": False,
        "heldout_final_test_used": False,
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
        "protocol_core_source_sha256": _sha256(CORE_SOURCE_PATH),
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
    print(f"design_status={metadata['design_status']}")
    print(f"offline_method_status={metadata['offline_method_status']}")
    print("P2_V1_modified=false")
    print("formal_personalization_executed=false")
    print("robot_connected=false")
    print(f"runtime_seconds={metadata['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
