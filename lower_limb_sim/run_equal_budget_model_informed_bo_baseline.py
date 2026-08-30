"""Run the frozen equal-budget residual-GP BO baseline."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import run_final_model_screened_finite_sequential_validation as frozen_v1
from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    build_trajectory_component_cache,
    evaluate_truth_map,
)
from .equal_budget_model_informed_bo_baseline import (
    ALPHA_LOWER, ALPHA_UPPER, BO_VARIANTS, BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED, BUDGETS, DEFAULT_ENABLED, EI_XI,
    GP_RANDOM_BASE_SEED, KERNEL_SPEC, NOT_HUMAN_READY,
    NOT_ROBOT_APPROVED, OFFLINE_ONLY, PRIMARY_VARIANT,
    SECONDARY_VARIANT, STAGE_ID, TIE_TOLERANCE,
    acquisition_table, bootstrap_mean_ci, deterministic_seed,
    run_bo_sequence,
)
from .final_model_screened_finite_sequential_validation import (
    assert_complete_candidate_trajectory,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_PATH, ACTIVE_REFERENCE_SHA256, ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION, validate_active_reference_file,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_decision_rule_semantics_audit import sha256_file
from .p2_v2_prospective_offline_validation import (
    dynamic_subject_for_id, registered_prospective_subject,
)
from .research_decision_guarded_sequential_personalization import _actual_objective
from .run_model_trust_finite_validation_stress_test import (
    FROZEN_V1_DIRECTORY, FROZEN_V1_MANIFEST_SHA256, _case_plan,
    _identity_lattice, _prepare_all, _prediction_eligible, _scenario_role,
)
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)
from .sequential_personalization import Stage45CVirtualTruthOracle

MODULE_DIR = Path(__file__).resolve().parent
CORE_PATH = MODULE_DIR / "equal_budget_model_informed_bo_baseline.py"
RUNNER_PATH = MODULE_DIR / "run_equal_budget_model_informed_bo_baseline.py"
STRESS_DIRECTORY = MODULE_DIR / "formal_artifacts" / "model_trust_finite_validation_stress_test_v1"
STRESS_METADATA_PATH = STRESS_DIRECTORY / "metadata.json"
STRESS_PROTOCOL_SHA256 = "abaad1f7ab56c7ccd97124ec995810f21f1f331b4e87d3243feb45db0a426b04"
STRESS_RESULT_SHA256 = "5580a85ddf413f7d17f1c0cdea643e0494b50cfa45c7cd114f3c930cb65424a4"
DEFAULT_OUTPUT_DIRECTORY = MODULE_DIR / "formal_artifacts" / "equal_budget_model_informed_bo_baseline_v1"
FIGURES = (
    "figures/EQUAL_BUDGET_FINAL_REGRET_COMPARISON.png",
    "figures/BO_VS_TOP1_PAIRED_REGRET.png",
    "figures/BO_VS_FROZEN_TOP3_PAIRED_REGRET.png",
    "figures/TRIAL_BUDGET_MODEL_SCREENING_VS_BO.png",
    "figures/BO_RESCUE_BY_MISMATCH_SCENARIO.png",
    "figures/ALPHA_SPACE_BO_QUERY_SEQUENCE.png",
)
OUTPUTS = (
    "BO_PROTOCOL.json", "FROZEN_BO_CONFIG.json", "BO_QUERY_LOG.csv",
    "PER_CASE_BO_RESULTS.csv", "BO_POSTERIOR_RECOMMENDATION.csv",
    "METHOD_COMPARISON.csv", "PAIRED_COMPARISONS.csv", "RESCUE_ANALYSIS.csv",
    "TRUTH_ACCESS_AUDIT.csv", "BO_DESIGN_AUDIT.md",
    "EQUAL_BUDGET_MODEL_INFORMED_BO_BASELINE_REPORT.md", *FIGURES,
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(_safe(dict(payload)), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False) + "\n"
    _atomic(path, data.encode())


def _csv(path: Path, table: pd.DataFrame) -> None:
    _atomic(path, table.to_csv(index=False, lineterminator="\n",
                               float_format="%.12g").encode())


def _text(path: Path, value: str) -> None:
    _atomic(path, value.encode())


def _dir_hashes(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): sha256_file(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def _sources() -> dict[str, str]:
    paths = {
        "bo_core": CORE_PATH, "bo_runner": RUNNER_PATH,
        "v1_core": MODULE_DIR / "final_model_screened_finite_sequential_validation.py",
        "v1_runner": MODULE_DIR / "run_final_model_screened_finite_sequential_validation.py",
        "stress_core": MODULE_DIR / "model_trust_finite_validation_stress_test.py",
        "stress_runner": MODULE_DIR / "run_model_trust_finite_validation_stress_test.py",
        "objective": MODULE_DIR / "mechanical_objective.py",
        "generator": MODULE_DIR / "continuous_reference_neighborhood.py",
        "estimator": MODULE_DIR / "parameter_estimator.py",
        "mismatch": MODULE_DIR / "mismatch_scenarios.py",
    }
    return {k: sha256_file(v) for k, v in paths.items()}


def _preflight() -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    stress = json.loads(STRESS_METADATA_PATH.read_text())
    if stress["protocol_sha256"] != STRESS_PROTOCOL_SHA256 or stress["scientific_result_sha256"] != STRESS_RESULT_SHA256:
        raise RuntimeError("frozen stress-test prerequisite changed")
    expected = {
        "model_screening_conclusion": "MODEL_SCREENING_SUPPORTED",
        "top1_top3_conclusion": "TOP1_SUFFICIENT",
        "budget_conclusion": "FINITE_VALIDATION_BUDGET_SATURATES_AT_K_0",
        "model_trust_limit_conclusion": "MODEL_TRUST_LIMIT_NOT_IDENTIFIED",
    }
    for key, value in expected.items():
        if stress["conclusions"][key] != value:
            raise RuntimeError("prior formal conclusion changed")
    for relative, digest in stress["artifact_sha256"].items():
        if sha256_file(STRESS_DIRECTORY / relative) != digest:
            raise RuntimeError(f"stress artifact changed: {relative}")
    v1 = json.loads((FROZEN_V1_DIRECTORY / "metadata.json").read_text())
    if v1["manifest_sha256"] != FROZEN_V1_MANIFEST_SHA256:
        raise RuntimeError("frozen V1 changed")
    validate_active_reference_file(ACTIVE_REFERENCE_PATH)
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("reference changed")
    return ({"prior_conclusions": expected, "stress_protocol_sha256": STRESS_PROTOCOL_SHA256,
             "stress_result_sha256": STRESS_RESULT_SHA256,
             "v1_manifest_sha256": FROZEN_V1_MANIFEST_SHA256},
            _dir_hashes(STRESS_DIRECTORY), _dir_hashes(FROZEN_V1_DIRECTORY))


def _pool_hash(table: pd.DataFrame) -> str:
    columns = ["trajectory_id", "hip_delta", "knee_delta", "phase_delta", "J_pred"]
    payload = table[columns].to_csv(index=False, lineterminator="\n",
                                    float_format="%.12g")
    return hashlib.sha256(payload.encode()).hexdigest()


def _protocol(preflight: Mapping[str, Any], sources: Mapping[str, str],
              cases: pd.DataFrame) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID, "default_enabled": DEFAULT_ENABLED,
        "evidence_level": OFFLINE_ONLY, "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED, "preflight": dict(preflight),
        "source_sha256": dict(sources), "case_plan": cases.to_dict(orient="records"),
        "candidate_grid_size": 21025,
        "candidate_semantics": "IDENTICAL_GEOMETRY_VALID_NONZERO_FROZEN_ALPHA_GRID",
        "bo_variants": list(BO_VARIANTS), "primary_variant": PRIMARY_VARIANT,
        "budgets": list(BUDGETS), "first_query": "FROZEN_MODEL_PREDICTED_C1",
        "model_prior_mean": "J_pred(alpha)", "gp_target": "J_truth-J_pred",
        "residual_prior_mean_before_first_query": 0.0,
        "alpha_scaling": {"formula": "2*(alpha-lower)/(upper-lower)-1",
                          "lower": ALPHA_LOWER.tolist(), "upper": ALPHA_UPPER.tolist(),
                          "output_range": [-1.0, 1.0]},
        "kernel": KERNEL_SPEC,
        "acquisition": {"name": "EXPECTED_IMPROVEMENT_MINIMIZATION", "xi": EI_XI,
                        "incumbent": "min(reference=1,queried truth)",
                        "std_epsilon": 1e-12},
        "reference_fallback": True, "posterior_recommendation_secondary": True,
        "gp_seed_base": GP_RANDOM_BASE_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS, "tie_tolerance": TIE_TOLERANCE,
        "truth_access": "STRICT_QUERY_ORDER_ORACLE_AFTER_BOTH_VARIANTS",
        "interpretation_rules": {
            "bo_value": "mean gain>0, bootstrap95 lower>0, no losses",
            "equality": "all paired differences within 1e-12",
            "frozen_wins": "mean gain<0, bootstrap95 upper<0, no BO wins",
        },
        "no_posthoc_tuning": True, "bo_may_win": True,
    }


def _config(prepared: Sequence[Any], protocol_sha: str) -> dict[str, Any]:
    rows = []
    for item in prepared:
        prediction = item.initial_prediction_map
        neutral = (np.isclose(prediction["hip_delta"], 0) &
                   np.isclose(prediction["knee_delta"], 0) &
                   np.isclose(prediction["phase_delta"], 0))
        pools = {PRIMARY_VARIANT: prediction.loc[~neutral].copy(),
                 SECONDARY_VARIANT: _prediction_eligible(prediction)}
        c1 = item.shortlist.candidates[0]
        for variant, pool in pools.items():
            if c1.trajectory_id not in set(pool["trajectory_id"].astype(str)):
                raise RuntimeError("C1 absent from BO pool")
            rows.append({
                "case_id": item.role["case_id"], "bo_variant": variant,
                "seed": deterministic_seed(str(item.role["case_id"]), variant),
                "candidate_pool_size": len(pool),
                "candidate_pool_sha256": _pool_hash(pool),
                "first_query_trajectory_id": c1.trajectory_id,
                "first_query_alpha": [c1.hip_delta, c1.knee_delta, c1.phase_delta],
                "first_query_J_pred": c1.initial_J_pred,
                "truth_read_before_freeze": False,
            })
    return {"config_id": "FROZEN_BO_CONFIG_V1", "stage_id": STAGE_ID,
            "protocol_sha256": protocol_sha, "records": rows,
            "all_configuration_frozen_before_truth": True}


def _context(role: Mapping[str, Any]):
    if str(role["development_origin"]) == "POST_REJECTION_DEVELOPMENT":
        return registered_prospective_subject(
            dynamic_subject_for_id(str(role["subject_id"])))
    return nullcontext()


def _run_variant(item: Any, pool: pd.DataFrame, variant: str,
                 config_sha: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    role = item.role
    backend = Stage45CVirtualTruthOracle(str(role["subject_id"]),
                                         str(role["scenario_name"]))
    reference, _ = frozen_v1._reference_execution(backend)
    access = [{"case_id": role["case_id"], "bo_variant": variant,
               "access_order": 0, "trajectory_id": "REFERENCE",
               "access_role": "NORMALIZATION_REFERENCE",
               "query_frozen_before_truth": True}]

    def query(row: Mapping[str, Any], index: int, token: str) -> float:
        generated = frozen_v1.generate_personalized_trajectory(
            hip_amplitude_delta_deg=float(row["hip_delta"]),
            knee_amplitude_delta_deg=float(row["knee_delta"]),
            knee_phase_shift=float(row["phase_delta"]))
        trajectory = generated.trajectory.copy(deep=True)
        trajectory["trajectory_id"] = str(row["trajectory_id"])
        assert_complete_candidate_trajectory(
            trajectory, expected_trajectory_id=str(row["trajectory_id"]))
        actual = _actual_objective(str(row["trajectory_id"]),
                                   backend.simulate(trajectory),
                                   reference.actual_metrics)
        access.append({"case_id": role["case_id"], "bo_variant": variant,
                       "access_order": index, "trajectory_id": row["trajectory_id"],
                       "access_role": "SEQUENTIAL_BO_QUERY",
                       "truth_authorization_token": token,
                       "query_frozen_before_truth": True})
        return float(actual.mechanical_cost_j_rms)

    log, posterior = run_bo_sequence(
        pool, case_id=str(role["case_id"]), variant=variant,
        first_trajectory_id=item.shortlist.candidates[0].trajectory_id,
        config_sha256=config_sha, truth_query=query, max_budget=5)
    return log, posterior, access


def _selected(prefix: pd.DataFrame) -> Mapping[str, Any]:
    reference = {"trajectory_id": "REFERENCE", "J_truth": 1.0,
                 "hip_delta": 0.0, "knee_delta": 0.0, "phase_delta": 0.0}
    candidates = prefix[["trajectory_id", "J_truth", "hip_delta",
                         "knee_delta", "phase_delta"]].to_dict(orient="records")
    return min([reference, *candidates],
               key=lambda row: (float(row["J_truth"]), str(row["trajectory_id"])))


def _evaluate(prepared: Sequence[Any], config_sha: str, cache: Any,
              stress: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame, pd.DataFrame]:
    logs, results, posterior_rows, access_rows = [], [], [], []
    top1 = stress.loc[stress["method"].eq("B3_MODEL_TOP1_VALIDATION")].set_index("case_id")
    for item in prepared:
        role = {**item.role, **_scenario_role(str(item.role["scenario_name"]))}
        prediction = item.initial_prediction_map
        neutral = (np.isclose(prediction["hip_delta"], 0) &
                   np.isclose(prediction["knee_delta"], 0) &
                   np.isclose(prediction["phase_delta"], 0))
        pools = {PRIMARY_VARIANT: prediction.loc[~neutral].copy(),
                 SECONDARY_VARIANT: _prediction_eligible(prediction)}
        case_logs = {}
        with _context(role):
            for variant in BO_VARIANTS:
                log, posterior, access = _run_variant(
                    item, pools[variant], variant, config_sha)
                case_logs[variant] = log
                logs.append(log.assign(**{k: v for k, v in role.items()
                                          if k not in log.columns}))
                access_rows.extend(access)
        with _context(role):
            truth_map, truth_meta = evaluate_truth_map(
                prediction, item.initial_model, cache)
        if truth_meta["truth_used_for_pre_evaluation_ranking"]:
            raise RuntimeError("oracle leaked into ranking")
        oracle = truth_map.sort_values(
            ["J_truth", "trajectory_id"], kind="mergesort").iloc[0]
        access_rows.append({"case_id": role["case_id"],
                            "bo_variant": "POST_ALL_BO_VARIANTS",
                            "access_order": 999, "trajectory_id": "ALL_21025",
                            "access_role": "POST_RUN_ORACLE_EVALUATION",
                            "query_frozen_before_truth": True})
        for variant, log in case_logs.items():
            for budget in BUDGETS:
                prefix = log.head(budget)
                chosen = _selected(prefix)
                regret = max(0.0, float(chosen["J_truth"]) -
                             float(oracle["J_truth"]))
                obs = prefix.assign(truth_was_queried=True)
                posterior_map, kernel = acquisition_table(
                    pools[variant], obs,
                    seed=deterministic_seed(str(role["case_id"]), variant))
                rec = posterior_map.sort_values(
                    ["posterior_total_J_mean", "trajectory_id"],
                    kind="mergesort").iloc[0]
                posterior_rows.append(pd.DataFrame([{
                    "case_id": role["case_id"], "bo_variant": variant,
                    "budget": budget,
                    "posterior_recommended_trajectory_id": rec["trajectory_id"],
                    "posterior_recommended_J_mean": rec["posterior_total_J_mean"],
                    "posterior_recommended_std": rec["posterior_std"],
                    "fitted_kernel": kernel,
                    "recommendation_is_primary_physical_decision": False,
                    "unqueried_truth_used": False,
                }]))
                results.append({
                    **role, "method": f"{variant}_K{budget}",
                    "bo_variant": variant, "budget": budget,
                    "query_ids_json": json.dumps(
                        prefix["trajectory_id"].astype(str).tolist(),
                        separators=(",", ":")),
                    "final_selected_trajectory": chosen["trajectory_id"],
                    "final_alpha_json": json.dumps(
                        [chosen["hip_delta"], chosen["knee_delta"],
                         chosen["phase_delta"]], separators=(",", ":")),
                    "J_final_truth": float(chosen["J_truth"]),
                    "oracle_J": float(oracle["J_truth"]),
                    "final_regret": regret,
                    "first_query_residual": float(log.iloc[0]["residual"]),
                    "reference_fallback_available": True,
                    "unqueried_candidate_selected": False,
                })
        k1 = next(row for row in results
                  if row["case_id"] == role["case_id"] and
                  row["method"] == f"{PRIMARY_VARIANT}_K1")
        expected = top1.loc[str(role["case_id"])]
        if not np.isclose(k1["J_final_truth"], float(expected["J_final_truth"]),
                          atol=1e-11, rtol=0) or not np.isclose(
                              k1["final_regret"], float(expected["final_regret"]),
                              atol=1e-11, rtol=0):
            raise RuntimeError("BO_PROTOCOL_SANITY = FAIL")
    return (pd.concat(logs, ignore_index=True), pd.DataFrame(results),
            pd.concat(posterior_rows, ignore_index=True),
            pd.DataFrame(access_rows))


def _paired(bo: pd.DataFrame, stress: pd.DataFrame,
            random: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = stress.set_index(["case_id", "method"])
    primary = bo.loc[(bo["bo_variant"] == PRIMARY_VARIANT) &
                     (bo["budget"] == 3)].set_index("case_id")
    for case_id, row in primary.iterrows():
        comparators = {
            "MODEL_TOP1": float(base.loc[
                (case_id, "B3_MODEL_TOP1_VALIDATION"), "final_regret"]),
            "FROZEN_TOP3": float(base.loc[
                (case_id, "B4_FROZEN_TOP3_SEQUENTIAL"), "final_regret"]),
            "RANDOM3_MEAN": float(random.loc[
                random["case_id"].eq(case_id), "final_regret"].mean()),
        }
        for name, value in comparators.items():
            gain = value - float(row["final_regret"])
            outcome = ("BO_WIN" if gain > TIE_TOLERANCE else
                       "COMPARATOR_WIN" if gain < -TIE_TOLERANCE else "TIE")
            rows.append({"case_id": case_id,
                         "scenario_name": row["scenario_name"],
                         "mismatch_family": row["mismatch_family"],
                         "comparison": f"BO_K3_VS_{name}",
                         "bo_regret": row["final_regret"],
                         "comparator_regret": value,
                         "adaptivity_gain": gain, "outcome": outcome})
    return pd.DataFrame(rows)


def _summary(stress: pd.DataFrame, random: pd.DataFrame,
             bo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ("B0_REFERENCE", "B1_MODEL_ONLY",
                   "B3_MODEL_TOP1_VALIDATION",
                   "B4_FROZEN_TOP3_SEQUENTIAL", "B5_ORACLE"):
        table = stress.loc[stress["method"].eq(method)]
        values = table["final_regret"].to_numpy(float)
        low, high = bootstrap_mean_ci(values)
        rows.append({"method": method,
                     "budget": table["validation_budget"].dropna().iloc[0]
                     if table["validation_budget"].notna().any() else np.nan,
                     "case_count": len(table), "mean_regret": values.mean(),
                     "median_regret": np.median(values),
                     "P95_regret": np.quantile(values, .95),
                     "max_regret": values.max(),
                     "bootstrap95_low": low, "bootstrap95_high": high})
    values = random["final_regret"].to_numpy(float)
    low, high = bootstrap_mean_ci(values)
    rows.append({"method": "B2_RANDOM3_DISTRIBUTION", "budget": 3,
                 "case_count": random["case_id"].nunique(),
                 "mean_regret": values.mean(), "median_regret": np.median(values),
                 "P95_regret": np.quantile(values, .95),
                 "max_regret": values.max(),
                 "bootstrap95_low": low, "bootstrap95_high": high})
    for method, table in bo.groupby("method", sort=False):
        values = table["final_regret"].to_numpy(float)
        low, high = bootstrap_mean_ci(values)
        rows.append({"method": method, "budget": int(table["budget"].iloc[0]),
                     "case_count": len(table), "mean_regret": values.mean(),
                     "median_regret": np.median(values),
                     "P95_regret": np.quantile(values, .95),
                     "max_regret": values.max(),
                     "bootstrap95_low": low, "bootstrap95_high": high})
    return pd.DataFrame(rows)


def _stats(paired: pd.DataFrame, comparison: str) -> dict[str, Any]:
    table = paired.loc[paired["comparison"].eq(comparison)]
    low, high = bootstrap_mean_ci(table["adaptivity_gain"].to_numpy(float))
    return {"comparison": comparison,
            "bo_wins": int(table["outcome"].eq("BO_WIN").sum()),
            "ties": int(table["outcome"].eq("TIE").sum()),
            "comparator_wins": int(table["outcome"].eq("COMPARATOR_WIN").sum()),
            "mean_gain": float(table["adaptivity_gain"].mean()),
            "median_gain": float(table["adaptivity_gain"].median()),
            "bootstrap95_low": low, "bootstrap95_high": high}


def _conclusions(bo: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
    top1 = _stats(paired, "BO_K3_VS_MODEL_TOP1")
    frozen = _stats(paired, "BO_K3_VS_FROZEN_TOP3")
    random = _stats(paired, "BO_K3_VS_RANDOM3_MEAN")
    if top1["ties"] == 15:
        q1 = "ADAPTIVE_BO_NOT_EMPIRICALLY_JUSTIFIED_AT_K3"
    elif top1["mean_gain"] > 0 and top1["bootstrap95_low"] > 0 and top1["comparator_wins"] == 0:
        q1 = "LOW_BUDGET_BO_PROVIDES_ADDITIONAL_VALUE"
    else:
        q1 = "BO_ADDITIONAL_VALUE_NOT_ESTABLISHED"
    if frozen["ties"] == 15:
        q2 = "PRECOMMITMENT_HAS_NO_OBSERVED_PERFORMANCE_COST"
    elif frozen["mean_gain"] > 0 and frozen["bootstrap95_low"] > 0 and frozen["comparator_wins"] == 0:
        q2 = "ADAPTIVE_CANDIDATE_ADMISSION_OUTPERFORMS_FROZEN_SET"
    elif frozen["mean_gain"] < 0 and frozen["bootstrap95_high"] < 0 and frozen["bo_wins"] == 0:
        q2 = "FROZEN_MODEL_SCREENING_OUTPERFORMS_LOW_BUDGET_BO"
    else:
        q2 = "PRECOMMITMENT_COST_NOT_ESTABLISHED"
    primary = bo.loc[bo["bo_variant"].eq(PRIMARY_VARIANT)].pivot(
        index="case_id", columns="budget", values="final_regret")
    saturation = 5
    for budget in BUDGETS:
        later = [value for value in BUDGETS if value > budget]
        if all(np.allclose(primary[budget], primary[value],
                           atol=TIE_TOLERANCE, rtol=0) for value in later):
            saturation = budget
            break
    justified = (q1 == "LOW_BUDGET_BO_PROVIDES_ADDITIONAL_VALUE" or
                 q2 == "ADAPTIVE_CANDIDATE_ADMISSION_OUTPERFORMS_FROZEN_SET")
    return {"top1_comparison": q1, "frozen_comparison": q2,
            "primary_bo_regret_saturates_at_K": saturation,
            "bo_complexity_empirically_justified": justified,
            "top1_stats": top1, "frozen_stats": frozen,
            "random3_stats": random}


def _figures(output: Path, stress: pd.DataFrame, bo: pd.DataFrame,
             paired: pd.DataFrame, rescue: pd.DataFrame,
             query: pd.DataFrame) -> None:
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.size": 9})
    data = [
        stress.loc[stress["method"].eq("B3_MODEL_TOP1_VALIDATION"), "final_regret"],
        stress.loc[stress["method"].eq("B4_FROZEN_TOP3_SEQUENTIAL"), "final_regret"],
        bo.loc[bo["method"].eq(f"{PRIMARY_VARIANT}_K2"), "final_regret"],
        bo.loc[bo["method"].eq(f"{PRIMARY_VARIANT}_K3"), "final_regret"],
        bo.loc[bo["method"].eq(f"{SECONDARY_VARIANT}_K3"), "final_regret"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.boxplot(data, tick_labels=["Top-1", "Frozen Top-3", "BO-A K2",
                                 "BO-A K3", "BO-B K3"], showmeans=True)
    ax.axhline(0, color=".25", ls="--")
    ax.set_ylabel("Final regret, J - J_oracle")
    ax.set_title("Equal-budget final regret comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(folder / "EQUAL_BUDGET_FINAL_REGRET_COMPARISON.png")
    plt.close(fig)
    for comp, filename, title in (
        ("BO_K3_VS_MODEL_TOP1", "BO_VS_TOP1_PAIRED_REGRET.png",
         "BO K=3 versus Model Top-1"),
        ("BO_K3_VS_FROZEN_TOP3", "BO_VS_FROZEN_TOP3_PAIRED_REGRET.png",
         "BO K=3 versus Frozen Top-3"),
    ):
        table = paired.loc[paired["comparison"].eq(comp)]
        fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
        colors = np.where(table["outcome"].eq("BO_WIN"), "tab:blue",
                          np.where(table["outcome"].eq("TIE"), ".45", "tab:red"))
        ax.scatter(table["comparator_regret"], table["bo_regret"], c=colors)
        maximum = max(table["comparator_regret"].max(), table["bo_regret"].max())
        ax.plot([0, maximum], [0, maximum], "--", color=".2")
        ax.set_xlabel("Comparator regret")
        ax.set_ylabel("BO-A K=3 regret")
        ax.set_title(title)
        ax.grid(alpha=.25)
        fig.savefig(folder / filename)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    top1 = stress.loc[stress["method"].eq(
        "B3_MODEL_TOP1_VALIDATION"), "final_regret"].mean()
    frozen = stress.loc[stress["method"].eq(
        "B4_FROZEN_TOP3_SEQUENTIAL"), "final_regret"].mean()
    ax.plot(BUDGETS, [top1] * 4, "--", label="Model Top-1")
    ax.plot(BUDGETS, [frozen] * 4, ":", label="Frozen Top-3")
    for variant, label in ((PRIMARY_VARIANT, "BO-A full feasible"),
                           (SECONDARY_VARIANT, "BO-B screened")):
        means = bo.loc[bo["bo_variant"].eq(variant)].groupby(
            "budget")["final_regret"].mean().reindex(BUDGETS)
        ax.plot(BUDGETS, means, marker="o", label=label)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Full-cycle validation budget K")
    ax.set_ylabel("Mean final regret")
    ax.set_title("Trial budget: model screening versus BO")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(folder / "TRIAL_BUDGET_MODEL_SCREENING_VS_BO.png")
    plt.close(fig)
    ordered = rescue.sort_values("adaptivity_gain")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(ordered["case_id"], ordered["adaptivity_gain"],
            color=np.where(ordered["rescued"], "tab:blue", ".65"))
    ax.axvline(0, color=".2", ls="--")
    ax.set_xlabel("Top-1 regret - BO K=3 regret")
    ax.set_title("BO rescue by existing mismatch scenario")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=.25)
    fig.savefig(folder / "BO_RESCUE_BY_MISMATCH_SCENARIO.png")
    plt.close(fig)
    case = sorted(query["case_id"].unique())[0]
    table = query.loc[(query["bo_variant"] == PRIMARY_VARIANT) &
                      (query["case_id"] == case)]
    fig = plt.figure(figsize=(7, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(table["hip_delta"], table["knee_delta"],
                         table["phase_delta"], c=table["query_index"],
                         s=70, cmap="viridis")
    ax.plot(table["hip_delta"], table["knee_delta"], table["phase_delta"],
            color=".35")
    ax.set_xlabel("Hip delta (deg)")
    ax.set_ylabel("Knee delta (deg)")
    ax.set_zlabel("Phase delta")
    ax.set_title("Representative BO-A query sequence")
    fig.colorbar(scatter, ax=ax, label="Query order", shrink=.7)
    fig.savefig(folder / "ALPHA_SPACE_BO_QUERY_SEQUENCE.png")
    plt.close(fig)


def _audit() -> str:
    return f"""# BO_DESIGN_AUDIT

BO uses the identical 21,025-point geometry-valid alpha grid and the same
case-specific J_pred from theta_hat. BO-A searches all 21,024 non-zero points;
BO-B applies the unchanged 90% coverage and J_pred < 1 screen. Alpha is mapped
from {ALPHA_LOWER.tolist()} to {ALPHA_UPPER.tolist()} into [-1,1]^3 without truth.

The prior is J_pred plus a zero-mean residual Matérn-5/2 ARD GP. C1 is the
frozen model Top-1. Expected Improvement sees only queried residuals. Both BO
variants finish before the full Oracle landscape is revealed. No hyperparameter,
kernel, acquisition, budget, mismatch, or threshold search was performed.
"""


def _report(summary: pd.DataFrame, conclusions: Mapping[str, Any],
            rescue: pd.DataFrame, protocol_sha: str, config_sha: str) -> str:
    lookup = summary.set_index("method")
    top1 = lookup.loc["B3_MODEL_TOP1_VALIDATION"]
    frozen = lookup.loc["B4_FROZEN_TOP3_SEQUENTIAL"]
    bo2 = lookup.loc[f"{PRIMARY_VARIANT}_K2"]
    bo3 = lookup.loc[f"{PRIMARY_VARIANT}_K3"]
    return f"""# EQUAL_BUDGET_MODEL_INFORMED_BO_BASELINE_V1

- Protocol SHA-256: {protocol_sha}
- Frozen config SHA-256: {config_sha}
- BO_PROTOCOL_INTEGRITY = PASS
- OFFLINE_ONLY, NOT_HUMAN_READY, NOT_ROBOT_APPROVED

## Results

Mean regret: Model Top-1 {top1.mean_regret:.6f}; Frozen Top-3
{frozen.mean_regret:.6f}; BO-A K=2 {bo2.mean_regret:.6f}; BO-A K=3
{bo3.mean_regret:.6f}. K=1 equals Model Top-1 in every case.

BO K=3 versus Top-1: {conclusions['top1_stats']['bo_wins']} wins,
{conclusions['top1_stats']['ties']} ties,
{conclusions['top1_stats']['comparator_wins']} losses; mean gain
{conclusions['top1_stats']['mean_gain']:.6f}, bootstrap 95% CI
[{conclusions['top1_stats']['bootstrap95_low']:.6f},
{conclusions['top1_stats']['bootstrap95_high']:.6f}].

BO K=3 versus Frozen Top-3: {conclusions['frozen_stats']['bo_wins']} wins,
{conclusions['frozen_stats']['ties']} ties,
{conclusions['frozen_stats']['comparator_wins']} losses; mean gain
{conclusions['frozen_stats']['mean_gain']:.6f}, bootstrap 95% CI
[{conclusions['frozen_stats']['bootstrap95_low']:.6f},
{conclusions['frozen_stats']['bootstrap95_high']:.6f}].

BO K=3 versus per-case Random-3 mean: {conclusions['random3_stats']['bo_wins']}
wins, {conclusions['random3_stats']['ties']} ties,
{conclusions['random3_stats']['comparator_wins']} losses; mean gain
{conclusions['random3_stats']['mean_gain']:.6f}, bootstrap 95% CI
[{conclusions['random3_stats']['bootstrap95_low']:.6f},
{conclusions['random3_stats']['bootstrap95_high']:.6f}].

Rescue count versus Top-1: {int(rescue['rescued'].sum())}.

## Direct answers

### Q1 Does adaptive BO provide measurable benefit beyond Model Top-1?

{conclusions['top1_comparison']}.

### Q2 Does adaptive BO provide measurable benefit beyond Frozen Top-3?

{conclusions['frozen_comparison']}.

### Q3 Does precommitting candidates impose an observable performance cost?

{'YES' if conclusions['frozen_comparison']=='ADAPTIVE_CANDIDATE_ADMISSION_OUTPERFORMS_FROZEN_SET' else 'NO / NOT ESTABLISHED'}.

### Q4 At K=1/2/3, where does regret saturate?

Including K=5, primary BO saturates at K={conclusions['primary_bo_regret_saturates_at_K']}.

### Q5 Is the added complexity of BO empirically justified in the current cohort?

{'YES' if conclusions['bo_complexity_empirically_justified'] else 'NO'}.

Prior formal conclusions remain unchanged. No robot, human, prospective cohort,
PINN, RL, MPC, or further optimizer stage was started.
"""


def generate_artifacts(output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    preflight, stress_before, v1_before = _preflight()
    sources_before = _sources()
    cases = _case_plan()
    parameter_map = pd.read_csv(DEFAULT_PARAMETER_MAP_PATH)
    lattice = _identity_lattice(parameter_map)
    cache = build_trajectory_component_cache(lattice)
    prepared = _prepare_all(cases, lattice, cache)
    protocol = _protocol(preflight, sources_before, cases)
    _json(output / "BO_PROTOCOL.json", protocol)
    protocol_sha = sha256_file(output / "BO_PROTOCOL.json")
    config = _config(prepared, protocol_sha)
    _json(output / "FROZEN_BO_CONFIG.json", config)
    config_sha = sha256_file(output / "FROZEN_BO_CONFIG.json")
    stress = pd.read_csv(STRESS_DIRECTORY / "PER_CASE_RESULTS.csv")
    random = pd.read_csv(STRESS_DIRECTORY / "RANDOM3_RESULTS.csv")
    query, bo, posterior, access = _evaluate(
        prepared, config_sha, cache, stress)
    paired = _paired(bo, stress, random)
    summary = _summary(stress, random, bo)
    conclusions = _conclusions(bo, paired)
    primary = bo.loc[(bo["bo_variant"] == PRIMARY_VARIANT) &
                     (bo["budget"] == 3)].merge(
        stress.loc[stress["method"].eq("B3_MODEL_TOP1_VALIDATION"),
                   ["case_id", "final_regret"]].rename(
                       columns={"final_regret": "top1_regret"}),
        on="case_id", validate="one_to_one")
    primary["adaptivity_gain"] = primary["top1_regret"] - primary["final_regret"]
    primary["rescued"] = primary["adaptivity_gain"] > TIE_TOLERANCE
    q2 = query.loc[(query["bo_variant"] == PRIMARY_VARIANT) &
                   (query["query_index"] == 2)].set_index("case_id")
    q3 = query.loc[(query["bo_variant"] == PRIMARY_VARIANT) &
                   (query["query_index"] == 3)].set_index("case_id")
    primary["query2_id"] = primary["case_id"].map(q2["trajectory_id"])
    primary["query3_id"] = primary["case_id"].map(q3["trajectory_id"])
    tables = {
        "BO_QUERY_LOG.csv": query, "PER_CASE_BO_RESULTS.csv": bo,
        "BO_POSTERIOR_RECOMMENDATION.csv": posterior,
        "METHOD_COMPARISON.csv": summary,
        "PAIRED_COMPARISONS.csv": paired,
        "RESCUE_ANALYSIS.csv": primary, "TRUTH_ACCESS_AUDIT.csv": access,
    }
    for name, table in tables.items():
        _csv(output / name, table)
    _text(output / "BO_DESIGN_AUDIT.md", _audit())
    _figures(output, stress, bo, paired, primary, query)
    _text(output / "EQUAL_BUDGET_MODEL_INFORMED_BO_BASELINE_REPORT.md",
          _report(summary, conclusions, primary, protocol_sha, config_sha))
    if (_sources() != sources_before or
            _dir_hashes(STRESS_DIRECTORY) != stress_before or
            _dir_hashes(FROZEN_V1_DIRECTORY) != v1_before):
        raise RuntimeError("protected source or artifact changed")
    checks = "".join(f"{sha256_file(output / name)}  {name}\n"
                     for name in sorted(OUTPUTS))
    _text(output / "checksums.sha256", checks)
    metadata = {
        "stage_id": STAGE_ID, "default_enabled": False,
        "evidence_level": OFFLINE_ONLY, "human_ready": NOT_HUMAN_READY,
        "robot_approved": NOT_ROBOT_APPROVED,
        "BO_PROTOCOL_INTEGRITY": "PASS", "BO_PROTOCOL_SANITY": "PASS",
        "protocol_sha256": protocol_sha, "frozen_config_sha256": config_sha,
        "case_count": len(cases), "query_count": len(query),
        "conclusions": conclusions,
        "scientific_result_sha256": hashlib.sha256(checks.encode()).hexdigest(),
        "K1_equals_Model_Top1_all_cases": True,
        "oracle_revealed_after_all_bo_variants": True,
        "unqueried_truth_used": False, "prior_conclusions_unchanged": True,
        "existing_v1_and_stress_artifacts_unchanged": True,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "objective_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "model_support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "robot_connected": False, "runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": {name: sha256_file(output / name) for name in OUTPUTS},
    }
    _json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args(argv)
    print(json.dumps(generate_artifacts(args.output_dir),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
