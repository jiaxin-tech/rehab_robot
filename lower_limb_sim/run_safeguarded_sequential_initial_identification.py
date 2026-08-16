"""Generate formal offline artifacts for sequential initial identification.

No function in this file imports or calls robot, hardware, collection, safety,
or personalization execution code.  The stop rule below is an illustrative
virtual-research comparator because the repository has no approved complete
identifiability release rule.  It is not a selected or recommended scientific
threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .continuous_reference_neighborhood import generate_personalized_trajectory
from .config import L1
from .dynamic_subject import get_dynamic_subject
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
from .geometry_error_metrics import StateDomainBounds
from .mechanical_objective import compute_torque_metrics
from .parameter_estimator import (
    PARAMETER_NAMES,
    baseline_template_from_dynamic_subject,
    predict_joint_torque,
)
from .safeguarded_sequential_initial_identification import (
    AUTO_EXPAND_PATIENT_ENVELOPE,
    IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW,
    INITIAL_IDENTIFICATION_COMPLETE,
    MAX_INITIAL_IDENTIFICATION_TRIALS,
    PROTOCOL_ID,
    REAL_ROBOT_HARD_SAFEGUARD,
    SUPPORTED_PREDICTION,
    UNSUPPORTED_EXTRAPOLATION,
    PatientOperationalEnvelope,
    ResearchIdentifiabilityStopRule,
    SequentialIdentificationResult,
    VirtualIdentificationOracle,
    default_virtual_patient_envelope,
    default_virtual_research_candidate_pool,
    limited_rom_virtual_patient_envelope,
    predict_mechanical_cost,
    prediction_support,
    run_sequential_initial_identification,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "safeguarded_sequential_initial_identification_v1"
)
PARAMETER_MAP_PATH = (
    MODULE_DIR
    / "formal_artifacts"
    / "admissible_personalization_region_v1"
    / "parameter_space_admissibility.csv"
)
INITIAL_DOMAIN_COVERAGE_MINIMUM_PERCENT = 90.0

# These numbers are not an approved real-subject stop rule and are not claimed
# as a selected scientific threshold.  They only exercise the sequential
# software paths; formal research must review/register a rule before new data.
VIRTUAL_RESEARCH_COMPARATOR_RULE = ResearchIdentifiabilityStopRule(
    minimum_rank=5,
    minimum_singular_value=20.0,
    maximum_condition_number=50.0,
    maximum_abs_parameter_correlation=0.30,
    maximum_uncertainty_proxy=0.05,
    minimum_parameter_sensitivity=20.0,
    maximum_validation_rmse_nm=0.20,
)

FORMAL_VIRTUAL_CASES = (
    ("baseline", "matched_linear", "default"),
    ("hip_stiff", "matched_linear", "default"),
    ("knee_stiff", "matched_linear", "default"),
    ("heavy_leg", "matched_linear", "default"),
    ("baseline", "combined_mild", "default"),
    ("LIMITED_ROM_VIRTUAL_SUBJECT", "matched_linear", "limited"),
)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=MODULE_DIR.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _case_id(result: SequentialIdentificationResult) -> str:
    return f"{result.subject_id}__{result.truth_scenario}"


def _envelope(kind: str) -> PatientOperationalEnvelope:
    if kind == "default":
        return default_virtual_patient_envelope()
    if kind == "limited":
        return limited_rom_virtual_patient_envelope()
    raise ValueError(f"unknown envelope fixture {kind!r}")


def run_virtual_cases(
    *,
    stop_rule: ResearchIdentifiabilityStopRule = VIRTUAL_RESEARCH_COMPARATOR_RULE,
) -> tuple[list[SequentialIdentificationResult], dict[str, PatientOperationalEnvelope]]:
    results: list[SequentialIdentificationResult] = []
    envelopes: dict[str, PatientOperationalEnvelope] = {}
    for subject_id, scenario, envelope_kind in FORMAL_VIRTUAL_CASES:
        envelope = _envelope(envelope_kind)
        result = run_sequential_initial_identification(
            VirtualIdentificationOracle(subject_id, scenario),
            envelope,
            stop_rule=stop_rule,
        )
        results.append(result)
        envelopes[_case_id(result)] = envelope
    return results, envelopes


def _concat(results: Iterable[SequentialIdentificationResult], field: str) -> pd.DataFrame:
    frames = [getattr(result, field) for result in results]
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _subject_summary(results: Iterable[SequentialIdentificationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "case_id": _case_id(result),
                "subject_id": result.subject_id,
                "truth_scenario": result.truth_scenario,
                "status": result.status,
                "trials_required": result.trials_required,
                "theta_hat_0_frozen": result.theta_hat_0 is not None,
                "D_init_frozen": result.d_init is not None,
                "D_init_sha256": result.summary[
                    "initial_identification_dataset_sha"
                ],
                "personalization_interface_ready": result.summary[
                    "personalization_interface_ready"
                ],
                "personalization_executed": False,
                "real_robot_motion_executed": False,
                "stop_rule_status": result.summary[
                    "identifiability_stop_rule_status"
                ],
                "failure_reason": ";".join(
                    result.summary["completion_audit_reason"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _fit_initial_domain(d_init: pd.DataFrame) -> StateDomainBounds:
    source = (
        "q_hip_rad",
        "q_knee_rad",
        "dq_hip_rad_s",
        "dq_knee_rad_s",
        "ddq_hip_rad_s2",
        "ddq_knee_rad_s2",
    )
    target = (
        "q_hip_est_rad",
        "q_knee_est_rad",
        "dq_hip_est_rad_s",
        "dq_knee_est_rad_s",
        "ddq_hip_est_rad_s2",
        "ddq_knee_est_rad_s2",
    )
    values = d_init.loc[:, source].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    selected = values[finite]
    if selected.size == 0:
        raise ValueError("D_init has no finite state samples")
    return StateDomainBounds(
        columns=target,
        lower=tuple(np.min(selected, axis=0)),
        upper=tuple(np.max(selected, axis=0)),
        valid_training_samples=int(len(selected)),
    )


def _component_cache(parameter_map: pd.DataFrame) -> tuple[dict[float, pd.DataFrame], dict[tuple[float, float], pd.DataFrame]]:
    hip_cache: dict[float, pd.DataFrame] = {}
    knee_cache: dict[tuple[float, float], pd.DataFrame] = {}
    for hip in sorted(parameter_map["hip_delta"].astype(float).unique()):
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=float(hip),
            knee_amplitude_delta_deg=0.0,
            knee_phase_shift=0.0,
        ).trajectory
        hip_cache[float(hip)] = generated[
            ["q_hip_rad", "dq_hip_rad_s", "ddq_hip_rad_s2"]
        ].reset_index(drop=True)
    knee_pairs = (
        parameter_map[["knee_delta", "phase_delta"]]
        .drop_duplicates()
        .sort_values(["knee_delta", "phase_delta"])
    )
    for row in knee_pairs.itertuples(index=False):
        key = (float(row.knee_delta), float(row.phase_delta))
        generated = generate_personalized_trajectory(
            hip_amplitude_delta_deg=0.0,
            knee_amplitude_delta_deg=key[0],
            knee_phase_shift=key[1],
        ).trajectory
        knee_cache[key] = generated[
            ["q_knee_rad", "dq_knee_rad_s", "ddq_knee_rad_s2"]
        ].reset_index(drop=True)
    return hip_cache, knee_cache


def _combined_trajectory(
    time_s: np.ndarray,
    hip: pd.DataFrame,
    knee: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": time_s,
            "q_hip_rad": hip["q_hip_rad"].to_numpy(dtype=float),
            "q_knee_rad": knee["q_knee_rad"].to_numpy(dtype=float),
            "dq_hip_rad_s": hip["dq_hip_rad_s"].to_numpy(dtype=float),
            "dq_knee_rad_s": knee["dq_knee_rad_s"].to_numpy(dtype=float),
            "ddq_hip_rad_s2": hip["ddq_hip_rad_s2"].to_numpy(dtype=float),
            "ddq_knee_rad_s2": knee["ddq_knee_rad_s2"].to_numpy(dtype=float),
        }
    )


def build_full_prediction_map(
    results: Iterable[SequentialIdentificationResult],
    envelopes: dict[str, PatientOperationalEnvelope],
    *,
    parameter_map_path: str | Path = PARAMETER_MAP_PATH,
    maximum_points: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every patient-envelope point in the chosen deterministic lattice.

    ``maximum_points`` exists only for fast unit tests.  Formal generation uses
    the complete frozen 21,025-point lattice.
    """

    parameter_map = pd.read_csv(parameter_map_path)
    required = {
        "hip_delta",
        "knee_delta",
        "phase_delta",
        "trajectory_id",
        "domain_coverage",
        "global_rom_valid",
        "workspace_valid",
        "jacobian_valid",
        "force_mapping_valid",
        "closure_valid",
        "continuity_valid",
        "asymmetry_valid",
        "finite_valid",
        "parent_reference_sha256",
    }
    missing = required.difference(parameter_map.columns)
    if missing:
        raise ValueError(f"prediction lattice missing columns: {sorted(missing)}")
    if not parameter_map["parent_reference_sha256"].astype(str).eq(
        ACTIVE_REFERENCE_SHA256
    ).all():
        raise RuntimeError("prediction lattice parent reference SHA mismatch")
    if maximum_points is not None:
        parameter_map = parameter_map.iloc[: int(maximum_points)].copy()
    hip_cache, knee_cache = _component_cache(parameter_map)
    neutral = generate_personalized_trajectory().trajectory
    time_s = neutral["time_s"].to_numpy(dtype=float)
    baseline_template = baseline_template_from_dynamic_subject(
        get_dynamic_subject("baseline")
    )
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for result in results:
        if result.status != INITIAL_IDENTIFICATION_COMPLETE:
            summary_rows.append(
                {
                    "case_id": _case_id(result),
                    "subject_id": result.subject_id,
                    "truth_scenario": result.truth_scenario,
                    "theta_hat_0_available": False,
                    "full_prediction_map_generated": False,
                    "geometrically_admissible_points": 0,
                    "supported_prediction_points": 0,
                    "unsupported_extrapolation_points": 0,
                    "reason": "theta_hat_0_not_available",
                }
            )
            continue
        assert result.theta_hat_0 is not None and result.d_init is not None
        case = _case_id(result)
        envelope = envelopes[case]
        domain = _fit_initial_domain(result.d_init)
        neutral_hip, neutral_knee = predict_joint_torque(
            neutral,
            baseline_template,
            result.theta_hat_0,
            L1,
        )
        reference_metrics = compute_torque_metrics(
            time_s, neutral_hip, neutral_knee
        )
        case_rows: list[dict[str, Any]] = []
        for row in parameter_map.itertuples(index=False):
            hip = hip_cache[float(row.hip_delta)]
            knee = knee_cache[(float(row.knee_delta), float(row.phase_delta))]
            trajectory = _combined_trajectory(time_s, hip, knee)
            global_geometry = bool(
                row.global_rom_valid
                and row.workspace_valid
                and row.jacobian_valid
                and row.force_mapping_valid
                and row.closure_valid
                and row.continuity_valid
                and row.asymmetry_valid
                and row.finite_valid
            )
            patient_valid = envelope.contains(trajectory)
            geometrically_admissible = bool(global_geometry and patient_valid)
            if not geometrically_admissible:
                continue
            coverage, supported = prediction_support(
                trajectory,
                domain,
                minimum_coverage_percent=INITIAL_DOMAIN_COVERAGE_MINIMUM_PERCENT,
            )
            j_pred = predict_mechanical_cost(
                trajectory,
                baseline_template,
                result.theta_hat_0,
                reference_metrics,
            )
            case_rows.append(
                {
                    "case_id": case,
                    "subject_id": result.subject_id,
                    "truth_scenario": result.truth_scenario,
                    "trajectory_id": str(row.trajectory_id),
                    "hip_delta": float(row.hip_delta),
                    "knee_delta": float(row.knee_delta),
                    "phase_delta": float(row.phase_delta),
                    "geometrically_admissible": True,
                    "J_pred": j_pred,
                    "domain_coverage": coverage,
                    "model_supported": supported,
                    "prediction_label": (
                        SUPPORTED_PREDICTION
                        if supported
                        else UNSUPPORTED_EXTRAPOLATION
                    ),
                    "can_calculate_equals_can_trust": False,
                    "theta_hat_0_dataset_sha256": result.summary[
                        "initial_identification_dataset_sha"
                    ],
                    "parent_reference_sha256": ACTIVE_REFERENCE_SHA256,
                }
            )
        detail_rows.extend(case_rows)
        case_table = pd.DataFrame(case_rows)
        supported_count = (
            int(case_table["model_supported"].astype(bool).sum())
            if not case_table.empty
            else 0
        )
        summary_rows.append(
            {
                "case_id": case,
                "subject_id": result.subject_id,
                "truth_scenario": result.truth_scenario,
                "theta_hat_0_available": True,
                "full_prediction_map_generated": True,
                "geometrically_admissible_points": int(len(case_table)),
                "supported_prediction_points": supported_count,
                "unsupported_extrapolation_points": int(
                    len(case_table) - supported_count
                ),
                "minimum_J_pred": float(case_table["J_pred"].min()),
                "maximum_J_pred": float(case_table["J_pred"].max()),
                "can_calculate_equals_can_trust": False,
                "reason": "",
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def _save_figure(path: Path, figure: plt.Figure) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")
    boxes = [
        (0.08, 0.72, "Pre-supplied patient\noperational envelope"),
        (0.37, 0.72, "Select one constraint-valid\ninformative excitation"),
        (0.68, 0.72, "Virtually execute Trial i\n(i <= 5)"),
        (0.68, 0.39, "Audit all 5 parameters\nrank / SVD / corr / uncertainty"),
        (0.37, 0.39, "Sufficient under an\nexplicit reviewed rule?"),
        (0.08, 0.39, "YES: freeze theta_hat_0\nand D_init"),
        (0.37, 0.08, "NO: diagnose weakest\ndirection; choose next trial"),
        (0.73, 0.08, "After Trial 5: fail closed\nno theta_hat_0"),
    ]
    for x, y, label in boxes:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="#EAF2F8", ec="#1F4E79"),
        )
    arrows = [
        ((0.17, 0.72), (0.28, 0.72)),
        ((0.50, 0.72), (0.59, 0.72)),
        ((0.68, 0.64), (0.68, 0.49)),
        ((0.59, 0.39), (0.49, 0.39)),
        ((0.28, 0.39), (0.18, 0.39)),
        ((0.37, 0.31), (0.37, 0.17)),
        ((0.43, 0.13), (0.65, 0.13)),
        ((0.37, 0.17), (0.47, 0.64)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=1.7),
        )
    ax.text(0.22, 0.43, "YES", transform=ax.transAxes, color="#1B7F3A")
    ax.text(0.39, 0.29, "NO", transform=ax.transAxes, color="#A13B2D")
    ax.set_title(
        "Safeguarded Sequential Initial Identification (offline architecture)",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        0.97,
        "Hard safeguard remains independent: NOT_DEFINED_NOT_APPROVED",
        transform=ax.transAxes,
        ha="center",
        color="#A13B2D",
        fontsize=10,
    )
    _save_figure(path, fig)


def _plot_parameter_identifiability(path: Path, table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    primary = table.loc[
        table["truth_scenario"].eq("matched_linear")
        & ~table["subject_id"].eq("LIMITED_ROM_VIRTUAL_SUBJECT")
    ]
    aggregated = primary.groupby(["trial_id", "parameter"], as_index=False)[
        "uncertainty_proxy"
    ].mean()
    for parameter, group in aggregated.groupby("parameter", sort=False):
        ax.plot(
            group["trial_id"],
            group["uncertainty_proxy"],
            marker="o",
            label=parameter,
        )
    ax.set_xlabel("Executed identification trial")
    ax.set_ylabel("Design uncertainty proxy (lower is better)")
    ax.set_title("Five-parameter identifiability after each executed trial")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save_figure(path, fig)


def _plot_history_metric(
    path: Path,
    history: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (subject, scenario), group in history.groupby(
        ["subject_id", "truth_scenario"], sort=False
    ):
        label = subject if scenario == "matched_linear" else f"{subject}/{scenario}"
        ax.plot(group["trial_id"], group[column], marker="o", label=label)
    ax.set_xlabel("Executed identification trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save_figure(path, fig)


def _plot_excitation_sequence(path: Path, result: SequentialIdentificationResult) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for trial_id, group in result.executed_identification_data.groupby("trial_id"):
        duration = float(group["excitation_duration_s"].iloc[0])
        time_axis = np.linspace(0.0, duration, len(group))
        axes[0].plot(time_axis, np.rad2deg(group["q_hip_rad"]), label=f"Trial {trial_id}")
        axes[1].plot(time_axis, np.rad2deg(group["q_knee_rad"]), label=f"Trial {trial_id}")
    axes[0].set_ylabel("Hip angle (deg)")
    axes[1].set_ylabel("Knee angle (deg)")
    axes[1].set_xlabel("Excitation time (s)")
    axes[0].set_title("Actually selected excitation sequence (baseline virtual case)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    _save_figure(path, fig)


def _plot_envelopes(path: Path, envelopes: dict[str, PatientOperationalEnvelope]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.add_patch(
        Rectangle(
            (FORMAL_HIP_ROM_DEG[0], FORMAL_KNEE_ROM_DEG[0]),
            FORMAL_HIP_ROM_DEG[1] - FORMAL_HIP_ROM_DEG[0],
            FORMAL_KNEE_ROM_DEG[1] - FORMAL_KNEE_ROM_DEG[0],
            facecolor="#D9EAF7",
            edgecolor="#1F4E79",
            alpha=0.5,
            label="Global model ROM (not patient safety ROM)",
        )
    )
    default = next(value for key, value in envelopes.items() if "LIMITED" not in key)
    limited = next(value for key, value in envelopes.items() if "LIMITED" in key)
    for env, color, label in (
        (default, "#4C9F70", "Default synthetic operational envelope"),
        (limited, "#D9822B", "LIMITED_ROM synthetic fixture"),
    ):
        ax.add_patch(
            Rectangle(
                (env.patient_hip_min_deg, env.patient_knee_min_deg),
                env.patient_hip_max_deg - env.patient_hip_min_deg,
                env.patient_knee_max_deg - env.patient_knee_min_deg,
                fill=False,
                edgecolor=color,
                linewidth=2.2,
                label=label,
            )
        )
    ax.set_xlim(-3, 123)
    ax.set_ylim(1, 149)
    ax.set_xlabel("Hip flexion (deg)")
    ax.set_ylabel("Knee flexion (deg)")
    ax.set_title("Global model ROM vs patient-specific operational envelopes")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    _save_figure(path, fig)


def _plot_prediction_map(path: Path, prediction_map: pd.DataFrame) -> None:
    baseline = prediction_map.loc[
        prediction_map["case_id"].eq("baseline__matched_linear")
        & np.isclose(prediction_map["phase_delta"], 0.0)
    ]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(
        baseline["hip_delta"],
        baseline["knee_delta"],
        c=baseline["J_pred"],
        cmap="viridis",
        s=28,
        edgecolor=np.where(baseline["model_supported"], "none", "#D64541"),
        linewidth=0.8,
    )
    fig.colorbar(scatter, ax=ax, label="Predicted mechanical cost J")
    ax.set_xlabel("Hip amplitude perturbation (deg)")
    ax.set_ylabel("Knee amplitude perturbation (deg)")
    ax.set_title("theta_hat_0 to full prediction lattice (phase shift = 0)")
    ax.text(
        0.02,
        0.02,
        "Red outline: calculated but unsupported extrapolation\ncan calculate != can trust",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999999"),
    )
    ax.grid(alpha=0.2)
    _save_figure(path, fig)


def _architecture_document() -> str:
    return f"""# Safeguard / Identification / Personalization Architecture

Protocol: `{PROTOCOL_ID}`

Evidence level: deterministic offline virtual-subject software validation only.
Nothing in this artifact is robot, human, clinical, comfort, safety, or
effectiveness validation.

## Layer 1 — HARD SAFEGUARD

Status: `{REAL_ROBOT_HARD_SAFEGUARD}`.

This is an independent, future real-robot fail-closed protection layer.
Neither identification nor personalization may override it.  The global model
ROM (hip 0–120 deg, knee 5–145 deg) is not a universal patient-safe ROM.

## Layer 2 — SEQUENTIAL INITIAL IDENTIFICATION

One candidate is selected and virtually executed at a time.  Selection sees
only executed identification data, the current temporary five-parameter model,
global model constraints, and the current pre-supplied patient operational
envelope.  It ranks candidates lexicographically by constraint validity, rank,
minimum singular value, condition number, worst correlation, weakest-parameter
sensitivity, incremental state/regressor coverage, excursion, and stable ID.

After every trial, all five local equivalent dynamics parameters are audited.
The process stops immediately when a reviewed rule is met and never exceeds
{MAX_INITIAL_IDENTIFICATION_TRIALS} trials.  Failure produces no `theta_hat_0`.
The repository currently has no approved complete stop rule, so the saved
virtual experiment uses an explicitly non-frozen illustrative comparator.  The
authoritative default remains `{IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW}`.

`excitation_duration_s` is independent of the unchanged 24 s rehabilitation
reference.  The tested durations are a research design range, not human-safety
limits.

## Layer 3 — SEQUENTIAL PERSONALIZATION (future interface only)

On successful identification only:

```text
theta_hat_0 + D_init + full prediction map + initial known region
    -> future EXPLOIT / EXPLORE
    -> execute one approved personalization trial
    -> theta_hat_(k+1) and updated support map
```

No personalization is implemented or executed by this task.  Unsupported
geometrically admissible points retain a calculated `J_pred` but are labelled
`UNSUPPORTED_EXTRAPOLATION`: can calculate does not mean can trust.

## Constraint ownership

| Layer | Meaning | May this module change it? |
|---|---|---|
| GLOBAL_MODEL_CONSTRAINTS | Model ROM, workspace, Jacobian, force mapping, C2, finite values | No; validate only |
| PATIENT_SPECIFIC_OPERATIONAL_ENVELOPE | Pre-supplied conservative local region | No; `AUTO_EXPAND_PATIENT_ENVELOPE=false` |
| REAL_ROBOT_HARD_SAFEGUARD | Future independent hardware protection | No; not defined or approved |
"""


def _leakage_document(results: Iterable[SequentialIdentificationResult]) -> str:
    cases = "\n".join(
        f"- `{_case_id(result)}`: {result.trials_required} oracle calls, "
        f"selection-before-execution asserted, held-out test absent."
        for result in results
    )
    return f"""# Data Leakage Audit

## Selection inputs

- Executed identification observations through trial `i-1` only.
- Current temporary five-parameter estimate.
- Predeclared candidate specifications and their model-only sensitivity.
- Global model constraints and the unchanged patient operational envelope.

## Prohibited inputs

- Truth subject label or truth five-parameter vector.
- Future virtual trial outcomes.
- Held-out test data.
- Mechanical personalization objective `J`.
- Final prediction-map results.

The selector function has no truth-oracle or held-out-test argument.  The
virtual oracle is called only after a candidate has been selected.  Truth is
used only to generate the post-selection observation and to describe the case
afterward.  The within-identification validation rows are sampled from trials
that have already been executed and never enter candidate ranking as future
outcomes.

## Case audit

{cases}

## Stop-rule audit

The existing repository supplies numerical rank/SVD/correlation/uncertainty
metrics but no approved complete set of stopping thresholds.  Therefore the
default is `{IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW}`.  The values used in
this illustrative software comparison are recorded in metadata.  They are not
claimed as a selected scientific threshold and must not be promoted to a
real-subject release rule by interpreting virtual recovery results.
"""


def _method_migration_audit_document() -> str:
    return f"""# Sequential Initial Identification Method Migration Audit

## Active new method

- `safeguarded_sequential_initial_identification.py` is the only implementation
  of `{PROTOCOL_ID}`.
- `MAX_INITIAL_IDENTIFICATION_TRIALS={MAX_INITIAL_IDENTIFICATION_TRIALS}`;
  early stop is supported and a sixth trial is structurally prohibited.
- Every identification candidate owns an independent
  `excitation_duration_s`.  The tested values are explicitly
  `RESEARCH_DESIGN_RANGE_NOT_HUMAN_SAFETY_LIMIT`.
- Default completion remains fail-closed as
  `{IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW}`.

## Frozen 24 s content that must remain

- `reference_measured_asymmetric_closed_slow` remains a 24 s rehabilitation
  reference with SHA-256 `{ACTIVE_REFERENCE_SHA256}`.
- `continuous_reference_neighborhood.py` keeps `TOTAL_DURATION_S=24.0` and a
  fixed time scale because it defines the frozen rehabilitation-personalization
  family, not the new identification-duration policy.
- The new identification generator first obtains a C2 geometric member of that
  family, then applies an independent linear time change with exact chain-rule
  scaling of velocity and acceleration.  It does not edit the reference file or
  the existing generator equations.

## Historical method retained, not active for this task

- `sequential_personalization.py` and
  `formal_artifacts/sequential_personalization/` preserve the earlier workflow
  in which one pre-existing training dataset seeds personalization and metadata
  records 24 s.  These are historical prior-stage evidence; this task neither
  imports nor executes that personalization path.
- Existing Stage 4 identification trajectory families and speed profiles are
  retained as earlier software evidence.  They are not silently relabelled as
  the new patient-envelope-aware 1–5 trial protocol.

## Existing identifiability threshold audit

- `identifiability_analysis.py` contains a numerical SVD rank tolerance and a
  correlation reporting threshold.  These are numerical/diagnostic mechanisms,
  not an approved multi-criterion completion rule.
- `parameter_estimator.py` reports optimizer success, singular values,
  covariance-shaped uncertainty, standard errors, and residuals, but does not
  define when a new subject is sufficiently identified.
- No approved conjunction of rank, minimum singular value, condition number,
  worst correlation, all-five-parameter uncertainty/sensitivity, and validation
  residual was found.  No real-subject threshold was invented during migration.

## Constraint audit

- Global ROM/workspace/Jacobian/force mapping/finite/C2 checks remain model
  constraints and are never called patient-safety limits.
- Patient operational envelopes are supplied inputs, never expanded by
  constraint violation, large force, error, or pain.
- Real-robot hard safeguard status remains
  `{REAL_ROBOT_HARD_SAFEGUARD}`; therefore no physical execution is authorized.
"""


def generate_formal_artifacts(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    maximum_prediction_points: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_active_reference_file()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results, envelopes = run_virtual_cases()

    tables = {
        "identification_trial_history.csv": _concat(results, "trial_history"),
        "identification_trial_candidates.csv": _concat(results, "trial_candidates"),
        "parameter_identifiability_by_trial.csv": _concat(
            results, "parameter_identifiability"
        ),
        "parameter_estimates_by_trial.csv": _concat(results, "parameter_estimates"),
        "incremental_information_gain.csv": _concat(
            results, "incremental_information_gain"
        ),
        "patient_operational_envelope_history.csv": _concat(
            results, "patient_envelope_history"
        ),
        "subject_identification_summary.csv": _subject_summary(results),
    }
    summary = tables["subject_identification_summary.csv"]
    tables["failure_case_summary.csv"] = summary.loc[
        ~summary["theta_hat_0_frozen"].astype(bool)
    ].copy()

    for result in results:
        if result.d_init is not None:
            result.d_init.to_csv(output / f"D_init_{_case_id(result)}.csv", index=False)

    initial_models: dict[str, Any] = {}
    initial_known_regions: dict[str, Any] = {}
    for result in results:
        if result.theta_hat_0 is None or result.d_init is None:
            continue
        case = _case_id(result)
        domain = _fit_initial_domain(result.d_init)
        initial_models[case] = {
            "initial_subject_model": result.theta_hat_0,
            "parameter_order": list(PARAMETER_NAMES),
            "parameter_interpretation": "local_equivalent_dynamics_parameters",
            "is_tissue_material_constant": False,
            "initial_identification_trial_count": result.trials_required,
            "initial_identification_dataset_sha": result.summary[
                "initial_identification_dataset_sha"
            ],
            "stop_rule_status": result.summary[
                "identifiability_stop_rule_status"
            ],
        }
        initial_known_regions[case] = {
            "classification": "INITIAL_KNOWN_IDENTIFICATION_REGION_FROM_D_INIT",
            "columns": list(domain.columns),
            "lower": list(domain.lower),
            "upper": list(domain.upper),
            "valid_executed_samples": domain.valid_training_samples,
            "D_init_sha256": result.summary[
                "initial_identification_dataset_sha"
            ],
            "is_patient_safety_envelope": False,
            "does_not_expand_patient_operational_envelope": True,
        }
    _json_dump(output / "initial_subject_models.json", initial_models)
    _json_dump(output / "initial_known_regions.json", initial_known_regions)

    prediction_map, prediction_summary = build_full_prediction_map(
        results,
        envelopes,
        maximum_points=maximum_prediction_points,
    )
    tables["full_prediction_map_summary.csv"] = prediction_summary
    tables["full_prediction_map.csv"] = prediction_map
    for name, table in tables.items():
        table.to_csv(output / name, index=False)

    (output / "SAFEGUARD_IDENTIFICATION_PERSONALIZATION_ARCHITECTURE.md").write_text(
        _architecture_document(), encoding="utf-8"
    )
    (output / "DATA_LEAKAGE_AUDIT.md").write_text(
        _leakage_document(results), encoding="utf-8"
    )
    (output / "IDENTIFIABILITY_STOP_RULE_AUDIT.md").write_text(
        """# Identifiability Stop Rule Audit

The repository already computes numerical rank, singular values, condition
number, parameter correlation, information diagonals, covariance-shaped
uncertainty, optimizer standard errors, and residual metrics.  It does not
contain an approved complete conjunction of rank, conditioning, correlation,
per-parameter uncertainty/sensitivity, and validation thresholds for a new
subject.

Therefore the authoritative state is
`IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW`.  The numeric rule recorded in
`metadata.json` is an illustrative virtual-research comparator used only to
test early-stop and fail-closed behavior.  It is not a human, clinical, robot,
or safety release rule, and this artifact makes no scientific threshold-
selection claim.  Runtime selection never reads held-out test results.
""",
        encoding="utf-8",
    )
    (output / "METHOD_MIGRATION_AUDIT.md").write_text(
        _method_migration_audit_document(), encoding="utf-8"
    )

    _plot_flowchart(output / "sequential_initial_identification_flowchart.png")
    _plot_parameter_identifiability(
        output / "parameter_identifiability_by_trial.png",
        tables["parameter_identifiability_by_trial.csv"],
    )
    _plot_history_metric(
        output / "condition_number_by_trial.png",
        tables["identification_trial_history.csv"],
        "condition_number",
        "Condition number after each executed trial",
        "Condition number (lower is better)",
    )
    _plot_history_metric(
        output / "parameter_correlation_by_trial.png",
        tables["identification_trial_history.csv"],
        "maximum_abs_parameter_correlation",
        "Worst parameter correlation after each executed trial",
        "Maximum absolute off-diagonal correlation",
    )
    baseline = next(
        result
        for result in results
        if _case_id(result) == "baseline__matched_linear"
    )
    _plot_excitation_sequence(output / "excitation_trajectories_sequence.png", baseline)
    _plot_envelopes(output / "patient_envelope_vs_global_rom.png", envelopes)
    _plot_prediction_map(
        output / "identification_to_global_prediction_map.png", prediction_map
    )

    artifact_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "metadata.json"
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "evidence_level": "OFFLINE_VIRTUAL_SUBJECT_SOFTWARE_VALIDATION_ONLY",
        "git_commit_at_generation": _git_commit(),
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "global_model_rom": {
            "hip_deg": list(FORMAL_HIP_ROM_DEG),
            "knee_deg": list(FORMAL_KNEE_ROM_DEG),
            "is_universal_patient_safe_rom": False,
        },
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "active_reference_id": ACTIVE_REFERENCE_ID,
        "active_reference_path": str(ACTIVE_REFERENCE_PATH.relative_to(MODULE_DIR.parent)),
        "active_reference_sha256": sha256_file(ACTIVE_REFERENCE_PATH),
        "active_reference_duration_s": 24.0,
        "active_reference_modified": False,
        "maximum_initial_identification_trials": MAX_INITIAL_IDENTIFICATION_TRIALS,
        "auto_expand_patient_envelope": AUTO_EXPAND_PATIENT_ENVELOPE,
        "real_robot_hard_safeguard": REAL_ROBOT_HARD_SAFEGUARD,
        "real_robot_motion_executed": False,
        "real_patient_safety_thresholds_defined": False,
        "personalization_executed": False,
        "formal_personalization_implemented": False,
        "identifiability_threshold_audit": {
            "approved_complete_stop_rule_found": False,
            "authoritative_status": IDENTIFIABILITY_STOP_RULE_REQUIRES_REVIEW,
            "existing_metrics_reused": [
                "numerical_rank",
                "singular_values",
                "condition_number",
                "parameter_correlation",
                "uncertainty_proxy",
                "per_parameter_sensitivity",
                "training_residual",
                "within_identification_validation_residual",
            ],
            "virtual_research_comparator": {
                **VIRTUAL_RESEARCH_COMPARATOR_RULE.__dict__,
                "not_human_or_robot_release_rule": True,
                "threshold_selection_claimed": False,
                "heldout_test_used_by_runtime": False,
            },
        },
        "duration_design": {
            "field": "excitation_duration_s",
            "candidate_values_s": sorted(
                {
                    spec.excitation_duration_s
                    for spec in default_virtual_research_candidate_pool()
                }
            ),
            "status": "RESEARCH_DESIGN_RANGE_NOT_HUMAN_SAFETY_LIMIT",
            "coupled_to_reference_duration": False,
        },
        "virtual_cases": [result.summary for result in results],
        "full_prediction_map": {
            "lattice_source": str(PARAMETER_MAP_PATH.relative_to(MODULE_DIR.parent)),
            "lattice_source_sha256": sha256_file(PARAMETER_MAP_PATH),
            "formal_complete_lattice_used": maximum_prediction_points is None,
            "minimum_support_coverage_percent": INITIAL_DOMAIN_COVERAGE_MINIMUM_PERCENT,
            "can_calculate_equals_can_trust": False,
        },
        "legacy_method_status": {
            "single_initial_identification_trajectory": "SUPERSEDED_NOT_ACTIVE_METHOD",
            "identification_duration_fixed_to_reference_24s": "SUPERSEDED_NOT_ACTIVE_METHOD",
            "historical_sequential_personalization_artifacts_modified": False,
        },
        "artifact_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
        "runtime_s": time.perf_counter() - started,
    }
    _json_dump(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline sequential-identification formal artifacts."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    metadata = generate_formal_artifacts(arguments.output_dir)
    cases = metadata["virtual_cases"]
    completed = sum(case["status"] == INITIAL_IDENTIFICATION_COMPLETE for case in cases)
    print(
        f"{PROTOCOL_ID}: {completed}/{len(cases)} virtual cases completed; "
        f"artifacts={arguments.output_dir}; robot_motion=false"
    )


if __name__ == "__main__":
    main()
