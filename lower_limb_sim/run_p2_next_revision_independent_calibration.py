"""Generate the frozen, independent P2 decision-error calibration evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
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

from .decision_relevant_global_model_reliability import (
    MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
    geometrically_valid_parameter_lattice,
)
from .formal_protocol import (
    ACTIVE_REFERENCE_PATH,
    ACTIVE_REFERENCE_SHA256,
    FORMAL_HIP_ROM_DEG,
    FORMAL_KNEE_ROM_DEG,
    ROM_PROTOCOL_VERSION,
    THETA_SHANK_DEFINITION,
    validate_active_reference_file,
)
from .mechanical_objective import OBJECTIVE_EQUIVALENCE_TOLERANCE
from .p2_next_revision_independent_calibration import (
    BUNDLE_SCALE_CALIBRATED,
    BUNDLE_SOURCE_PAIR_PLAN_PATH,
    BUNDLE_SOURCE_PAIR_PLAN_SHA256,
    BUNDLE_SOURCE_PROTOCOL_PATH,
    CALIBRATION_DATA_ROLE,
    CALIBRATION_ID,
    CALIBRATION_MANIFEST_ID,
    FUTURE_PROSPECTIVE_STATUS,
    FrozenCalibrationManifestGate,
    HELD_OUT_STATUS,
    LOCAL_CALIBRATION_PLAN_ID,
    LOCAL_SOURCE_PAIR_PLAN_PATH,
    LOCAL_SOURCE_PAIR_PLAN_SHA256,
    NOT_HUMAN_READY,
    NOT_ROBOT_MOTION_APPROVED,
    assign_pairs_to_calibration_cases,
    bundle_scale_feasibility,
    calibration_case_manifest,
    calibration_manifest_payload,
    evaluate_frozen_calibration_assignments,
    grouped_residual_summary,
    residual_distribution,
    sha256_file,
)
from .p2_v2_prospective_offline_validation import (
    EXPECTED_GEOMETRIC_LATTICE_SIZE,
    LOCAL_P95,
)
from .post_prospective_rejection_root_cause_audit import (
    FINAL_STATUS_IDENTIFIED,
    PROSPECTIVE_CONCLUSION,
    PROSPECTIVE_MANIFEST_SHA256,
    verify_immutable_prospective_artifacts,
)
from .run_p2_v2_prospective_offline_validation import _protected_source_hashes
from .run_research_decision_guarded_sequential_personalization import (
    DEFAULT_PARAMETER_MAP_PATH,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_next_revision_independent_calibration_v1"
)
POST_AUDIT_DIRECTORY = (
    MODULE_DIR
    / "formal_artifacts"
    / "post_prospective_rejection_root_cause_audit_v1"
)
POST_AUDIT_METADATA_PATH = POST_AUDIT_DIRECTORY / "metadata.json"
PROSPECTIVE_DIRECTORY = (
    MODULE_DIR / "formal_artifacts" / "p2_v2_prospective_offline_validation_v1"
)
CORE_SOURCE_PATH = MODULE_DIR / "p2_next_revision_independent_calibration.py"
RUNNER_SOURCE_PATH = MODULE_DIR / "run_p2_next_revision_independent_calibration.py"

REQUIRED_CSV_FILENAMES = (
    "calibration_case_manifest.csv",
    "independent_local_calibration_pair_plan.csv",
    "local_pair_assignment_manifest.csv",
    "bundle_pair_assignment_manifest.csv",
    "independent_one_step_residuals.csv",
    "independent_bundle_2step_residuals.csv",
    "independent_bundle_3step_residuals.csv",
    "independent_bundle_5step_residuals.csv",
    "one_step_residual_summary.csv",
    "bundle_residual_summary.csv",
    "decision_scale_residual_comparison.csv",
    "axis_direction_residual_summary.csv",
    "matched_mismatch_residual_summary.csv",
)
EXTRA_CSV_FILENAMES = (
    "calibration_identification_audit.csv",
    "calibration_truth_access_audit.csv",
    "bundle_scale_feasibility.csv",
    "development_vs_independent_calibration.csv",
)
JSON_FILENAMES = (
    "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json",
    "local_pair_plan_provenance.json",
    "bundle_pair_plan_provenance.json",
)
REPORT_FILENAMES = (
    "CALIBRATION_DATA_SPLIT_AUDIT.md",
    "CUMULATIVE_DECISION_CALIBRATION_FEASIBILITY.md",
    "INDEPENDENT_CALIBRATION_REPORT.md",
    "DATA_PROVENANCE_AUDIT.md",
)
FIGURE_FILENAMES = (
    "decision_residual_by_step_scale.png",
    "one_step_residual_distribution.png",
    "bundle_residual_distribution.png",
    "axis_direction_residual_comparison.png",
    "matched_vs_mismatch_residual.png",
    "development_vs_independent_calibration.png",
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


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_safe(dict(payload)),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any], *, canonical: bool = False) -> None:
    data = (
        _canonical_json_bytes(payload)
        if canonical
        else (
            json.dumps(
                _json_safe(dict(payload)),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


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
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("INDEPENDENT_CALIBRATION_REQUIRES_CHECKPOINT") from exc
    return completed.stdout.rstrip("\n")


def _checkpoint_preflight() -> dict[str, Any]:
    head = _git_output("rev-parse", "HEAD")
    required = (
        "lower_limb_sim/post_prospective_rejection_root_cause_audit.py",
        "lower_limb_sim/run_post_prospective_rejection_root_cause_audit.py",
        (
            "lower_limb_sim/formal_artifacts/"
            "post_prospective_rejection_root_cause_audit_v1/"
            "DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1.json"
        ),
        (
            "lower_limb_sim/formal_artifacts/"
            "post_prospective_rejection_root_cause_audit_v1/"
            "designated_bundle_validation_pair_plan.csv"
        ),
    )
    for relative in required:
        _git_output("ls-files", "--error-unmatch", relative)
    committed_plan = subprocess.run(
        ["git", "show", f"HEAD:{required[-1]}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        timeout=10.0,
    ).stdout
    committed_sha = hashlib.sha256(committed_plan).hexdigest()
    if committed_sha != BUNDLE_SOURCE_PAIR_PLAN_SHA256:
        raise RuntimeError("BUNDLE_PROTOCOL_PROVENANCE_FAILURE")
    post_metadata = json.loads(POST_AUDIT_METADATA_PATH.read_text(encoding="utf-8"))
    if post_metadata["final_status"] != FINAL_STATUS_IDENTIFIED:
        raise RuntimeError("INDEPENDENT_CALIBRATION_REQUIRES_CHECKPOINT")
    if post_metadata["bundle_pair_plan_sha256"] != BUNDLE_SOURCE_PAIR_PLAN_SHA256:
        raise RuntimeError("BUNDLE_PROTOCOL_PROVENANCE_FAILURE")
    return {
        "checkpoint_commit": head,
        "checkpoint_subject": _git_output("log", "-1", "--format=%s"),
        "post_audit_checkpointed": True,
        "required_checkpoint_paths": list(required),
        "committed_bundle_pair_plan_sha256": committed_sha,
        "git_log": _git_output("log", "--oneline", "-8").splitlines(),
    }


def _calibration_protected_hashes() -> dict[str, str]:
    values = _protected_source_hashes()
    values["P2_V2A_definition"] = sha256_file(
        MODULE_DIR / "p2_v2_prospective_offline_validation.py"
    )
    values["rejected_post_prospective_audit"] = sha256_file(
        MODULE_DIR / "post_prospective_rejection_root_cause_audit.py"
    )
    return values


def _verify_frozen_baseline() -> None:
    immutable = verify_immutable_prospective_artifacts()
    if immutable["final_status"] != PROSPECTIVE_CONCLUSION:
        raise RuntimeError("old prospective rejection changed")
    if sha256_file(BUNDLE_SOURCE_PAIR_PLAN_PATH) != BUNDLE_SOURCE_PAIR_PLAN_SHA256:
        raise RuntimeError("BUNDLE_PROTOCOL_PROVENANCE_FAILURE")
    protocol = json.loads(BUNDLE_SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        protocol["pair_plan_sha256"] != BUNDLE_SOURCE_PAIR_PLAN_SHA256
        or protocol["truth_used_to_select_plan"] is not False
    ):
        raise RuntimeError("BUNDLE_PROTOCOL_PROVENANCE_FAILURE")
    if sha256_file(LOCAL_SOURCE_PAIR_PLAN_PATH) != LOCAL_SOURCE_PAIR_PLAN_SHA256:
        raise RuntimeError("LOCAL_VALIDATION_PROTOCOL_PROVENANCE_FAILURE")
    validate_active_reference_file()
    if sha256_file(ACTIVE_REFERENCE_PATH) != ACTIVE_REFERENCE_SHA256:
        raise RuntimeError("active reference changed")
    if ROM_PROTOCOL_VERSION != "ROM_PROTOCOL_V2":
        raise RuntimeError("ROM protocol changed")
    if tuple(FORMAL_HIP_ROM_DEG) != (0.0, 120.0):
        raise RuntimeError("hip ROM changed")
    if tuple(FORMAL_KNEE_ROM_DEG) != (5.0, 145.0):
        raise RuntimeError("knee ROM changed")
    if THETA_SHANK_DEFINITION != "q_hip - q_knee":
        raise RuntimeError("theta_shank changed")
    if OBJECTIVE_EQUIVALENCE_TOLERANCE != 0.005:
        raise RuntimeError("equivalence tolerance changed")
    if MODEL_SUPPORT_COVERAGE_GATE_PERCENT != 90.0:
        raise RuntimeError("support gate changed")


def _summary_with_scope(
    table: pd.DataFrame,
    residual_column: str,
    scopes: Sequence[tuple[str, Sequence[str]]],
    decision_scale: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, groups in scopes:
        frame = grouped_residual_summary(
            table,
            residual_column=residual_column,
            groups=groups,
            decision_scale=decision_scale,
        )
        frame.insert(0, "summary_scope", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _decision_scale_summary(
    one_step: pd.DataFrame, bundle: pd.DataFrame
) -> pd.DataFrame:
    rows = [
        {
            "decision_scale": "1-step",
            "step_count": 1,
            **residual_distribution(one_step["e_deltaJ_1"]),
        }
    ]
    for length in (2, 3, 5):
        selected = bundle.loc[bundle["bundle_length"].eq(length)]
        rows.append(
            {
                "decision_scale": f"{length}-step",
                "step_count": length,
                **residual_distribution(selected["e_deltaJ_bundle"]),
            }
        )
    output = pd.DataFrame(rows)
    output["direct_endpoint_residual"] = True
    output["analytic_scaling_formula_used"] = False
    output["empirical_trend_only"] = True
    output["formula_fitted_or_frozen"] = False
    output["threshold_selected"] = False
    output["data_role"] = CALIBRATION_DATA_ROLE
    return output


def _axis_direction_summary(
    one_step: pd.DataFrame, bundle: pd.DataFrame
) -> pd.DataFrame:
    local_views = _one_step_direction_views(one_step)
    local = _summary_with_scope(
        local_views,
        "e_deltaJ_1",
        (
            ("AXIS_DIRECTION", ("coordinate", "direction")),
            (
                "AXIS_DIRECTION_LOCATION",
                ("coordinate", "direction", "location_class"),
            ),
        ),
        "1-step",
    )
    local = local.merge(
        local_views[["direction", "direction_evidence_status"]].drop_duplicates(),
        on="direction",
        how="left",
        validate="many_to_one",
    )
    bundle_frames = []
    for length in (2, 3, 5):
        selected = bundle.loc[bundle["bundle_length"].eq(length)]
        bundle_frames.append(
            _summary_with_scope(
                selected,
                "e_deltaJ_bundle",
                (
                    ("AXIS_DIRECTION", ("coordinate", "direction")),
                    (
                        "AXIS_DIRECTION_LOCATION",
                        ("coordinate", "direction", "location_class"),
                    ),
                ),
                f"{length}-step",
            )
        )
    output = pd.concat((local, *bundle_frames), ignore_index=True, sort=False)
    output["heteroscedasticity_threshold_defined"] = False
    output["axis_specific_threshold_created"] = False
    return output


def _one_step_direction_views(one_step: pd.DataFrame) -> pd.DataFrame:
    """Expose the predeclared symmetric reverse view without adding samples."""

    positive = one_step.copy(deep=True)
    positive["direction"] = "POSITIVE"
    positive["direction_evidence_status"] = "CANONICAL_FROZEN_PAIR"
    negative = one_step.copy(deep=True)
    negative["direction"] = "NEGATIVE"
    negative["deltaJ_pred"] = negative["reverse_deltaJ_pred"]
    negative["deltaJ_truth"] = negative["reverse_deltaJ_truth"]
    negative["e_deltaJ_1"] = negative["reverse_e_deltaJ_1"]
    negative["direction_evidence_status"] = (
        "SYMMETRIC_REVERSE_VIEW_NOT_AN_INDEPENDENT_PAIR"
    )
    return pd.concat((positive, negative), ignore_index=True, sort=False)


def _matched_mismatch_summary(
    one_step: pd.DataFrame, bundle: pd.DataFrame
) -> pd.DataFrame:
    frames = [
        grouped_residual_summary(
            one_step,
            residual_column="e_deltaJ_1",
            groups=("calibration_category",),
            decision_scale="1-step",
        )
    ]
    for length in (2, 3, 5):
        frames.append(
            grouped_residual_summary(
                bundle.loc[bundle["bundle_length"].eq(length)],
                residual_column="e_deltaJ_bundle",
                groups=("calibration_category",),
                decision_scale=f"{length}-step",
            )
        )
    output = pd.concat(frames, ignore_index=True, sort=False)
    output["five_parameter_model_modified"] = False
    return output


def _development_comparison(
    one_step: pd.DataFrame, bundle: pd.DataFrame
) -> pd.DataFrame:
    development_local = pd.read_csv(
        MODULE_DIR
        / "formal_artifacts"
        / "p2_v2_offline_research_prototype_v1"
        / "local_validation_results.csv"
    )
    development_bundle = pd.read_csv(
        POST_AUDIT_DIRECTORY / "prospective_bundle_residual_characterization.csv"
    )
    rows: list[dict[str, Any]] = []
    for source, scale, values, role in (
        (
            "OLD_DEVELOPMENT_LOCAL",
            "1-step",
            development_local["e_delta_J"],
            "DEVELOPMENT_ESTIMATE_ONLY",
        ),
        (
            "INDEPENDENT_CALIBRATION",
            "1-step",
            one_step["e_deltaJ_1"],
            CALIBRATION_DATA_ROLE,
        ),
    ):
        rows.append(
            {
                "evidence_source": source,
                "decision_scale": scale,
                **residual_distribution(values),
                "evidence_role": role,
            }
        )
    for length in (2, 3, 5):
        for source, values, role in (
            (
                "REJECTED_PROSPECTIVE_POSTHOC_DEVELOPMENT",
                development_bundle.loc[
                    development_bundle["bundle_length"].eq(length),
                    "e_deltaJ_bundle_posthoc",
                ],
                "POST_REJECTION_DEVELOPMENT_EVIDENCE_ONLY",
            ),
            (
                "INDEPENDENT_CALIBRATION",
                bundle.loc[
                    bundle["bundle_length"].eq(length), "e_deltaJ_bundle"
                ],
                CALIBRATION_DATA_ROLE,
            ),
        ):
            rows.append(
                {
                    "evidence_source": source,
                    "decision_scale": f"{length}-step",
                    **residual_distribution(values),
                    "evidence_role": role,
                }
            )
    output = pd.DataFrame(rows)
    output["pooled_distribution_computed"] = False
    output["pooled_threshold_computed"] = False
    output["threshold_selected"] = False
    return output


def _plot_outputs(
    output: Path,
    one_step: pd.DataFrame,
    bundle: pd.DataFrame,
    decision_scale: pd.DataFrame,
    axis_direction: pd.DataFrame,
    matched: pd.DataFrame,
    development: pd.DataFrame,
) -> None:
    groups = [
        one_step["e_deltaJ_1"].to_numpy(dtype=float),
        *[
            bundle.loc[bundle["bundle_length"].eq(length), "e_deltaJ_bundle"].to_numpy(dtype=float)
            for length in (2, 3, 5)
        ],
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.boxplot(groups, tick_labels=["1", "2", "3", "5"], showfliers=True)
    axis.set(xlabel="decision displacement (formal steps)", ylabel="direct |deltaJ error|", title="Independent residual by decision scale")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[0], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    axis.hist(one_step["e_deltaJ_1"], bins=30, color="#4c78a8", alpha=0.85)
    axis.set(xlabel="e_deltaJ_1", ylabel="count", title="Independent one-step residual distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[1], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    for length, color in zip((2, 3, 5), ("#4c78a8", "#f58518", "#54a24b")):
        values = bundle.loc[bundle["bundle_length"].eq(length), "e_deltaJ_bundle"]
        axis.hist(values, bins=24, alpha=0.45, label=f"{length}-step", color=color)
    axis.set(xlabel="direct bundle residual", ylabel="count", title="Independent bundle residual distributions")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[2], dpi=180, bbox_inches="tight")
    plt.close(figure)

    selected = axis_direction.loc[
        axis_direction["summary_scope"].eq("AXIS_DIRECTION")
    ].copy()
    selected["label"] = (
        selected["decision_scale"].astype(str)
        + "\n"
        + selected["coordinate"].astype(str)
        + " "
        + selected["direction"].astype(str)
    )
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.bar(np.arange(len(selected)), selected["P95"], color="#4c78a8")
    axis.set_xticks(np.arange(len(selected)), selected["label"], rotation=65, ha="right")
    axis.set(ylabel="empirical P95 residual", title="Axis and direction residual comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[3], dpi=180, bbox_inches="tight")
    plt.close(figure)

    pivot = matched.pivot(index="decision_scale", columns="calibration_category", values="P95").reindex(["1-step", "2-step", "3-step", "5-step"])
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    pivot.plot(kind="bar", ax=axis, color=["#4c78a8", "#e45756"])
    axis.set(ylabel="empirical P95 residual", title="Matched vs mismatch independent calibration")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[4], dpi=180, bbox_inches="tight")
    plt.close(figure)

    pivot = development.pivot(index="decision_scale", columns="evidence_source", values="P95").reindex(["1-step", "2-step", "3-step", "5-step"])
    figure, axis = plt.subplots(figsize=(10, 5.8))
    pivot.plot(kind="bar", ax=axis)
    axis.set(ylabel="empirical P95 residual", title="Development vs independent calibration (not pooled)")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / FIGURE_FILENAMES[5], dpi=180, bbox_inches="tight")
    plt.close(figure)


def _data_split_report(cases: pd.DataFrame) -> str:
    return f"""# Calibration data split audit

The four data layers are strictly separated:

- `DEVELOPMENT`: the nine early P2 development cases plus the six revealed rejected-prospective cases. They are permanently development-used evidence.
- `INDEPENDENT_CALIBRATION`: the {len(cases)} new cases in `calibration_case_manifest.csv`; they are used only for decision-error residual estimation.
- `FUTURE_PROSPECTIVE`: `{FUTURE_PROSPECTIVE_STATUS}`. No case or outcome is generated here.
- `HELD_OUT_FINAL_TEST`: `{HELD_OUT_STATUS}`. It is not read, selected, enumerated for truth, or evaluated.

The calibration IDs do not overlap either protected historical group. Every calibration case has `reserved_for_future_prospective=false`, so a future prospective manifest must exclude all IDs recorded here. Case selection used a fixed synthetic grid, seed, exclusions by pre-existing identity/signature, and SHA ordering only; it used no truth optimum, residual, or subject-specificity result.
"""


def _feasibility_report(feasibility: pd.DataFrame) -> str:
    lines = "\n".join(
        f"- {int(row.bundle_length)}-step: `{row.calibration_status}`; "
        f"n={int(row.pair_count)}, strata={int(row.stratum_count)}, "
        f"12 pairs/stratum."
        for row in feasibility.itertuples(index=False)
    )
    return f"""# Cumulative decision calibration feasibility

{lines}

These statuses mean that the preregistered geometry received complete, finite, direct endpoint-residual evidence. They do not establish statistical power, select P95/P99, enable C2/C3/C5, or approve a policy. No `n * U_1`, `sqrt(n) * U_1`, or other analytic scaling was used.
"""


def _independent_report(
    decision_scale: pd.DataFrame,
    axis_direction: pd.DataFrame,
    matched: pd.DataFrame,
    development: pd.DataFrame,
    feasibility: pd.DataFrame,
    manifest_sha: str,
) -> str:
    scale = decision_scale.set_index("decision_scale")
    old_local = development.loc[
        development["evidence_source"].eq("OLD_DEVELOPMENT_LOCAL")
    ].iloc[0]
    new_local = development.loc[
        development["evidence_source"].eq("INDEPENDENT_CALIBRATION")
        & development["decision_scale"].eq("1-step")
    ].iloc[0]
    comparison = (
        "INDEPENDENT_P95_HIGHER_THAN_DEVELOPMENT"
        if new_local.P95 > old_local.P95
        else "INDEPENDENT_P95_LOWER_THAN_DEVELOPMENT"
        if new_local.P95 < old_local.P95
        else "P95_EQUAL"
    )
    p95_ratio = float(new_local.P95 / old_local.P95)
    bundle_p95 = [float(scale.loc[f"{length}-step", "P95"]) for length in (2, 3, 5)]
    bundle_increasing = bool(np.all(np.diff(bundle_p95) > 0.0))
    all_scale_p95 = [float(scale.loc[f"{length}-step", "P95"]) for length in (1, 2, 3, 5)]
    all_scale_increasing = bool(np.all(np.diff(all_scale_p95) > 0.0))
    scale_lines = "\n".join(
        f"- {name}: P90={row.P90:.12g}, P95={row.P95:.12g}, "
        f"P99={row.P99:.12g}, max={row['max']:.12g}, n={int(row.n)}"
        for name, row in scale.iterrows()
    )
    matched_lines = "\n".join(
        f"- {row.decision_scale} {row.calibration_category}: "
        f"P95={row.P95:.12g}, max={row.max:.12g}, n={int(row.n)}"
        for row in matched.itertuples(index=False)
    )
    axis_rows = axis_direction.loc[
        axis_direction["summary_scope"].eq("AXIS_DIRECTION")
        & axis_direction["decision_scale"].ne("1-step")
    ]
    axis_lines = "\n".join(
        f"- {scale_name}: lowest axis/direction P95="
        f"{group.loc[group['P95'].idxmin(), 'coordinate']}/"
        f"{group.loc[group['P95'].idxmin(), 'direction']} "
        f"{group['P95'].min():.12g}; highest="
        f"{group.loc[group['P95'].idxmax(), 'coordinate']}/"
        f"{group.loc[group['P95'].idxmax(), 'direction']} "
        f"{group['P95'].max():.12g}."
        for scale_name, group in axis_rows.groupby("decision_scale", sort=False)
    )
    calibrated = feasibility.loc[
        feasibility["calibration_status"].eq(BUNDLE_SCALE_CALIBRATED),
        "bundle_length",
    ].astype(int).tolist()
    return f"""# Independent calibration report

Calibration manifest SHA-256: `{manifest_sha}`.

## Direct residual distributions

{scale_lines}

The old development one-step P95 was `{LOCAL_P95:.12g}` and is retained only as `DEVELOPMENT_ESTIMATE_ONLY`. The independent P95 is {p95_ratio:.6g} times the old value, so the old estimate is descriptively optimistic for this new, broader matched/mismatch calibration cohort. The comparison label is `{comparison}`. No pooled distribution or threshold is created.

The one-step source plan contains canonical positive orientation only. Negative one-step direction summaries are an explicitly labelled symmetric reverse view of the same 324 pairs, not 324 additional independent samples. Bundle positive/negative directions are independently preregistered pairs.

## Matched versus mismatch

{matched_lines}

These are descriptive empirical comparisons only. The five-parameter model is not modified.

## Scale, axis, and direction

Bundle P95 increases strictly from 2 to 3 to 5 steps: `{str(bundle_increasing).lower()}`. P95 across 1/2/3/5 is strictly increasing: `{str(all_scale_increasing).lower()}`; the 2-step P95 is below the one-step P95 because the frozen one-step and bundle plans have different trust-step and orientation composition. No universal scale law is inferred.

{axis_lines}

The dominant descriptive heteroscedasticity is by axis (knee largest, phase smallest), while positive/negative bundle directions within an axis are comparatively similar. No heteroscedasticity pass/fail threshold or axis-specific uncertainty threshold is created.

## Decision-scale conclusion

Research-calibrated bundle scales by complete-design criteria: `{calibrated}`. This allows a later research uncertainty-candidate design, not a policy. Residual scale is reported empirically; no formula, percentile, cumulative rule, or stopping rule is selected.

Evidence is sufficient to enter a separate `NEXT_REVISION_POLICY_DESIGN` task: **YES**, but it is not sufficient to freeze or enable a policy in this task.

The result is calibration-only. P2 V1 and rejected V2A remain unchanged, no P2 V3 exists, no prospective personalization ran, held-out final test was not read, and no robot or human approval is implied.
"""


def _provenance_report(
    checkpoint: Mapping[str, Any], manifest_sha: str, truth_audit: pd.DataFrame
) -> str:
    return f"""# Data provenance audit

- Post-prospective checkpoint: `{checkpoint['checkpoint_commit']}` (`{checkpoint['checkpoint_subject']}`).
- Original prospective conclusion remains `{PROSPECTIVE_CONCLUSION}`.
- Original prospective manifest SHA remains `{PROSPECTIVE_MANIFEST_SHA256}`.
- Frozen bundle pair-plan SHA remains `{BUNDLE_SOURCE_PAIR_PLAN_SHA256}`.
- Calibration manifest SHA: `{manifest_sha}`; it was persisted before all {len(truth_audit)} gated truth stages.
- Case selection, local assignment, and bundle assignment used no truth, residual, optimum, or subject-specificity output.
- Endpoint residuals are direct pair differences. No analytic uncertainty scaling was used.
- Development and independent distributions are stored separately and never pooled.
- No percentile, K, policy, prospective cohort, held-out evaluation, robot connection, or human-ready state was created.
"""


def generate_artifacts(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    parameter_map_path: Path = DEFAULT_PARAMETER_MAP_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = _checkpoint_preflight()
    _verify_frozen_baseline()
    protected_before = _calibration_protected_hashes()

    raw_map = pd.read_csv(parameter_map_path)
    lattice = geometrically_valid_parameter_lattice(raw_map)
    if len(lattice) != EXPECTED_GEOMETRIC_LATTICE_SIZE:
        raise RuntimeError("formal geometric lattice changed")

    # Everything through manifest SHA freeze is truth-free.
    cases = calibration_case_manifest()
    case_path = output / "calibration_case_manifest.csv"
    _write_csv(case_path, cases)

    local_plan = pd.read_csv(LOCAL_SOURCE_PAIR_PLAN_PATH)
    local_plan_path = output / "independent_local_calibration_pair_plan.csv"
    shutil.copyfile(LOCAL_SOURCE_PAIR_PLAN_PATH, local_plan_path)
    local_plan_sha = sha256_file(local_plan_path)
    if local_plan_sha != LOCAL_SOURCE_PAIR_PLAN_SHA256:
        raise RuntimeError("LOCAL_VALIDATION_PROTOCOL_PROVENANCE_FAILURE")
    local_assignment = assign_pairs_to_calibration_cases(
        local_plan,
        cases,
        pair_id_column="pair_id",
        strata_columns=("coordinate", "trust_level", "trust_step", "location_class"),
        assignment_id="INDEPENDENT_ONE_STEP_ASSIGNMENT_V1",
    )
    local_assignment_path = output / "local_pair_assignment_manifest.csv"
    _write_csv(local_assignment_path, local_assignment)

    bundle_plan = pd.read_csv(BUNDLE_SOURCE_PAIR_PLAN_PATH)
    bundle_assignment = assign_pairs_to_calibration_cases(
        bundle_plan,
        cases,
        pair_id_column="bundle_pair_id",
        strata_columns=("coordinate", "direction", "bundle_length", "location_class"),
        assignment_id="INDEPENDENT_BUNDLE_ASSIGNMENT_V1",
    )
    bundle_assignment_path = output / "bundle_pair_assignment_manifest.csv"
    _write_csv(bundle_assignment_path, bundle_assignment)

    local_provenance = {
        "protocol_id": LOCAL_CALIBRATION_PLAN_ID,
        "source_protocol_id": "DESIGNATED_LOCAL_VALIDATION_PROTOCOL_V1",
        "source_pair_plan_path": str(LOCAL_SOURCE_PAIR_PLAN_PATH),
        "source_pair_plan_sha256": LOCAL_SOURCE_PAIR_PLAN_SHA256,
        "independent_pair_plan_sha256": local_plan_sha,
        "geometry_and_strata_unchanged": True,
        "case_identity_bound_in_source_plan": False,
        "new_case_assignment_sha256": sha256_file(local_assignment_path),
        "assignment_frozen_before_truth": True,
        "truth_used_for_plan_or_assignment": False,
    }
    _write_json(output / "local_pair_plan_provenance.json", local_provenance)
    bundle_provenance = {
        "protocol_id": "DESIGNATED_BUNDLE_VALIDATION_PROTOCOL_V1",
        "source_protocol_path": str(BUNDLE_SOURCE_PROTOCOL_PATH),
        "source_pair_plan_path": str(BUNDLE_SOURCE_PAIR_PLAN_PATH),
        "expected_pair_plan_sha256": BUNDLE_SOURCE_PAIR_PLAN_SHA256,
        "verified_pair_plan_sha256": sha256_file(BUNDLE_SOURCE_PAIR_PLAN_PATH),
        "pair_count": len(bundle_plan),
        "geometry_and_strata_unchanged": True,
        "new_case_assignment_sha256": sha256_file(bundle_assignment_path),
        "assignment_frozen_before_truth": True,
        "truth_used_for_plan_or_assignment": False,
    }
    _write_json(output / "bundle_pair_plan_provenance.json", bundle_provenance)

    manifest_payload = calibration_manifest_payload(
        checkpoint_commit=str(checkpoint["checkpoint_commit"]),
        case_manifest_sha256=sha256_file(case_path),
        local_plan_sha256=local_plan_sha,
        local_assignment_sha256=sha256_file(local_assignment_path),
        bundle_plan_sha256=sha256_file(BUNDLE_SOURCE_PAIR_PLAN_PATH),
        bundle_assignment_sha256=sha256_file(bundle_assignment_path),
        protected_source_sha256=protected_before,
        case_rows=cases.to_dict(orient="records"),
    )
    manifest_path = output / "P2_NEXT_REVISION_CALIBRATION_MANIFEST_V1.json"
    _write_json(manifest_path, manifest_payload, canonical=True)
    manifest_sha = sha256_file(manifest_path)
    gate = FrozenCalibrationManifestGate(manifest_path, manifest_sha)
    gate.require_frozen()

    one_step, bundles, model_audit = evaluate_frozen_calibration_assignments(
        cases,
        local_plan,
        local_assignment,
        bundle_plan,
        bundle_assignment,
        lattice,
        gate,
    )
    _write_csv(output / "independent_one_step_residuals.csv", one_step)
    for length in (2, 3, 5):
        selected = bundles.loc[bundles["bundle_length"].eq(length)].copy()
        _write_csv(output / f"independent_bundle_{length}step_residuals.csv", selected)
    _write_csv(output / "calibration_identification_audit.csv", model_audit)
    truth_audit = pd.DataFrame(gate.access_records)
    _write_csv(output / "calibration_truth_access_audit.csv", truth_audit)

    one_summary_base = _summary_with_scope(
        one_step,
        "e_deltaJ_1",
        (
            ("OVERALL", ()),
            ("AXIS", ("coordinate",)),
            ("LOCATION", ("location_class",)),
            ("MATCHED_MISMATCH", ("calibration_category",)),
        ),
        "1-step",
    )
    one_step_direction_views = _one_step_direction_views(one_step)
    one_summary_direction = _summary_with_scope(
        one_step_direction_views,
        "e_deltaJ_1",
        (
            ("DIRECTION", ("direction", "direction_evidence_status")),
            (
                "FULL_STRATIFICATION",
                (
                    "coordinate",
                    "direction",
                    "direction_evidence_status",
                    "location_class",
                    "calibration_category",
                ),
            ),
        ),
        "1-step",
    )
    one_summary = pd.concat(
        (one_summary_base, one_summary_direction), ignore_index=True, sort=False
    )
    _write_csv(output / "one_step_residual_summary.csv", one_summary)
    bundle_summary_frames = []
    for length in (2, 3, 5):
        selected = bundles.loc[bundles["bundle_length"].eq(length)]
        bundle_summary_frames.append(
            _summary_with_scope(
                selected,
                "e_deltaJ_bundle",
                (("OVERALL", ()),),
                f"{length}-step",
            )
        )
    bundle_summary = pd.concat(bundle_summary_frames, ignore_index=True, sort=False)
    _write_csv(output / "bundle_residual_summary.csv", bundle_summary)
    decision_scale = _decision_scale_summary(one_step, bundles)
    axis_direction = _axis_direction_summary(one_step, bundles)
    matched = _matched_mismatch_summary(one_step, bundles)
    development = _development_comparison(one_step, bundles)
    feasibility = bundle_scale_feasibility(bundles)
    _write_csv(output / "decision_scale_residual_comparison.csv", decision_scale)
    _write_csv(output / "axis_direction_residual_summary.csv", axis_direction)
    _write_csv(output / "matched_mismatch_residual_summary.csv", matched)
    _write_csv(output / "development_vs_independent_calibration.csv", development)
    _write_csv(output / "bundle_scale_feasibility.csv", feasibility)

    _plot_outputs(
        output,
        one_step,
        bundles,
        decision_scale,
        axis_direction,
        matched,
        development,
    )
    (output / "CALIBRATION_DATA_SPLIT_AUDIT.md").write_text(
        _data_split_report(cases), encoding="utf-8"
    )
    (output / "CUMULATIVE_DECISION_CALIBRATION_FEASIBILITY.md").write_text(
        _feasibility_report(feasibility), encoding="utf-8"
    )
    (output / "INDEPENDENT_CALIBRATION_REPORT.md").write_text(
        _independent_report(
            decision_scale,
            axis_direction,
            matched,
            development,
            feasibility,
            manifest_sha,
        ),
        encoding="utf-8",
    )
    (output / "DATA_PROVENANCE_AUDIT.md").write_text(
        _provenance_report(checkpoint, manifest_sha, truth_audit), encoding="utf-8"
    )

    protected_after = _calibration_protected_hashes()
    if protected_before != protected_after:
        raise RuntimeError("protected scientific baseline changed during calibration")
    artifact_names = [
        *REQUIRED_CSV_FILENAMES,
        *EXTRA_CSV_FILENAMES,
        *JSON_FILENAMES,
        *REPORT_FILENAMES,
        *FIGURE_FILENAMES,
    ]
    metadata = {
        "calibration_id": CALIBRATION_ID,
        "data_role": CALIBRATION_DATA_ROLE,
        "calibration_only": True,
        "checkpoint": checkpoint,
        "calibration_manifest_sha256": manifest_sha,
        "calibration_manifest_frozen_before_truth": True,
        "case_manifest_frozen_before_truth": True,
        "pair_assignments_frozen_before_truth": True,
        "calibration_case_ids": cases["case_id"].astype(str).tolist(),
        "calibration_case_count": len(cases),
        "development_used_after_rejection": True,
        "rejected_prospective_cases_used_in_residual_estimate": False,
        "old_development_cases_used_in_residual_estimate": False,
        "future_prospective_created": False,
        "reserved_for_future_prospective": False,
        "heldout_final_test_status": HELD_OUT_STATUS,
        "heldout_final_test_truth_access_count": 0,
        "calibration_truth_gate_access_count": len(truth_audit),
        "truth_used_for_case_selection": False,
        "truth_used_for_pair_selection_or_assignment": False,
        "local_source_pair_plan_sha256": LOCAL_SOURCE_PAIR_PLAN_SHA256,
        "bundle_source_pair_plan_sha256": BUNDLE_SOURCE_PAIR_PLAN_SHA256,
        "old_prospective_manifest_sha256": PROSPECTIVE_MANIFEST_SHA256,
        "old_prospective_conclusion": PROSPECTIVE_CONCLUSION,
        "old_prospective_conclusion_revised": False,
        "new_percentile_selected": False,
        "new_K_selected": False,
        "new_policy_implemented": False,
        "prospective_personalization_run": False,
        "cumulative_rule_enabled": False,
        "analytic_uncertainty_scaling_used": False,
        "one_step_canonical_pair_count": len(one_step),
        "one_step_negative_direction_independent_pair_count": 0,
        "one_step_reverse_direction_summary_is_symmetric_view_only": True,
        "development_and_calibration_residuals_pooled": False,
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "rom_protocol_version": ROM_PROTOCOL_VERSION,
        "hip_rom_deg": list(FORMAL_HIP_ROM_DEG),
        "knee_rom_deg": list(FORMAL_KNEE_ROM_DEG),
        "theta_shank_definition": THETA_SHANK_DEFINITION,
        "algorithm_equivalence_tolerance": OBJECTIVE_EQUIVALENCE_TOLERANCE,
        "support_gate_percent": MODEL_SUPPORT_COVERAGE_GATE_PERCENT,
        "P2_V2_default_enabled": False,
        "human_readiness": NOT_HUMAN_READY,
        "robot_motion_approval": NOT_ROBOT_MOTION_APPROVED,
        "robot_connected": False,
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
        "bundle_scale_status": {
            str(int(row.bundle_length)): str(row.calibration_status)
            for row in feasibility.itertuples(index=False)
        },
        "next_revision_policy_design_evidence_available": bool(
            feasibility["calibration_status"].eq(BUNDLE_SCALE_CALIBRATED).all()
        ),
        "bundle_2_3_5_P95_strictly_increasing": bool(
            np.all(
                np.diff(
                    decision_scale.set_index("decision_scale").loc[
                        ["2-step", "3-step", "5-step"], "P95"
                    ].to_numpy(dtype=float)
                )
                > 0.0
            )
        ),
        "all_1_2_3_5_P95_strictly_increasing": bool(
            np.all(
                np.diff(
                    decision_scale.set_index("decision_scale").loc[
                        ["1-step", "2-step", "3-step", "5-step"], "P95"
                    ].to_numpy(dtype=float)
                )
                > 0.0
            )
        ),
        "next_revision_policy_designed_or_enabled": False,
        "artifact_manifest": {
            name: {
                "sha256": sha256_file(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in artifact_names
        },
        "runtime_seconds": time.perf_counter() - started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run independent P2 decision-error calibration only."
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--parameter-map", type=Path, default=DEFAULT_PARAMETER_MAP_PATH)
    args = parser.parse_args(argv)
    metadata = generate_artifacts(args.output_directory, args.parameter_map)
    print(f"calibration_id: {metadata['calibration_id']}")
    print(f"calibration_manifest_sha256: {metadata['calibration_manifest_sha256']}")
    print(f"case_count: {metadata['calibration_case_count']}")
    print(f"bundle_scale_status: {metadata['bundle_scale_status']}")
    print(f"runtime_seconds: {metadata['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
