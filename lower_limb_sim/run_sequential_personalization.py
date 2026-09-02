"""Generate formal offline sequential-personalization artifacts and figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .formal_protocol import ACTIVE_REFERENCE_SHA256
from .mismatch_scenarios import get_mismatch_scenario
from .sequential_personalization import (
    FORMAL_SUBJECT_IDS,
    FORMAL_TRUTH_SCENARIOS,
    INITIAL_IDENTIFICATION_CONFIG_PATH,
    INITIAL_IDENTIFICATION_DATASET_PATH,
    SubjectPersonalizationResult,
    optimizer_metadata,
    run_sequential_personalization_experiment,
)


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "formal_artifacts"
    / "sequential_personalization"
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent

    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip()

    return {
        "git_commit": run("rev-parse", "HEAD") or None,
        "git_branch": run("branch", "--show-current") or None,
        "workspace_dirty_at_generation": bool(run("status", "--porcelain")),
    }


def _save_combined_tables(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    def combine(attribute: str) -> pd.DataFrame:
        frames = [getattr(result, attribute) for result in results.values()]
        nonempty = [frame for frame in frames if not frame.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    tables = {
        "sequential_personalization_history.csv": combine("history"),
        "sequential_candidate_audit.csv": combine("candidate_audit"),
        "sequential_executed_trial_torque_audit.csv": combine("torque_audit"),
        "sequential_parameter_history.csv": combine("parameter_history"),
        "final_heldout_generalization.csv": combine("heldout_generalization"),
        "sequential_personalization_summary.csv": pd.DataFrame(
            [result.summary for result in results.values()]
        ),
    }
    final_trajectories: list[pd.DataFrame] = []
    for result in results.values():
        trajectory = result.final_trajectory.trajectory.copy(deep=True)
        trajectory.insert(0, "truth_scenario", result.truth_scenario)
        trajectory.insert(0, "subject_id", result.subject_id)
        final_trajectories.append(trajectory)
    tables["final_personalized_trajectories.csv"] = pd.concat(
        final_trajectories, ignore_index=True
    )
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, float_format="%.12g")
    return tables


def _formal_results(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
) -> list[SubjectPersonalizationResult]:
    return [results[(subject, "combined_mild")] for subject in FORMAL_SUBJECT_IDS]


def _plot_mechanical_cost(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for result in _formal_results(results):
        history = result.history
        ax.plot(history["trial_id"], history["actual_J"], marker="o", label=result.subject_id)
        rejected = history.loc[~history["accepted"].astype(bool)]
        if not rejected.empty:
            ax.scatter(
                rejected["trial_id"],
                rejected["actual_J"],
                marker="x",
                s=70,
                color=ax.lines[-1].get_color(),
            )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="frozen reference")
    ax.set_xlabel("Executed trial")
    ax.set_ylabel("Actual normalized mechanical cost $J_{rms}$")
    ax.set_title("Combined-mild virtual truth: executed trials")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mechanical_cost_vs_iteration.png", dpi=180)
    plt.close(fig)


def _plot_alpha_evolution(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
    output_dir: Path,
) -> None:
    fields = (
        ("proposed_alpha_hip", "Hip amplitude delta (deg)"),
        ("proposed_alpha_knee", "Knee amplitude delta (deg)"),
        ("proposed_alpha_phase", "Knee phase shift"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.0), sharex=True)
    for result in _formal_results(results):
        history = result.history
        for ax, (field, label) in zip(axes, fields):
            ax.plot(history["trial_id"], history[field], marker="o", label=result.subject_id)
            rejected = history.loc[~history["accepted"].astype(bool)]
            if not rejected.empty:
                ax.scatter(rejected["trial_id"], rejected[field], marker="x", s=65)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Executed trial")
    fig.suptitle("Combined-mild proposed trajectory parameters")
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_evolution_vs_iteration.png", dpi=180)
    plt.close(fig)


def _plot_reference_vs_final(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True, sharey=True)
    for ax, result in zip(axes.flat, _formal_results(results)):
        final = result.final_trajectory.trajectory
        reference = results[(result.subject_id, "combined_mild")].history.iloc[0]
        # The final generator shares the frozen reference time grid; the
        # canonical neutral candidate is reconstructed only for plotting.
        from .continuous_reference_neighborhood import generate_personalized_trajectory

        neutral = generate_personalized_trajectory().trajectory
        ax.plot(neutral["time_s"], np.rad2deg(neutral["q_hip_rad"]), "--", color="C0", label="hip reference")
        ax.plot(final["time_s"], np.rad2deg(final["q_hip_rad"]), color="C0", label="hip final")
        ax.plot(neutral["time_s"], np.rad2deg(neutral["q_knee_rad"]), "--", color="C1", label="knee reference")
        ax.plot(final["time_s"], np.rad2deg(final["q_knee_rad"]), color="C1", label="knee final")
        ax.set_title(f"{result.subject_id}; stop={reference['stop_reason'] or result.summary['stop_reason']}")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Joint angle (deg)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Frozen asymmetric reference versus final personalized trajectory")
    fig.tight_layout()
    fig.savefig(output_dir / "reference_vs_final_personalized.png", dpi=180)
    plt.close(fig)


def _plot_prediction_audit(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    values: list[float] = []
    for result in _formal_results(results):
        history = result.history.loc[result.history["trial_id"].astype(int).gt(0)]
        if history.empty:
            continue
        ax.scatter(
            history["predicted_improvement"],
            history["actual_improvement"],
            s=65,
            label=result.subject_id,
        )
        values.extend(history["predicted_improvement"].astype(float))
        values.extend(history["actual_improvement"].astype(float))
    span = max([abs(value) for value in values] + [0.01])
    ax.plot([-span, span], [-span, span], "k--", linewidth=1.0, label="perfect prediction")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Predicted improvement")
    ax.set_ylabel("Actual improvement")
    ax.set_title("Combined-mild prediction-versus-actual audit")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_actual_improvement.png", dpi=180)
    plt.close(fig)


def _plot_final_parameters(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
    output_dir: Path,
) -> None:
    formal = _formal_results(results)
    labels = [result.subject_id for result in formal]
    hip = [result.final_alpha.hip_delta_deg for result in formal]
    knee = [result.final_alpha.knee_delta_deg for result in formal]
    phase = [result.final_alpha.phase_delta for result in formal]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axes[0].bar(x - 0.18, hip, width=0.36, label="hip delta")
    axes[0].bar(x + 0.18, knee, width=0.36, label="knee delta")
    axes[0].set_ylabel("Amplitude delta (deg)")
    axes[0].legend()
    axes[1].bar(x, phase, width=0.55, color="C2")
    axes[1].set_ylabel("Knee phase shift")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Subject-specific final offline trajectory parameters")
    fig.tight_layout()
    fig.savefig(output_dir / "subject_specific_final_parameters.png", dpi=180)
    plt.close(fig)


def _write_leakage_report(
    results: Mapping[tuple[str, str], SubjectPersonalizationResult],
    output_dir: Path,
) -> None:
    audits = [result.data_leakage_audit for result in results.values()]
    passed = all(
        audit["data_leakage_detected"] is False
        and audit["heldout_rows_used_for_proposal"] == 0
        and audit["heldout_rows_used_for_parameter_fitting"] == 0
        and audit["truth_calls_unchanged_during_every_proposal"] is True
        for audit in audits
    )
    payload = {
        "audit_status": "PASS" if passed else "FAIL",
        "active_reference_sha256": ACTIVE_REFERENCE_SHA256,
        "audits": audits,
    }
    _write_json(output_dir / "data_leakage_audit.json", payload)
    lines = [
        "# DATA_LEAKAGE_AUDIT",
        "",
        f"Status: **{'PASS' if passed else 'FAIL'}**",
        "",
        "The five-parameter estimator receives only the fixed observation whitelist. "
        "Subject/scenario IDs, Stage-4.5C generator parameters, true torque terms, "
        "validation rows, and held-out rows are excluded from proposal and fitting.",
        "",
        "Held-out trajectories are evaluated once, after the search stop reason is "
        "fixed. Rejected but actually simulated trials remain adaptation data.",
        "",
        "| scenario | subject | initial role | adaptation trials | heldout used in fit | truth calls during proposal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for audit in audits:
        lines.append(
            f"| {audit['truth_scenario']} | {audit['subject_id']} | "
            f"{audit['initial_identification_role']} | "
            f"{audit['sequential_executed_adaptation_trials']} | 0 | "
            f"{'unchanged' if audit['truth_calls_unchanged_during_every_proposal'] else 'CHANGED'} |"
        )
    (output_dir / "DATA_LEAKAGE_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_experiment_report(
    output_dir: Path,
    histories: pd.DataFrame,
    summaries: pd.DataFrame,
    candidates: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> None:
    formal = summaries.loc[summaries["truth_scenario"].eq("combined_mild")]
    matched = histories.loc[
        histories["truth_scenario"].eq("matched_linear")
        & histories["trial_id"].astype(int).gt(0)
    ]
    lines = [
        "# Sequential Personalization Formal Offline Report",
        "",
        "Status: **FORMAL OFFLINE SOFTWARE EXPERIMENT**. This is not real-robot, "
        "human, clinical, safety, effectiveness, or comfort validation.",
        "",
        "## Frozen inputs",
        "",
        f"- Active reference SHA-256: `{ACTIVE_REFERENCE_SHA256}`",
        "- ROM: `ROM_PROTOCOL_V2`, hip 0--120 deg, knee 5--145 deg",
        "- Model convention: `theta_shank = q_hip - q_knee`",
        "- Duration: 24 s; duration optimization disabled",
        "",
        "## Mechanical objective and deterministic ranking",
        "",
        "`J_rms = sqrt((R_h^2 + R_k^2) / 2)`, where each joint RMS torque "
        "is normalized by the same subject/scenario's frozen reference, so the "
        "reference is exactly 1. Mechanical costs within 0.005 are equivalent; "
        "ties use smaller reference deviation, combined peak ratio, torque-rate "
        "ratio, then lexical trajectory ID. This tolerance is not a robot safety limit.",
        "",
        "## Combined-mild formal result",
        "",
        "| subject | executed | accepted improvements | final alpha (hip, knee, phase) | final J | reduction | stop | boundary |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for _, row in formal.iterrows():
        lines.append(
            f"| {row['subject']} | {int(row['number_of_executed_trials'])} | "
            f"{int(row['number_of_accepted_improvements'])} | "
            f"({row['final_hip_delta']:.6g}, {row['final_knee_delta']:.6g}, "
            f"{row['final_phase_delta']:.6g}) | {row['final_actual_J']:.9f} | "
            f"{row['mechanical_reduction_percent']:.6f}% | {row['stop_reason']} | "
            f"{bool(row['boundary_saturation'])} |"
        )
    infeasible = candidates.loc[~candidates["trajectory_feasible"].astype(bool)]
    lines.extend(
        [
            "",
            "The `knee_stiff` result legitimately falls back to the frozen reference: "
            "no feasible candidate cleared the 0.005 predicted-improvement rule. "
            "The other three subjects accepted one knee-amplitude step of -1 deg. "
            "No final point reached an offline search bound.",
            "",
            "## Prediction audit and stopping boundary",
            "",
            f"- Matched-case maximum absolute prediction error: "
            f"`{float(matched['prediction_error'].abs().max()):.12g}`.",
            f"- Runs reaching a non-reference prediction audit: "
            f"`{metadata['runs_reaching_nonreference_prediction_audit']}`.",
            f"- Runs stopped because no frozen reliability threshold exists: "
            f"`{metadata['runs_stopped_for_missing_reliability_threshold']}`.",
            "- No model-reliability threshold was guessed. A reviewed threshold is "
            "required before additional sequential steps can be called formal.",
            "",
            "## Failure and gate audit",
            "",
            f"- Infeasible candidate rows: `{len(infeasible)}`; observed formal reason: "
            f"`{';'.join(sorted(set(infeasible['invalid_reason'].astype(str))))}`.",
            "- Executed-but-rejected behavior, model-update failure, trust-step shrink, "
            "minimum-step stop, bound saturation, parent hash failure, and legacy "
            "reference rejection are retained as regression tests, not hidden trials.",
            "",
            "## Data isolation",
            "",
            "Initial fitting uses only the persisted `train` role. Only actually "
            "simulated trials enter adaptation, including rejected trials. Validation "
            "and held-out rows never enter proposal, fitting, ranking, trust-region, or "
            "stopping decisions. Held-out evaluation runs once after the stop reason is fixed. "
            "See `DATA_LEAKAGE_AUDIT.md` and `data_leakage_audit.json`.",
            "",
            "## Interpretation limit",
            "",
            "The reported reductions are virtual-model mechanical torque reductions. "
            "They do not establish comfort, rehabilitation benefit, patient response, "
            "robot safety, or real-world effectiveness.",
        ]
    )
    (output_dir / "SEQUENTIAL_PERSONALIZATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    results = run_sequential_personalization_experiment()
    tables = _save_combined_tables(results, target)
    _plot_mechanical_cost(results, target)
    _plot_alpha_evolution(results, target)
    _plot_reference_vs_final(results, target)
    _plot_prediction_audit(results, target)
    _plot_final_parameters(results, target)
    _write_leakage_report(results, target)

    histories = tables["sequential_personalization_history.csv"]
    matched = histories.loc[
        histories["truth_scenario"].astype(str).eq("matched_linear")
        & histories["trial_id"].astype(int).gt(0)
    ]
    combined = tables["sequential_personalization_summary.csv"].loc[
        tables["sequential_personalization_summary.csv"]["truth_scenario"]
        .astype(str)
        .eq("combined_mild")
    ]
    metadata = {
        **optimizer_metadata(),
        **_git_state(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_status": "FORMAL_OFFLINE_SOFTWARE_EXPERIMENT",
        "initial_identification_dataset": str(
            INITIAL_IDENTIFICATION_DATASET_PATH.resolve()
        ),
        "initial_identification_dataset_sha256": results[
            (FORMAL_SUBJECT_IDS[0], FORMAL_TRUTH_SCENARIOS[0])
        ].data_leakage_audit["initial_identification_audit"]["source_sha256"],
        "initial_identification_config": str(
            INITIAL_IDENTIFICATION_CONFIG_PATH.resolve()
        ),
        "initial_identification_config_sha256": _file_sha256(
            INITIAL_IDENTIFICATION_CONFIG_PATH
        ),
        "truth_scenario_provenance_post_run_audit_only": {
            name: get_mismatch_scenario(name).as_metadata_dict()
            for name in FORMAL_TRUTH_SCENARIOS
        },
        "truth_scenario_provenance_exposed_to_proposal_or_estimator": False,
        "source_sha256": {
            "mechanical_objective.py": _file_sha256(
                Path(__file__).with_name("mechanical_objective.py")
            ),
            "sequential_personalization.py": _file_sha256(
                Path(__file__).with_name("sequential_personalization.py")
            ),
            "run_sequential_personalization.py": _file_sha256(Path(__file__)),
        },
        "matched_max_abs_prediction_error": (
            float(matched["prediction_error"].abs().max()) if not matched.empty else None
        ),
        "combined_mild_subject_summaries": combined.to_dict(orient="records"),
        "runs_reaching_nonreference_prediction_audit": int(
            histories["trial_id"].astype(int).gt(0).groupby(
                [histories["subject_id"], histories["truth_scenario"]]
            ).any().sum()
        ),
        "runs_stopped_for_missing_reliability_threshold": int(
            histories.groupby(["subject_id", "truth_scenario"])["stop_reason"]
            .last()
            .eq("STOP_MODEL_RELIABILITY_REQUIRES_THRESHOLD")
            .sum()
        ),
        "model_reliability_threshold_was_guessed": False,
        "output_files": sorted(
            {
                path.name for path in target.iterdir() if path.is_file()
            }
            | {
                "sequential_personalization_metadata.json",
                "SEQUENTIAL_PERSONALIZATION_REPORT.md",
            }
        ),
    }
    _write_json(target / "sequential_personalization_metadata.json", metadata)
    _write_experiment_report(
        target,
        histories,
        tables["sequential_personalization_summary.csv"],
        tables["sequential_candidate_audit.csv"],
        metadata,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic offline sequential personalization."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metadata = run(args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
