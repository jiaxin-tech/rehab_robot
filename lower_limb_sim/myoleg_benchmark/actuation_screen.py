"""Development-only screen of controlled torque subtraction, without a learner.

All endpoints use the same subject's unassisted, duration=1 reference. The
evaluator also scores the best shared feasible assisted candidate: improvement
over no assistance alone is not evidence of a need for personalization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lower_limb_sim.mechanical_endpoints import branch_rms_components
from .controlled_actuation import (
    ControlledActuationBackend, ControlledActuationConfig, ControlledActuationDomain,
)
from .experiment import is_feasible, response_metrics
from .simulation import COHORT_MANIFEST, ROOT, SimulatorBackend, development_ids, file_sha


DEFAULT_OUTPUT = ROOT / "outputs/myoleg_controlled_actuation_pilot_v3"
DEFAULT_CACHE = ROOT / ".cache/myoleg-benchmark-v1"
TIER = 0.01
TIE_TOLERANCE = 1e-10
PRACTICAL_RELATIVE_GAP = 0.005
UNASSISTED_REFERENCE = "UNASSISTED_REFERENCE"


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluator-only shared-policy/oracle comparison; fail closed on missing data."""
    if rows.empty or rows.duplicated(["subject_id", "candidate_id"]).any():
        raise ValueError("EMPTY_OR_DUPLICATED_ACTUATION_LANDSCAPE")
    pivot = rows.pivot(index="candidate_id", columns="subject_id", values="E3")
    if pivot.isna().any().any() or not np.isfinite(pivot.to_numpy()).all():
        raise ValueError("INCOMPLETE_ACTUATION_LANDSCAPE")
    if not rows.valid.all():
        raise ValueError("INVALID_ACTUATION_LANDSCAPE")
    subjects = sorted(rows.subject_id.unique())
    feasible = rows.pivot(index="candidate_id", columns="subject_id", values="feasible")
    common_ids = feasible.index[feasible.all(axis=1)]
    means = pivot.loc[common_ids].mean(axis=1)
    # Same decision for all subjects. This diagnostic uses pilot truth and is
    # not an independently trained common policy for a future test cohort.
    common_id = None if means.empty else str(means.sort_values(kind="stable").index[0])
    fallback_common_id = (common_id if common_id is not None and float(means[common_id]) < 1.0
                          else UNASSISTED_REFERENCE)
    summaries = []
    for subject_id, group in rows.groupby("subject_id", sort=True):
        pool = group[group.feasible].sort_values(["E3", "candidate_index"], kind="stable")
        record = {"subject_id": subject_id, "candidate_count": len(group),
                  "feasible_count": len(pool), "common_candidate_id": common_id,
                  "oracle_candidate_id": None, "oracle_E3": None,
                  "oracle_status": "NO_FEASIBLE_CANDIDATE", "tie_count": 0,
                  "reference_E3": 1.0, "reference_regret": None,
                  "common_E3": None, "common_regret": None, "common_relative_regret": None}
        if not pool.empty:
            best = float(pool.E3.min())
            tied = pool[pool.E3 <= best + TIE_TOLERANCE].sort_values("candidate_index")
            record.update(oracle_candidate_id=str(tied.iloc[0].candidate_id),
                          oracle_E3=best, oracle_status="FEASIBLE_ORACLE",
                          tie_count=len(tied), reference_regret=1.0 - best)
            if common_id is not None:
                common_e3 = float(pivot.loc[common_id, subject_id])
                gap = max(common_e3 - best, 0.0)
                record.update(common_E3=common_e3, common_regret=gap,
                              common_relative_regret=gap / common_e3 if common_e3 > 0 else 0.0)
        fallback_oracle_e3 = min(float(record["oracle_E3"]), 1.0) if not pool.empty else 1.0
        fallback_common_e3 = (float(pivot.loc[fallback_common_id, subject_id])
                              if fallback_common_id != UNASSISTED_REFERENCE else 1.0)
        fallback_gap = max(fallback_common_e3 - fallback_oracle_e3, 0.0)
        record.update(
            fallback_common_candidate_id=fallback_common_id,
            fallback_common_E3=fallback_common_e3,
            fallback_oracle_candidate_id=(record["oracle_candidate_id"] if fallback_oracle_e3 < 1.0 else UNASSISTED_REFERENCE),
            fallback_oracle_E3=fallback_oracle_e3,
            fallback_common_relative_regret=fallback_gap / fallback_common_e3 if fallback_common_e3 > 0 else 0.0,
        )
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    correlations = []
    for i, left in enumerate(subjects):
        for right in subjects[i + 1:]:
            a, b = pivot[left].rank(), pivot[right].rank()
            if a.nunique() == b.nunique() == 1:
                correlations.append(1.0)
            elif min(a.nunique(), b.nunique()) > 1:
                correlations.append(float(a.corr(b)))
    rank_median = float(np.median(correlations)) if correlations else None
    rank_min = float(np.min(correlations)) if correlations else None
    oracle_count = int(summary.oracle_candidate_id.nunique())
    reference_fraction = float((summary.reference_regret.fillna(0) >= PRACTICAL_RELATIVE_GAP).mean())
    complete_feasible = bool((summary.feasible_count > 0).all())
    legacy_signal = bool(len(subjects) >= 2 and complete_feasible and oracle_count > 1 and
                         ((rank_median is not None and rank_median < 0.99) or reference_fraction >= 0.2))
    common_relative = summary.common_relative_regret.dropna().astype(float)
    # Supplement the old reference-based rule before the first V3 responses:
    # a large common assistance effect must not be called personalization.
    common_fraction = float((common_relative >= PRACTICAL_RELATIVE_GAP).sum() / len(subjects))
    fallback_relative = summary.fallback_common_relative_regret
    fallback_fraction = float((fallback_relative >= PRACTICAL_RELATIVE_GAP).mean())
    practical_signal = bool(legacy_signal and common_id is not None and fallback_fraction >= 0.2)
    metrics = {
        "subject_count": len(subjects), "candidate_count": len(pivot),
        "valid_trace_count": len(rows), "oracle_unique_count": oracle_count,
        "subjects_with_feasible_oracle": int((summary.feasible_count > 0).sum()),
        "shared_feasible_candidate_count": len(common_ids), "common_candidate_id": common_id,
        "rank_spearman_median": rank_median, "rank_spearman_min": rank_min,
        "rank_pair_count": len(correlations),
        "reference_relative_regret_ge_0_005_fraction": reference_fraction,
        "common_relative_regret_median": float(common_relative.median()) if len(common_relative) else None,
        "common_relative_regret_max": float(common_relative.max()) if len(common_relative) else None,
        "common_relative_regret_ge_0_005_fraction": common_fraction,
        "fallback_common_candidate_id": fallback_common_id,
        "fallback_common_relative_regret_median": float(fallback_relative.median()),
        "fallback_common_relative_regret_max": float(fallback_relative.max()),
        "fallback_common_relative_regret_ge_0_005_fraction": fallback_fraction,
        "exploratory_reference_rule_pass": legacy_signal,
        "practical_personalization_signal": practical_signal,
        "decision": "DEVELOPMENT_FOLLOWUP_ELIGIBLE" if practical_signal else "HOLD_NO_PRACTICAL_PERSONALIZATION_SIGNAL",
        "confirmatory_ready": False,
    }
    return summary, metrics


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _subject_rows(subject_id, domain, config, cache_dir):
    native = SimulatorBackend(subject_id, domain, cache_dir)
    backend = ControlledActuationBackend(native, domain, config)
    reference = domain.reference
    phases = domain.subject_reference.phases
    # Deliberately bypass the actuation wrapper for the common denominator.
    baseline = native.requested(reference)
    components = np.asarray(branch_rms_components(*baseline.T, reference.time_s, phases))
    peaks = np.max(np.abs(baseline), axis=0)
    if not np.isfinite(components).all() or np.any(components <= 0) or np.any(peaks <= 0):
        raise ValueError("INVALID_UNASSISTED_REFERENCE")
    records, waveforms = [], []
    for point in domain:
        net = backend.requested(point)
        trace = backend.trace_for(point)
        diagnostic = trace["diagnostics"]
        measured = response_metrics(net, point, phases, components, peaks)
        unassisted = response_metrics(trace["native_tau_nm"], point, phases, components, peaks)
        net_components = branch_rms_components(*net.T, point.time_s, phases)
        native_components = branch_rms_components(*trace["native_tau_nm"].T, point.time_s, phases)
        row = {
            "subject_id": subject_id, "family": domain.family,
            "candidate_id": point.candidate_id, "candidate_index": point.candidate_index,
            "duration_scale": point.duration_scale, "assistance_timing": point.assistance_timing,
            "hip_share": point.hip_share, **measured, "native_E3": unassisted["E3"],
            "assistance_E3_change": measured["E3"] - unassisted["E3"],
            "valid": True,
            "feasible": bool(diagnostic["actuation_valid"] and is_feasible(measured, TIER)),
            "actuation_valid": bool(diagnostic["actuation_valid"]),
            "actuation_violations": json.dumps(diagnostic["constraint_violations"]),
            "assist_peak_l1_nm": diagnostic["assist_peak_l1_nm"],
        }
        for j, joint in enumerate(("hip", "knee")):
            row[f"{joint}_assist_peak_nm"] = diagnostic["joint_assist_peak_nm"][j]
            row[f"{joint}_abs_impulse_nm_s"] = diagnostic["joint_abs_impulse_nm_s"][j]
            row[f"{joint}_signed_work_j"] = diagnostic["joint_signed_work_j"][j]
            power = trace["assistance_tau_nm"][:, j] * point.trajectory.dq[:, j]
            row[f"{joint}_positive_work_j"] = float(np.trapezoid(np.maximum(power, 0), point.time_s))
            row[f"{joint}_negative_work_j"] = float(np.trapezoid(np.minimum(power, 0), point.time_s))
            row[f"{joint}_squared_torque_integral_nm2_s"] = diagnostic["joint_squared_torque_integral_nm2_s"][j]
            row[f"{joint}_native_peak_nm"] = diagnostic["native_peak_nm"][j]
            row[f"{joint}_net_peak_nm"] = diagnostic["net_peak_nm"][j]
        for j, name in enumerate(("hip_flex", "hip_extend", "knee_flex", "knee_extend")):
            row[f"{name}_net_rms_nm"] = float(net_components[j])
            row[f"{name}_native_rms_nm"] = float(native_components[j])
            row[f"{name}_reference_rms_nm"] = float(components[j])
        records.append(row)
        waveforms.append(trace["assistance_tau_nm"])
    metadata = {"subject_id": subject_id, "native_provenance": native.provenance,
                "reference_kind": "UNASSISTED_DURATION_1", "reference_components_nm": components.tolist(),
                "reference_peaks_nm": peaks.tolist(), "fresh_simulations": native.fresh_simulations,
                "disk_hits": native.disk_hits, "max_decomposition_residual_nm": native.max_decomposition_residual_nm}
    return records, metadata, np.asarray(waveforms)


def _report(metrics: dict, summary: pd.DataFrame) -> str:
    def fmt(value):
        return "undefined" if value is None else f"{value:.6g}"
    lines = ["# CONTROLLED_ACTUATION_V3 development pilot", "",
             "Controlled torque subtraction on prescribed-state MyoLeg traces; no forward actuation, muscle recruitment, patient or hardware validation.",
             "No adaptive algorithm was run. All endpoints use the same subject's unassisted duration=1 reference.", "",
             f"- subjects / assisted candidates: {metrics['subject_count']} / {metrics['candidate_count']}",
             f"- valid traces: {metrics['valid_trace_count']}",
             f"- unique constrained oracles: {metrics['oracle_unique_count']}",
             f"- shared feasible candidates: {metrics['shared_feasible_candidate_count']}",
             f"- E3 rank Spearman median / min: {fmt(metrics['rank_spearman_median'])} / {fmt(metrics['rank_spearman_min'])}",
             f"- best shared assisted candidate: {metrics['common_candidate_id']}",
             f"- shared-candidate relative regret median / max: {fmt(metrics['common_relative_regret_median'])} / {fmt(metrics['common_relative_regret_max'])}",
             f"- shared policy including no-assistance fallback: {metrics['fallback_common_candidate_id']}",
             f"- relative regret with no-assistance fallback median / max: {fmt(metrics['fallback_common_relative_regret_median'])} / {fmt(metrics['fallback_common_relative_regret_max'])}",
             f"- decision: {metrics['decision']}", "",
             "The common candidate was selected with complete pilot truth, for diagnostic comparison only; it is not an independently fitted policy.",
             "The previously documented reference-based rule is exploratory. V3 additionally requires a >=0.5% gap over the best shared feasible policy in >=20% of pilot subjects, allowing no assistance for both shared and individual choices.",
             "A pilot pass permits further development, never immediate confirmation: algorithm comparison, null/positive controls and independent protocol freezing remain outstanding.", "",
             "| Subject | Feasible | Oracle | Oracle E3 | Shared relative regret |",
             "| --- | ---: | --- | ---: | ---: |"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['subject_id']} | {row['feasible_count']} | {row['oracle_candidate_id']} | {fmt(row['oracle_E3'])} | {fmt(row['common_relative_regret'])} |")
    lines += ["", "Fixed strength means a 2 Nm total L1 peak, not fixed torque impulse. Joint impulse and signed work are reported in landscape.csv.",
              "Assistance direction follows public reference movement, not the sign of hidden subject torque; it can increase required load.",
              "V1 30-seed expansion remains paused. No sealed subjects were accessed."]
    return "\n".join(lines) + "\n"


def run_screen(*, subjects=None, output_dir=DEFAULT_OUTPUT, cache_dir=DEFAULT_CACHE):
    allowed = sorted(development_ids())
    subjects = allowed[:3] if subjects is None else list(subjects)
    if not subjects or len(subjects) != len(set(subjects)) or not set(subjects).issubset(allowed):
        raise ValueError("ONLY_UNIQUE_FROZEN_DEVELOPMENT_SUBJECTS_ALLOWED")
    domain, config = ControlledActuationDomain(), ControlledActuationConfig()
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "experiment_id": "CONTROLLED_ACTUATION_V3_DEVELOPMENT_PILOT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CONTROLLED_ACTUATION; algebraic net=native-assistance; not forward simulation",
        "subject_ids": subjects, "default_subject_selection": "first three sorted frozen development IDs, without outcome selection",
        "candidate_family": domain.family, "candidate_count": len(domain),
        "candidate_factors": {"duration_scale": [0.9, 1.0, 1.1], "assistance_timing": [0.25, 0.5, 0.75], "hip_share": [0.25, 0.5, 0.75]},
        "config": config.as_dict(), "normalization": "same-subject unassisted duration=1 reference for every candidate",
        "constraints": {"E2_max": 1.01, "peak_ratio_max": 1.1}, "noise_std": 0.0,
        "oracle_tie_tolerance": TIE_TOLERANCE,
        "common_rule": "evaluator-only mean E3 minimum over pilot-universally feasible assisted candidates",
        "exploratory_rule": ">=2 subjects; every subject has a feasible oracle; >1 oracle; median Spearman<0.99 OR >=20% reference gaps >=0.005",
        "v3_practical_rule": "exploratory rule AND a shared feasible assisted candidate exists AND >=20% subjects have relative common-policy regret >=0.005 including no-assistance fallback for both common and oracle; not a significance test",
        "fallback_rule": "common: best mean E3 shared feasible assisted candidate or no assistance; oracle: best feasible assisted candidate or no assistance; prefer no assistance when E3>=1",
        "confirmation_rule": "never automatic from pilot; requires policy/null/positive validation and separately frozen independent protocol",
        "learner_access": "none; evaluator-only screen", "confirmatory_access": False, "v1_expansion": "PAUSED",
        "cache_dir": str(cache_dir), "cohort_manifest_sha256": file_sha(COHORT_MANIFEST),
        "code_sha256": {name: file_sha(Path(__file__).with_name(name)) for name in (
            "actuation_screen.py", "controlled_actuation.py", "mechanism_candidates.py", "experiment.py", "simulation.py")},
    }
    _write_json(output_dir / "protocol.json", protocol)
    protocol_sha = file_sha(output_dir / "protocol.json")
    all_rows, provenance, first_waveforms = [], [], None
    try:
        for subject_id in subjects:
            records, metadata, waveforms = _subject_rows(subject_id, domain, config, cache_dir)
            if first_waveforms is None:
                first_waveforms = waveforms
            elif not np.array_equal(first_waveforms, waveforms):
                raise ValueError("SUBJECT_DEPENDENT_ASSISTANCE_WAVEFORM")
            all_rows.extend(records)
            provenance.append(metadata)
            print(f"{subject_id}: {len(records)} controlled responses", flush=True)
        rows = pd.DataFrame(all_rows)
        summary, metrics = summarize(rows)
        rows.to_csv(output_dir / "landscape.csv", index=False)
        summary.to_csv(output_dir / "oracle_summary.csv", index=False)
        np.savez_compressed(output_dir / "assistance_waveforms.npz",
                            candidate_ids=np.asarray([p.candidate_id for p in domain]),
                            time_s=np.asarray([p.time_s for p in domain]),
                            assistance_tau_nm=first_waveforms)
        _write_json(output_dir / "summary.json", metrics)
        _write_json(output_dir / "provenance.json", {"subjects": provenance})
        (output_dir / "REPORT.md").write_text(_report(metrics, summary), encoding="utf-8")
        _write_json(output_dir / "completion.json", {"complete": True, "protocol_sha256": protocol_sha,
                                                     "responses": len(rows), "confirmatory_access": False})
    except Exception as error:
        _write_json(output_dir / "completion.json", {"complete": False, "protocol_sha256": protocol_sha,
                                                     "error": f"{type(error).__name__}: {error}"})
        raise
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    path = run_screen(subjects=args.subjects, output_dir=args.output_dir, cache_dir=args.cache_dir)
    print(json.dumps({"complete": True, "output": str(path)}))


if __name__ == "__main__":
    main()
