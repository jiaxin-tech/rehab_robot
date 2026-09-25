"""Development screen of an additional spring/damper on native MyoLeg replay.

No learner, confirmation split or robot is used. Each native development model
is crossed with the same four declared mechanical configurations. These are
controlled attachments, not identified patient parameters or independent people.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lower_limb_sim.force_mapping import endpoint_force_from_joint_torque
from lower_limb_sim.mechanical_endpoints import branch_rms_components
from .actuation_screen import summarize
from .controlled_actuation import ControlledActuationConfig, ControlledActuationDomain
from .experiment import is_feasible, response_metrics
from .physical_interaction import (
    PHYSICAL_INTERACTION_FORMULA_VERSION, PhysicalInteractionConfig, PhysicalInteractionProfile,
)
from .physical_interaction_backend import PhysicalInteractionBackend
from .simulation import COHORT_MANIFEST, ROOT, SimulatorBackend, development_ids, file_sha


DEFAULT_OUTPUT = ROOT / "outputs/myoleg_physical_interaction_pilot_v1"
DEFAULT_CACHE = ROOT / ".cache/myoleg-benchmark-v1"
GEOMETRY = {"L1_m": 0.42, "L2_m": 0.30, "force_limit_n": 500.0}


def development_profiles(reference_q_rad):
    """Fixed exploratory values, declared before any candidate response.

    K is Nm/rad and B is Nm*s/rad. The rest pose is the public ROM reference's
    start, never fitted to native torque or a candidate oracle.
    """
    rest = tuple(float(x) for x in reference_q_rad)
    return tuple(PhysicalInteractionProfile(name, stiffness, damping, rest) for name, stiffness, damping in (
        ("ZERO_INTERACTION", (0.0, 0.0), (0.0, 0.0)),
        ("HIP_SPRING", (6.0, 2.0), (0.3, 0.3)),
        ("KNEE_SPRING", (2.0, 6.0), (0.3, 0.3)),
        ("DAMPING", (2.0, 2.0), (1.0, 1.0)),
    ))


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _force(q, tau):
    mapped = endpoint_force_from_joint_torque(
        q[:, 0], q[:, 1], tau[:, 0], tau[:, 1], GEOMETRY["L1_m"], GEOMETRY["L2_m"],
        force_limit_n=GEOMETRY["force_limit_n"],
    )
    return mapped


def _configuration_rows(native, domain, profile, interaction_config, assistance_config, trace_path):
    backend = PhysicalInteractionBackend(
        native, domain, profile, config=interaction_config, assistance_config=assistance_config,
    )
    reference, phases = domain.reference, domain.subject_reference.phases
    backend.requested(reference)
    reference_trace = backend.trace_for(reference)
    baseline = reference_trace["native_tau_nm"] - reference_trace["interaction_tau_nm"]
    components = np.asarray(branch_rms_components(*baseline.T, reference.time_s, phases))
    peaks = np.max(np.abs(baseline), axis=0)
    if not np.isfinite(components).all() or np.any(components <= 0) or np.any(peaks <= 0):
        raise ValueError("INVALID_PHYSICAL_UNASSISTED_REFERENCE")
    baseline_force = _force(reference.trajectory.q, baseline)
    configuration_id = f"{native.subject_id}::{profile.profile_id}"
    records, arrays = [], {key: [] for key in (
        "time_s", "q_rad", "dq_rad_s", "native_tau_nm", "interaction_tau_nm", "assistance_tau_nm",
        "net_tau_nm", "elastic_tau_nm", "viscous_tau_nm", "potential_energy_j",
        "net_equivalent_force_n", "interaction_equivalent_force_n", "force_mapping_valid",
        "net_force_mapping_valid", "interaction_force_mapping_valid",
        "net_force_mapping_invalid_reason", "interaction_force_mapping_invalid_reason",
    )}
    by_duration = {}
    for point in domain:
        net = backend.requested(point)
        trace = backend.trace_for(point)
        q, dq, times = point.trajectory.q, point.trajectory.dq, point.time_s
        native_tau, interaction, assistance = (trace[key] for key in (
            "native_tau_nm", "interaction_tau_nm", "assistance_tau_nm"))
        # An assistance-only change must never change the external attachment.
        previous = by_duration.setdefault(point.duration_scale, interaction.copy())
        if not np.array_equal(previous, interaction):
            raise ValueError("ASSISTANCE_CANDIDATE_CHANGED_SUBJECT_RESISTANCE")
        if not np.array_equal(interaction, trace["elastic_tau_nm"] + trace["viscous_tau_nm"]):
            raise ValueError("INTERACTION_COMPONENT_BALANCE_FAILED")
        closure = float(np.max(np.abs(net - (native_tau - interaction - assistance))))
        damper_power = trace["viscous_tau_nm"] * dq
        if closure > 1e-12 or np.max(damper_power) > 1e-12:
            raise ValueError("TORQUE_BALANCE_OR_DAMPER_PASSIVITY_FAILED")
        metrics = response_metrics(net, point, phases, components, peaks)
        unassisted = response_metrics(native_tau - interaction, point, phases, components, peaks)
        native_metrics = response_metrics(native_tau, point, phases, components, peaks)
        net_force, ext_force = _force(q, net), _force(q, interaction)
        force_valid = np.asarray(net_force.force_mapping_valid) & np.asarray(ext_force.force_mapping_valid)
        force_reasons = sorted(set(np.asarray(net_force.invalid_reason).tolist()) | set(np.asarray(ext_force.invalid_reason).tolist()))
        force_reasons = [reason for reason in force_reasons if reason]
        records.append({
            "subject_id": configuration_id, "native_subject_id": native.subject_id,
            "profile_id": profile.profile_id, "candidate_id": point.candidate_id,
            "candidate_index": point.candidate_index, "duration_scale": point.duration_scale,
            "assistance_timing": point.assistance_timing, "hip_share": point.hip_share,
            **metrics, "unassisted_E3": unassisted["E3"],
            "native_E3_same_denominator": native_metrics["E3"],
            "assistance_E3_change": metrics["E3"] - unassisted["E3"],
            "valid": True, "feasible": is_feasible(metrics, 0.01),
            "force_mapping_valid": bool(np.all(force_valid)),
            "force_mapping_invalid_samples": int(np.sum(~force_valid)),
            "force_mapping_invalid_reasons": ";".join(force_reasons),
            "net_equivalent_force_peak_n": (float(np.max(net_force.force_magnitude_n))
                                             if np.all(net_force.force_mapping_valid) else None),
            "interaction_equivalent_force_peak_n": (float(np.max(ext_force.force_magnitude_n))
                                                     if np.all(ext_force.force_mapping_valid) else None),
            "interaction_peak_nm": float(np.max(np.abs(interaction))),
            "assist_peak_l1_nm": float(np.max(np.sum(np.abs(assistance), axis=1))),
            "balance_error_nm": closure,
            "spring_work_j": float(np.sum(np.trapezoid(trace["elastic_tau_nm"] * dq, times, axis=0))),
            "damper_work_j": float(np.sum(np.trapezoid(damper_power, times, axis=0))),
            "stored_energy_change_j": float(np.sum(trace["potential_energy_j"][-1] - trace["potential_energy_j"][0])),
        })
        for key in arrays:
            if key == "q_rad":
                value = q
            elif key == "dq_rad_s":
                value = dq
            elif key == "net_equivalent_force_n":
                value = np.column_stack((net_force.fx_robot_on_leg_n, net_force.fz_robot_on_leg_n))
            elif key == "interaction_equivalent_force_n":
                value = np.column_stack((ext_force.fx_robot_on_leg_n, ext_force.fz_robot_on_leg_n))
            elif key == "force_mapping_valid":
                value = force_valid
            elif key == "net_force_mapping_valid":
                value = net_force.force_mapping_valid
            elif key == "interaction_force_mapping_valid":
                value = ext_force.force_mapping_valid
            elif key == "net_force_mapping_invalid_reason":
                value = net_force.invalid_reason
            elif key == "interaction_force_mapping_invalid_reason":
                value = ext_force.invalid_reason
            else:
                value = trace[key]
            arrays[key].append(np.asarray(value))
    np.savez_compressed(trace_path, **{key: np.asarray(value) for key, value in arrays.items()},
                        candidate_ids=np.asarray([p.candidate_id for p in domain]),
                        phases=np.asarray(phases), unassisted_reference_tau_nm=baseline,
                        reference_components_nm=components, reference_peaks_nm=peaks)
    return records, {
        "configuration_id": configuration_id, "native_subject_id": native.subject_id,
        "profile": asdict(profile), "unassisted_reference_components_nm": components.tolist(),
        "unassisted_reference_peaks_nm": peaks.tolist(),
        "unassisted_reference_force_mapping_valid": bool(np.all(baseline_force.force_mapping_valid)),
        "unassisted_reference_force_invalid_reasons": sorted(set(np.asarray(baseline_force.invalid_reason).tolist()) - {""}),
        "trace_file": trace_path.name, "trace_sha256": file_sha(trace_path),
    }


def _report(summary, metrics, grouped):
    lines = ["# Physical interaction development pilot", "",
        "Native MyoLeg prescribed-state torque plus a declared external spring/damper and fixed 2 Nm assistance.",
        "This is a mechanical attachment sensitivity experiment, not a patient model, forward simulation or hardware measurement.",
        "Candidate timing/share change assistance only. Actual dq in rad/s already includes duration scaling.",
        "Each configuration uses its own unassisted duration=1 reference including the same external attachment.", "",
        f"Native bases: {metrics['native_subject_count']}; declared attachments per base: {metrics['profile_count']}; responses: {metrics['valid_trace_count']}.",
        "The crossed configurations share native bases. Do not treat them or candidate rows as independent patients; no population CI is claimed.",
        "Shared policy and oracle use the complete development landscape for screening only; neither is a tested learner recommendation.", "",
        "| Native base / attachment | Feasible / 27 | Oracle | Shared-policy regret, with unassisted fallback |",
        "| --- | ---: | --- | ---: |"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['subject_id']} | {row['feasible_count']} | {row['fallback_oracle_candidate_id']} | {100 * row['fallback_common_relative_regret']:.6f}% |")
    lines += ["", f"Maximum shared-policy relative regret: {100 * metrics['fallback_common_relative_regret_max']:.6f}%.",
              f"Overall screening decision: {metrics['decision']}.",
              "The predeclared practical gap is 0.5%, required in at least 20% of configurations, with a shared feasible candidate and distinct oracles.",
              f"Equivalent planar force mapping valid for all traces and baselines: {metrics['all_force_mappings_valid']}.",
              "Force mapping uses the declared 0.42/0.30 m two-link geometry and a 500 N numerical screen; it is not measured cuff force or a clinical limit.", "",
              "Separate native-base contrasts within each fixed attachment:"]
    for name, values in grouped.items():
        lines.append(f"- {name}: max shared-policy regret {100 * values['fallback_common_relative_regret_max']:.6f}%; {values['decision']}.")
    lines += ["", "No algorithm comparison was run: a useful mechanical signal is a prerequisite, not an assumed outcome.",
              "No measurement noise or repeatability claim is made. Confirmatory data remain unused; V1 expansion paused; robot motion NO-GO."]
    return "\n".join(lines) + "\n"


def run_screen(*, subjects=None, output_dir=DEFAULT_OUTPUT, cache_dir=DEFAULT_CACHE):
    allowed = sorted(development_ids())
    subjects = allowed[:3] if subjects is None else list(subjects)
    if not subjects or len(subjects) != len(set(subjects)) or not set(subjects).issubset(allowed):
        raise ValueError("ONLY_UNIQUE_FROZEN_DEVELOPMENT_SUBJECTS_ALLOWED")
    domain = ControlledActuationDomain()
    profiles = development_profiles(domain.reference.trajectory.q[0])
    interaction_config, assistance_config = PhysicalInteractionConfig(), ControlledActuationConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "physical_interaction.py", "physical_interaction_backend.py", "controlled_actuation.py",
        "actuation_screen.py", "mechanism_candidates.py", "simulation.py", "experiment.py")),
        ROOT / "lower_limb_sim/force_mapping.py", ROOT / "lower_limb_sim/jacobian.py",
        ROOT / "lower_limb_sim/mechanical_endpoints.py"]
    protocol = {
        "experiment_id": "PHYSICAL_INTERACTION_DEVELOPMENT_PILOT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "formula_version": PHYSICAL_INTERACTION_FORMULA_VERSION,
        "formula": "interaction=-scale*(K*(q-q0)+B*dq); net=native-interaction-assistance",
        "parameter_source": "exploratory external attachment values declared in code before responses; not identified physiology",
        "candidate_dependence": "interaction uses q,dq only; assistance uses timing/share; duration applied once by trajectory generator",
        "native_passive_accounting": "already included in native required-drive torque; never subtracted again",
        "native_subject_ids": subjects, "selection": "first three sorted development IDs unless explicitly supplied",
        "profiles": [asdict(profile) for profile in profiles], "interaction_config": asdict(interaction_config),
        "assistance_config": assistance_config.as_dict(), "geometry_proxy": GEOMETRY,
        "candidate_family": domain.family, "candidates": [{"candidate_id": p.candidate_id, "features": p.features} for p in domain],
        "normalization": "same native base and attachment, unassisted duration=1 reference",
        "constraints": {"E2_max": 1.01, "peak_ratio_max": 1.10, "scope": "mechanical comparison ratios, not clinical safety"},
        "screen_rule": "actuation_screen.summarize: common relative regret >=0.005 in >=20% configurations, shared feasible candidate, distinct oracles; allow no-assistance fallback; all planar mappings valid",
        "statistical_unit": "native-base x fixed mechanical attachment; crossed sensitivity design, no independent patient inference",
        "noise_std": 0.0, "learner_access": "none; evaluator-only full-domain screen",
        "common_policy": "evaluator-only mean E3 minimum over universally feasible candidates; not held-out-fitted common policy",
        "confirmatory_access": False, "v1_expansion": "PAUSED", "robot_motion": "NO-GO",
        "cohort_manifest_sha256": file_sha(COHORT_MANIFEST),
        "code_sha256": {str(path.relative_to(ROOT)).replace('\\', '/'): file_sha(path) for path in sources},
    }
    _json(output_dir / "protocol.json", protocol)
    protocol_sha = file_sha(output_dir / "protocol.json")
    all_rows, configurations, provenance = [], [], []
    try:
        for subject_id in subjects:
            native = SimulatorBackend(subject_id, domain, cache_dir)
            for profile in profiles:
                records, metadata = _configuration_rows(
                    native, domain, profile, interaction_config, assistance_config,
                    output_dir / f"traces_{subject_id}_{profile.profile_id}.npz")
                all_rows.extend(records)
                configurations.append(metadata)
                print(f"{subject_id} / {profile.profile_id}: {len(records)} responses", flush=True)
            provenance.append({"native": native.provenance, "fresh_simulations": native.fresh_simulations,
                               "disk_hits": native.disk_hits})
        rows = pd.DataFrame(all_rows)
        summary, metrics = summarize(rows)
        metrics.update(native_subject_count=len(subjects), profile_count=len(profiles),
                       configuration_count=len(configurations),
                       all_force_mappings_valid=bool(rows.force_mapping_valid.all() and all(
                           item["unassisted_reference_force_mapping_valid"] for item in configurations)),
                       max_balance_error_nm=float(rows.balance_error_nm.max()),
                       max_abs_spring_energy_balance_error_j=float(np.max(np.abs(rows.spring_work_j + rows.stored_energy_change_j))),
                       max_damper_work_j=float(rows.damper_work_j.max()),
                       e2_violation_count=int((rows.E2 > 1.01 + 1e-12).sum()),
                       peak_violation_count=int((rows.peak_ratio > 1.10 + 1e-12).sum()),
                       invalid_force_mapping_trace_count=int((~rows.force_mapping_valid).sum()))
        metrics["mechanical_signal_decision"] = metrics["decision"]
        if not metrics["all_force_mappings_valid"]:
            metrics["decision"] = "HOLD_EQUIVALENT_FORCE_MAPPING_INVALID"
        grouped = {name: summarize(group)[1] for name, group in rows.groupby("profile_id", sort=True)}
        rows.to_csv(output_dir / "landscape.csv", index=False)
        summary.to_csv(output_dir / "oracle_summary.csv", index=False)
        _json(output_dir / "summary.json", metrics)
        _json(output_dir / "within_profile_summary.json", grouped)
        _json(output_dir / "provenance.json", {"native_bases": provenance, "configurations": configurations})
        (output_dir / "REPORT.md").write_text(_report(summary, metrics, grouped), encoding="utf-8")
        artifacts = {p.name: file_sha(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
        _json(output_dir / "completion.json", {"complete": True, "protocol_sha256": protocol_sha,
                                               "responses": len(rows), "confirmatory_access": False,
                                               "artifact_sha256": artifacts})
    except Exception as error:
        _json(output_dir / "completion.json", {"complete": False, "protocol_sha256": protocol_sha,
                                               "error": f"{type(error).__name__}: {error}"})
        raise
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    print(run_screen(subjects=args.subjects, output_dir=args.output_dir, cache_dir=args.cache_dir))


if __name__ == "__main__":
    main()
