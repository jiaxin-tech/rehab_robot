"""Reproduce the October 17-18 measurement evidence, without device access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lower_limb_sim.config import L1, L2
from lower_limb_sim.kinematics import forward_kinematics
from measurement_validation.leg_episode import read_leg_episode

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def stats(values):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("nonempty finite samples required")
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)), max=float(a.max()))


def finite_max(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return float(finite.max()) if len(finite) else None


def linear_fit(angles, counts):
    angles, counts = np.asarray(angles, float), np.asarray(counts, float)
    if len(angles) < 2 or len(np.unique(angles)) < 2 or not np.isfinite([angles, counts]).all():
        raise ValueError("at least two distinct finite calibration angles required")
    slope, zero = np.polyfit(angles, counts, 1)
    if abs(slope) < 1e-12:
        raise ValueError("zero calibration slope")
    residuals = (counts-zero)/slope-angles
    return dict(slope_counts_per_deg=float(slope), zero_counts=float(zero),
                rmse_deg=float(np.sqrt(np.mean(residuals**2))),
                max_abs_deg=float(np.max(np.abs(residuals))))


def leave_one_angle_out(angles, counts):
    """Hold out both directions at the same angle; do not leak its paired point."""
    angles, counts = np.asarray(angles, float), np.asarray(counts, float)
    predictions = np.full(len(angles), np.nan)
    for angle in np.unique(angles):
        held = angles == angle
        fit = linear_fit(angles[~held], counts[~held])
        predictions[held] = (counts[held]-fit["zero_counts"])/fit["slope_counts_per_deg"]
    errors = predictions-angles
    return dict(predictions_deg=predictions.tolist(), errors_deg=errors.tolist(),
        rmse_deg=float(np.sqrt(np.mean(errors**2))), max_abs_deg=float(np.max(np.abs(errors))))


def segment(data_root, spec, joint):
    path = data_root / spec["file"]
    if digest(path).lower() != spec["sha256"].lower():
        raise ValueError("source checksum mismatch: " + str(path))
    episode = read_leg_episode(path)
    time_s = episode.host_time_s-episode.host_time_s[0]
    mask = (time_s >= spec["start_s"]) & (time_s < spec["end_s"])
    selected = episode.joints[joint].position_counts[mask]
    if len(selected) < 3 or not np.isfinite(selected).all():
        raise ValueError("invalid recorded static interval: " + spec["id"])
    t = time_s[mask]
    return dict(id=spec["id"], joint=joint, file=spec["file"], angle_deg=spec["angle_deg"],
        direction=spec.get("direction", "unspecified"), role=spec.get("role", "scan_fit"),
        rows=len(selected), mean_counts=float(selected.mean()), range_counts=float(np.ptp(selected)),
        slope_counts_per_s=float(np.polyfit(t-t[0], selected, 1)[0]),
        max_gap_ms=float(np.diff(t).max()*1000), duration_s=float(t[-1]-t[0]))


def ads_summary(path):
    f = pd.read_csv(path)
    ticks = f.host_monotonic_ns.to_numpy(np.int64)
    dt = np.diff(ticks)/1e6
    if np.any(dt <= 0):
        raise ValueError("duplicate or backward monotonic time: " + str(path))
    duration = (int(ticks[-1])-int(ticks[0]))/1e9
    reads = (f.query_end_ns.to_numpy(np.int64)-f.query_start_ns.to_numpy(np.int64))/1e6
    if np.any(reads < 0):
        raise ValueError("negative ADS read duration")
    summary = dict(segment=path.parent.name, rows=len(f), duration_s=duration,
        rate_hz=(len(f)-1)/duration, interval_ms=stats(dt), read_ms=stats(reads),
        gaps_over_4ms=int(np.sum(dt > 4)), gaps_over_10ms=int(np.sum(dt > 10)),
        fraction_over_4ms=float(np.mean(dt > 4)), required_missing={}, joints={})
    for axis, name in ((1, "hip"), (4, "knee")):
        p, tau = f[f"axis{axis}_position_raw"], f[f"axis{axis}_torque_3c1a_nm"]
        for col in (f"axis{axis}_position_raw", f"axis{axis}_torque_3c1a_nm"):
            summary["required_missing"][col] = int(f[col].isna().sum())
        n = max(1, len(f)//10)
        summary["joints"][name] = dict(position_range_counts=float(p.max()-p.min()),
            torque_mean_nm=float(tau.mean()), torque_sd_nm=float(tau.std()),
            torque_end_minus_start_nm=float(tau.iloc[-n:].mean()-tau.iloc[:n].mean()),
            faults=sorted(f[f"axis{axis}_fault"].astype(str).unique().tolist()),
            enabled=sorted(f[f"axis{axis}_operation_enabled"].astype(str).unique().tolist()),
            error_codes=sorted(f[f"axis{axis}_error_code_hex"].astype(str).unique().tolist()))
    return summary, dt


def empirical_cdf(ax, values, label):
    x = np.sort(np.asarray(values))
    ax.step(x, np.arange(1, len(x)+1)/len(x), where="post", label=label)


def save_figure(fig, directory, stem):
    fig.savefig(directory / (stem+".png"), dpi=180)
    fig.savefig(directory / (stem+".pdf"))
    plt.close(fig)


def analyze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_root = root / "results/lab_visit"
    raw = sorted(data_root.rglob("*.csv"))
    configs = [root / "config" / name for name in ("leg_joint_calibration.json", "leg_round_analysis.json",
        "robot_leg_round.json", "live_observation_acquisition_20261018.json")]
    docs = sorted((root / "site_evidence").glob("*.md"))
    diagnostics = sorted((root / "diagnostics").glob("*20261018*"))
    sources = sorted(set(raw + configs + docs + [p for p in diagnostics if p.is_file()] +
        [data_root / "knee_sweep_20261017.json", data_root / "points_20261017.json"]))
    before = {str(p.relative_to(root)): digest(p) for p in sources}
    cal = load(root / "config/leg_joint_calibration.json")
    sweep = load(data_root / "knee_sweep_20261017.json")
    scan = pd.DataFrame([segment(data_root, p, "knee") for p in sweep["points"]])
    angle, counts = scan.angle_deg.to_numpy(float), scan.mean_counts.to_numpy(float)
    fit = linear_fit(angle, counts)
    scan["predicted_deg"] = (counts-fit["zero_counts"])/fit["slope_counts_per_deg"]
    scan["error_deg"] = scan.predicted_deg-angle
    loo = leave_one_angle_out(angle, counts)
    scan["leave_angle_out_predicted_deg"] = loo["predictions_deg"]
    scan.to_csv(output / "knee_scan_recomputed.csv", index=False)
    points = load(data_root / "points_20261017.json")
    checks = pd.DataFrame([segment(data_root, p, p["joint"]) for p in points["points"]])
    checks["prediction_deg"] = [cal["joints"][r.joint]["flexion_sign"] *
        (r.mean_counts-cal["joints"][r.joint]["zero_counts"])/cal["joints"][r.joint]["counts_per_degree"]
        for r in checks.itertuples()]
    checks["error_deg"] = checks.prediction_deg-checks.angle_deg
    checks.to_csv(output / "out_of_fit_checks.csv", index=False)
    knee_checks = checks[checks.joint == "knee"]
    checks_rmse = float(np.sqrt(np.mean(knee_checks.error_deg**2)))
    angle_error = float(np.abs(knee_checks.error_deg).max())
    b = linear_fit(checks.loc[checks.id.isin(["knee_A", "knee_B"]), "angle_deg"],
                   checks.loc[checks.id.isin(["knee_A", "knee_B"]), "mean_counts"])
    directions = {}
    for name, mask in (("flexion_including_start", scan.direction.isin(["start", "flexion"])),
                       ("extension_only", scan.direction == "extension"),
                       ("extension_including_shared_60deg_turn", (scan.direction == "extension") | (angle == angle.max()))):
        directions[name] = linear_fit(angle[mask], counts[mask])
    statics, intervals = [], []
    for label in ("static_10s", "static_30s", "stability_240s"):
        path, = sorted(data_root.glob(label+"*/leg.csv"))
        s, dt = ads_summary(path)
        statics.append(s)
        intervals.append(dt)
    pd.json_normalize(statics).to_csv(output / "static_acquisition_summary.csv", index=False)
    wrench_path = root / "diagnostics/wrench_query_timing_20261018T092232Z.csv"
    wrench = pd.read_csv(wrench_path)
    budget = load(root / "config/live_observation_acquisition_20261018.json")["budgets"]
    queries = wrench.query_duration_ms.to_numpy(float)
    qbudget = budget["query_s"]*1000
    query_summary = dict(query_duration_ms=stats(queries), budget_ms=qbudget,
        over_budget_count=int(np.sum(queries > qbudget)), sample_count=len(wrench),
        note="Static diagnostic calls, not independent-process capture or moving robot validation.")
    wrench.loc[wrench.query_duration_ms > qbudget].to_csv(output / "wrench_query_overruns.csv", index=False)
    snapshot_rows = []
    for p in sorted((root / "diagnostics").glob("snapshot_alignment_20261018*.csv")):
        f = pd.read_csv(p)
        finite = np.isfinite(f[["robot_state_age_ms", "force_sample_age_ms", "state_internal_skew_ms"]]).all(axis=1)
        nonfuture = (f[["robot_state_age_ms", "force_sample_age_ms", "state_internal_skew_ms"]] >= 0).all(axis=1)
        valid = f.valid.astype(str).str.lower().eq("true")
        policy = finite & nonfuture & (f.robot_state_age_ms <= budget["state_age_s"]*1000) & (
            f.force_sample_age_ms <= budget["sample_age_s"]*1000) & (
            f.state_internal_skew_ms <= budget["skew_s"]*1000) & f.force_estimate_valid.astype(str).str.lower().eq("true")
        # Only timestamp/force conditions are reconstructed here, not the full safety gate.
        snapshot_rows.append(dict(file=p.name, rows=len(f), original_valid=int(valid.sum()),
            temporary_timing_only_valid=int(policy.sum()),
            max_wrench_age_ms=finite_max(f.force_sample_age_ms),
            max_skew_ms=finite_max(f.state_internal_skew_ms)))
    pd.DataFrame(snapshot_rows).to_csv(output / "snapshot_timing_checks.csv", index=False)
    reference_path = root / "reference_release/reference_measured_asymmetric_closed_slow.csv"
    before[str(reference_path.relative_to(root))] = digest(reference_path)
    ref = pd.read_csv(reference_path)
    qh, qk = ref.q_hip_rad.to_numpy(float), ref.q_knee_rad.to_numpy(float)
    outside = np.rad2deg(qk) > angle.max()
    outside_time_fraction = float(np.trapezoid(outside.astype(float), ref.time_s)/
                                  (ref.time_s.iloc[-1]-ref.time_s.iloc[0]))
    _, _, x0, z0 = forward_kinematics(qh, qk, L1, L2)
    hip_error = float(checks.loc[checks.joint == "hip", "error_deg"].abs().max())
    scenarios = []
    for label, dh, dk in (("Observed-error magnitudes", hip_error, angle_error), ("Declared tolerance", 15., 15.)):
        displacement = []
        for sh, sk in itertools.product((-1, 1), repeat=2):
            _, _, x, z = forward_kinematics(qh+np.deg2rad(sh*dh), qk+np.deg2rad(sk*dk), L1, L2)
            displacement.append(np.hypot(x-x0, z-z0)*1000)
        scenarios.append(dict(label=label, hip_perturbation_deg=dh, knee_perturbation_deg=dk,
            max_pullpoint_shift_mm=float(np.max(displacement)),
            scenario_only_not_measured_or_probabilistic_bound=True))
    dependency_audit = []
    round_config = load(root / "config/robot_leg_round.json")
    dep_items = [(str(Path(round_config["algorithm_root"])/name), sha)
                 for name, sha in round_config["algorithm_sha256"].items()]
    dep_items += list(round_config["algorithm_dependencies"].items())
    for name, sha in dep_items:
        path = Path(name)
        # Resolve only explicit known repository suffixes, then verify bytes;
        # do not change the runtime config or accept a different digest.
        alternative = None
        parts = name.replace("\\", "/").split("/rehab_robot/", 1)
        if len(parts) == 2:
            alternative = root / parts[1]
        found = path if path.is_file() else alternative
        actual = digest(found) if found and found.is_file() else None
        lf = hashlib.sha256(found.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if actual else None
        dependency_audit.append(dict(configured_path=name, inspected_path=str(found) if found else None,
            expected_sha256=sha, actual_sha256=actual, byte_exact=actual == sha,
            lf_normalized_matches=lf == sha, config_path_exists=path.is_file()))
    write(output / "algorithm_dependency_audit.json", dependency_audit)
    # Validate the exact frozen learner on this PC, with no measurements or
    # hardware calls. Path relocation is accepted only after matching bytes.
    algorithm_validation = dict(status="NOT_ATTEMPTED", model_trained=False)
    if all(a["byte_exact"] for a in dependency_audit):
        candidate = next(c for c in round_config["candidates"] if c["candidate_id"] == round_config["reference_candidate_id"])
        if digest(reference_path) == candidate["sha256"]:
            request = dict(algorithm_root=round_config["algorithm_root"],
                algorithm_sha256=round_config["algorithm_sha256"],
                algorithm_dependencies={a["inspected_path"]: a["expected_sha256"] for a in dependency_audit[3:]},
                geometry_m=[L1, L2], setup_id="OFFLINE_INITIALIZATION_ONLY_NOT_SITE_APPROVAL",
                setup_sha256=digest(root / "config/robot_leg_round.json"),
                candidates=[dict(candidate, path=str(reference_path))],
                reference_candidate_id=candidate["candidate_id"], reference_sha256=digest(reference_path), history=[])
            write(output / "algorithm_initialization_request.json", request)
            with (output / "algorithm_initialization.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.train_robot_leg_round",
                    str(output / "algorithm_initialization_request.json"), "--validate-only", "--output",
                    str(output / "algorithm_ready.json")], cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            algorithm_validation = dict(status="INITIALIZED" if completed.returncode == 0 else "INITIALIZATION_FAILED",
                returncode=completed.returncode, model_trained=False, note="Exact source, local paths; no site configuration edited.")
    latest_rt = load(root / "diagnostics/rt_second_session_20261018.json")
    result = dict(knee_scan_fit=fit, knee_leave_one_angle_out=loo, knee_direction_fits=directions,
        knee_out_of_fit_rmse_deg=checks_rmse, knee_out_of_fit_max_abs_deg=angle_error,
        old_AB_slope=b["slope_counts_per_deg"], slope_change_percent=100*(fit["slope_counts_per_deg"]/b["slope_counts_per_deg"]-1),
        knee_scan_angle_range_deg=[float(angle.min()), float(angle.max())],
        reference_knee_range_deg=[float(np.rad2deg(qk).min()), float(np.rad2deg(qk).max())],
        reference_time_fraction_above_knee_calibration_range=outside_time_fraction,
        statics=statics, wrench=query_summary, snapshots=snapshot_rows, geometric_scenarios=scenarios,
        raw_csv_count=len(raw), leg_csv_count=sum(p.name == "leg.csv" for p in raw),
        robot_leg_round_directory_present=(root / "results/robot_leg_round").exists(),
        latest_rt_session=latest_rt, algorithm_validation=algorithm_validation,
        limitations=["One installation/session, no independent devices or subjects.",
            "Phone-reference placement uncertainty is not measured.",
            "Out-of-fit checks were assessed retrospectively; they are not a prospective locked test set.",
            "No moving robot/leg trial or real algorithm optimization outcomes are present.",
            "Sampling intervals are host receive intervals; no exact device frame loss claim."])
    write(output / "analysis.json", result)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for name, mask in (("Flexion/start", scan.direction.isin(["start", "flexion"])), ("Extension", scan.direction == "extension")):
        axes[0].scatter(angle[mask], counts[mask]/1e6, label=name)
    a = np.linspace(0, 60, 100)
    axes[0].plot(a, (fit["zero_counts"]+fit["slope_counts_per_deg"]*a)/1e6, c="black", lw=1, label="Pooled fit")
    axes[0].set(xlabel="Phone reference angle (deg)", ylabel="Axis 4 position (million counts)")
    axes[0].legend(fontsize=8)
    axes[1].scatter(angle, scan.error_deg, label="Scan fit residual", color="#287D8E")
    axes[1].scatter(knee_checks.angle_deg, knee_checks.error_deg, marker="x", label="Prior out-of-fit check", color="#BD3B48")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set(xlabel="Phone reference angle (deg)", ylabel="Predicted minus reference (deg)")
    axes[1].legend(fontsize=8)
    axes[2].plot(ref.time_s, np.rad2deg(qk), label="Frozen reference", color="#287D8E")
    axes[2].axhspan(0, 60, color="#72B788", alpha=.2, label="Measured knee calibration range")
    axes[2].set(xlabel="Reference time (s)", ylabel="Knee angle (deg)")
    axes[2].legend(fontsize=8)
    save_figure(fig, output, "figure_1_knee_calibration")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for label, dt in zip(("10 s", "30 s", "240 s"), intervals):
        empirical_cdf(axes[0], dt, label)
    axes[0].axvline(2, color="gray", lw=.8, ls="--")
    axes[0].set(xscale="log", xlabel="ADS receive interval (ms)", ylabel="Empirical CDF")
    axes[0].legend()
    axes[1].plot(np.arange(len(queries)), queries, lw=.7, color="#287D8E")
    axes[1].axhline(qbudget, color="#BD3B48", ls="--", label="Temporary 50 ms budget")
    axes[1].set(xlabel="Static wrench query index", ylabel="SDK query duration (ms)")
    axes[1].legend(fontsize=8)
    empirical_cdf(axes[2], queries, "30 s static diagnostic")
    axes[2].axvline(qbudget, color="#BD3B48", ls="--")
    axes[2].set(xscale="log", xlabel="SDK query duration (ms)", ylabel="Empirical CDF")
    axes[2].legend(fontsize=8)
    save_figure(fig, output, "figure_2_acquisition_timing")
    changed = [name for name, sha in before.items() if digest(root / name) != sha]
    if changed:
        raise RuntimeError("source changed during analysis: " + str(changed))
    provenance = dict(generated_at=datetime.now(timezone.utc).isoformat(), script_sha256=digest(__file__),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        python=sys.version, packages={n: importlib.metadata.version(n) for n in ("numpy", "pandas", "matplotlib")},
        source_sha256=before, source_files_unchanged=True,
        fixed_intervals="Existing recorded static intervals reused; no new interval search.")
    write(output / "provenance.json", provenance)
    print(json.dumps({k:v for k,v in result.items() if k not in ("statics", "latest_rt_session", "snapshots")}, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.root, args.output_dir)


if __name__ == "__main__":
    main()
