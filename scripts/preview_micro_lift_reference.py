"""Create an offline scalar upward-task reference, never a robot motion plan.

The output has no TCP pose, orientation, start anchor or trajectory identifier.
A scalar upward displacement does not resolve rigid leg/robot kinematics,
directional leg resistance, or the installation's unverified stops and limits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PERIOD_S = .02
PROTECTED_FILES = (
    "control/continuous_taught_trajectory.py",
    "control/continuous_taught_timing.py",
    "control/taught_tcp_trajectory.py",
    "config/continuous_taught_90s.empty.json",
    "config/continuous_taught_90s.loaded_diagnostic.json",
)
CSV_COLUMNS = ("time_s", "task_delta_z_m", "task_vz_m_s", "task_az_m_s2")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bounded(value, low, high, name, *, inclusive_low=True):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value > high
            or (value < low if inclusive_low else value <= low)):
        bracket = "[" if inclusive_low else "("
        raise ValueError(f"{name} must be a finite number in {bracket}{low}, {high}]")
    return float(value)


def preview(output_dir, *, duration_s=10., lift_mm=1.):
    """Write a non-executable scalar task reference and its numerical checks."""
    duration_s = _bounded(duration_s, 10., 60., "duration_s")
    lift_mm = _bounded(lift_mm, 0., 2., "lift_mm", inclusive_low=False)
    intervals = round(duration_s / SAMPLE_PERIOD_S)
    if not math.isclose(intervals * SAMPLE_PERIOD_S, duration_s, abs_tol=1e-10, rel_tol=0):
        raise ValueError("duration_s must be a multiple of the 0.02-second sample period")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("choose a new output directory; previous reference files are preserved")
    protected_before = {name: _sha(ROOT / name) for name in PROTECTED_FILES}
    time_s = np.linspace(0., duration_s, intervals + 1)
    phase = time_s / duration_s
    lift_m = lift_mm / 1000.
    delta_z = lift_m * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    velocity = lift_m / duration_s * (30 * phase**2 - 60 * phase**3 + 30 * phase**4)
    acceleration = lift_m / duration_s**2 * (60 * phase - 180 * phase**2 + 120 * phase**3)
    vmax = 1.875 * lift_m / duration_s
    amax = (10 * math.sqrt(3) / 3) * lift_m / duration_s**2
    matrix = np.column_stack((time_s, delta_z, velocity, acceleration))
    numerical_checks = dict(
        finite=bool(np.isfinite(matrix).all()),
        fixed_sample_period=bool(np.allclose(np.diff(time_s), SAMPLE_PERIOD_S, atol=1e-12, rtol=0)),
        endpoint_inclusive=bool(time_s[0] == 0. and time_s[-1] == duration_s),
        monotonic_upward=bool(np.all(np.diff(delta_z) >= -1e-15)),
        exact_displacement_endpoints=bool(delta_z[0] == 0. and math.isclose(delta_z[-1], lift_m, abs_tol=1e-15)),
        rest_endpoints=bool(np.max(np.abs(np.r_[velocity[[0, -1]], acceleration[[0, -1]]])) < 1e-15),
        sampled_extrema_within_analytic_bounds=bool(np.max(np.abs(velocity)) <= vmax * (1+1e-12)
            and np.max(np.abs(acceleration)) <= amax * (1+1e-12)),
    )
    if not all(numerical_checks.values()):
        raise RuntimeError("scalar-reference numeric check failed")
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "scalar_task_reference.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(matrix)
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 8.), sharex=True, constrained_layout=True)
    for axis, values, label in zip(axes, (delta_z * 1000, velocity * 1000, acceleration * 1000),
            ("Upward task displacement (mm)", "Task vertical speed (mm/s)", "Task vertical acceleration (mm/s²)")):
        axis.plot(time_s, values, color="#247e9b", linewidth=1.8)
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"Scalar task reference: +{lift_mm:g} mm in {duration_s:g} s\n"
                   "Offline only | robot pose and physical compatibility unresolved", fontsize=11)
    figure_path = output / "scalar_task_reference.png"
    figure.savefig(figure_path, dpi=250)
    plt.close(figure)
    protected_after = {name: _sha(ROOT / name) for name in PROTECTED_FILES}
    if protected_before != protected_after:
        raise RuntimeError("protected input changed during generation; regenerate in a stable checkout")
    result = dict(
        schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        status="BLOCKED_PENDING_RIGID_POSE_AND_SETUP_REVIEW",
        hardware_accessed=False, motion_commanded=False, ready_for_execution=False,
        reference_scope="scalar_upward_task_only_not_a_robot_pose_trajectory",
        physical_compatibility_verified=False,
        duration_s=duration_s, lift_mm=lift_mm, sample_period_s=SAMPLE_PERIOD_S,
        rows=len(time_s), endpoint_inclusive=True, csv_columns=list(CSV_COLUMNS),
        clock_basis="offline relative task time; no robot or sensor timestamps",
        profile="quintic: h(s)=10*s^3-15*s^4+6*s^5, s=t/T",
        kinematics=dict(
            initial_displacement_mm=float(delta_z[0] * 1000),
            final_displacement_mm=float(delta_z[-1] * 1000),
            endpoint_speed_mm_s=[float(value) * 1000 for value in velocity[[0, -1]]],
            endpoint_acceleration_mm_s2=[float(value) * 1000 for value in acceleration[[0, -1]]],
            analytic_peak_speed_mm_s=vmax * 1000,
            analytic_peak_absolute_acceleration_mm_s2=amax * 1000,
            analytic_peak_speed_time_s=duration_s / 2,
            analytic_acceleration_extrema_times_s=[duration_s * (3-math.sqrt(3))/6,
                                                 duration_s * (3+math.sqrt(3))/6],
            sampled_peak_speed_mm_s=float(np.max(np.abs(velocity)) * 1000),
            sampled_peak_absolute_acceleration_mm_s2=float(np.max(np.abs(acceleration)) * 1000),
        ),
        numerical_checks=numerical_checks,
        blockers=[
            "Rigid shank coupling: compatible robot position and orientation are unresolved; fixed orientation cannot be assumed",
            "Previously reported hip torque target -5 N m and one-way upward resistance remain unresolved; user now reports both drives disabled and torque targets zero",
            "Actual leg limits, soft limits, directional resistance and stop behavior have not been validated",
            "No actual installed robot start anchor or coupled-pose model was supplied",
            "A scalar upward task is not proof of a kinematically feasible or safe connected-leg robot motion",
        ],
        configs_unchanged=True, protected_source_sha256=protected_after,
        generator_sha256=_sha(__file__),
        output_files_sha256={path.name: _sha(path) for path in (csv_path, figure_path)},
    )
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory, never overwritten")
    parser.add_argument("--duration-s", type=float, default=10., help="10 to 60 s, in multiples of 0.02 s")
    parser.add_argument("--lift-mm", type=float, default=1., help="scalar task displacement: >0 and <=2 mm")
    args = parser.parse_args(argv)
    result = preview(args.output_dir, duration_s=args.duration_s, lift_mm=args.lift_mm)
    print(json.dumps({key: result[key] for key in ("status", "hardware_accessed", "motion_commanded",
                     "ready_for_execution", "duration_s", "lift_mm", "rows", "kinematics")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
