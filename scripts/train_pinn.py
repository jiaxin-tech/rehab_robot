# scripts/train_pinn.py
# Offline PINN validation on collected CSV episodes.

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import settings
from models.pinn import run_pinn
from utils.logger import get_logger

logger = get_logger("TrainPINN")


PARAM_KEYS = ["Mx", "My", "Mz", "Bx", "By", "Bz", "Kx", "Ky", "Kz"]


def load_episode(fpath: str):
    """Load one collected excitation episode as t, xyz, F arrays."""
    with open(fpath, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 10:
        raise ValueError(f"episode too short: {fpath}")
    trajectory_type = rows[0].get("trajectory_type", "")
    if trajectory_type and trajectory_type != "excitation":
        raise ValueError(f"skip non-excitation episode: {fpath} ({trajectory_type})")

    t = np.array([float(r["t"]) for r in rows], dtype=np.float32)
    xyz = np.array(
        [[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows],
        dtype=np.float32,
    )
    force = np.array(
        [[float(r["fx"]), float(r["fy"]), float(r["fz"])] for r in rows],
        dtype=np.float32,
    )
    return t, xyz, force


def _is_excitation_file(fpath: str) -> bool:
    with open(fpath, newline="") as f:
        first = next(csv.DictReader(f), None)
    if first is None:
        return False
    return first.get("trajectory_type", "excitation") == "excitation"


def run_validation(data_dir: str, n_episodes: int = 5, epochs: int = settings.PINN_EPOCHS):
    """Run PINN only on excitation episodes and summarize M/B/K consistency."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    files = [f for f in all_files if _is_excitation_file(f)]
    if not files:
        raise FileNotFoundError(f"No excitation CSV files found under {data_dir}")

    files = files[:n_episodes]
    results = []
    for i, fpath in enumerate(files, start=1):
        logger.info(f"Episode {i}/{len(files)}: {os.path.basename(fpath)}")
        t, xyz, force = load_episode(fpath)
        _, params = run_pinn(
            t_data=t,
            xyz_data=xyz,
            F_data=force,
            epochs=epochs,
            verbose=True,
        )
        results.append(params)

    if len(results) <= 1:
        return results

    logger.info("PINN parameter summary:")
    for key in PARAM_KEYS:
        values = np.array([r[key] for r in results], dtype=float)
        logger.info(
            f"{key}: mean={values.mean():.4g} std={values.std():.4g} "
            f"cv={values.std() / (abs(values.mean()) + 1e-8):.3f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Offline PINN validation")
    parser.add_argument("--data-dir", default=settings.DATA_DIR)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=settings.PINN_EPOCHS)
    args = parser.parse_args()
    run_validation(args.data_dir, args.n_episodes, args.epochs)


if __name__ == "__main__":
    main()
