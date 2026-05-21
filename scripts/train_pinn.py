# scripts/train_pinn.py
# 入口：PINN离线验证（用采集数据验证参数辨识精度）

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv, glob
import numpy as np
from config import settings
from models.pinn import train_offline
from utils.signal_processing import smooth_differentiate
from utils.logger import get_logger

logger = get_logger("TrainPINN")


def load_episode(fpath: str):
    """加载单个episode的CSV，返回 t, x, F"""
    with open(fpath) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"])  for r in rows])
    x = np.array([float(r["x"])  for r in rows])   # 末端x方向位置
    F = np.array([float(r["fx"]) for r in rows])   # x方向力
    return t, x, F


def run_validation(data_dir: str, n_episodes: int = 5):
    """
    用多个episode验证PINN辨识精度
    同一患者不同episode的M/B/K应该相近
    """
    files = sorted(glob.glob(
        os.path.join(data_dir, "**/*.csv"), recursive=True))[:n_episodes]

    if not files:
        logger.error(f"未找到数据: {data_dir}")
        return

    results = []
    for i, fpath in enumerate(files):
        logger.info(f"\n── Episode {i+1}: {os.path.basename(fpath)}")
        t, x, F = load_episode(fpath)

        _, params = train_offline(
            t_data  = t,
            x_data  = x,
            F_data  = F,
            epochs  = settings.PINN_EPOCHS,
            verbose = True,
        )
        results.append(params)

    # 统计辨识结果的一致性
    if len(results) > 1:
        Ms = [r["M"] for r in results]
        Bs = [r["B"] for r in results]
        Ks = [r["K"] for r in results]
        logger.info("\n── 辨识结果汇总 ──")
        logger.info(f"M: mean={np.mean(Ms):.3f}  std={np.std(Ms):.3f}")
        logger.info(f"B: mean={np.mean(Bs):.3f}  std={np.std(Bs):.3f}")
        logger.info(f"K: mean={np.mean(Ks):.3f}  std={np.std(Ks):.3f}")
        cv_M = np.std(Ms) / (np.mean(Ms) + 1e-8)
        if cv_M < 0.1:
            logger.info("✓ M辨识一致性良好 (变异系数<10%)")
        else:
            logger.warning(f"⚠️  M辨识波动较大 (CV={cv_M:.2f})，建议增加激励轨迹多样性")


def main():
    parser = argparse.ArgumentParser(description="PINN离线验证")
    parser.add_argument("--data-dir",   default=settings.DATA_DIR)
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="用于验证的episode数量")
    args = parser.parse_args()
    run_validation(args.data_dir, args.n_episodes)


if __name__ == "__main__":
    main()
