# scripts/train_comfort.py
# 入口：训练舒适度神经网络

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from models.comfort_net import train, ComfortPredictor, load_dataset
from utils.logger import get_logger

logger = get_logger("TrainComfort")


def evaluate(model_path: str, data_dir: str):
    """训练完后评估准确率"""
    predictor = ComfortPredictor(model_path)
    X, y, _ = load_dataset(
        data_dir,
        mode=predictor.mode,
        normalize=False,
    )

    preds  = predictor.predict_batch(X)
    binary = (preds >= 0.5).astype(float)
    acc    = (binary == y).mean()
    tp     = ((binary == 1) & (y == 1)).sum()
    tn     = ((binary == 0) & (y == 0)).sum()
    fp     = ((binary == 1) & (y == 0)).sum()
    fn     = ((binary == 0) & (y == 1)).sum()

    logger.info(f"评估结果：准确率={acc:.3f}")
    logger.info(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    logger.info(f"  精确率={tp/(tp+fp+1e-8):.3f}  召回率={tp/(tp+fn+1e-8):.3f}")

    # 重点关注FN（把不舒适判成舒适），这是最危险的错误
    if fn > 0:
        logger.warning(f"  ⚠️  有{fn}个不舒适样本被误判为舒适，建议增加不舒适数据或降低阈值")


def main():
    parser = argparse.ArgumentParser(description="训练舒适度网络")
    parser.add_argument("--data-dir",   default=settings.DATA_DIR)
    parser.add_argument("--model-path", default=settings.COMFORT_MODEL_PATH)
    parser.add_argument("--epochs",     type=int, default=settings.COMFORT_EPOCHS)
    parser.add_argument("--lr",         type=float, default=settings.COMFORT_LR)
    parser.add_argument("--eval-only",  action="store_true", help="只评估不训练")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model_path, args.data_dir)
    else:
        logger.info(f"开始训练舒适度网络，数据路径: {args.data_dir}")
        train(
            data_dir   = args.data_dir,
            model_path = args.model_path,
            epochs     = args.epochs,
            lr         = args.lr,
        )
        evaluate(args.model_path, args.data_dir)


if __name__ == "__main__":
    main()
