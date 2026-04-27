# models/comfort_net.py
# 舒适度神经网络：离线训练，在线推理

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from config import settings
from utils.logger import get_logger

logger = get_logger("ComfortNet")


# ── 网络定义 ─────────────────────────────────────────
class ComfortNet(nn.Module):
    """
    输入: [Fx, Fy, Fz, x, y, z, vx, vy, vz]  (9维)
    输出: 舒适度概率 (0~1)，越大越舒适
    """

    def __init__(self, input_dim=settings.COMFORT_INPUT_DIM,
                 hidden=settings.COMFORT_HIDDEN):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 数据加载 ─────────────────────────────────────────
def load_dataset(data_dir: str):
    """
    从data_dir下的所有CSV加载数据
    comfort=0  → label=1（舒适）
    comfort=1,2 → label=0（不舒适）
    comfort=-1 → 跳过（未标注）

    Returns:
        X: (N, 9) numpy array
        y: (N,)   numpy array，0或1
        norm_params: dict，含mean/std，推理时对齐用
    """
    import csv, glob
    X_list, y_list = [], []

    files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"未找到CSV文件：{data_dir}")

    for fpath in files:
        with open(fpath) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            c = int(r["comfort"])
            if c == -1:
                continue
            feat = [
                float(r["fx"]), float(r["fy"]), float(r["fz"]),
                float(r["x"]),  float(r["y"]),  float(r["z"]),
                float(r["vx"]), float(r["vy"]), float(r["vz"]),
            ]
            X_list.append(feat)
            y_list.append(1.0 if c == 0 else 0.0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(f"加载数据: {len(X)} 行，舒适={y.sum():.0f}，不适={(1-y).sum():.0f}")

    # Z-score归一化
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X    = (X - mean) / std

    return X, y, {"mean": mean, "std": std}


# ── 训练 ─────────────────────────────────────────────
def train(data_dir: str,
          model_path: str = settings.COMFORT_MODEL_PATH,
          epochs: int     = settings.COMFORT_EPOCHS,
          lr: float       = settings.COMFORT_LR,
          batch: int      = settings.COMFORT_BATCH):
    """
    离线训练舒适度网络
    保存模型权重 + 归一化参数到 model_path
    """
    X, y, norm = load_dataset(data_dir)

    # 转Tensor
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)
    ds  = TensorDataset(X_t, y_t)

    # 8:2划分训练/验证集
    n_val   = int(len(ds) * 0.2)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch)

    model     = ComfortNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5)

    best_val_loss = float("inf")
    for ep in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        scheduler.step(val_loss)

        if ep % 20 == 0:
            logger.info(f"Epoch {ep:4d} | train={train_loss:.4f} | val={val_loss:.4f}")

        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "norm_mean":   norm["mean"],
                "norm_std":    norm["std"],
            }, model_path)

    logger.info(f"训练完成，最优验证损失={best_val_loss:.4f}，模型已保存: {model_path}")
    return model


# ── 推理封装 ─────────────────────────────────────────
class ComfortPredictor:
    """
    在线推理封装，控制循环里直接调用 predict()
    """

    def __init__(self, model_path: str = settings.COMFORT_MODEL_PATH):
        ckpt = torch.load(model_path, map_location="cpu")
        self.model = ComfortNet()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.mean = ckpt["norm_mean"]
        self.std  = ckpt["norm_std"]
        logger.info(f"舒适度模型已加载: {model_path}")

    def predict(self, fx, fy, fz, x, y, z, vx, vy, vz) -> float:
        """
        返回舒适度分数 (0~1)
        直接传入各标量值，方便在控制循环中调用
        """
        feat = np.array([fx, fy, fz, x, y, z, vx, vy, vz], dtype=np.float32)
        feat = (feat - self.mean) / self.std
        with torch.no_grad():
            score = self.model(torch.tensor(feat).unsqueeze(0)).item()
        return score

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """批量推理，X形状 (N, 9)"""
        X_norm = (X.astype(np.float32) - self.mean) / self.std
        with torch.no_grad():
            return self.model(torch.tensor(X_norm)).numpy().squeeze()
