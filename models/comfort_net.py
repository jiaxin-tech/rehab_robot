# models/comfort_net.py
# 舒适度神经网络：离线训练，在线推理
#
# 输入模式（settings.COMFORT_INPUT_MODE）：
#   "pinn_force"    → [Mx..Kz, Fx,Fy,Fz]                       维度=12

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from config import settings
from utils.logger import get_logger

logger = get_logger("ComfortNet")


def _load_checkpoint(model_path: str) -> dict:
    """
    Load ComfortNet checkpoints across PyTorch versions.

    PyTorch 2.6 changed torch.load() to weights_only=True by default. Older
    checkpoints in this project stored numpy arrays for normalization stats,
    so they need one trusted fallback load.
    """
    try:
        return torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location="cpu")
    except pickle.UnpicklingError:
        logger.warning(
            "检测到旧版ComfortNet checkpoint含numpy对象，正在使用兼容模式加载。"
            "请只加载本项目训练生成的可信模型文件。"
        )
        return torch.load(model_path, map_location="cpu", weights_only=False)


# ── 输入维度计算 ──────────────────────────────────────
def get_input_dim(mode: str = settings.COMFORT_INPUT_MODE) -> int:
    if mode == "pinn_force":
        return 12                                   # M/B/K(9) + Fx,Fy,Fz(3)
    elif mode == "force":
        return 9
    elif mode == "mbk":
        return 9                                    # Mx,My,Mz,Bx,By,Bz,Kx,Ky,Kz
    elif mode == "mbk+force":
        return 18                                   # mbk(9) + force特征(9)
    elif mode == "tactile":
        return 3 + settings.TACTILE_DIM
    elif mode == "force+tactile":
        return 9 + settings.TACTILE_DIM
    else:
        raise ValueError(f"未知input_mode: {mode}，可选: pinn_force/force/mbk/mbk+force/tactile/force+tactile")


def build_feature(fx=0., fy=0., fz=0.,
                  x=0.,  y=0.,  z=0.,
                  vx=0., vy=0., vz=0.,
                  Mx=0., My=0., Mz=0.,
                  Bx=0., By=0., Bz=0.,
                  Kx=0., Ky=0., Kz=0.,
                  tactile: np.ndarray = None,
                  mode: str = settings.COMFORT_INPUT_MODE) -> np.ndarray:
    """
    根据mode拼接特征向量。

    mbk模式：需要传入PINN推出的M/B/K参数
    mbk+force模式：M/B/K + 原始力/位置/速度，信息最全
    触觉传感器到位后再用含tactile的模式，做消融实验对比
    """
    force_feat   = np.array([fx, fy, fz, x, y, z, vx, vy, vz], dtype=np.float32)
    raw_force_feat = np.array([fx, fy, fz], dtype=np.float32)
    mbk_feat     = np.array([Mx, My, Mz, Bx, By, Bz, Kx, Ky, Kz], dtype=np.float32)
    tactile_feat = np.array(tactile, dtype=np.float32) if tactile is not None \
                   else np.zeros(settings.TACTILE_DIM, dtype=np.float32)
    pos_feat     = np.array([x, y, z], dtype=np.float32)

    if mode == "pinn_force":
        return np.concatenate([mbk_feat, raw_force_feat])
    elif mode == "force":
        return force_feat
    elif mode == "mbk":
        return mbk_feat
    elif mode == "mbk+force":
        return np.concatenate([mbk_feat, force_feat])
    elif mode == "tactile":
        return np.concatenate([pos_feat, tactile_feat])
    elif mode == "force+tactile":
        return np.concatenate([force_feat, tactile_feat])
    else:
        raise ValueError(f"未知mode: {mode}")


# ── 网络定义 ─────────────────────────────────────────
class ComfortNet(nn.Module):
    def __init__(self,
                 input_dim: int = None,
                 hidden: list   = settings.COMFORT_HIDDEN,
                 mode: str      = settings.COMFORT_INPUT_MODE):
        super().__init__()
        self.mode = mode
        if input_dim is None:
            input_dim = get_input_dim(mode)

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 数据加载 ─────────────────────────────────────────
def load_dataset(data_dir: str,
                 mode: str = settings.COMFORT_INPUT_MODE,
                 normalize: bool = True):
    """
    从data_dir下所有CSV加载数据。

    CSV必须包含的列（pinn_force默认模式）：
        Mx,My,Mz, Bx,By,Bz, Kx,Ky,Kz, fx,fy,fz, comfort

    mbk/mbk+force/pinn_force模式需要：
        Mx,My,Mz, Bx,By,Bz, Kx,Ky,Kz
        （由康复轨迹采集时调用 OnlinePINN.infer_mbk() 写入CSV）

    comfort: 0=舒适→label=1，1/2=不适→label=0，-1=跳过
    """
    import csv, glob
    X_list, y_list = [], []
    mbk_missing_warned = False

    files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"未找到CSV文件：{data_dir}")

    for fpath in files:
        with open(fpath) as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0].get("trajectory_type") != "rehab":
            continue
        for r in rows:
            c = int(r["comfort"])
            if c == -1:
                continue

            # 读取M/B/K（列不存在时填0并警告一次）
            mbk_keys = ["Mx","My","Mz","Bx","By","Bz","Kx","Ky","Kz"]
            if mode in ("pinn_force", "mbk", "mbk+force") and not mbk_missing_warned:
                missing = [k for k in mbk_keys if k not in r]
                if missing:
                    raise ValueError(
                        f"CSV缺少M/B/K列 {missing}: {fpath}。"
                        "请用 --collect-kind comfort 重新采集，让康复轨迹数据写入PINN参数。"
                    )
                mbk_missing_warned = True

            Mx = float(r.get("Mx", settings.PINN_M_INIT)); My = float(r.get("My", settings.PINN_M_INIT)); Mz = float(r.get("Mz", settings.PINN_M_INIT))
            Bx = float(r.get("Bx", settings.PINN_B_INIT)); By = float(r.get("By", settings.PINN_B_INIT)); Bz = float(r.get("Bz", settings.PINN_B_INIT))
            Kx = float(r.get("Kx", settings.PINN_K_INIT)); Ky = float(r.get("Ky", settings.PINN_K_INIT)); Kz = float(r.get("Kz", settings.PINN_K_INIT))

            tactile = np.array([
                float(r.get(f"tactile_{i}", 0.0))
                for i in range(settings.TACTILE_DIM)
            ], dtype=np.float32)

            feat = build_feature(
                fx=float(r["fx"]), fy=float(r["fy"]), fz=float(r["fz"]),
                x =float(r["x"]),  y =float(r["y"]),  z =float(r["z"]),
                vx=float(r["vx"]), vy=float(r["vy"]), vz=float(r["vz"]),
                Mx=Mx, My=My, Mz=Mz,
                Bx=Bx, By=By, Bz=Bz,
                Kx=Kx, Ky=Ky, Kz=Kz,
                tactile=tactile,
                mode=mode,
            )
            X_list.append(feat)
            y_list.append(1.0 if c == 0 else 0.0)

    if not X_list:
        raise ValueError(
            "没有找到有效的comfort标注样本。请先采集数据并打分。"
        )

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(
        f"[{mode}模式] 加载数据: {len(X)}行，"
        f"舒适={int(y.sum())}，不适={int((1-y).sum())}，"
        f"特征维度={X.shape[1]}"
    )

    if len(np.unique(y)) < 2:
        logger.warning("comfort标注只有一类，训练前请补充另一类样本。")

    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    if normalize:
        X = (X - mean) / std

    return X, y, {"mean": mean, "std": std}


# ── 训练 ─────────────────────────────────────────────
def train(data_dir:   str,
          model_path: str   = settings.COMFORT_MODEL_PATH,
          mode:       str   = settings.COMFORT_INPUT_MODE,
          epochs:     int   = settings.COMFORT_EPOCHS,
          lr:         float = settings.COMFORT_LR,
          batch:      int   = settings.COMFORT_BATCH):
    X, y, norm = load_dataset(data_dir, mode=mode)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "需要两类样本才能训练：comfort=0（舒适）和comfort=1/2（不适）。"
        )
    input_dim = X.shape[1]

    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)
    ds  = TensorDataset(X_t, y_t)

    n_val   = max(1, int(len(ds) * 0.2))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch)

    model     = ComfortNet(input_dim=input_dim, mode=mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_loss = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        scheduler.step(val_loss)

        if ep % 20 == 0:
            logger.info(f"Epoch {ep:4d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "norm_mean":   norm["mean"].astype(np.float32).tolist(),
                "norm_std":    norm["std"].astype(np.float32).tolist(),
                "input_mode":  mode,
                "input_dim":   input_dim,
            }, model_path)

    logger.info(f"训练完成 [{mode}模式]，最优验证损失={best_val_loss:.4f}，已保存: {model_path}")
    return model


# ── 推理封装 ─────────────────────────────────────────
class ComfortPredictor:
    def __init__(self, model_path: str = settings.COMFORT_MODEL_PATH):
        ckpt       = _load_checkpoint(model_path)
        self.mode  = ckpt["input_mode"]
        input_dim  = int(ckpt["input_dim"])
        self.model = ComfortNet(input_dim=input_dim, mode=self.mode)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.mean  = np.asarray(ckpt["norm_mean"], dtype=np.float32)
        self.std   = np.asarray(ckpt["norm_std"], dtype=np.float32)
        logger.info(f"舒适度模型已加载: {model_path}  [mode={self.mode}, dim={input_dim}]")

    def predict(self,
                fx=0., fy=0., fz=0.,
                x=0.,  y=0.,  z=0.,
                vx=0., vy=0., vz=0.,
                Mx=0., My=0., Mz=0.,
                Bx=0., By=0., Bz=0.,
                Kx=0., Ky=0., Kz=0.,
                tactile: np.ndarray = None) -> float:
        """
        返回舒适度分数 (0~1)。

        示例：
            # pinn_force模式（推荐）
            score = predictor.predict(Mx=1.2, My=1.1, Mz=0.9,
                                      Bx=0.3, By=0.3, Bz=0.2,
                                      Kx=5.1, Ky=4.8, Kz=4.9,
                                      fx=2.1, fy=0.3, fz=5.0)
        """
        feat = build_feature(
            fx=fx, fy=fy, fz=fz,
            x=x,   y=y,   z=z,
            vx=vx, vy=vy, vz=vz,
            Mx=Mx, My=My, Mz=Mz,
            Bx=Bx, By=By, Bz=Bz,
            Kx=Kx, Ky=Ky, Kz=Kz,
            tactile=tactile,
            mode=self.mode,
        )
        feat_norm = (feat - self.mean) / self.std
        with torch.no_grad():
            return self.model(
                torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)
            ).item()

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X.astype(np.float32) - self.mean) / self.std
        with torch.no_grad():
            return self.model(torch.tensor(X_norm)).numpy().squeeze()
