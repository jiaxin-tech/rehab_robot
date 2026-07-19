# models/comfort_net.py
# 舒适度神经网络：离线训练，在线推理
# 支持三种输入模式（通过 settings.COMFORT_INPUT_MODE 切换）：
# 所有运动特征均使用 base 坐标系和 SI 单位（N、m、m/s）。
#   "force"         → [Fx,Fy,Fz, x,y,z, vx,vy,vz]            维度=9
#   "tactile"       → [x,y,z, tactile_0...tactile_N]           维度=3+TACTILE_DIM
#   "force+tactile" → [Fx,Fy,Fz, x,y,z, vx,vy,vz, tactile...] 维度=9+TACTILE_DIM

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from config import settings
from utils.logger import get_logger

logger = get_logger("ComfortNet")
VERIFIED_BASE_WRENCH_TRANSFORMS = {
    "sdk_base",
    "rotation_only_verified_by_project_procedure",
}


# ── 输入维度计算 ──────────────────────────────────────
def get_input_dim(mode: str = settings.COMFORT_INPUT_MODE) -> int:
    """根据输入模式返回特征维度"""
    if mode == "force":
        return 9                                    # Fx,Fy,Fz + x,y,z + vx,vy,vz
    elif mode == "tactile":
        return 3 + settings.TACTILE_DIM            # x,y,z + tactile向量
    elif mode == "force+tactile":
        return 9 + settings.TACTILE_DIM            # force特征 + tactile向量
    else:
        raise ValueError(f"未知的input_mode: {mode}，可选: force / tactile / force+tactile")


def build_feature(fx=0., fy=0., fz=0.,
                  x=0.,  y=0.,  z=0.,
                  vx=0., vy=0., vz=0.,
                  tactile: np.ndarray = None,
                  mode: str = settings.COMFORT_INPUT_MODE) -> np.ndarray:
    """
    根据mode把各传感器数据拼成特征向量，供推理时调用。
    力为 N，位置为 m，速度为 m/s，且均在 robot base 坐标系。
    tactile: (TACTILE_DIM,) 触觉向量，mode含tactile时必须传入
    """
    force_feat   = np.array([fx, fy, fz, x, y, z, vx, vy, vz], dtype=np.float32)
    tactile_feat = np.array(tactile, dtype=np.float32) if tactile is not None \
                   else np.zeros(settings.TACTILE_DIM, dtype=np.float32)
    pos_feat     = np.array([x, y, z], dtype=np.float32)

    if mode == "force":
        return force_feat
    elif mode == "tactile":
        return np.concatenate([pos_feat, tactile_feat])
    elif mode == "force+tactile":
        return np.concatenate([force_feat, tactile_feat])
    else:
        raise ValueError(f"未知mode: {mode}")


# ── 网络定义 ─────────────────────────────────────────
class ComfortNet(nn.Module):
    """
    输入维度由 mode 决定（见 get_input_dim）
    输出: 舒适度概率 (0~1)，越大越舒适
    """

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
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
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
    从data_dir下所有CSV加载数据，根据mode选取对应列。

    CSV必须包含的列（force模式）：
        fx_base_n,fy_base_n,fz_base_n, x_m,y_m,z_m,
        vx_mps,vy_mps,vz_mps, valid, force_estimate_valid, comfort

    CSV可选列（触觉模式）：
        tactile_0, tactile_1, ..., tactile_{TACTILE_DIM-1}
        （触觉传感器到位前这些列不存在，tactile模式下会用0填充）

    comfort: 0=舒适→label=1，1/2=不适→label=0，-1=跳过
    """
    import csv, glob
    X_list, y_list = [], []

    files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"未找到CSV文件：{data_dir}")

    for fpath in files:
        with open(fpath) as f:
            rows = list(csv.DictReader(f))
        for row_index, r in enumerate(rows, start=2):
            if int(r.get("schema_version", -1)) != settings.DATA_SCHEMA_VERSION:
                raise ValueError(
                    f"数据 schema 不兼容: {fpath}；需要版本 "
                    f"{settings.DATA_SCHEMA_VERSION}"
                )
            if r.get("frame") != settings.CONTROL_FRAME:
                raise ValueError(
                    f"数据坐标系不兼容: {fpath}；需要 {settings.CONTROL_FRAME}"
                )
            if r.get("units") != settings.SI_UNITS:
                raise ValueError(f"数据单位元数据不兼容: {fpath}:{row_index}")
            if r.get("valid") != "1" or r.get("force_estimate_valid") != "1":
                continue
            if r.get("base_wrench_transform_kind") not in VERIFIED_BASE_WRENCH_TRANSFORMS:
                continue
            try:
                c = int(r["comfort"])
            except (KeyError, TypeError, ValueError):
                continue
            if c == -1:
                continue

            # 读取触觉数据（列不存在时用0填充）
            tactile = np.array([
                float(r.get(f"tactile_{i}", 0.0))
                for i in range(settings.TACTILE_DIM)
            ], dtype=np.float32)

            try:
                feat = build_feature(
                    fx=float(r["fx_base_n"]), fy=float(r["fy_base_n"]),
                    fz=float(r["fz_base_n"]),
                    x=float(r["x_m"]), y=float(r["y_m"]), z=float(r["z_m"]),
                    vx=float(r["vx_mps"]), vy=float(r["vy_mps"]),
                    vz=float(r["vz_mps"]),
                    tactile=tactile,
                    mode=mode,
                )
            except (KeyError, TypeError, ValueError):
                # A partial/invalid capture row must not be converted to zeros.
                continue
            if not np.all(np.isfinite(feat)):
                continue
            X_list.append(feat)
            y_list.append(1.0 if c == 0 else 0.0)

    # BatchNorm in ComfortNet needs at least two training rows after the
    # validation split; three total rows is the smallest safe split (2 train,
    # 1 validation).
    if len(X_list) < 3:
        raise ValueError(
            f"有效舒适度标注不足3条：{data_dir}；请先完成 episode 标注"
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(
        f"[{mode}模式] 加载数据: {len(X)}行，"
        f"舒适={int(y.sum())}，不适={int((1-y).sum())}，"
        f"特征维度={X.shape[1]}"
    )

    # Z-score归一化。评估已保存模型时可返回原始特征，让 Predictor
    # 使用训练时保存的 mean/std，避免重复归一化或数据泄漏。
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
    """
    离线训练舒适度网络。
    model_path 中会同时保存：模型权重、归一化参数、input_mode。
    加载时自动恢复mode，不会出现维度不匹配。
    """
    X, y, norm = load_dataset(data_dir, mode=mode)
    input_dim  = X.shape[1]

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5)

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
                "norm_mean":   norm["mean"],
                "norm_std":    norm["std"],
                "input_mode":  mode,          # ← 保存mode，加载时自动恢复
                "input_dim":   input_dim,
                "data_schema_version": settings.DATA_SCHEMA_VERSION,
                "frame": settings.CONTROL_FRAME,
                "units": "SI",
                "force_feature_source": "cartesian_force_base_corrected",
            }, model_path)

    logger.info(f"训练完成 [{mode}模式]，最优验证损失={best_val_loss:.4f}，已保存: {model_path}")
    return model


# ── 推理封装 ─────────────────────────────────────────
class ComfortPredictor:
    """
    在线推理封装。
    加载时自动读取训练时的mode，无需手动指定。
    """

    def __init__(self, model_path: str = settings.COMFORT_MODEL_PATH):
        ckpt       = torch.load(model_path, map_location="cpu")
        if ckpt.get("data_schema_version") != settings.DATA_SCHEMA_VERSION:
            raise ValueError(
                "舒适度模型使用旧数据；请用 schema v3 的 SI/base 数据重新训练"
            )
        if ckpt.get("frame") != settings.CONTROL_FRAME or ckpt.get("units") != "SI":
            raise ValueError("舒适度模型的单位或坐标系与当前项目不一致")
        if ckpt.get("force_feature_source") != "cartesian_force_base_corrected":
            raise ValueError("舒适度模型未声明使用去偏 base-frame 内部力")
        self.mode  = ckpt["input_mode"]
        input_dim  = ckpt["input_dim"]
        self.model = ComfortNet(input_dim=input_dim, mode=self.mode)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.mean  = ckpt["norm_mean"]
        self.std   = ckpt["norm_std"]
        logger.info(f"舒适度模型已加载: {model_path}  [mode={self.mode}, dim={input_dim}]")

    def predict(self,
                fx=0., fy=0., fz=0.,
                x=0.,  y=0.,  z=0.,
                vx=0., vy=0., vz=0.,
                tactile: np.ndarray = None) -> float:
        """
        返回舒适度分数 (0~1)。
        tactile 参数在 mode 含 "tactile" 时传入，否则忽略。

        示例：
            # 只用力传感器（mode="force"）
            score = predictor.predict(fx=2.1, fy=0.3, fz=5.0,
                                      x=.301, y=-.200, z=.350,
                                      vx=.010, vy=0., vz=.005)

            # 力+触觉（mode="force+tactile"）
            score = predictor.predict(fx=2.1, ..., tactile=tactile_array)
        """
        feat = build_feature(fx, fy, fz, x, y, z, vx, vy, vz,
                             tactile=tactile, mode=self.mode)
        feat_norm = (feat - self.mean) / self.std
        with torch.no_grad():
            score = self.model(
                torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)
            ).item()
        return score

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        批量推理，X形状 (N, input_dim)，列顺序与训练时一致。
        """
        X_norm = (X.astype(np.float32) - self.mean) / self.std
        with torch.no_grad():
            return self.model(torch.tensor(X_norm)).numpy().squeeze()
