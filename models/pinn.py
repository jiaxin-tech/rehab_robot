# models/pinn.py
# Inverse PINN：三维任务空间，从激励轨迹的力/运动数据反推等效参数 M, B, K
#
# 物理方程（三维对角形式）：
#   Mi*ẍi + Bi*ẋi + Ki*xi = Fi，i in {x, y, z}
# 共9个可辨识参数：Mx/My/Mz, Bx/By/Bz, Kx/Ky/Kz
# 项目内部统一使用mm：
#   x: mm, v: mm/s, a: mm/s^2
#   M: N*s^2/mm, B: N*s/mm, K: N/mm

import numpy as np
import torch
import torch.nn as nn
from config import settings
from utils.logger import get_logger

logger = get_logger("PINN")


# ── 网络定义 ─────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, hidden=settings.PINN_HIDDEN_LAYERS):
        super().__init__()

        layers = []
        prev = 1
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

        for axis in ["x", "y", "z"]:
            setattr(self, f"_log_M{axis}",
                    nn.Parameter(torch.log(torch.tensor(float(settings.PINN_M_INIT)))))
            setattr(self, f"_log_B{axis}",
                    nn.Parameter(torch.log(torch.tensor(float(settings.PINN_B_INIT)))))
            setattr(self, f"_log_K{axis}",
                    nn.Parameter(torch.log(torch.tensor(float(settings.PINN_K_INIT)))))

    def _get(self, name): return torch.exp(getattr(self, f"_log_{name}"))

    @property
    def Mx(self): return self._get("Mx")
    @property
    def My(self): return self._get("My")
    @property
    def Mz(self): return self._get("Mz")
    @property
    def Bx(self): return self._get("Bx")
    @property
    def By(self): return self._get("By")
    @property
    def Bz(self): return self._get("Bz")
    @property
    def Kx(self): return self._get("Kx")
    @property
    def Ky(self): return self._get("Ky")
    @property
    def Kz(self): return self._get("Kz")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    def physics_residual(self, t, F, xyz_mean=None, xyz_std=None, t_scale=1.0):
        t = t.requires_grad_(True)
        pos_norm = self.forward(t)
        if xyz_mean is None:
            xyz_mean = np.zeros(3, dtype=np.float32)
        if xyz_std is None:
            xyz_std = np.ones(3, dtype=np.float32)
        xyz_mean_t = torch.as_tensor(xyz_mean, dtype=t.dtype, device=t.device).view(1, 3)
        xyz_std_t  = torch.as_tensor(xyz_std,  dtype=t.dtype, device=t.device).view(1, 3)
        t_scale_t  = torch.as_tensor(float(t_scale), dtype=t.dtype, device=t.device)
        pos = pos_norm * xyz_std_t + xyz_mean_t

        residuals = []
        for i, (M, B, K) in enumerate(zip(
            [self.Mx, self.My, self.Mz],
            [self.Bx, self.By, self.Bz],
            [self.Kx, self.Ky, self.Kz],
        )):
            xi = pos[:, i:i+1]
            vi = torch.autograd.grad(xi, t, grad_outputs=torch.ones_like(xi), create_graph=True)[0] / t_scale_t
            ai = torch.autograd.grad(vi, t, grad_outputs=torch.ones_like(vi), create_graph=True)[0] / t_scale_t
            residuals.append(M * ai + B * vi + K * xi - F[:, i:i+1])

        return torch.cat(residuals, dim=1)

    def get_params(self) -> dict:
        return {
            "Mx": self.Mx.item(), "My": self.My.item(), "Mz": self.Mz.item(),
            "Bx": self.Bx.item(), "By": self.By.item(), "Bz": self.Bz.item(),
            "Kx": self.Kx.item(), "Ky": self.Ky.item(), "Kz": self.Kz.item(),
        }


# ── 核心训练函数 ─────────────────────────────────────
def run_pinn(t_data, xyz_data, F_data,
             epochs=settings.PINN_EPOCHS,
             lr=settings.PINN_LR,
             lam=settings.PINN_LAMBDA,
             verbose=True):
    t_data   = np.asarray(t_data,   dtype=np.float32).reshape(-1)
    xyz_data = np.asarray(xyz_data, dtype=np.float32)
    F_data   = np.asarray(F_data,   dtype=np.float32)

    if xyz_data.ndim != 2 or xyz_data.shape[1] != 3:
        raise ValueError(f"xyz_data shape must be (N,3), got {xyz_data.shape}")
    if F_data.ndim != 2 or F_data.shape[1] != 3:
        raise ValueError(f"F_data shape must be (N,3), got {F_data.shape}")
    if len(t_data) != len(xyz_data) or len(t_data) != len(F_data):
        raise ValueError("t_data, xyz_data, F_data must have same length")
    if len(t_data) < 10:
        raise ValueError("PINN needs at least 10 samples")

    t0 = float(t_data.min())
    t_scale = float(t_data.max() - t0)
    if t_scale <= 1e-8:
        raise ValueError("t_data duration too short")
    t_norm = (t_data - t0) / t_scale

    # 项目内部统一使用mm。这里只去掉绝对坐标偏置，不再换算成m。
    xyz_ref  = xyz_data.mean(axis=0)
    xyz_disp = xyz_data - xyz_ref
    xyz_mean = xyz_disp.mean(axis=0)
    xyz_std  = xyz_disp.std(axis=0) + 1e-8
    xyz_norm = (xyz_disp - xyz_mean) / xyz_std

    F_dyn = F_data - F_data.mean(axis=0)

    t_t   = torch.tensor(t_norm,   dtype=torch.float32).unsqueeze(1)
    xyz_t = torch.tensor(xyz_norm, dtype=torch.float32)
    F_t   = torch.tensor(F_dyn,    dtype=torch.float32)

    model     = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        loss_data = ((model(t_t) - xyz_t) ** 2).mean()
        loss_phys = (model.physics_residual(t_t.clone(), F_t, xyz_mean, xyz_std, t_scale) ** 2).mean()
        loss = loss_data + lam * loss_phys
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and ep % 200 == 0:
            p = model.get_params()
            logger.info(
                f"Epoch {ep:5d} | data={loss_data.item():.4f} phys={loss_phys.item():.5f} | "
                f"M=({p['Mx']:.6f},{p['My']:.6f},{p['Mz']:.6f}) | "
                f"B=({p['Bx']:.6f},{p['By']:.6f},{p['Bz']:.6f}) | "
                f"K=({p['Kx']:.6f},{p['Ky']:.6f},{p['Kz']:.6f})"
            )

    params = model.get_params()
    logger.info(
        f"PINN辨识完成 | "
        f"M=({params['Mx']:.6f},{params['My']:.6f},{params['Mz']:.6f}) | "
        f"B=({params['Bx']:.6f},{params['By']:.6f},{params['Bz']:.6f}) | "
        f"K=({params['Kx']:.6f},{params['Ky']:.6f},{params['Kz']:.6f})"
    )
    return model, params


# ── 在线辨识封装 ─────────────────────────────────────
class OnlinePINN:
    """
    两种用途：
      1. update()      → 用激励轨迹辨识关节全局参数（训练/标定阶段）
      2. infer_mbk()   → 给一段康复轨迹推M/B/K，供ComfortNet训练数据生成
    """

    def __init__(self):
        self._params = {
            f"{prefix}{axis}": val
            for prefix, val in [("M", settings.PINN_M_INIT),
                                 ("B", settings.PINN_B_INIT),
                                 ("K", settings.PINN_K_INIT)]
            for axis in ["x", "y", "z"]
        }
        self._model = None
        self._ready = False

    def update(self, t_buf, xyz_buf, F_buf, epochs=500):
        """用激励轨迹辨识关节全局参数，结果存入 self._params"""
        if len(t_buf) < 30:
            return
        t   = np.array(t_buf,   dtype=np.float32)
        xyz = np.array(xyz_buf, dtype=np.float32)
        F   = np.array(F_buf,   dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            logger.error(f"xyz_buf shape错误: {xyz.shape}")
            return
        if F.ndim != 2 or F.shape[1] != 3:
            logger.error(f"F_buf shape错误: {F.shape}")
            return
        self._model, self._params = run_pinn(t, xyz, F, epochs=epochs, verbose=False)
        self._ready = True

    def infer_mbk(self, t_buf, xyz_buf, F_buf, epochs=300) -> dict | None:
        """
        给一段康复轨迹推M/B/K，用于ComfortNet训练数据生成。

        和 update() 的区别：
          - update()    用激励轨迹，目的是辨识关节的全局物理参数
          - infer_mbk() 用康复轨迹，目的是估计该段运动的等效参数
                        作为ComfortNet的输入特征

        Returns:
            dict {Mx,My,Mz,Bx,By,Bz,Kx,Ky,Kz} 或 None（数据不足）
        """
        if len(t_buf) < 30:
            logger.warning("infer_mbk: 数据不足30帧，跳过")
            return None
        t   = np.array(t_buf,   dtype=np.float32)
        xyz = np.array(xyz_buf, dtype=np.float32)
        F   = np.array(F_buf,   dtype=np.float32)
        _, params = run_pinn(t, xyz, F, epochs=epochs, verbose=False)
        return params

    def get_params(self) -> dict:
        return dict(self._params)

    @property
    def is_ready(self) -> bool:
        return self._ready
