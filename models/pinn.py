# models/pinn.py
# Inverse PINN：三维任务空间，从力/运动数据反推关节等效参数 M, B, K
#
# 物理方程（三维对角形式）：
#   [Mx  0   0 ] [ẍ]   [Bx  0   0 ] [ẋ]   [Kx  0   0 ] [x]   [Fx]
#   [0   My  0 ] [ÿ] + [0   By  0 ] [ẏ] + [0   Ky  0 ] [y] = [Fy]
#   [0   0   Mz] [z̈]   [0   0   Bz] [ż]   [0   0   Kz] [z]   [Fz]
#
# 即每个轴独立：Mi*ẍi + Bi*ẋi + Ki*xi = Fi，i ∈ {x, y, z}
# 共9个可辨识参数：Mx/My/Mz, Bx/By/Bz, Kx/Ky/Kz

import numpy as np
import torch
import torch.nn as nn
from config import settings
from utils.logger import get_logger

logger = get_logger("PINN")


# ── 网络定义 ─────────────────────────────────────────
class PINN(nn.Module):
    """
    三维 Inverse PINN
    网络：时间 t(1维) → 末端位置 [x, y, z](3维)
    可训练物理参数：Mx/My/Mz, Bx/By/Bz, Kx/Ky/Kz（各轴独立）
    """

    def __init__(self, hidden=settings.PINN_HIDDEN_LAYERS):
        super().__init__()

        # 位置预测网络：t(1) → [x, y, z](3)
        layers = []
        prev = 1
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 3))   # ← 输出3维
        self.net = nn.Sequential(*layers)

        # 9个可训练物理参数，用log保证正数
        # 初始值：三轴用相同初始猜测
        for axis in ["x", "y", "z"]:
            setattr(self, f"_log_M{axis}",
                    nn.Parameter(torch.log(torch.tensor(float(settings.PINN_M_INIT)))))
            setattr(self, f"_log_B{axis}",
                    nn.Parameter(torch.log(torch.tensor(float(settings.PINN_B_INIT)))))
            setattr(self, f"_log_K{axis}",
                    nn.Parameter(torch.log(torch.tensor(float(settings.PINN_K_INIT)))))

    # ── 参数访问 ─────────────────────────────────────
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
        """t: (N,1) → pos: (N,3)，列顺序 [x, y, z]"""
        return self.net(t)

    def physics_residual(self, t: torch.Tensor,
                         F: torch.Tensor,
                         xyz_mean=None,
                         xyz_std=None,
                         t_scale: float = 1.0) -> torch.Tensor:
        """
        三轴物理方程残差
        t: (N,1)  requires_grad=True
        F: (N,3)  [Fx, Fy, Fz]
        返回: (N,3) 每轴的残差
        """
        t = t.requires_grad_(True)
        pos_norm = self.forward(t)   # (N,3)
        if xyz_mean is None:
            xyz_mean = np.zeros(3, dtype=np.float32)
        if xyz_std is None:
            xyz_std = np.ones(3, dtype=np.float32)
        xyz_mean_t = torch.as_tensor(xyz_mean, dtype=t.dtype, device=t.device).view(1, 3)
        xyz_std_t = torch.as_tensor(xyz_std, dtype=t.dtype, device=t.device).view(1, 3)
        t_scale_t = torch.as_tensor(float(t_scale), dtype=t.dtype, device=t.device)
        pos = pos_norm * xyz_std_t + xyz_mean_t

        residuals = []
        M_list = [self.Mx, self.My, self.Mz]
        B_list = [self.Bx, self.By, self.Bz]
        K_list = [self.Kx, self.Ky, self.Kz]

        for i, (M, B, K) in enumerate(zip(M_list, B_list, K_list)):
            xi = pos[:, i:i+1]   # (N,1) 取第i轴

            vi = torch.autograd.grad(
                xi, t,
                grad_outputs=torch.ones_like(xi),
                create_graph=True,
            )[0] / t_scale_t
            ai = torch.autograd.grad(
                vi, t,
                grad_outputs=torch.ones_like(vi),
                create_graph=True,
            )[0] / t_scale_t

            res_i = M * ai + B * vi + K * xi - F[:, i:i+1]
            residuals.append(res_i)

        return torch.cat(residuals, dim=1)   # (N,3)

    def get_params(self) -> dict:
        """返回9个辨识参数的dict（detach后的标量）"""
        return {
            "Mx": self.Mx.item(), "My": self.My.item(), "Mz": self.Mz.item(),
            "Bx": self.Bx.item(), "By": self.By.item(), "Bz": self.Bz.item(),
            "Kx": self.Kx.item(), "Ky": self.Ky.item(), "Kz": self.Kz.item(),
        }


# ── 核心训练函数 ─────────────────────────────────────
def run_pinn(t_data: np.ndarray,
             xyz_data: np.ndarray,
             F_data: np.ndarray,
             epochs: int   = settings.PINN_EPOCHS,
             lr: float     = settings.PINN_LR,
             lam: float    = settings.PINN_LAMBDA,
             verbose: bool = True):
    """
    PINN训练/辨识的核心函数，离线验证和在线辨识共用同一套逻辑。

    Args:
        t_data:   时间序列     (N,)
        xyz_data: 末端三维位置 (N,3)  单位mm，列顺序 [x, y, z]
        F_data:   三维力       (N,3)  单位N，列顺序 [Fx, Fy, Fz]
        epochs:   迭代次数（在线用少，离线验证用多）
    Returns:
        model:  训练好的PINN实例
        params: dict，9个辨识参数
    """
    # ── 数据归一化 ────────────────────────────────────
    t_data = np.asarray(t_data, dtype=np.float32).reshape(-1)
    xyz_data = np.asarray(xyz_data, dtype=np.float32)
    F_data = np.asarray(F_data, dtype=np.float32)
    if xyz_data.ndim != 2 or xyz_data.shape[1] != 3:
        raise ValueError(f"xyz_data shape must be (N,3), got {xyz_data.shape}")
    if F_data.ndim != 2 or F_data.shape[1] != 3:
        raise ValueError(f"F_data shape must be (N,3), got {F_data.shape}")
    if len(t_data) != len(xyz_data) or len(t_data) != len(F_data):
        raise ValueError("t_data, xyz_data, and F_data must have the same length")
    if len(t_data) < 10:
        raise ValueError("PINN training needs at least 10 samples")

    t0 = float(t_data.min())
    t_scale = float(t_data.max() - t0)
    if t_scale <= 1e-8:
        raise ValueError("t_data duration is too short for PINN training")
    t_norm  = (t_data - t0) / t_scale                   # → [0,1]

    xyz_m = xyz_data / 1000.0
    xyz_ref = xyz_m.mean(axis=0)
    xyz_disp = xyz_m - xyz_ref
    xyz_mean = xyz_disp.mean(axis=0)
    xyz_std  = xyz_disp.std(axis=0) + 1e-8
    xyz_norm = (xyz_disp - xyz_mean) / xyz_std          # → 零均值单位方差

    # 力不归一化，保留物理量纲，让M/B/K的量纲也保持真实
    # （若力量级差异很大可考虑归一化，但需同步调整参数初值）

    # ── 转Tensor ─────────────────────────────────────
    F_dyn = F_data - F_data.mean(axis=0)

    t_t   = torch.tensor(t_norm,   dtype=torch.float32).unsqueeze(1)   # (N,1)
    xyz_t = torch.tensor(xyz_norm, dtype=torch.float32)                 # (N,3)
    F_t   = torch.tensor(F_dyn,    dtype=torch.float32)                 # (N,3)

    # ── 训练 ─────────────────────────────────────────
    model     = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()

        # 数据损失：预测位置 vs 测量位置（三轴均方误差）
        xyz_pred  = model(t_t)
        loss_data = ((xyz_pred - xyz_t) ** 2).mean()

        # 物理损失：三轴方程残差均趋近0
        residual  = model.physics_residual(
            t_t.clone(), F_t,
            xyz_mean=xyz_mean,
            xyz_std=xyz_std,
            t_scale=t_scale,
        )
        loss_phys = (residual ** 2).mean()

        loss = loss_data + lam * loss_phys
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and ep % 200 == 0:
            p = model.get_params()
            logger.info(
                f"Epoch {ep:5d} | data={loss_data.item():.4f} "
                f"phys={loss_phys.item():.5f} | "
                f"Mx={p['Mx']:.3f} My={p['My']:.3f} Mz={p['Mz']:.3f} | "
                f"Bx={p['Bx']:.3f} By={p['By']:.3f} Bz={p['Bz']:.3f} | "
                f"Kx={p['Kx']:.3f} Ky={p['Ky']:.3f} Kz={p['Kz']:.3f}"
            )

    params = model.get_params()
    logger.info(
        f"PINN辨识完成 | "
        f"M=({params['Mx']:.3f},{params['My']:.3f},{params['Mz']:.3f}) | "
        f"B=({params['Bx']:.3f},{params['By']:.3f},{params['Bz']:.3f}) | "
        f"K=({params['Kx']:.3f},{params['Ky']:.3f},{params['Kz']:.3f})"
    )
    return model, params


# ── 在线辨识封装 ─────────────────────────────────────
class OnlinePINN:
    """
    在线参数辨识封装，供控制循环调用。

    使用方式：
        1. 机械臂做3~5s持续激励运动，同时往缓冲区喂数据
        2. 每隔N步调用 update() 触发辨识
        3. 调用 get_params() 把最新9个参数传给MPC
    """

    def __init__(self):
        # 初始值：三轴用相同猜测值
        self._params = {
            axis: val
            for prefix, val in [("M", settings.PINN_M_INIT),
                                 ("B", settings.PINN_B_INIT),
                                 ("K", settings.PINN_K_INIT)]
            for axis in [f"{prefix}x", f"{prefix}y", f"{prefix}z"]
        }
        self._model = None
        self._ready = False

    def update(self,
               t_buf:   list,
               xyz_buf: list,
               F_buf:   list,
               epochs:  int = 500):
        """
        用滑动窗口内的最新数据重新辨识参数。

        Args:
            t_buf:   时间列表       (N,)
            xyz_buf: 末端位置列表   (N,3) 或 list of [x,y,z]
            F_buf:   三维力列表     (N,3) 或 list of [Fx,Fy,Fz]
            epochs:  在线迭代次数，建议300~500，保证速度
        """
        if len(t_buf) < 30:
            return

        t   = np.array(t_buf,   dtype=np.float32)
        xyz = np.array(xyz_buf, dtype=np.float32)   # (N,3)
        F   = np.array(F_buf,   dtype=np.float32)   # (N,3)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            logger.error(f"xyz_buf形状错误: {xyz.shape}，期望(N,3)")
            return
        if F.ndim != 2 or F.shape[1] != 3:
            logger.error(f"F_buf形状错误: {F.shape}，期望(N,3)")
            return

        self._model, self._params = run_pinn(
            t, xyz, F, epochs=epochs, verbose=False)
        self._ready = True

    def get_params(self) -> dict:
        """返回最新的9个参数字典 {Mx,My,Mz,Bx,By,Bz,Kx,Ky,Kz}"""
        return dict(self._params)

    @property
    def is_ready(self) -> bool:
        return self._ready
