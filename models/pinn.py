# models/pinn.py
# Inverse PINN：三维任务空间，从经验证的 base 系力/运动数据反推等效参数 M, B, K
#
# 物理方程（三维对角形式）：
#   [Mx  0   0 ] [ẍ]   [Bx  0   0 ] [ẋ]   [Kx  0   0 ] [x]   [Fx]
#   [0   My  0 ] [ÿ] + [0   By  0 ] [ẏ] + [0   Ky  0 ] [y] = [Fy]
#   [0   0   Mz] [z̈]   [0   0   Bz] [ż]   [0   0   Kz] [z]   [Fz]
#
# 即每个轴独立：Mi*ẍi + Bi*ẋi + Ki*xi = Fi，i ∈ {x, y, z}
# 共9个可辨识参数：Mx/My/Mz, Bx/By/Bz, Kx/Ky/Kz

import math
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

    def physics_residual(
        self,
        t: torch.Tensor,
        F: torch.Tensor,
        time_scale_s: torch.Tensor,
        position_scale_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        三轴物理方程残差
        t: (N,1)  归一化时间，requires_grad=True
        F: (N,3)  [Fx, Fy, Fz]，单位 N，base 坐标系
        time_scale_s: 时间归一化尺度，单位 s
        position_scale_m: 各轴位置归一化尺度，单位 m
        返回: (N,3) 每轴的残差
        """
        t = t.requires_grad_(True)
        pos = self.forward(t)   # (N,3)

        residuals = []
        M_list = [self.Mx, self.My, self.Mz]
        B_list = [self.Bx, self.By, self.Bz]
        K_list = [self.Kx, self.Ky, self.Kz]

        for i, (M, B, K) in enumerate(zip(M_list, B_list, K_list)):
            xi = pos[:, i:i+1]   # (N,1) 取第i轴

            vi_normalized = torch.autograd.grad(
                xi, t,
                grad_outputs=torch.ones_like(xi),
                create_graph=True,
            )[0]
            ai_normalized = torch.autograd.grad(
                vi_normalized, t,
                grad_outputs=torch.ones_like(vi_normalized),
                create_graph=True,
            )[0]

            scale_m = position_scale_m[i]
            displacement_m = xi * scale_m
            velocity_m_s = vi_normalized * scale_m / time_scale_s
            acceleration_m_s2 = (
                ai_normalized * scale_m / (time_scale_s ** 2)
            )
            res_i = (
                M * acceleration_m_s2
                + B * velocity_m_s
                + K * displacement_m
                - F[:, i:i+1]
            )
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
        xyz_data: base 系末端位置 (N,3)，单位m，列顺序 [x, y, z]
        F_data:   base 系三维力 (N,3)，单位N，列顺序 [Fx, Fy, Fz]
        epochs:   迭代次数（在线用少，离线验证用多）
    Returns:
        model:  训练好的PINN实例
        params: dict，9个辨识参数
    """
    t_data = np.asarray(t_data, dtype=np.float64).reshape(-1)
    xyz_data = np.asarray(xyz_data, dtype=np.float64)
    F_data = np.asarray(F_data, dtype=np.float64)
    if (
        len(t_data) < 8
        or xyz_data.shape != (len(t_data), 3)
        or F_data.shape != (len(t_data), 3)
    ):
        raise ValueError("Cartesian PINN needs >=8 matching (t, xyz, force) samples")
    if not (
        np.all(np.isfinite(t_data))
        and np.all(np.isfinite(xyz_data))
        and np.all(np.isfinite(F_data))
    ):
        raise ValueError("Cartesian PINN inputs must be finite")
    if np.any(np.diff(t_data) <= 0):
        raise ValueError("Cartesian PINN timestamps must be strictly increasing")

    # ── 数据归一化 ────────────────────────────────────
    t_origin = t_data.min()
    t_scale = max(float(t_data.max() - t_origin), 1e-8)
    t_norm = (t_data - t_origin) / t_scale             # → [0,1]

    xyz_mean = xyz_data.mean(axis=0)                    # (3,)
    xyz_std  = xyz_data.std(axis=0) + 1e-8              # (3,)
    xyz_norm = (xyz_data - xyz_mean) / xyz_std          # → 零均值单位方差

    # 力不归一化，保留物理量纲，让M/B/K的量纲也保持真实
    # （若力量级差异很大可考虑归一化，但需同步调整参数初值）

    # ── 转Tensor ─────────────────────────────────────
    t_t   = torch.tensor(t_norm,   dtype=torch.float32).unsqueeze(1)   # (N,1)
    xyz_t = torch.tensor(xyz_norm, dtype=torch.float32)                 # (N,3)
    F_t   = torch.tensor(F_data,   dtype=torch.float32)                 # (N,3)
    t_scale_t = torch.tensor(t_scale, dtype=torch.float32)
    xyz_std_t = torch.tensor(xyz_std, dtype=torch.float32)

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
        residual = model.physics_residual(
            t_t.clone(), F_t, t_scale_t, xyz_std_t
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
            xyz_buf: base 系末端位置 (N,3)，单位m
            F_buf:   base 系三维力   (N,3)，单位N
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


# ── 沿康复轨迹的标量 PINN ───────────────────────────────
class TangentialPINN(nn.Module):
    """Inverse PINN for ``M s¨ + B s˙ + K(s-s_eq) = F_tangent``.

    ``s`` is physical arc length in metres, not the dimensionless CSV
    ``trajectory_s`` fraction.  This model is the appropriate default for the
    project's coupled x-z rehabilitation arc; the existing three-axis PINN is
    retained for explicitly validated Cartesian experiments.
    """

    def __init__(self, hidden=settings.PINN_HIDDEN_LAYERS):
        super().__init__()
        layers = []
        previous = 1
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.Tanh()]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)
        self._log_M = nn.Parameter(torch.log(torch.tensor(float(settings.PINN_M_INIT))))
        self._log_B = nn.Parameter(torch.log(torch.tensor(float(settings.PINN_B_INIT))))
        self._log_K = nn.Parameter(torch.log(torch.tensor(float(settings.PINN_K_INIT))))

    @property
    def M(self):
        return torch.exp(self._log_M)

    @property
    def B(self):
        return torch.exp(self._log_B)

    @property
    def K(self):
        return torch.exp(self._log_K)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    def physics_residual(
        self,
        t: torch.Tensor,
        force_tangent_n: torch.Tensor,
        time_scale_s: torch.Tensor,
        arc_std_m: torch.Tensor,
        arc_mean_m: torch.Tensor,
        equilibrium_arc_m: torch.Tensor,
    ) -> torch.Tensor:
        t = t.requires_grad_(True)
        normalized_arc = self.forward(t)
        velocity_normalized = torch.autograd.grad(
            normalized_arc,
            t,
            grad_outputs=torch.ones_like(normalized_arc),
            create_graph=True,
        )[0]
        acceleration_normalized = torch.autograd.grad(
            velocity_normalized,
            t,
            grad_outputs=torch.ones_like(velocity_normalized),
            create_graph=True,
        )[0]
        arc_m = normalized_arc * arc_std_m + arc_mean_m
        velocity_mps = velocity_normalized * arc_std_m / time_scale_s
        acceleration_mps2 = acceleration_normalized * arc_std_m / (time_scale_s ** 2)
        return (
            self.M * acceleration_mps2
            + self.B * velocity_mps
            + self.K * (arc_m - equilibrium_arc_m)
            - force_tangent_n
        )

    def get_params(self) -> dict[str, float]:
        return {"M": self.M.item(), "B": self.B.item(), "K": self.K.item()}


def run_tangential_pinn(
    t_data: np.ndarray,
    arc_length_m_data: np.ndarray,
    force_tangent_n_data: np.ndarray,
    *,
    equilibrium_arc_m: float | None = None,
    epochs: int = settings.PINN_EPOCHS,
    lr: float = settings.PINN_LR,
    lam: float = settings.PINN_LAMBDA,
    verbose: bool = True,
):
    """Fit a scalar PINN from valid base-frame trajectory projections only."""
    t = np.asarray(t_data, dtype=np.float64).reshape(-1)
    arc = np.asarray(arc_length_m_data, dtype=np.float64).reshape(-1)
    force = np.asarray(force_tangent_n_data, dtype=np.float64).reshape(-1)
    if len(t) < 8 or len(arc) != len(t) or len(force) != len(t):
        raise ValueError("Tangential PINN needs >=8 equal-length t/arc/force samples")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(arc)) and np.all(np.isfinite(force))):
        raise ValueError("Tangential PINN inputs must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Tangential PINN timestamps must be strictly increasing")
    time_scale_s = float(t[-1] - t[0])
    if time_scale_s <= 0:
        raise ValueError("Tangential PINN needs positive duration")
    arc_mean_m = float(np.mean(arc))
    arc_std_m = float(np.std(arc))
    if arc_std_m <= 1e-9:
        raise ValueError("Tangential PINN needs non-degenerate arc excitation")
    equilibrium = arc_mean_m if equilibrium_arc_m is None else float(equilibrium_arc_m)
    if not math.isfinite(equilibrium):
        raise ValueError("equilibrium_arc_m must be finite")

    t_normalized = ((t - t[0]) / time_scale_s).astype(np.float32)
    arc_normalized = ((arc - arc_mean_m) / arc_std_m).astype(np.float32)
    t_tensor = torch.tensor(t_normalized, dtype=torch.float32).unsqueeze(1)
    arc_tensor = torch.tensor(arc_normalized, dtype=torch.float32).unsqueeze(1)
    force_tensor = torch.tensor(force.astype(np.float32), dtype=torch.float32).unsqueeze(1)
    time_scale_tensor = torch.tensor(time_scale_s, dtype=torch.float32)
    arc_std_tensor = torch.tensor(arc_std_m, dtype=torch.float32)
    arc_mean_tensor = torch.tensor(arc_mean_m, dtype=torch.float32)
    equilibrium_tensor = torch.tensor(equilibrium, dtype=torch.float32)

    model = TangentialPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predicted_arc = model(t_tensor)
        data_loss = ((predicted_arc - arc_tensor) ** 2).mean()
        residual = model.physics_residual(
            t_tensor.clone(),
            force_tensor,
            time_scale_tensor,
            arc_std_tensor,
            arc_mean_tensor,
            equilibrium_tensor,
        )
        physics_loss = (residual ** 2).mean()
        loss = data_loss + lam * physics_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        if verbose and epoch % 200 == 0:
            params = model.get_params()
            logger.info(
                "Tangential PINN epoch=%d data=%.5f phys=%.5f M=%.3f B=%.3f K=%.3f",
                epoch, data_loss.item(), physics_loss.item(), params["M"], params["B"], params["K"],
            )
    params = model.get_params()
    params.update({"equilibrium_arc_m": equilibrium, "coordinate": "trajectory_arc_length_m"})
    return model, params


class OnlineTangentialPINN:
    """Non-control-thread wrapper for scalar, base-frame trajectory identification."""

    def __init__(self) -> None:
        self._params = {
            "M": float(settings.PINN_M_INIT),
            "B": float(settings.PINN_B_INIT),
            "K": float(settings.PINN_K_INIT),
            "equilibrium_arc_m": 0.0,
            "coordinate": "trajectory_arc_length_m",
        }
        self._model = None
        self._ready = False

    def update(
        self,
        time_s: list[float],
        arc_length_m: list[float],
        force_tangent_n: list[float],
        *,
        epochs: int = 100,
    ) -> None:
        if len(time_s) < 30:
            return
        self._model, self._params = run_tangential_pinn(
            np.asarray(time_s),
            np.asarray(arc_length_m),
            np.asarray(force_tangent_n),
            epochs=epochs,
            verbose=False,
        )
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_params(self) -> dict:
        return dict(self._params)
