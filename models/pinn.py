# models/pinn.py
# Inverse PINN：从力/运动数据反推关节等效参数 M, B, K

import numpy as np
import torch
import torch.nn as nn
from config import settings
from utils.logger import get_logger

logger = get_logger("PINN")


# ── 网络定义 ─────────────────────────────────────────
class PINN(nn.Module):
    """
    Inverse PINN for 1-DOF joint dynamics:
        M * ẍ + B * ẋ + K * x = F_external

    网络：时间 t → 末端位置 x(t)
    可训练物理参数：M（惯量）、B（阻尼）、K（刚度）

    训练后：
        M, B, K = 当前患者的关节等效参数
        作为MPC内部模型的输入
    """

    def __init__(self, hidden=settings.PINN_HIDDEN_LAYERS):
        super().__init__()

        # 位置预测网络：t → x
        layers = []
        prev = 1
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # 可训练物理参数（用log保证正数）
        self._log_M = nn.Parameter(torch.log(torch.tensor(settings.PINN_M_INIT)))
        self._log_B = nn.Parameter(torch.log(torch.tensor(settings.PINN_B_INIT)))
        self._log_K = nn.Parameter(torch.log(torch.tensor(settings.PINN_K_INIT)))

    @property
    def M(self): return torch.exp(self._log_M)
    @property
    def B(self): return torch.exp(self._log_B)
    @property
    def K(self): return torch.exp(self._log_K)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """输入时间t，输出预测位置x"""
        return self.net(t)

    def physics_residual(self, t: torch.Tensor,
                         F: torch.Tensor) -> torch.Tensor:
        """
        计算物理方程残差：M*ẍ + B*ẋ + K*x - F
        利用autograd对t求导得到速度和加速度
        """
        t = t.requires_grad_(True)
        x = self.forward(t)

        # 自动微分求速度和加速度
        v = torch.autograd.grad(
            x, t,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
        )[0]
        a = torch.autograd.grad(
            v, t,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
        )[0]

        residual = self.M * a + self.B * v + self.K * x - F
        return residual


# ── 离线训练（用于验证/离线分析）────────────────────
def train_offline(t_data: np.ndarray,
                  x_data: np.ndarray,
                  F_data: np.ndarray,
                  epochs: int   = settings.PINN_EPOCHS,
                  lr: float     = settings.PINN_LR,
                  lam: float    = settings.PINN_LAMBDA,
                  verbose: bool = True):
    """
    离线训练PINN，用于验证参数辨识精度

    Args:
        t_data: 时间序列 (N,)
        x_data: 测量位置序列 (N,)  单位mm
        F_data: 测量力序列 (N,)    单位N
    Returns:
        model: 训练好的PINN
        params: dict {M, B, K}
    """
    # 时间归一化到 [0,1]（提升网络收敛性）
    t_scale = t_data.max()
    t_norm  = t_data / t_scale

    # 位置归一化到 [-1,1]
    x_mean, x_std = x_data.mean(), x_data.std() + 1e-8
    x_norm = (x_data - x_mean) / x_std

    # 转Tensor
    t_t = torch.tensor(t_norm,  dtype=torch.float32).unsqueeze(1)
    x_t = torch.tensor(x_norm,  dtype=torch.float32).unsqueeze(1)
    F_t = torch.tensor(F_data,  dtype=torch.float32).unsqueeze(1)

    model     = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()

        # 数据损失：网络预测位置 vs 测量位置
        x_pred = model(t_t)
        loss_data = ((x_pred - x_t) ** 2).mean()

        # 物理损失：方程残差趋近0
        residual  = model.physics_residual(t_t.clone(), F_t)
        loss_phys = (residual ** 2).mean()

        loss = loss_data + lam * loss_phys
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and ep % 200 == 0:
            logger.info(
                f"Epoch {ep:5d} | data={loss_data.item():.4f} "
                f"phys={loss_phys.item():.4f} "
                f"M={model.M.item():.3f} "
                f"B={model.B.item():.3f} "
                f"K={model.K.item():.3f}"
            )

    params = dict(M=model.M.item(), B=model.B.item(), K=model.K.item())
    logger.info(f"PINN辨识结果: M={params['M']:.3f} kg, "
                f"B={params['B']:.3f} N·s/m, K={params['K']:.3f} N/m")
    return model, params


# ── 在线辨识封装 ─────────────────────────────────────
class OnlinePINN:
    """
    在线参数辨识封装
    在控制系统中的用法：
        1. 机械臂做短暂探测运动（3~5s持续激励）
        2. 调用 update(t, x, F) 更新参数估计
        3. 调用 get_params() 获取最新 M/B/K 供MPC使用
    """

    def __init__(self):
        self._params = dict(
            M=settings.PINN_M_INIT,
            B=settings.PINN_B_INIT,
            K=settings.PINN_K_INIT,
        )
        self._model   = None
        self._ready   = False

    def update(self, t_buf: list, x_buf: list, F_buf: list,
               epochs: int = 500):
        """
        用最近一段时间窗口的数据重新辨识参数
        t_buf, x_buf, F_buf: 时间、位置、力的列表（滑动窗口）
        epochs较少，保证在线速度
        """
        if len(t_buf) < 20:
            return  # 数据不够

        t = np.array(t_buf, dtype=np.float32)
        x = np.array(x_buf, dtype=np.float32)
        F = np.array(F_buf, dtype=np.float32)

        self._model, self._params = train_offline(
            t, x, F,
            epochs=epochs,
            verbose=False,
        )
        self._ready = True
        logger.info(f"[OnlinePINN] 更新: M={self._params['M']:.3f} "
                    f"B={self._params['B']:.3f} K={self._params['K']:.3f}")

    def get_params(self) -> dict:
        """返回当前最新的 {M, B, K}"""
        return dict(self._params)

    @property
    def is_ready(self) -> bool:
        """首次辨识完成后为True"""
        return self._ready
