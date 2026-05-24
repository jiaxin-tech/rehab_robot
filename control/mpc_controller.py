# control/mpc_controller.py
# MPC控制器：轨迹跟踪 + jerk平滑 + 舒适度调制

import numpy as np
from scipy.optimize import minimize
from config import settings
from utils.logger import get_logger

logger = get_logger("MPC")


class MPCController:
    """
    多目标MPC控制器
    状态量：x = [p1..pn, v1..vn]      (2*dim维，n维任务空间)
    控制量：u = [a1..an]              (dim维末端加速度指令)

    代价函数：
        J = w_track_eff(comfort) * ||x - x_ref||²
          + w_comfort * (1 - comfort_score) * ||u||²
          + w_jerk_eff(comfort) * ||du/dt||²

    comfort低时降低跟踪权重、提高jerk权重，让控制更柔和；
    M/B/K与力数据只用于ComfortNet生成comfort_score。
    """

    def __init__(self,
                 horizon:    int   = settings.MPC_HORIZON,
                 dt:         float = settings.MPC_DT,
                 w_tracking: float = settings.MPC_W_TRACKING,
                 w_comfort:  float = settings.MPC_W_COMFORT,
                 w_jerk:     float = settings.MPC_W_JERK,
                 dim:        int   = settings.MPC_DIM):
        self.N          = horizon
        self.dt         = dt
        self.w_tracking = w_tracking
        self.w_comfort  = w_comfort
        self.w_jerk     = w_jerk
        self.dim        = int(dim)
        if self.dim < 1:
            raise ValueError("MPC dim must be >= 1")

        logger.info(f"MPC初始化: horizon={horizon}, dt={dt}s, dim={self.dim}")

    def _comfort_weights(self, comfort_score: float) -> tuple[float, float]:
        """
        根据舒适度动态调整权重。

        comfort=1: 使用基础tracking/jerk权重
        comfort=0: tracking降到最小比例，jerk提高到最大增益
        """
        comfort_score = float(np.clip(comfort_score, 0.0, 1.0))
        discomfort = 1.0 - comfort_score

        min_track_scale = float(np.clip(settings.MPC_TRACKING_MIN_SCALE, 0.0, 1.0))
        tracking_scale = min_track_scale + (1.0 - min_track_scale) * comfort_score
        jerk_scale = 1.0 + settings.MPC_JERK_COMFORT_GAIN * discomfort

        return self.w_tracking * tracking_scale, self.w_jerk * jerk_scale

    def _reshape_u(self, u_flat: np.ndarray) -> np.ndarray:
        """把优化器的一维变量还原为 (N, dim) 加速度序列。"""
        u_flat = np.asarray(u_flat, dtype=float)
        expected = self.N * self.dim
        if u_flat.size != expected:
            raise ValueError(f"u size must be {expected}, got {u_flat.size}")
        return u_flat.reshape(self.N, self.dim)

    def _prepare_last_acc(self, last_acc) -> np.ndarray:
        acc = np.asarray(last_acc, dtype=float).reshape(-1)
        if acc.size == 1 and self.dim > 1:
            return np.full(self.dim, float(acc[0]))
        if acc.size != self.dim:
            raise ValueError(f"last_acc size must be {self.dim}, got {acc.size}")
        return acc

    def _predict_states(self, x0: np.ndarray,
                        u_seq: np.ndarray) -> np.ndarray:
        """
        给定初始状态x0和控制序列u_seq，用运动学模型滚动预测未来N步状态。

        x0:    (2*dim,) [p1..pn, v1..vn]
        u_seq: (N, dim) 加速度指令序列
        Returns: (N+1, 2*dim) 状态序列
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != 2 * self.dim:
            raise ValueError(f"x0 size must be {2 * self.dim}, got {x0.size}")

        states    = np.zeros((self.N + 1, 2 * self.dim))
        states[0] = x0
        for i in range(self.N):
            pos = states[i, :self.dim]
            vel = states[i, self.dim:]
            acc = u_seq[i]
            states[i+1, :self.dim] = pos + vel * self.dt + 0.5 * acc * self.dt ** 2
            states[i+1, self.dim:] = vel + acc * self.dt
        return states

    def _cost(self, u_flat: np.ndarray,
              x0: np.ndarray,
              ref_traj: np.ndarray,
              comfort_score: float,
              last_acc: float = 0.0) -> float:
        """
        MPC代价函数（传给scipy.optimize.minimize）
        u_flat:       (N*dim,) 展平的控制序列
        x0:           当前状态 [p1..pn, v1..vn]
        ref_traj:     (N+1, 2*dim) 参考轨迹
        comfort_score: 当前舒适度分数，0~1，越大越舒适
        last_acc:      上一控制周期已执行的加速度，用于第一步jerk惩罚
        """
        u_seq  = self._reshape_u(u_flat)
        ref_traj = np.asarray(ref_traj, dtype=float)
        if ref_traj.shape != (self.N + 1, 2 * self.dim):
            raise ValueError(
                f"ref_traj shape must be {(self.N + 1, 2 * self.dim)}, got {ref_traj.shape}"
            )
        states = self._predict_states(x0, u_seq)
        w_tracking_eff, w_jerk_eff = self._comfort_weights(comfort_score)

        # 1. 轨迹跟踪代价
        pos_err = states[:, :self.dim] - ref_traj[:, :self.dim]
        vel_err = states[:, self.dim:] - ref_traj[:, self.dim:]
        pos_err_norm = pos_err / max(settings.MPC_POS_SCALE, 1e-6)
        vel_err_norm = vel_err / max(settings.MPC_VEL_SCALE, 1e-6)
        cost_track = w_tracking_eff * (
            settings.MPC_W_POS * np.sum(pos_err_norm ** 2)
            + settings.MPC_W_VEL * np.sum(vel_err_norm ** 2)
        )

        comfort_score = float(np.clip(comfort_score, 0.0, 1.0))
        u_norm = u_seq / max(settings.MPC_A_MAX, 1e-6)
        cost_comfort = self.w_comfort * (1.0 - comfort_score) * np.sum(u_norm ** 2)

        # 2. jerk惩罚：jerk = 加速度变化率，归一化后避免单位尺度主导优化
        last_acc_vec = self._prepare_last_acc(last_acc)
        prev_u = np.vstack([last_acc_vec, u_seq[:-1]])
        jerk = (u_seq - prev_u) / max(self.dt, 1e-6)
        jerk_norm = jerk / max(settings.MPC_A_MAX / max(self.dt, 1e-6), 1e-6)
        cost_jerk = w_jerk_eff * np.sum(jerk_norm ** 2)

        return cost_track + cost_comfort + cost_jerk

    def solve(self, x0: np.ndarray,
              ref_traj: np.ndarray,
              comfort_score: float = 1.0,
              u_init: np.ndarray = None,
              last_acc: float = 0.0) -> np.ndarray:
        """
        求解MPC，返回最优控制序列的第一步

        Args:
            x0:            当前状态 [p1..pn, v1..vn]
            ref_traj:      (N+1, 2*dim) 参考轨迹（包含当前时刻）
            comfort_score: 当前舒适度分数，0~1
            u_init:        控制序列初始猜测，None则用零初始化，可为(N,dim)或(N*dim,)
            last_acc:      上一控制周期已执行的加速度

        Returns:
            u_opt: 最优加速度指令 (mm/s²)，dim=1时为标量，否则为(dim,)
        """
        if u_init is None:
            u_init_flat = np.zeros(self.N * self.dim)
        else:
            u_init_flat = np.asarray(u_init, dtype=float).reshape(-1)
            if u_init_flat.size != self.N * self.dim:
                raise ValueError(
                    f"u_init size must be {self.N * self.dim}, got {u_init_flat.size}"
                )

        # 加速度幅值约束（物理可行）
        a_max  = settings.MPC_A_MAX
        bounds = [(-a_max, a_max)] * (self.N * self.dim)

        result = minimize(
            fun=self._cost,
            x0=u_init_flat,
            args=(x0, ref_traj, comfort_score, last_acc),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-4},
        )

        if not result.success:
            logger.warning(f"MPC求解警告: {result.message}")

        u_seq = self._reshape_u(result.x)
        u_first = u_seq[0, 0] if self.dim == 1 else u_seq[0].copy()
        return u_first, u_seq   # (u_first, full_sequence)

    def acceleration_to_pose(self, current_pose: np.ndarray,
                             current_vel: np.ndarray,
                             acc,
                             axes=None,
                             axis: int = None) -> np.ndarray:
        """
        把加速度指令转换为下一时刻的末端位姿（供ServoP使用）
        axes: acc各维对应的pose/velocity索引，默认按前dim个xyz轴
        """
        if axes is None:
            axes = [axis] if axis is not None else list(range(self.dim))
        axes = list(axes)
        acc_vec = np.asarray(acc, dtype=float).reshape(-1)
        if acc_vec.size == 1 and len(axes) == 1:
            pass
        elif acc_vec.size != len(axes):
            raise ValueError(f"acc size must match axes length: {acc_vec.size} vs {len(axes)}")

        next_pose = current_pose.copy()
        next_vel  = current_vel.copy()
        for a, ax in zip(acc_vec, axes):
            next_pose[ax] += next_vel[ax] * self.dt + 0.5 * a * self.dt ** 2
            next_vel[ax]  += a * self.dt
        return next_pose, next_vel
