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
    状态量：x = [position, velocity]  (2维，单轴)
    控制量：u = 末端加速度指令        (1维，单轴)

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
                 w_jerk:     float = settings.MPC_W_JERK):
        self.N          = horizon
        self.dt         = dt
        self.w_tracking = w_tracking
        self.w_comfort  = w_comfort
        self.w_jerk     = w_jerk

        logger.info(f"MPC初始化: horizon={horizon}, dt={dt}s")

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

    def _predict_states(self, x0: np.ndarray,
                        u_seq: np.ndarray) -> np.ndarray:
        """
        给定初始状态x0和控制序列u_seq，用运动学模型滚动预测未来N步状态。

        x0:    [position, velocity]
        u_seq: (N,) 加速度指令序列
        Returns: (N+1, 2) 状态序列
        """
        states    = np.zeros((self.N + 1, 2))
        states[0] = x0
        for i in range(self.N):
            pos, vel = states[i]
            acc = u_seq[i]
            states[i+1, 0] = pos + vel * self.dt
            states[i+1, 1] = vel + acc * self.dt
        return states

    def _cost(self, u_flat: np.ndarray,
              x0: np.ndarray,
              ref_traj: np.ndarray,
              comfort_score: float,
              last_acc: float = 0.0) -> float:
        """
        MPC代价函数（传给scipy.optimize.minimize）
        u_flat:       (N,) 展平的控制序列
        x0:           当前状态 [pos, vel]
        ref_traj:     (N+1, 2) 参考轨迹
        comfort_score: 当前舒适度分数，0~1，越大越舒适
        last_acc:      上一控制周期已执行的加速度，用于第一步jerk惩罚
        """
        u_seq  = u_flat
        states = self._predict_states(x0, u_seq)
        w_tracking_eff, w_jerk_eff = self._comfort_weights(comfort_score)

        # 1. 轨迹跟踪代价
        track_err   = states - ref_traj
        cost_track  = w_tracking_eff * np.sum(track_err ** 2)

        comfort_score = float(np.clip(comfort_score, 0.0, 1.0))
        u_norm = u_seq / max(settings.MPC_A_MAX, 1e-6)
        cost_comfort = self.w_comfort * (1.0 - comfort_score) * np.sum(u_norm ** 2)

        # 2. jerk惩罚：jerk = 加速度变化率，归一化后避免单位尺度主导优化
        prev_u = np.concatenate(([float(last_acc)], u_seq[:-1]))
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
            x0:            当前状态 [position(mm), velocity(mm/s)]
            ref_traj:      (N+1, 2) 参考轨迹（包含当前时刻）
            comfort_score: 当前舒适度分数，0~1
            u_init:        控制序列初始猜测，None则用零初始化
            last_acc:      上一控制周期已执行的加速度

        Returns:
            u_opt: 最优加速度指令 (mm/s²)，只返回第一步
        """
        if u_init is None:
            u_init = np.zeros(self.N)

        # 加速度幅值约束（物理可行）
        a_max  = settings.MPC_A_MAX
        bounds = [(-a_max, a_max)] * self.N

        result = minimize(
            fun=self._cost,
            x0=u_init,
            args=(x0, ref_traj, comfort_score, last_acc),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-4},
        )

        if not result.success:
            logger.warning(f"MPC求解警告: {result.message}")

        # 只返回第一步控制量
        return result.x[0], result.x   # (u_first, full_sequence)

    def acceleration_to_pose(self, current_pose: np.ndarray,
                             current_vel: np.ndarray,
                             acc: float,
                             axis: int = 0) -> np.ndarray:
        """
        把加速度指令转换为下一时刻的末端位姿（供ServoJ使用）
        axis: 0=x轴运动
        """
        next_pose = current_pose.copy()
        next_vel  = current_vel.copy()
        next_vel[axis]  += acc * self.dt
        next_pose[axis] += next_vel[axis] * self.dt
        return next_pose, next_vel
