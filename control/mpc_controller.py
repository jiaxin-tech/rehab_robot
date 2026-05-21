# control/mpc_controller.py
# MPC多目标控制器：轨迹跟踪 + 舒适度最大化 + 安全约束

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
        J = w_track  * ||x - x_ref||²      ← 轨迹跟踪
          + w_comfort * (1 - comfort_score) ← 舒适度最大化
          + w_force   * ||F_pred||²         ← 受力最小化

    硬约束：
        comfort_score >= COMFORT_THRESHOLD  ← 舒适度下限
        |F_pred|       <= MAX_FORCE_N       ← 力绝对上限
    """

    def __init__(self,
                 horizon:    int   = settings.MPC_HORIZON,
                 dt:         float = settings.MPC_DT,
                 w_tracking: float = settings.MPC_W_TRACKING,
                 w_comfort:  float = settings.MPC_W_COMFORT,
                 w_force:    float = settings.MPC_W_FORCE):
        self.N          = horizon
        self.dt         = dt
        self.w_tracking = w_tracking
        self.w_comfort  = w_comfort
        self.w_force    = w_force

        # 从PINN获取的患者参数（默认值，会被实时更新）
        self.M = settings.PINN_M_INIT
        self.B = settings.PINN_B_INIT
        self.K = settings.PINN_K_INIT

        # 舒适度预测器（由外部注入）
        self.comfort_predictor = None

        logger.info(f"MPC初始化: horizon={horizon}, dt={dt}s")

    def set_patient_params(self, M: float, B: float, K: float):
        """更新PINN辨识出的患者参数"""
        self.M, self.B, self.K = M, B, K
        logger.debug(f"MPC更新患者参数: M={M:.3f} B={B:.3f} K={K:.3f}")

    def set_comfort_predictor(self, predictor):
        """注入舒适度预测器"""
        self.comfort_predictor = predictor

    def _predict_states(self, x0: np.ndarray,
                        u_seq: np.ndarray) -> np.ndarray:
        """
        给定初始状态x0和控制序列u_seq，
        用患者动力学模型滚动预测未来N步状态

        x0:    [position, velocity]
        u_seq: (N,) 加速度指令序列
        Returns: (N+1, 2) 状态序列
        """
        states    = np.zeros((self.N + 1, 2))
        states[0] = x0
        for i in range(self.N):
            pos, vel = states[i]
            # 离散化动力学：M*a + B*v + K*x = F → a = (F - B*v - K*x)/M
            # 这里u是加速度指令，F = M*u + B*v + K*x（内力）
            acc          = u_seq[i]
            F_pred       = self.M * acc + self.B * vel + self.K * pos
            states[i+1, 0] = pos + vel * self.dt
            states[i+1, 1] = vel + acc * self.dt
        return states

    def _cost(self, u_flat: np.ndarray,
              x0: np.ndarray,
              ref_traj: np.ndarray,
              current_force: np.ndarray) -> float:
        """
        MPC代价函数（传给scipy.optimize.minimize）
        u_flat:       (N,) 展平的控制序列
        x0:           当前状态 [pos, vel]
        ref_traj:     (N+1, 2) 参考轨迹
        current_force: 当前力传感器读数 [fx,fy,fz]
        """
        u_seq  = u_flat
        states = self._predict_states(x0, u_seq)

        # 1. 轨迹跟踪代价
        track_err   = states - ref_traj
        cost_track  = self.w_tracking * np.sum(track_err ** 2)

        # 2. 舒适度代价（用当前状态近似，不逐步预测）
        cost_comfort = 0.0
        if self.comfort_predictor is not None:
            for i in range(1, self.N + 1):
                pos, vel = states[i]
                score = self.comfort_predictor.predict(
                    current_force[0], current_force[1], current_force[2],
                    pos, 0.0, 0.0,   # 单轴简化，y/z为0
                    vel, 0.0, 0.0,
                )
                cost_comfort += self.w_comfort * (1.0 - score)

        # 3. 受力最小化代价
        F_seq       = self.M * u_seq + self.B * states[1:, 1] + self.K * states[1:, 0]
        cost_force  = self.w_force * np.sum(F_seq ** 2)

        return cost_track + cost_comfort + cost_force

    def _constraints(self, u_flat: np.ndarray,
                     x0: np.ndarray,
                     current_force: np.ndarray) -> list:
        """
        硬约束列表（scipy格式，>= 0 表示满足）
        1. 舒适度 >= COMFORT_THRESHOLD
        2. 预测力 <= MAX_FORCE_N
        """
        u_seq  = u_flat
        states = self._predict_states(x0, u_seq)
        cons   = []

        for i in range(1, self.N + 1):
            pos, vel = states[i]

            # 舒适度约束
            if self.comfort_predictor is not None:
                score = self.comfort_predictor.predict(
                    current_force[0], current_force[1], current_force[2],
                    pos, 0.0, 0.0, vel, 0.0, 0.0,
                )
                cons.append(score - settings.COMFORT_THRESHOLD)

            # 力约束
            F_pred = abs(self.M * u_seq[i-1] +
                         self.B * vel + self.K * pos)
            cons.append(settings.MAX_FORCE_N - F_pred)

        return cons

    def solve(self, x0: np.ndarray,
              ref_traj: np.ndarray,
              current_force: np.ndarray,
              u_init: np.ndarray = None) -> np.ndarray:
        """
        求解MPC，返回最优控制序列的第一步

        Args:
            x0:            当前状态 [position(mm), velocity(mm/s)]
            ref_traj:      (N+1, 2) 参考轨迹（包含当前时刻）
            current_force: 当前力传感器值 [fx,fy,fz]
            u_init:        控制序列初始猜测，None则用零初始化

        Returns:
            u_opt: 最优加速度指令 (mm/s²)，只返回第一步
        """
        if u_init is None:
            u_init = np.zeros(self.N)

        # 加速度幅值约束（物理可行）
        a_max  = 500.0   # mm/s²，根据实际调整
        bounds = [(-a_max, a_max)] * self.N

        constraints = [{
            "type": "ineq",
            "fun":  lambda u: self._constraints(u, x0, current_force),
        }]

        result = minimize(
            fun=self._cost,
            x0=u_init,
            args=(x0, ref_traj, current_force),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
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
