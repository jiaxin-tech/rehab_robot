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
                 w_force:    float = settings.MPC_W_FORCE,
                 control_axis: int = settings.CONTROL_AXIS,
                 equilibrium_position_m: float | None = None):
        self.N          = horizon
        self.dt         = dt
        self.w_tracking = w_tracking
        self.w_comfort  = w_comfort
        self.w_force    = w_force
        if control_axis not in (0, 1, 2):
            raise ValueError("control_axis must be 0, 1, or 2")
        self.control_axis = control_axis
        if equilibrium_position_m is None:
            equilibrium_position_m = settings.JOINT_CENTER[control_axis]
            if control_axis == 0:
                equilibrium_position_m += settings.JOINT_RADIUS * np.cos(
                    settings.JOINT_NEUTRAL
                )
            elif control_axis == 2:
                equilibrium_position_m += settings.JOINT_RADIUS * np.sin(
                    settings.JOINT_NEUTRAL
                )
        self.equilibrium_position_m = float(equilibrium_position_m)

        # 从PINN获取的患者参数（默认值，会被实时更新）
        self.M = settings.PINN_M_INIT
        self.B = settings.PINN_B_INIT
        self.K = settings.PINN_K_INIT

        # 舒适度预测器（由外部注入）
        self.comfort_predictor = None
        # Optional mapping from the scalar controlled coordinate to a full
        # base-frame Cartesian context.  Tangential MPC must install this; the
        # legacy axis substitution below is only for explicitly Cartesian use.
        self.scalar_state_mapper = None

        logger.info(
            "MPC初始化: horizon=%d, dt=%.3fs, axis=%d, equilibrium=%.4fm",
            horizon,
            dt,
            self.control_axis,
            self.equilibrium_position_m,
        )

    def set_patient_params(self, M: float, B: float, K: float):
        """更新PINN辨识出的患者参数"""
        self.M, self.B, self.K = M, B, K
        logger.debug(f"MPC更新患者参数: M={M:.3f} B={B:.3f} K={K:.3f}")

    def set_comfort_predictor(self, predictor):
        """注入舒适度预测器"""
        self.comfort_predictor = predictor

    def set_scalar_state_mapper(self, mapper) -> None:
        """Set ``mapper(arc_m, tangent_velocity_mps) -> (xyz_m, vxyz_mps)``.

        A rehabilitation arc changes more than one Cartesian coordinate.  This
        prevents a scalar arc length from being incorrectly written into one
        Cartesian axis before calling ComfortNet.
        """
        self.scalar_state_mapper = mapper

    def set_equilibrium_position(self, position_m: float) -> None:
        """Set the zero-displacement coordinate in the controlled scalar domain."""
        self.equilibrium_position_m = float(position_m)

    def _predict_comfort(
        self,
        current_force: np.ndarray,
        position_m: float,
        velocity_m_s: float,
        pose_context: np.ndarray | None,
        velocity_context: np.ndarray | None,
    ) -> float:
        if self.scalar_state_mapper is not None:
            pose, velocity = self.scalar_state_mapper(position_m, velocity_m_s)
            pose = np.asarray(pose, dtype=float).reshape(-1)
            velocity = np.asarray(velocity, dtype=float).reshape(-1)
            if (
                pose.shape != (3,)
                or velocity.shape != (3,)
                or not np.all(np.isfinite(pose))
                or not np.all(np.isfinite(velocity))
            ):
                raise ValueError("scalar_state_mapper must return finite xyz and velocity")
        else:
            # Legacy single-axis Cartesian context. New tangential users must
            # install a mapper instead of interpreting arc length as x/y/z.
            pose = np.zeros(3) if pose_context is None else np.asarray(pose_context[:3]).copy()
            velocity = (
                np.zeros(3)
                if velocity_context is None
                else np.asarray(velocity_context[:3]).copy()
            )
            pose[self.control_axis] = position_m
            velocity[self.control_axis] = velocity_m_s
        return self.comfort_predictor.predict(
            fx=current_force[0], fy=current_force[1], fz=current_force[2],
            x=pose[0], y=pose[1], z=pose[2],
            vx=velocity[0], vy=velocity[1], vz=velocity[2],
        )

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
            # x is the configured scalar control coordinate; the elastic term
            # uses displacement relative to the explicit equilibrium value.
            acc = u_seq[i]
            next_vel = vel + acc * self.dt
            # Match acceleration_to_trajectory_pose's semi-implicit integration
            # exactly, so the first MPC prediction and commanded path target do
            # not disagree by one control interval.
            states[i+1, 0] = pos + next_vel * self.dt
            states[i+1, 1] = next_vel
        return states

    def _cost(self, u_flat: np.ndarray,
              x0: np.ndarray,
              ref_traj: np.ndarray,
              current_force: np.ndarray,
              pose_context: np.ndarray | None = None,
              velocity_context: np.ndarray | None = None) -> float:
        """
        MPC代价函数（传给scipy.optimize.minimize）
        u_flat:       (N,) 展平的控制序列
        x0:           当前状态 [pos, vel]
        ref_traj:     (N+1, 2) 参考轨迹
        current_force: 当前有效 base 系内部 wrench 的力分量 [fx,fy,fz]
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
                score = self._predict_comfort(
                    current_force, pos, vel, pose_context, velocity_context
                )
                cost_comfort += self.w_comfort * (1.0 - score)

        # 3. 受力最小化代价
        displacement = states[1:, 0] - self.equilibrium_position_m
        F_seq = self.M * u_seq + self.B * states[1:, 1] + self.K * displacement
        cost_force  = self.w_force * np.sum(F_seq ** 2)

        return cost_track + cost_comfort + cost_force

    def _constraints(self, u_flat: np.ndarray,
                     x0: np.ndarray,
                     current_force: np.ndarray,
                     pose_context: np.ndarray | None = None,
                     velocity_context: np.ndarray | None = None) -> list:
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
                score = self._predict_comfort(
                    current_force, pos, vel, pose_context, velocity_context
                )
                cons.append(score - settings.COMFORT_THRESHOLD)

            # 力约束
            F_pred = abs(self.M * u_seq[i-1] +
                         self.B * vel +
                         self.K * (pos - self.equilibrium_position_m))
            cons.append(settings.MAX_FORCE_N - F_pred)

        return cons

    def solve(self, x0: np.ndarray,
              ref_traj: np.ndarray,
              current_force: np.ndarray,
              u_init: np.ndarray = None,
              pose_context: np.ndarray | None = None,
              velocity_context: np.ndarray | None = None) -> np.ndarray:
        """
        求解MPC，返回最优控制序列的第一步

        Args:
            x0:            当前状态 [标量位置(m), 标量速度(m/s)]；切向模式为弧长坐标
            ref_traj:      (N+1, 2) 参考轨迹（包含当前时刻）
            current_force: 当前有效 base 系内部估计力 [fx,fy,fz]
            u_init:        控制序列初始猜测，None则用零初始化

        Returns:
            u_opt: 最优加速度指令 (m/s²)，只返回第一步
        """
        if u_init is None:
            u_init = np.zeros(self.N)

        # 加速度幅值约束（物理可行）
        a_max = settings.MPC_MAX_ACCEL_M_S2
        bounds = [(-a_max, a_max)] * self.N

        constraints = [{
            "type": "ineq",
            "fun": lambda u: self._constraints(
                u, x0, current_force, pose_context, velocity_context
            ),
        }]

        result = minimize(
            fun=self._cost,
            x0=u_init,
            args=(
                x0,
                ref_traj,
                current_force,
                pose_context,
                velocity_context,
            ),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-4},
        )

        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"MPC solve failed: {result.message}")

        # 只返回第一步控制量
        return result.x[0], result.x   # (u_first, full_sequence)

    def acceleration_to_pose(self, current_pose: np.ndarray,
                             current_vel: np.ndarray,
                             acc: float,
                             axis: int | None = None) -> np.ndarray:
        """
        把加速度指令转换为下一时刻的末端位姿（供ServoJ使用）
        axis: base 坐标轴；默认使用 settings.CONTROL_AXIS。该方法仅供旧的
              单轴 Cartesian 流程使用，康复弧线请用 acceleration_to_trajectory_pose。
        """
        if axis is None:
            axis = self.control_axis
        next_pose = current_pose.copy()
        next_vel  = current_vel.copy()
        next_vel[axis]  += acc * self.dt
        next_pose[axis] += next_vel[axis] * self.dt
        return next_pose, next_vel

    def acceleration_to_trajectory_pose(
        self,
        trajectory,
        current_arc_length_m: float,
        current_velocity_tangent_mps: float,
        acceleration_tangent_mps2: float,
    ) -> tuple[np.ndarray, float, float]:
        """Convert a scalar tangent acceleration into a full base-frame target.

        ``current_arc_length_m`` is physical arc length in metres.  This
        prevents the former fixed-z implementation from
        freezing the simultaneously varying x component of the rehab arc.
        """
        total_arc_m = float(trajectory.total_arc_length_m)
        if not np.isfinite(total_arc_m) or total_arc_m <= 1e-12:
            raise ValueError("Trajectory must have positive arc length")
        next_velocity = float(current_velocity_tangent_mps) + float(
            acceleration_tangent_mps2
        ) * self.dt
        next_arc_m = min(
            total_arc_m,
            max(0.0, float(current_arc_length_m) + next_velocity * self.dt),
        )
        next_s = next_arc_m / total_arc_m
        return trajectory.pose_at_normalized_s(next_s), next_velocity, next_arc_m
