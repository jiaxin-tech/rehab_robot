# scripts/run_control.py
# 入口：运行完整控制系统（PINN在线辨识 + MPC实时控制）

import argparse
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import settings
from hardware.dobot_cr5 import DobotCR5
from hardware.force_sensor import ForceSensor
from collection.safety_guard import SafetyGuard
from collection.trajectory import generate_rehab_trajectory
from models.comfort_net import ComfortPredictor
from models.pinn import OnlinePINN
from control.mpc_controller import MPCController
from utils.logger import get_logger

logger = get_logger("RunControl")

# PINN在线更新的滑动窗口大小
PINN_WINDOW = 200   # 200帧 @ 50Hz = 4秒数据


def _make_robot(robot_ip: str) -> DobotCR5:
    return DobotCR5(
        ip_address=robot_ip,
        dashboard_port=settings.ROBOT_DASH_PORT,
        move_port=settings.ROBOT_MOVE_PORT,
        feedback_port=settings.ROBOT_FEED_PORT,
    )


def _set_speed(robot, ratio: int):
    if hasattr(robot, "set_speed"):
        return robot.set_speed(ratio)
    return robot.set_speed_ratio(ratio)


def _move_l(robot, pose):
    if hasattr(robot, "move_l"):
        return robot.move_l(pose)
    return robot.mov_l(*[float(v) for v in pose[:6]])


def _servo_p(robot, pose):
    if hasattr(robot, "servo_p"):
        return robot.servo_p(*[float(v) for v in pose[:6]])
    return _move_l(robot, pose)


def _get_tool_orientation(robot):
    configured = getattr(settings, "TOOL_DOWN_ORIENTATION", None)
    if configured is not None:
        orientation = [float(v) for v in configured[:3]]
        logger.info(
            "使用 settings.TOOL_DOWN_ORIENTATION 锁定末端姿态: "
            f"rx={orientation[0]:.2f}, ry={orientation[1]:.2f}, rz={orientation[2]:.2f}"
        )
        return orientation

    pose = robot.get_cartesian_pose()
    if pose is None or len(pose) < 6:
        raise RuntimeError("无法读取当前末端姿态，请检查机器人反馈连接")

    orientation = [float(v) for v in pose[3:6]]
    logger.info(
        "已将当前末端姿态锁定为夹爪向下姿态: "
        f"rx={orientation[0]:.2f}, ry={orientation[1]:.2f}, rz={orientation[2]:.2f}"
    )
    return orientation


def _stop_robot(robot):
    if hasattr(robot, "stop"):
        return robot.stop()
    return robot.stop_move()


def run(robot_ip: str, sensor_ip: str, subject_id: str):

    # ── 初始化硬件 ────────────────────────────────────
    robot = _make_robot(robot_ip)
    force = ForceSensor(ip=sensor_ip)
    robot.connect()
    force.connect()
    logger.info("等待反馈稳定 2s...")
    time.sleep(2.0)

    robot.clear_error()
    robot.enable(load=0.0)
    _set_speed(robot, settings.INIT_SPEED_RATIO)
    time.sleep(1.0)

    force.start_streaming()
    time.sleep(0.5)
    force.set_bias()
    time.sleep(0.3)
    tool_orientation = _get_tool_orientation(robot)

    # ── 安全守卫 ─────────────────────────────────────
    guard = SafetyGuard(force, robot)
    guard.start()

    # ── 加载舒适度模型 ────────────────────────────────
    comfort_pred = ComfortPredictor(settings.COMFORT_MODEL_PATH)

    # ── 初始化PINN和MPC ───────────────────────────────
    online_pinn = OnlinePINN()
    mpc         = MPCController()

    # ── 生成参考轨迹 ──────────────────────────────────
    ref_wps = generate_rehab_trajectory(orientation=tool_orientation)   # (N_total, 6)

    # 参考轨迹转为状态序列 [position, velocity]（单轴x）
    ref_x   = ref_wps[:, 0]
    ref_vx  = np.gradient(ref_x, settings.COLLECT_DT)
    ref_states = np.stack([ref_x, ref_vx], axis=1)   # (N_total, 2)

    # ── 滑动窗口缓存（供PINN在线更新用）─────────────
    # xyz_buf: list of [x,y,z]   F_buf: list of [Fx,Fy,Fz]
    t_buf, xyz_buf, F_buf = [], [], []
    t0 = time.time()

    # ── 主控制循环 ─────────────────────────────────────
    logger.info("=== 控制循环启动 ===")
    step        = 0
    u_warm      = None   # MPC热启动：上一次的控制序列
    last_acc    = 0.0    # 上一拍已执行加速度，用于jerk惩罚
    prev_pose   = None
    prev_time   = None

    try:
        while step < len(ref_states) - settings.MPC_HORIZON - 1:
            loop_start = time.time()
            guard.check()

            # 1. 读取当前状态
            now = time.time()
            f_data = force.get()
            cur_pose = np.array(robot.get_cartesian_pose(), dtype=float)
            if prev_pose is None:
                cur_vel = np.zeros(3, dtype=float)
            else:
                dt_pose = max(now - prev_time, 1e-6)
                cur_vel = (cur_pose[:3] - prev_pose[:3]) / dt_pose

            x0 = np.array([cur_pose[0], cur_vel[0]])   # 单轴
            F_vec = np.array([f_data["fx"], f_data["fy"], f_data["fz"]])

            # 2. 更新PINN滑动窗口（三维）
            t_now = time.time() - t0
            t_buf.append(t_now)
            xyz_buf.append(cur_pose[:3].tolist())                    # [x, y, z] mm
            F_buf.append([f_data["fx"], f_data["fy"], f_data["fz"]])  # [Fx,Fy,Fz]

            if len(t_buf) > PINN_WINDOW:
                t_buf.pop(0); xyz_buf.pop(0); F_buf.pop(0)

            # 每100步更新一次PINN
            if step % 100 == 0 and len(t_buf) >= 50:
                online_pinn.update(t_buf, xyz_buf, F_buf, epochs=300)
                if online_pinn.is_ready:
                    p = online_pinn.get_params()
                    logger.info(
                        f"[Step {step}] PINN更新 | "
                        f"M=({p['Mx']:.6f},{p['My']:.6f},{p['Mz']:.6f}) "
                        f"B=({p['Bx']:.6f},{p['By']:.6f},{p['Bz']:.6f}) "
                        f"K=({p['Kx']:.6f},{p['Ky']:.6f},{p['Kz']:.6f})"
                    )

            # 3. 实时舒适度评分
            pinn_params = online_pinn.get_params()
            comfort_score = comfort_pred.predict(
                fx=f_data["fx"], fy=f_data["fy"], fz=f_data["fz"],
                Mx=pinn_params["Mx"], My=pinn_params["My"], Mz=pinn_params["Mz"],
                Bx=pinn_params["Bx"], By=pinn_params["By"], Bz=pinn_params["Bz"],
                Kx=pinn_params["Kx"], Ky=pinn_params["Ky"], Kz=pinn_params["Kz"],
            )

            # 4. MPC求解：只受轨迹与comfort_score影响
            ref_horizon = ref_states[step: step + settings.MPC_HORIZON + 1]
            u_opt, u_warm = mpc.solve(
                x0, ref_horizon, comfort_score, u_warm, last_acc)
            next_pose, _ = mpc.acceleration_to_pose(
                cur_pose, cur_vel, u_opt, axis=0)
            next_pose[3:6] = tool_orientation
            last_acc = u_opt

            # 5. 发送运动指令
            _servo_p(robot, next_pose)

            prev_pose = cur_pose
            prev_time = now

            # 6. 日志（每50步）
            if step % 50 == 0:
                logger.info(
                    f"[Step {step:4d}] "
                    f"pos={cur_pose[0]:.1f}mm  "
                    f"F={np.linalg.norm(F_vec):.1f}N  "
                    f"comfort={comfort_score:.3f}  "
                    f"pinn_ready={online_pinn.is_ready}"
                )

            step += 1

            # 7. 维持控制频率
            elapsed = time.time() - loop_start
            sleep_t = settings.MPC_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif elapsed > settings.MPC_DT * 1.5:
                logger.warning(f"控制循环超时: {elapsed*1000:.1f}ms > {settings.MPC_DT*1000:.0f}ms")

    except KeyboardInterrupt:
        logger.info("用户中止控制")
        _stop_robot(robot)
    except RuntimeError as e:
        logger.error(f"安全停止: {e}")
    finally:
        try:
            _stop_robot(robot)
        except Exception as e:
            logger.warning(f"停止机器人失败: {e}")
        guard.stop()
        force.disconnect()
        robot.disable()
        robot.disconnect()
        logger.info("=== 控制系统已停止 ===")


def main():
    parser = argparse.ArgumentParser(description="康复机器人控制系统")
    parser.add_argument("--robot-ip",  default=settings.ROBOT_IP)
    parser.add_argument("--sensor-ip", default=settings.SENSOR_IP)
    parser.add_argument("--subject",   default="subject_001")
    args = parser.parse_args()
    run(args.robot_ip, args.sensor_ip, args.subject)


if __name__ == "__main__":
    main()
