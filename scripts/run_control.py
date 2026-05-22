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

    # ── 安全守卫 ─────────────────────────────────────
    guard = SafetyGuard(force, robot)
    guard.start()

    # ── 加载舒适度模型 ────────────────────────────────
    comfort_pred = ComfortPredictor(settings.COMFORT_MODEL_PATH)

    # ── 初始化PINN和MPC ───────────────────────────────
    online_pinn = OnlinePINN()
    mpc         = MPCController()
    mpc.set_comfort_predictor(comfort_pred)

    # ── 生成参考轨迹 ──────────────────────────────────
    ref_wps = generate_rehab_trajectory()   # (N_total, 6)

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
                    mpc.set_patient_params(p["Mx"], p["Bx"], p["Kx"])  # MPC暂用x轴参数
                    logger.info(
                        f"[Step {step}] PINN更新 | "
                        f"M=({p['Mx']:.2f},{p['My']:.2f},{p['Mz']:.2f}) "
                        f"B=({p['Bx']:.2f},{p['By']:.2f},{p['Bz']:.2f}) "
                        f"K=({p['Kx']:.2f},{p['Ky']:.2f},{p['Kz']:.2f})"
                    )

            # 3. 实时舒适度评分
            # 触觉传感器到位后在此传入 tactile=tactile_array
            comfort_score = comfort_pred.predict(
                fx=f_data["fx"], fy=f_data["fy"], fz=f_data["fz"],
                x=cur_pose[0],   y=cur_pose[1],   z=cur_pose[2],
                vx=cur_vel[0],   vy=cur_vel[1],   vz=cur_vel[2],
                tactile=None,    # ← 触觉传感器到位后改为 tactile=read_tactile()
            )

            # 4. MPC求解（只在PINN就绪后启动）
            if online_pinn.is_ready:
                ref_horizon = ref_states[step: step + settings.MPC_HORIZON + 1]
                u_opt, u_warm = mpc.solve(x0, ref_horizon, F_vec, u_warm)
                next_pose, _ = mpc.acceleration_to_pose(
                    cur_pose, cur_vel, u_opt, axis=0)
            else:
                # PINN未就绪：直接跟踪参考轨迹
                next_pose    = ref_wps[step + 1]
                u_warm       = None

            # 5. 发送运动指令
            _move_l(robot, next_pose)

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
