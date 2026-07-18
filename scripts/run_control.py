# scripts/run_control.py
# 入口：运行完整控制系统（PINN在线辨识 + MPC实时控制）

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from collection.safety_guard import SafetyGuard
from collection.trajectory import generate_excitation_trajectory
from config import settings
from control.mpc_controller import MPCController
from hardware.windows import RokaeForceSensor, RokaeRobot
from models.comfort_net import ComfortPredictor
from models.pinn import OnlinePINN
from utils.logger import get_logger


logger = get_logger("RunControl")

# PINN在线更新的滑动窗口大小
PINN_WINDOW = 200  # 200帧 @ 50Hz = 4秒数据


def _make_robot(robot_ip: str) -> RokaeRobot:
    return RokaeRobot(
        ip_address=robot_ip,
        local_ip=settings.ROBOT_LOCAL_IP,
        robot_class=settings.ROBOT_CLASS,
        state_interval_ms=settings.ROBOT_STATE_MS,
        max_linear_speed_mm_s=settings.ROBOT_MAX_SPEED,
        command_cache_size=settings.ROBOT_CMD_CACHE,
        rt_network_tolerance_percent=settings.ROBOT_RT_NETWORK_TOLERANCE,
        rt_filter_hz=settings.ROBOT_RT_FILTER_HZ,
    )


def run(robot_ip: str, subject_id: str) -> None:
    # 先加载算法和参考轨迹，避免机械臂上电后才发现模型文件不可用。
    comfort_pred = ComfortPredictor(settings.COMFORT_MODEL_PATH)
    online_pinn = OnlinePINN()
    mpc = MPCController()
    mpc.set_comfort_predictor(comfort_pred)

    ref_wps = generate_excitation_trajectory()
    ref_x = ref_wps[:, 0]
    ref_vx = np.gradient(ref_x, settings.MPC_DT)
    ref_states = np.stack([ref_x, ref_vx], axis=1)

    robot = _make_robot(robot_ip)
    force = RokaeForceSensor(robot)
    guard = None

    try:
        robot.connect()
        force.connect()
        logger.info("等待反馈稳定 2s...")
        time.sleep(2.0)

        robot.clear_error()
        robot.enable(load=0.0)
        robot.set_speed(settings.INIT_SPEED_RATIO)
        time.sleep(1.0)

        force.start_streaming()
        time.sleep(0.5)
        force.set_bias()
        time.sleep(0.3)

        guard = SafetyGuard(force, robot)
        guard.start()

        # 先以 NRT MoveL 安全到达参考起点，再切换实时回调，避免第一帧
        # MPC/参考目标相对当前位姿产生大幅跳变。
        robot.move_l(ref_wps[0])
        robot.wait_idle()
        guard.check()
        robot.disable()
        robot.enable_realtime(load=0.0)
        guard.check()
        robot.start_realtime_cartesian(ref_wps[0])

        t_buf: list[float] = []
        xyz_buf: list[list[float]] = []
        force_buf: list[list[float]] = []
        t0 = time.perf_counter()
        step = 0
        u_warm = None

        logger.info("=== 控制循环启动：subject=%s ===", subject_id)
        while step < len(ref_states) - settings.MPC_HORIZON - 1:
            loop_start = time.perf_counter()
            guard.check()

            state = robot.get_state()
            f_data = force.get()
            cur_pose = np.asarray(state["cartesian_pose"], dtype=float)
            cur_vel = np.asarray(state["tcp_speed"], dtype=float)
            x0 = np.array([cur_pose[0], cur_vel[0]])
            force_vector = np.array(
                [f_data["fx"], f_data["fy"], f_data["fz"]]
            )

            t_buf.append(time.perf_counter() - t0)
            xyz_buf.append(cur_pose[:3].tolist())
            force_buf.append(force_vector.tolist())
            if len(t_buf) > PINN_WINDOW:
                t_buf.pop(0)
                xyz_buf.pop(0)
                force_buf.pop(0)

            if step % 100 == 0 and len(t_buf) >= 50:
                online_pinn.update(t_buf, xyz_buf, force_buf, epochs=300)
                if online_pinn.is_ready:
                    params = online_pinn.get_params()
                    mpc.set_patient_params(
                        params["Mx"], params["Bx"], params["Kx"]
                    )
                    logger.info(
                        "[Step %d] PINN更新 | M=(%.2f,%.2f,%.2f) "
                        "B=(%.2f,%.2f,%.2f) K=(%.2f,%.2f,%.2f)",
                        step,
                        params["Mx"],
                        params["My"],
                        params["Mz"],
                        params["Bx"],
                        params["By"],
                        params["Bz"],
                        params["Kx"],
                        params["Ky"],
                        params["Kz"],
                    )

            comfort_score = comfort_pred.predict(
                fx=f_data["fx"],
                fy=f_data["fy"],
                fz=f_data["fz"],
                x=cur_pose[0],
                y=cur_pose[1],
                z=cur_pose[2],
                vx=cur_vel[0],
                vy=cur_vel[1],
                vz=cur_vel[2],
                tactile=None,
            )

            if online_pinn.is_ready:
                ref_horizon = ref_states[
                    step : step + settings.MPC_HORIZON + 1
                ]
                u_opt, u_warm = mpc.solve(
                    x0, ref_horizon, force_vector, u_warm
                )
                next_pose, _ = mpc.acceleration_to_pose(
                    cur_pose, cur_vel, u_opt, axis=0
                )
            else:
                next_pose = ref_wps[step + 1]
                u_warm = None

            # 这里只更新共享目标；1 ms 指令调度由 xCoreSDK 回调线程完成。
            robot.set_realtime_cartesian_target(next_pose)

            if step % 50 == 0:
                logger.info(
                    "[Step %4d] pos=%.1fmm F=%.1fN comfort=%.3f "
                    "pinn_ready=%s",
                    step,
                    cur_pose[0],
                    np.linalg.norm(force_vector),
                    comfort_score,
                    online_pinn.is_ready,
                )

            step += 1
            elapsed = time.perf_counter() - loop_start
            sleep_time = settings.MPC_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > settings.MPC_DT * 1.5:
                logger.warning(
                    "控制循环超时: %.1fms > %.0fms",
                    elapsed * 1000.0,
                    settings.MPC_DT * 1000.0,
                )

    except KeyboardInterrupt:
        logger.info("用户中止控制")
        robot.stop()
    except RuntimeError as exc:
        logger.error("安全停止: %s", exc)
        robot.stop()
    finally:
        if guard is not None:
            guard.stop()
        try:
            robot.stop_realtime()
        except Exception as exc:
            logger.warning("停止实时模式失败: %s", exc)
        force.disconnect()
        try:
            robot.disable()
        except Exception as exc:
            logger.warning("机械臂下电失败: %s", exc)
        robot.disconnect()
        logger.info("=== 控制系统已停止 ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="康复机器人控制系统")
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--subject", default="subject_001")
    args = parser.parse_args()
    run(args.robot_ip, args.subject)


if __name__ == "__main__":
    main()
