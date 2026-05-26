# scripts/run_control.py
# 入口：运行完整控制系统（PINN在线辨识 + MPC实时控制）

import argparse
import time
import sys, os
import queue
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import settings
from hardware.dobot_cr5 import DobotCR5
from hardware.force_sensor import ForceSensor
from collection.safety_guard import SafetyGuard
from collection.trajectory import (
    generate_real_trajectory_from_csv,
    generate_rehab_trajectory,
)
from models.comfort_net import ComfortPredictor
from models.pinn import OnlinePINN, run_pinn
from control.mpc_controller import MPCController
from utils.logger import get_logger

logger = get_logger("RunControl")

# PINN在线更新的滑动窗口大小
PINN_WINDOW = 200   # 200帧 @ 50Hz = 4秒数据


class AsyncPINNUpdater:
    """后台执行PINN训练，控制循环只轮询结果，不等待训练完成。"""

    def __init__(self, epochs: int = settings.PINN_ONLINE_EPOCHS):
        self.epochs = epochs
        self._busy = False
        self._lock = threading.Lock()
        self._results = queue.Queue()

    def start(self, t_buf, xyz_buf, F_buf) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        t = np.array(t_buf, dtype=np.float32)
        xyz = np.array(xyz_buf, dtype=np.float32)
        force = np.array(F_buf, dtype=np.float32)
        worker = threading.Thread(
            target=self._run,
            args=(t, xyz, force),
            daemon=True,
        )
        worker.start()
        return True

    def _run(self, t, xyz, force):
        try:
            _, params = run_pinn(
                t_data=t,
                xyz_data=xyz,
                F_data=force,
                epochs=self.epochs,
                verbose=False,
            )
            self._results.put(("ok", params))
        except Exception as e:
            self._results.put(("error", e))
        finally:
            with self._lock:
                self._busy = False

    def poll(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy


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


def _wait_idle(robot, timeout: float = 30.0):
    if hasattr(robot, "wait_idle"):
        return robot.wait_idle()

    time.sleep(0.1)
    deadline = time.time() + timeout
    running_modes = {"RUNNING", "JOG", "RECORDING"}
    idle_seen_at = None
    while time.time() < deadline:
        if getattr(robot, "robot_mode", "") not in running_modes:
            if idle_seen_at is None:
                idle_seen_at = time.time()
            elif time.time() - idle_seen_at >= 0.2:
                return True
        else:
            idle_seen_at = None
        time.sleep(0.05)
    logger.warning(f"等待机器人空闲超时 {timeout:.1f}s，继续执行")
    return False


def _as_valid_pose(values, name: str = "cartesian_pose") -> np.ndarray:
    pose = np.asarray(values, dtype=float).reshape(-1)
    if pose.size < 6 or not np.all(np.isfinite(pose[:6])):
        raise RuntimeError(f"{name}反馈无效: {values}")
    return pose[:6].copy()


def _limit_position_step(current_xyz: np.ndarray, target_xyz: np.ndarray, max_step_mm: float) -> np.ndarray:
    delta = np.asarray(target_xyz, dtype=float) - np.asarray(current_xyz, dtype=float)
    dist = float(np.linalg.norm(delta))
    if dist <= max_step_mm or dist <= 1e-9:
        return np.asarray(target_xyz, dtype=float).copy()
    return np.asarray(current_xyz, dtype=float) + delta * (max_step_mm / dist)


def _comfort_progress_scale(comfort_score: float) -> float:
    min_scale = float(np.clip(settings.CONTROL_MIN_PROGRESS_SCALE, 0.0, 1.0))
    comfort_score = float(np.clip(comfort_score, 0.0, 1.0))
    return min_scale + (1.0 - min_scale) * comfort_score


def _comfort_motion_scale(comfort_score: float) -> float:
    comfort_score = float(np.clip(comfort_score, 0.0, 1.0))
    power = max(float(getattr(settings, "CONTROL_COMFORT_SPEED_POWER", 1.0)), 1.0)
    return float(comfort_score ** power)


def _comfort_target_step_mm(comfort_score: float) -> float:
    max_step = max(float(settings.CONTROL_MAX_TARGET_STEP_MM), 0.0)
    min_step = float(np.clip(getattr(settings, "CONTROL_MIN_TARGET_STEP_MM", 0.0), 0.0, max_step))
    return min_step + (max_step - min_step) * _comfort_motion_scale(comfort_score)


def _control_comfort_score(raw_comfort: float, force_norm: float = 0.0) -> float:
    raw_comfort = float(np.clip(raw_comfort, 0.0, 1.0))
    power = max(float(getattr(settings, "CONTROL_COMFORT_RECOVERY_POWER", 1.0)), 1.0)
    recovered = 1.0 - (1.0 - raw_comfort) ** power

    force_floor_n = max(float(getattr(settings, "CONTROL_COMFORT_FORCE_FLOOR_N", 0.0)), 0.0)
    low_force_floor = float(np.clip(getattr(settings, "CONTROL_COMFORT_LOW_FORCE_FLOOR", 0.0), 0.0, 1.0))
    if force_floor_n > 1e-6 and low_force_floor > 0.0:
        force_norm = max(float(force_norm), 0.0)
        force_scale = np.clip(1.0 - force_norm / force_floor_n, 0.0, 1.0)
        recovered = max(recovered, low_force_floor * force_scale)

    return float(np.clip(recovered, 0.0, 1.0))


def _load_reference_waypoints(tool_orientation, trajectory_source: str = None,
                              anchor_xyz=None,
                              real_trajectory_csv: str = None) -> np.ndarray:
    source = trajectory_source or getattr(settings, "CONTROL_TRAJECTORY_SOURCE", "math")
    source = str(source).strip().lower()

    if source in ("real", "real_csv", "csv"):
        ref_wps = generate_real_trajectory_from_csv(
            csv_path=real_trajectory_csv,
            dt=settings.MPC_DT,
            orientation=tool_orientation,
            anchor_xyz=anchor_xyz,
        )
        csv_path = real_trajectory_csv or getattr(settings, "REAL_TRAJECTORY_CSV_PATH", "")
        mode = getattr(settings, "REAL_TRAJECTORY_MODE", "point")
        point = getattr(settings, "REAL_TRAJECTORY_POINT", "")
        joints = getattr(settings, "REAL_TRAJECTORY_JOINTS", ())
        logger.info(
            "control参考轨迹来源: 真实CSV "
            f"path={csv_path}, mode={mode}, point={point}, joints={joints}, points={len(ref_wps)}"
        )
        return ref_wps

    if source in ("math", "generated", "rehab"):
        ref_wps = generate_rehab_trajectory(
            dt=settings.MPC_DT,
            orientation=tool_orientation,
        )
        logger.info(f"control参考轨迹来源: 数学康复轨迹 points={len(ref_wps)}")
        return ref_wps

    raise ValueError(f"未知control参考轨迹来源: {trajectory_source}")


def run(
    robot_ip: str,
    sensor_ip: str,
    subject_id: str,
    online_pinn_enabled: bool = None,
    trajectory_repeats: int = None,
    trajectory_source: str = None,
    real_trajectory_csv: str = None,
):

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
    pinn_updater = AsyncPINNUpdater()
    mpc         = MPCController(dim=3)
    if online_pinn_enabled is None:
        online_pinn_enabled = settings.PINN_ONLINE_ENABLED
    logger.info(f"在线PINN后台更新: {'开启' if online_pinn_enabled else '关闭'}")

    cur_pose0 = _as_valid_pose(robot.get_cartesian_pose())

    # ── 加载参考轨迹 ──────────────────────────────────
    ref_wps_base = _load_reference_waypoints(
        tool_orientation=tool_orientation,
        trajectory_source=trajectory_source,
        anchor_xyz=cur_pose0[:3],
        real_trajectory_csv=real_trajectory_csv,
    )   # (N_total, 6)
    if trajectory_repeats is None:
        trajectory_repeats = getattr(settings, "CONTROL_TRAJECTORY_REPEATS", 1)
    trajectory_repeats = max(1, int(trajectory_repeats))
    logger.info(f"control参考轨迹重复次数: {trajectory_repeats}")

    start_error = float(np.linalg.norm(cur_pose0[:3] - ref_wps_base[0, :3]))
    if start_error > settings.CONTROL_START_TOLERANCE_MM:
        logger.info(
            f"当前位置距控制轨迹起点 {start_error:.1f}mm，先移动到起点 "
            f"({ref_wps_base[0,0]:.1f},{ref_wps_base[0,1]:.1f},{ref_wps_base[0,2]:.1f})"
        )
        guard.check()
        _move_l(robot, ref_wps_base[0])
        _wait_idle(robot)
        time.sleep(0.3)

    # 参考轨迹转为状态序列 [x,y,z, vx,vy,vz]
    ref_wps = np.vstack([ref_wps_base] * trajectory_repeats)
    ref_pos = ref_wps[:, :3]
    ref_vel = np.gradient(ref_pos, settings.MPC_DT, axis=0)
    ref_states = np.hstack([ref_pos, ref_vel])   # (N_total, 6)
    cmd_pose = ref_wps[0].copy()
    cmd_vel = np.zeros(3, dtype=float)

    # ── 滑动窗口缓存（供PINN在线更新用）─────────────
    # xyz_buf: list of [x,y,z]   F_buf: list of [Fx,Fy,Fz]
    t_buf, xyz_buf, F_buf = [], [], []
    t0 = time.time()

    # ── 主控制循环 ─────────────────────────────────────
    logger.info("=== 控制循环启动 ===")
    step        = 0
    u_warm      = None   # MPC热启动：上一次的控制序列
    last_acc    = np.zeros(3, dtype=float)   # 上一拍已执行加速度，用于jerk惩罚
    prev_pose   = None
    prev_time   = None
    filtered_vel = None
    timeout_count = 0
    timeout_max_ms = 0.0
    last_timeout_log = time.time()

    try:
        ref_cursor = 0.0
        while ref_cursor < len(ref_states) - settings.MPC_HORIZON - 1:
            loop_start = time.time()
            guard.check()

            # 1. 读取当前状态
            now = time.time()
            f_data = force.get()
            cur_pose = _as_valid_pose(robot.get_cartesian_pose())
            if prev_pose is None:
                cur_vel_meas = np.zeros(3, dtype=float)
            else:
                pose_jump = float(np.linalg.norm(cur_pose[:3] - prev_pose[:3]))
                if pose_jump > settings.CONTROL_MAX_FEEDBACK_JUMP_MM:
                    raise RuntimeError(
                        f"机器人反馈位姿跳变异常: {pose_jump:.1f}mm > "
                        f"{settings.CONTROL_MAX_FEEDBACK_JUMP_MM:.1f}mm"
                    )
                dt_pose = max(now - prev_time, 1e-6)
                cur_vel_meas = (cur_pose[:3] - prev_pose[:3]) / dt_pose

            vel_alpha = float(np.clip(getattr(settings, "CONTROL_VEL_FILTER_ALPHA", 1.0), 0.0, 1.0))
            if filtered_vel is None:
                filtered_vel = cur_vel_meas.copy()
            else:
                filtered_vel = vel_alpha * cur_vel_meas + (1.0 - vel_alpha) * filtered_vel
            cur_vel = filtered_vel.copy()

            F_vec = np.array([f_data["fx"], f_data["fy"], f_data["fz"]])

            # 2. 更新PINN滑动窗口（三维）
            t_now = time.time() - t0
            t_buf.append(t_now)
            xyz_buf.append(cur_pose[:3].tolist())                    # [x, y, z] mm
            F_buf.append([f_data["fx"], f_data["fy"], f_data["fz"]])  # [Fx,Fy,Fz]

            if len(t_buf) > PINN_WINDOW:
                t_buf.pop(0); xyz_buf.pop(0); F_buf.pop(0)

            result = pinn_updater.poll()
            if result is not None:
                status, payload = result
                if status == "ok":
                    online_pinn.set_params(payload)
                    p = online_pinn.get_params()
                    logger.info(
                        f"[Step {step}] PINN后台更新完成 | "
                        f"M=({p['Mx']:.6f},{p['My']:.6f},{p['Mz']:.6f}) "
                        f"B=({p['Bx']:.6f},{p['By']:.6f},{p['Bz']:.6f}) "
                        f"K=({p['Kx']:.6f},{p['Ky']:.6f},{p['Kz']:.6f})"
                    )
                else:
                    logger.warning(f"PINN后台更新失败: {payload}")

            # 异步启动PINN更新；控制循环不等待训练完成。
            if (
                online_pinn_enabled
                and step % settings.PINN_ONLINE_UPDATE_EVERY == 0
                and len(t_buf) >= settings.PINN_ONLINE_MIN_SAMPLES
            ):
                if pinn_updater.start(t_buf, xyz_buf, F_buf):
                    logger.info(
                        f"[Step {step}] PINN后台更新已启动 "
                        f"({len(t_buf)}帧, epochs={settings.PINN_ONLINE_EPOCHS})"
                    )

            # 3. 实时舒适度评分
            pinn_params = online_pinn.get_params()
            comfort_raw = comfort_pred.predict(
                fx=f_data["fx"], fy=f_data["fy"], fz=f_data["fz"],
                x=cur_pose[0], y=cur_pose[1], z=cur_pose[2],
                vx=cur_vel[0], vy=cur_vel[1], vz=cur_vel[2],
                Mx=pinn_params["Mx"], My=pinn_params["My"], Mz=pinn_params["Mz"],
                Bx=pinn_params["Bx"], By=pinn_params["By"], Bz=pinn_params["Bz"],
                Kx=pinn_params["Kx"], Ky=pinn_params["Ky"], Kz=pinn_params["Kz"],
            )
            comfort_score = _control_comfort_score(comfort_raw, np.linalg.norm(F_vec))
            motion_scale = _comfort_motion_scale(comfort_score)
            x0_vel = cur_vel[:3] * motion_scale
            x0_cmd = np.concatenate([cur_pose[:3], x0_vel])   # [x,y,z,vx,vy,vz]
            if comfort_score < getattr(settings, "CONTROL_LOW_COMFORT_RESET_THRESHOLD", 0.0):
                u_warm = None

            # 4. MPC求解：舒适度会改变MPC权重，MPC输出再积分成实际下发目标。
            ref_base_idx = int(min(ref_cursor, len(ref_states) - settings.MPC_HORIZON - 1))
            ref_horizon = ref_states[ref_base_idx: ref_base_idx + settings.MPC_HORIZON + 1].copy()
            ref_horizon[:, 3:] *= motion_scale
            u_opt, u_warm = mpc.solve(
                x0_cmd, ref_horizon, comfort_score, u_warm, last_acc * motion_scale)
            next_pose, _ = mpc.acceleration_to_pose(
                cur_pose, x0_vel, u_opt, axes=(0, 1, 2))
            ref_target_idx = min(ref_base_idx + 1, len(ref_wps) - 1)
            ref_target_xyz = ref_wps[ref_target_idx, :3]
            next_pose[:3] = _limit_position_step(
                ref_target_xyz, next_pose[:3], settings.CONTROL_MAX_MPC_REF_DEVIATION_MM
            )
            target_step_mm = _comfort_target_step_mm(comfort_score)
            next_pose[:3] = _limit_position_step(
                cur_pose[:3], next_pose[:3], target_step_mm
            )
            next_pose[3:6] = tool_orientation
            cmd_vel = (next_pose[:3] - cur_pose[:3]) / settings.MPC_DT
            next_cmd_speed = float(np.linalg.norm(cmd_vel))
            cmd_pose = next_pose.copy()
            last_acc = u_opt
            track_error = float(np.linalg.norm(cur_pose[:3] - next_pose[:3]))
            if track_error > settings.CONTROL_MAX_TRACK_ERROR_MM:
                raise RuntimeError(
                    f"轨迹跟踪误差过大: {track_error:.1f}mm > "
                    f"{settings.CONTROL_MAX_TRACK_ERROR_MM:.1f}mm"
                )

            # 5. 发送运动指令
            _servo_p(robot, next_pose)

            prev_pose = cur_pose
            prev_time = now

            # 6. 日志（每50步）
            if step % 50 == 0:
                repeat_idx = ref_base_idx // len(ref_wps_base) + 1
                repeat_ref_idx = ref_base_idx % len(ref_wps_base)
                logger.info(
                    f"[Step {step:4d}] "
                    f"pos=({cur_pose[0]:.1f},{cur_pose[1]:.1f},{cur_pose[2]:.1f})mm  "
                    f"mpc_target=({next_pose[0]:.1f},{next_pose[1]:.1f},{next_pose[2]:.1f})mm  "
                    f"err={track_error:.1f}mm  "
                    f"repeat={repeat_idx}/{trajectory_repeats}  "
                    f"ref_idx={repeat_ref_idx}  "
                    f"v_cmd={next_cmd_speed:.1f}mm/s  "
                    f"|u|={np.linalg.norm(u_opt):.1f}mm/s²  "
                    f"F={np.linalg.norm(F_vec):.1f}N  "
                    f"comfort_raw={comfort_raw:.3f}  "
                    f"comfort_ctrl={comfort_score:.3f}  "
                    f"motion={motion_scale:.3f}  "
                    f"step_lim={target_step_mm:.1f}mm  "
                    f"pinn_ready={online_pinn.is_ready} "
                    f"pinn_busy={pinn_updater.busy}"
                )

            step += 1
            ref_cursor += _comfort_progress_scale(comfort_score)

            # 7. 维持控制频率
            elapsed = time.time() - loop_start
            sleep_t = settings.MPC_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif elapsed > settings.MPC_DT * 1.5:
                timeout_count += 1
                timeout_max_ms = max(timeout_max_ms, elapsed * 1000.0)
                now_log = time.time()
                if now_log - last_timeout_log >= 1.0:
                    logger.warning(
                        f"控制循环超时 {timeout_count} 次/秒，"
                        f"最大={timeout_max_ms:.1f}ms > {settings.MPC_DT*1000:.0f}ms"
                    )
                    timeout_count = 0
                    timeout_max_ms = 0.0
                    last_timeout_log = now_log

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
    parser.add_argument("--online-pinn", dest="online_pinn", action="store_true", default=None,
                        help="强制开启后台PINN更新；未指定则使用 config/settings.py 中的 PINN_ONLINE_ENABLED")
    parser.add_argument("--no-online-pinn", dest="online_pinn", action="store_false",
                        help="强制关闭后台PINN更新")
    parser.add_argument("--repeats", type=int, default=None,
                        help="control参考轨迹重复次数；未指定则使用 config/settings.py 中的 CONTROL_TRAJECTORY_REPEATS")
    parser.add_argument("--trajectory-source", choices=("real_csv", "math"), default=None,
                        help="control参考轨迹来源；默认使用 config/settings.py 中的 CONTROL_TRAJECTORY_SOURCE")
    parser.add_argument("--real-trajectory-csv", default=None,
                        help="真实轨迹CSV路径；未指定则使用 config/settings.py 中的 REAL_TRAJECTORY_CSV_PATH")
    args = parser.parse_args()
    run(
        args.robot_ip,
        args.sensor_ip,
        args.subject,
        online_pinn_enabled=args.online_pinn,
        trajectory_repeats=args.repeats,
        trajectory_source=args.trajectory_source,
        real_trajectory_csv=args.real_trajectory_csv,
    )


if __name__ == "__main__":
    main()
