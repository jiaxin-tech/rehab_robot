# hardware/dobot_cr5.py
# Dobot CR5 TCP接口，字节偏移与ZJFDobotCR5.m完全对应

import socket
import struct
import threading
import time
import numpy as np
from config import settings


class DobotCR5:
    """
    Dobot CR5 TCP控制接口
    端口：Dashboard(29999) / Move(30003) / Feedback(30004)
    反馈包：1440字节，固定格式，字节偏移与MATLAB文件一致
    """

    FEED_PACKET_SIZE = 1440

    def __init__(self, ip=settings.ROBOT_IP):
        self.ip = ip
        self._dash = None
        self._move = None
        self._feed = None
        self._lock = threading.Lock()

        # 状态缓存（对应MATLAB属性）
        self.joint_angles   = np.zeros(6)   # deg, J1~J6
        self.cartesian_pose = np.zeros(6)   # mm/deg, X Y Z Rx Ry Rz
        self.joint_speeds   = np.zeros(6)   # deg/s
        self.tcp_speed      = np.zeros(6)   # mm/s
        self.robot_mode     = 0             # 参考settings里的模式码

        self._feed_running = False
        self._feed_thread  = None

    # ── 连接管理 ─────────────────────────────────────
    def connect(self):
        self._dash = self._make_tcp(settings.ROBOT_DASH_PORT, timeout=5.0)
        self._move = self._make_tcp(settings.ROBOT_MOVE_PORT, timeout=5.0)
        self._feed = self._make_tcp(settings.ROBOT_FEED_PORT, timeout=2.0)
        self._feed_running = True
        self._feed_thread  = threading.Thread(target=self._feed_loop, daemon=True)
        self._feed_thread.start()
        print(f"[DobotCR5] 已连接 {self.ip}")

    def disconnect(self):
        self._feed_running = False
        for s in [self._dash, self._move, self._feed]:
            try:
                s.close()
            except Exception:
                pass
        print("[DobotCR5] 已断开")

    def _make_tcp(self, port, timeout=5.0):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((self.ip, port))
        return s

    # ── 反馈解析（对应MATLAB FeedBackTcpCallback）────
    @staticmethod
    def _b2d(data: bytes, matlab_idx: int) -> float:
        """MATLAB TransBytes2Double(n) → Python（0-based偏移）"""
        i = matlab_idx - 1
        return struct.unpack_from("<d", data, i)[0]

    def _parse_feedback(self, data: bytes):
        b2d = self._b2d
        with self._lock:
            self.robot_mode = data[24]   # MATLAB index 25
            self.joint_angles = np.array([
                b2d(data, 433), b2d(data, 441), b2d(data, 449),
                b2d(data, 457), b2d(data, 465), b2d(data, 473),
            ])
            self.cartesian_pose = np.array([
                b2d(data, 625), b2d(data, 633), b2d(data, 641),
                b2d(data, 649), b2d(data, 657), b2d(data, 665),
            ])
            self.joint_speeds = np.array([
                b2d(data, 481), b2d(data, 489), b2d(data, 497),
                b2d(data, 505), b2d(data, 513), b2d(data, 521),
            ])
            self.tcp_speed = np.array([
                b2d(data, 673), b2d(data, 681), b2d(data, 689),
                b2d(data, 697), b2d(data, 705), b2d(data, 713),
            ])

    def _feed_loop(self):
        buf = b""
        while self._feed_running:
            try:
                chunk = self._feed.recv(4096)
                if chunk:
                    buf += chunk
                    while len(buf) >= self.FEED_PACKET_SIZE:
                        self._parse_feedback(buf[:self.FEED_PACKET_SIZE])
                        buf = buf[self.FEED_PACKET_SIZE:]
            except socket.timeout:
                pass
            except Exception as e:
                if self._feed_running:
                    print(f"[DobotCR5] 反馈错误: {e}")

    # ── 状态读取 ─────────────────────────────────────
    def get_state(self) -> dict:
        """返回当前状态快照（线程安全）"""
        with self._lock:
            return dict(
                joint_angles   = self.joint_angles.copy(),
                cartesian_pose = self.cartesian_pose.copy(),
                joint_speeds   = self.joint_speeds.copy(),
                tcp_speed      = self.tcp_speed.copy(),
                robot_mode     = self.robot_mode,
                ts             = time.time(),
            )

    def is_idle(self) -> bool:
        """机械臂是否处于空闲状态（mode=5）"""
        with self._lock:
            return self.robot_mode == 5

    def wait_idle(self, timeout=15.0) -> bool:
        """阻塞等待机械臂空闲"""
        t0 = time.time()
        time.sleep(0.3)
        while time.time() - t0 < timeout:
            if self.is_idle():
                return True
            time.sleep(0.05)
        print("[DobotCR5] wait_idle 超时")
        return False

    # ── Dashboard指令 ────────────────────────────────
    def _send_dash(self, cmd: str) -> str:
        self._dash.sendall((cmd + "\n").encode())
        time.sleep(0.02)
        try:
            resp = self._dash.recv(1024).decode().strip()
        except Exception:
            resp = ""
        return resp

    def enable(self, load=0.0):
        return self._send_dash(f"EnableRobot({load})")

    def disable(self):
        return self._send_dash("DisableRobot()")

    def clear_error(self):
        return self._send_dash("ClearError()")

    def set_speed(self, ratio: int):
        """设置速度比例 1~100"""
        return self._send_dash(f"SpeedFactor({int(ratio)})")

    # ── 运动指令 ─────────────────────────────────────
    def _send_move(self, cmd: str):
        self._move.sendall((cmd + "\n").encode())
        time.sleep(0.002)

    def stop(self):
        """立即停止所有运动"""
        self._send_move("StopScript()")
        print("[DobotCR5] !! 运动已停止 !!")

    def move_j(self, pose: np.ndarray):
        """关节插值运动到笛卡尔坐标 [x,y,z,rx,ry,rz]"""
        self._send_move(
            f"MovJ({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f},"
            f"{pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f})"
        )

    def move_l(self, pose: np.ndarray):
        """直线运动到笛卡尔坐标"""
        self._send_move(
            f"MovL({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f},"
            f"{pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f})"
        )

    def servo_j(self, joints: np.ndarray, t=0.02,
                lookahead_time=0.1, gain=300):
        """
        实时关节伺服控制（高频发点用）
        joints: [J1~J6] deg
        t: 每步时间(s)，建议等于控制周期
        """
        self._send_move(
            f"ServoJ({joints[0]:.4f},{joints[1]:.4f},{joints[2]:.4f},"
            f"{joints[3]:.4f},{joints[4]:.4f},{joints[5]:.4f},"
            f"t={t},lookahead_time={lookahead_time},gain={gain})"
        )
