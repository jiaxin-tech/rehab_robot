# hardware/force_sensor.py
# ATI力传感器 UDP接口，翻译自C++源码

import socket
import struct
import threading
import time
from config import settings


class ForceSensor:
    """
    ATI六轴力/力矩传感器
    通信协议：UDP，端口49152
    数据格式：36字节响应包，包含Fx/Fy/Fz/Tx/Ty/Tz
    """

    # 控制指令码
    CMD_START  = 0x0002
    CMD_STOP   = 0x0000
    CMD_BIAS   = 0x0042
    CMD_FILTER = 0x0081
    CMD_SPEED  = 0x0082

    def __init__(self, ip=settings.SENSOR_IP, port=settings.SENSOR_PORT):
        self.ip   = ip
        self.port = port
        self._sock     = None
        self._lock     = threading.Lock()
        self._latest   = dict(fx=0.0, fy=0.0, fz=0.0,
                              tx=0.0, ty=0.0, tz=0.0, ts=0.0)
        self._running  = False
        self._thread   = None

    # ── 连接 ─────────────────────────────────────────
    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.connect((self.ip, self.port))
        self._sock.settimeout(1.0)
        print(f"[ForceSensor] 已连接 {self.ip}:{self.port}")

    def disconnect(self):
        self._running = False
        if self._sock:
            try:
                self._send_command(self.CMD_STOP, 0)
                self._sock.close()
            except Exception:
                pass
        print("[ForceSensor] 已断开")

    # ── 底层通信 ─────────────────────────────────────
    def _send_command(self, command: int, data: int):
        """发送8字节控制帧"""
        pkt = struct.pack(">HHI", 0x1234, command, data)
        self._sock.send(pkt)
        time.sleep(0.005)

    def _recv_frame(self):
        """接收并解析一帧36字节数据"""
        raw = self._sock.recv(36)
        if len(raw) < 36:
            return None
        _, _, status, fx, fy, fz, tx, ty, tz = struct.unpack(">IIIiiiiii", raw)
        return dict(
            fx = fx / settings.FORCE_DIV,
            fy = fy / settings.FORCE_DIV,
            fz = fz / settings.FORCE_DIV,
            tx = tx / settings.TORQUE_DIV,
            ty = ty / settings.TORQUE_DIV,
            tz = tz / settings.TORQUE_DIV,
            ts = time.time(),
        )

    # ── 流式采集 ─────────────────────────────────────
    def start_streaming(self):
        """配置传感器参数并启动后台接收线程"""
        speed_code = int(1000 / settings.SENSOR_HZ)
        self._send_command(self.CMD_SPEED,  speed_code)
        self._send_command(self.CMD_FILTER, settings.SENSOR_FILTER)
        self._send_command(self.CMD_BIAS,   0x00)
        self._send_command(self.CMD_START,  0xFFFFFFFF)  # 持续流式
        self._running = True
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[ForceSensor] 开始流式采集 @ {settings.SENSOR_HZ}Hz")

    def _recv_loop(self):
        while self._running:
            try:
                frame = self._recv_frame()
                if frame:
                    with self._lock:
                        self._latest = frame
            except socket.timeout:
                pass
            except Exception as e:
                if self._running:
                    print(f"[ForceSensor] 接收错误: {e}")

    def set_bias(self):
        """以当前值为零点（去除重力/安装偏置），开始采集前必须调用"""
        self._send_command(self.CMD_BIAS, 0xFF)
        print("[ForceSensor] 已设置偏置零点")

    # ── 数据读取 ─────────────────────────────────────
    def get(self) -> dict:
        """获取最新一帧（线程安全），返回 {fx,fy,fz,tx,ty,tz,ts}"""
        with self._lock:
            return dict(self._latest)

    def get_force_magnitude(self) -> float:
        """返回合力大小 (N)"""
        d = self.get()
        return (d["fx"]**2 + d["fy"]**2 + d["fz"]**2) ** 0.5

    def get_array(self):
        """返回 [Fx, Fy, Fz, Tx, Ty, Tz] numpy数组，供模型使用"""
        import numpy as np
        d = self.get()
        return np.array([d["fx"], d["fy"], d["fz"],
                         d["tx"], d["ty"], d["tz"]], dtype=float)
