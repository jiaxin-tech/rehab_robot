# collection/safety_guard.py
# 独立线程持续监测合力，超限立即停机

import threading
import time
from config import settings
from utils.logger import get_logger

logger = get_logger("SafetyGuard")


class SafetyGuard:
    """
    独立200Hz监测线程
    合力超过阈值 → 立刻调robot.stop() → 设置triggered标志
    所有运动循环在每次迭代检查 guard.triggered
    """

    def __init__(self, force_sensor, robot,
                 threshold: float = settings.MAX_FORCE_N):
        self.fs        = force_sensor
        self.robot     = robot
        self.threshold = threshold
        self.triggered = False
        self._active   = False
        self._thread   = None

    def start(self):
        self.triggered = False
        self._active   = True
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"安全守卫已启动，力阈值={self.threshold}N")

    def _loop(self):
        dt = 1.0 / settings.SAFETY_CHECK_HZ
        while self._active:
            try:
                f = self.fs.get_force_magnitude()
                if f > self.threshold:
                    logger.warning(f"安全停止！合力={f:.1f}N > {self.threshold}N")
                    self._stop_robot()
                    self.triggered = True
                    self._active   = False
                    break
            except Exception as e:
                logger.error(f"安全守卫异常: {e}")
            time.sleep(dt)

    def stop(self):
        self._active = False
        logger.info("安全守卫已停止")

    def check(self):
        """在运动循环内调用，触发后抛异常"""
        if self.triggered:
            raise RuntimeError("安全守卫已触发，运动已停止")

    def _stop_robot(self):
        """兼容不同机器人封装里的停止接口。"""
        if hasattr(self.robot, "stop"):
            return self.robot.stop()
        if hasattr(self.robot, "stop_move"):
            return self.robot.stop_move()
        raise AttributeError("robot 缺少 stop/stop_move 停止接口")
