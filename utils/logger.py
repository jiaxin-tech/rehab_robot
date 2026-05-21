# utils/logger.py
# 统一日志，所有模块用同一个logger

import logging
import os
from datetime import datetime
from config import settings


def get_logger(name: str) -> logging.Logger:
    """
    获取logger，自动同时输出到控制台和文件
    文件名格式：logs/YYYYMMDD_HHMMSS.log
    """
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加handler

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    log_path = os.path.join(
        settings.LOG_DIR,
        datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
