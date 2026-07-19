# utils/signal_processing.py
# 滤波、数值微分等信号处理工具

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


def smooth_differentiate(values: np.ndarray, times: np.ndarray,
                          window: int = 11, poly: int = 3) -> np.ndarray:
    """
    对时间序列求导（S-G滤波 + 数值微分）
    先平滑再微分，避免噪声放大

    Args:
        values: 待微分的序列，如速度
        times:  对应时间戳
        window: S-G滤波窗口长度（必须为奇数）
        poly:   S-G多项式阶数
    Returns:
        导数序列，与输入等长
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(times)

    if len(values) != n:
        raise ValueError("values 和 times 的长度必须一致")
    if n < 2:
        raise ValueError("至少需要两个真实时间戳才能计算导数")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)):
        raise ValueError("values 和 times 必须为有限数")

    # 这是离线工具，不能为了让 gradient 运行而重建真实采样时间。调用方
    # 应先按 CSV 的 valid/invalid_reason 分段或剔除异常帧。
    if np.any(np.diff(times) <= 0):
        raise ValueError("times 必须严格递增；不会重建采集时间轴")

    # 窗口长度必须为奇数、不能超过样本数，且 polyorder < window_length。
    wl = min(window, n if n % 2 == 1 else n - 1)
    if wl < 3:
        return np.gradient(values, times, axis=0)
    poly = min(poly, wl - 1)

    smoothed = savgol_filter(values, window_length=wl, polyorder=poly, axis=0)
    return np.gradient(smoothed, times, axis=0)


def lowpass_filter(values: np.ndarray, cutoff_hz: float,
                   sample_hz: float, order: int = 4) -> np.ndarray:
    """
    Butterworth低通滤波
    用于离线数据的噪声过滤；实时采集保留原始值和真实时间戳。
    """
    nyq  = 0.5 * sample_hz
    norm = cutoff_hz / nyq
    b, a = butter(order, norm, btype="low")
    return filtfilt(b, a, values)


def normalize(arr: np.ndarray, mean=None, std=None):
    """
    Z-score标准化
    如果提供mean/std则用给定值（用于推理时和训练集对齐）
    返回 (normalized, mean, std)
    """
    if mean is None:
        mean = arr.mean(axis=0)
    if std is None:
        std  = arr.std(axis=0) + 1e-8
    return (arr - mean) / std, mean, std


def compute_acceleration(cartesian_poses: np.ndarray,
                         times: np.ndarray) -> np.ndarray:
    """
    从末端笛卡尔位姿序列计算加速度
    cartesian_poses: (N, 3/6) 数组，前三列为 base 系 [x,y,z]，单位m
    times:           (N,)   时间戳
    返回:            (N, 3) base 系线加速度 [ax, ay, az]，单位m/s²
    """
    accel = np.zeros((len(times), 3))
    for i in range(3):   # 只对xyz做，姿态角不需要
        vel   = smooth_differentiate(cartesian_poses[:, i], times)
        accel[:, i] = smooth_differentiate(vel, times)
    return accel
