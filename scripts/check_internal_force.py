"""Read Rokae internal force estimates without powering or moving the robot."""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from hardware.windows import RokaeForceSensor, RokaeRobot


def make_robot(robot_ip: str) -> RokaeRobot:
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


def run(robot_ip: str, duration: float, software_bias: bool) -> None:
    robot = make_robot(robot_ip)
    force = RokaeForceSensor(robot)
    samples: list[list[float]] = []
    timestamps: list[float] = []

    try:
        print(f"Connecting to Rokae robot at {robot_ip} (no power, no motion)...")
        robot.connect()
        force.connect()
        force.start_streaming()
        if software_bias:
            print("Keep the robot unloaded and still while software bias is measured...")
            force.set_bias()

        deadline = time.monotonic() + duration
        last_timestamp = None
        next_report = time.monotonic()
        while time.monotonic() < deadline:
            data = force.get()
            if data["ts"] != last_timestamp:
                last_timestamp = data["ts"]
                timestamps.append(data["ts"])
                samples.append(
                    [
                        data["fx"],
                        data["fy"],
                        data["fz"],
                        data["tx"],
                        data["ty"],
                        data["tz"],
                    ]
                )
            if time.monotonic() >= next_report:
                print(
                    "F[N]=({fx:+8.3f}, {fy:+8.3f}, {fz:+8.3f})  "
                    "T[Nm]=({tx:+8.3f}, {ty:+8.3f}, {tz:+8.3f})".format(**data)
                )
                next_report += 1.0
            time.sleep(0.002)
    finally:
        force.disconnect()
        robot.disconnect()

    if not samples:
        raise RuntimeError("No internal force samples were received")

    labels = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    print(f"\nReceived {len(samples)} unique samples in {duration:.1f}s")
    if len(timestamps) > 1:
        span = timestamps[-1] - timestamps[0]
        if span > 0:
            print(f"Measured update rate: {(len(timestamps) - 1) / span:.1f} Hz")
    print("Channel       mean          std          min          max")
    for index, label in enumerate(labels):
        values = [sample[index] for sample in samples]
        print(
            f"{label:>3}  {statistics.fmean(values):+11.4f}  "
            f"{statistics.pstdev(values):10.4f}  "
            f"{min(values):+11.4f}  {max(values):+11.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="珞石内置关节力矩/末端六维力静态诊断（不上电、不运动）"
    )
    parser.add_argument("--robot-ip", default=settings.ROBOT_IP)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--software-bias",
        action="store_true",
        help="读取前用静止样本做软件去零",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    run(args.robot_ip, args.duration, args.software_bias)


if __name__ == "__main__":
    main()
