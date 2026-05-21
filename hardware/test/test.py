# test_hardware_get.py

import time
from hardware.dobot_cr5 import DobotCR5
from hardware.force_sensor import ForceSensor

ROBOT_IP  = "192.168.50.104"
SENSOR_IP = "192.168.1.x"   # 改成实际传感器IP

def test_robot_get(robot: DobotCR5, n: int = 5):
    print("\n=== DobotCR5 get 测试 ===")
    for i in range(n):
        pose = robot.get_cartesian_pose()
        jnt  = robot.get_joint_angles()
        djnt = robot.get_actual_joint_speeds()

        print(f"[{i+1}]")
        print(f"  cartesian_pose      : {[round(v,4) for v in pose]}")
        print(f"  joint_angles   (deg): {[round(v,4) for v in jnt]}")
        print(f"  joint_speeds (deg/s): {[round(v,4) for v in djnt]}")

        # 基本断言
        assert len(pose) == 6, "cartesian_pose 应有6个元素"
        assert len(jnt)  == 6, "joint_angles 应有6个元素"
        assert len(djnt) == 6, "joint_speeds 应有6个元素"

        time.sleep(0.2)
    print(">>> DobotCR5 get 测试通过\n")


def test_force_get(sensor: ForceSensor, n: int = 5):
    print("\n=== ForceSensor get 测试 ===")
    for i in range(n):
        d   = sensor.get()
        mag = sensor.get_force_magnitude()
        arr = sensor.get_array()

        print(f"[{i+1}]")
        print(f"  get()               : fx={d['fx']:.3f} fy={d['fy']:.3f} fz={d['fz']:.3f} "
              f"tx={d['tx']:.3f} ty={d['ty']:.3f} tz={d['tz']:.3f}")
        print(f"  get_force_magnitude : {mag:.4f} N")
        print(f"  get_array           : {arr.round(4)}")

        # 基本断言
        assert set(d.keys()) >= {"fx","fy","fz","tx","ty","tz","ts"}, "get() 字段缺失"
        assert arr.shape == (6,), "get_array 应返回长度6的数组"
        assert mag >= 0,          "合力大小不能为负"

        time.sleep(0.2)
    print(">>> ForceSensor get 测试通过\n")


if __name__ == "__main__":
    # ── 机器人 ──
    robot = DobotCR5(ip_address=ROBOT_IP)
    robot.connect()
    time.sleep(2)   # 等待反馈线程稳定

    try:
        test_robot_get(robot)
    finally:
        robot.disconnect()

    # ── 力传感器 ──
    sensor = ForceSensor(ip=SENSOR_IP)
    sensor.connect()
    sensor.start_streaming()
    time.sleep(1)   # 等待第一帧到达

    try:
        test_force_get(sensor)
    finally:
        sensor.disconnect()