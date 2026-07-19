import unittest
import sys

from hardware.windows.rokae_xcore import RokaeRobot, _load_sdk


class _FakeRealtimeController:
    def __init__(self):
        self.callback = None
        self.calls = []

    def setFilterFrequency(self, joint_hz, cartesian_hz, torque_hz, ec):
        self.calls.append(
            ("setFilterFrequency", joint_hz, cartesian_hz, torque_hz)
        )

    def setControlLoopCar(self, callback, priority, use_state):
        self.callback = callback
        self.calls.append(("setControlLoopCar", priority, use_state))

    def startMove(self, mode):
        self.calls.append(("startMove", mode))

    def startLoop(self, blocking):
        self.calls.append(("startLoop", blocking))

    def hasMotionError(self):
        return False

    def stopLoop(self):
        self.calls.append(("stopLoop",))

    def stopMove(self):
        self.calls.append(("stopMove",))


class _FakeNativeRobot:
    def __init__(self, sdk):
        self.sdk = sdk
        self.rt = _FakeRealtimeController()
        self.calls = []

    def setOperateMode(self, mode, ec):
        self.calls.append(("setOperateMode", mode))

    def setRtNetworkTolerance(self, percent, ec):
        self.calls.append(("setRtNetworkTolerance", percent))

    def setMotionControlMode(self, mode, ec):
        self.calls.append(("setMotionControlMode", mode))

    def setPowerState(self, powered, ec):
        self.calls.append(("setPowerState", powered))

    def getRtMotionController(self):
        self.calls.append(("getRtMotionController",))
        return self.rt

    def operationState(self, ec):
        return self.sdk.OperationState.idle


class RokaeXCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "win32":
            raise unittest.SkipTest("bundled xCoreSDK tests require Windows")
        cls.sdk = _load_sdk()

    def make_robot(self, local_ip="192.168.50.10"):
        robot = RokaeRobot("192.168.50.102", local_ip=local_ip)
        native = _FakeNativeRobot(self.sdk)
        robot._sdk = self.sdk
        robot._robot = native
        robot.is_connected = True
        robot.cartesian_pose = [0.3, -0.2, 0.35, 0.0, 1.5707963267948966, 0.0]
        robot._state_ready.set()
        return robot, native

    def test_bundled_sdk_is_0_7_0(self):
        self.assertEqual(self.sdk.BaseRobot.sdkVersion(), "0.7.0")
        self.assertTrue(hasattr(self.sdk.xMateRobot, "getRtMotionController"))

    def test_pose_unit_round_trip(self):
        pose = [0.1234, -0.0567, 0.8901, 0.1, -0.2, 0.3]
        restored = RokaeRobot._pose_from_sdk(RokaeRobot._pose_to_sdk(pose))
        for actual, expected in zip(restored, pose):
            self.assertAlmostEqual(actual, expected)

    def test_realtime_callback_holds_latest_target(self):
        robot, native = self.make_robot()
        robot.enable_realtime()
        robot.start_realtime_cartesian()

        first = native.rt.callback()
        self.assertEqual(first.trans, [0.3, -0.2, 0.35])
        self.assertAlmostEqual(first.rpy[1], 1.5707963267948966)

        robot.set_realtime_cartesian_target(
            [0.310, -0.190, 0.340, 0.1, 0.2, 0.3]
        )
        updated = native.rt.callback()
        self.assertEqual(updated.trans, [0.31, -0.19, 0.34])
        self.assertEqual(updated.rpy, [0.1, 0.2, 0.3])
        self.assertIn(("startLoop", False), native.rt.calls)

        robot.stop_realtime(switch_to_nrt=False)
        self.assertIn(("stopLoop",), native.rt.calls)
        self.assertIn(("stopMove",), native.rt.calls)

    def test_realtime_requires_local_ip(self):
        robot, _ = self.make_robot(local_ip="")
        with self.assertRaisesRegex(ValueError, "ROBOT_LOCAL_IP"):
            robot.enable_realtime()


if __name__ == "__main__":
    unittest.main()
