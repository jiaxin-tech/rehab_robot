"""Read-only child-owned SDK session; imported only inside a live child.

Reuse the Windows wrench conversion, but do not start its RT stream or invoke
the motion-capable wrapper's lifecycle. No power/mode/fault/motion API here.
"""
import os
import re

from hardware.rokae_adapter import RokaeRobotAdapter


class WrenchSDKError(RuntimeError):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class WrenchSession:
    def __init__(self, config):
        self.config = config
        self.owner_pid = os.getpid()
        self.robot = None
        self.wrapper = None
        self.adapter = None

    def _call(self, method):
        ec = {}
        result = method(ec)
        if int(ec.get("ec", 0)):
            raise WrenchSDKError(str(ec.get("message")), int(ec["ec"]))
        return result

    def connect(self):
        from hardware.windows.rokae_xcore import RokaeRobot, _load_sdk
        sdk = _load_sdk()
        self.robot = getattr(sdk, self.config["robot_class"])()
        self.robot.connectToRobot(self.config["robot_ip"], self.config["local_ip"])
        info = self._call(self.robot.robotInfo)
        identity = dict(robot_model=str(info.type), robot_serial=str(info.id),
                        controller_version=str(info.version), sdk_version=str(sdk.BaseRobot.sdkVersion()))
        if identity != self.config["expected_identity"]:
            raise RuntimeError("wrench_session_identity_mismatch")
        if self._call(self.robot.operationState).name.lower() != "idle":
            raise RuntimeError("stationary_wrench_session_requires_idle")
        if self._call(self.robot.powerState).name.lower() not in self.config["allowed_power_states"]:
            raise RuntimeError("wrench_session_power_state_mismatch")
        self.wrapper = RokaeRobot(self.config["robot_ip"], local_ip=self.config["local_ip"],
                                  robot_class=self.config["robot_class"])
        self.wrapper._sdk = sdk
        self.wrapper._robot = self.robot
        self.wrapper._force_control = self.robot.forceControl()
        self.wrapper.is_connected = True
        self.adapter = RokaeRobotAdapter(native_robot=self.wrapper)
        return identity

    def query(self):
        if os.getpid() != self.owner_pid:
            raise RuntimeError("SDK owner PID mismatch")
        try:
            return self.adapter.read_internal_wrench()
        except Exception as exc:
            match = re.search(r"\((\d+)\)", str(exc))
            raise WrenchSDKError(str(exc), int(match[1]) if match else None) from exc

    def disconnect(self):
        if self.robot is not None:
            self._call(self.robot.disconnectFromRobot)
            self.robot = None
