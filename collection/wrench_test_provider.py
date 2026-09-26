"""Explicit OFFLINE stubs. Never imported by the live session branch."""
import ctypes
import os
import time

from hardware.rokae_adapter import RobotWrenchFrame


class TestSDKError(RuntimeError):
    code = 263


def query(config, query_id, started_ns):
    behavior = config.get("behavior", "normal")
    if query_id <= config.get("good_queries", 0):
        behavior = "normal"
    if behavior == "crash":
        os._exit(73)
    if behavior == "never":
        while True:
            time.sleep(1)
    if behavior == "gil_block":
        # PyDLL retains the GIL during a foreign call (unlike CDLL).
        # Only the spawned OFFLINE child invokes this test-only function.
        if os.name == "nt":
            native = ctypes.PyDLL("kernel32.dll")
            native.Sleep.argtypes = [ctypes.c_ulong]
            native.Sleep.restype = None
            native.Sleep(int(config.get("delay_s", 10) * 1000))
        else:
            native = ctypes.PyDLL(None)
            native.sleep.argtypes = [ctypes.c_uint]
            native.sleep(int(config.get("delay_s", 10)))
    elif behavior in ("block", "error263"):
        time.sleep(config.get("delay_s", .1))
    if behavior == "error263":
        raise TestSDKError("OFFLINE delayed SDK 263")
    end = time.perf_counter_ns() / 1e9
    return RobotWrenchFrame(query_id, started_ns / 1e9, end, end,
        (started_ns / 1e9 + end) / 2, "", "OFFLINE_HOST_QUERY", True, "", "world",
        (1., 2., 3.), (.1, .2, .3), (1.,) * 6, (.5,) * 6)
