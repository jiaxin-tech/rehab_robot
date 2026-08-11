"""Opt-in Windows/robot observation-session integration smoke test.

It executes only when all three are present: Windows,
``ROKAE_HARDWARE_TEST=1``, and ``ROKAE_TEST_IP``.  Project code never enables,
powers, clears errors, calibrates, or sends a motion target.  Vendor object
initialization/connect/disconnect may nevertheless have controller-session
side effects and require an idle robot plus supervised first validation.
"""

from __future__ import annotations

import os
import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "win32"
    or os.environ.get("ROKAE_HARDWARE_TEST") != "1"
    or not os.environ.get("ROKAE_TEST_IP"),
    reason="requires explicit Windows real-robot opt-in",
)


def test_observation_session_connect_state_and_wrench_smoke():
    from hardware.rokae_adapter import RokaeRobotAdapter

    adapter = RokaeRobotAdapter(os.environ["ROKAE_TEST_IP"])
    try:
        adapter.connect()
        adapter.start_state_stream()
        assert adapter.is_connected()
        assert len(adapter.read_tcp_pose()) == 6
        assert len(adapter.read_joint_positions()) == 6
        wrench = adapter.read_internal_wrench()
        assert wrench.valid, wrench.invalid_reason
    finally:
        try:
            adapter.stop_state_stream()
        finally:
            adapter.disconnect()
