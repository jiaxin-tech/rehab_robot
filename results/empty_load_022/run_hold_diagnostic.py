from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from config.bundle import configuration
from control.execution_preflight import OPERATOR_CONFIRMATION
from control.start_anchor import load_start_anchor
from scripts.acquire_robot_data import (
    _configured_adapter_factory,
    load_live_wrench_config,
)
from scripts.run_rehab_experiment import run_execute


def main() -> None:
    session = Path(__file__).resolve().parent
    config_path = Path("config/experiment.json").resolve()
    config = configuration(config_path)
    robot = config["robot"]
    acquisition = config["acquisition"]
    anchor = load_start_anchor(session / "anchor.json")
    reference = pd.read_csv(session / "hold_diagnostic_reference.csv")
    state: dict[str, object] = {"hold_started_ns": None, "dispatches": []}

    def dispatched(trajectory_time_s: float, start_ns: int, end_ns: int) -> None:
        dispatches = state["dispatches"]
        assert isinstance(dispatches, list)
        dispatches.append((float(trajectory_time_s), start_ns, end_ns))
        if float(trajectory_time_s) == 0.0:
            state["hold_started_ns"] = end_ns

    def guard() -> str | None:
        started_ns = state["hold_started_ns"]
        if isinstance(started_ns, int):
            elapsed_s = (time.perf_counter_ns() - started_ns) / 1e9
            if elapsed_s >= 4.25:
                return "hold_diagnostic_complete"
        return None

    factory = _configured_adapter_factory(
        local_ip=robot["local_ip"],
        robot_class=robot["robot_class"],
        state_interval_ms=None,
        collision_source=robot["collision_source"],
        collision_event_max_age_s=robot["collision_event_max_age_s"],
    )
    expected_stop = False
    failure: BaseException | None = None
    try:
        run_execute(
            robot_ip=robot["ip"],
            local_ip=robot["local_ip"],
            episode_dir=session / "hold_diagnostic_robot_008",
            anchor_path=session / "anchor.json",
            requested_anchor_id=anchor.anchor_id,
            frame_config_path=config_path,
            safety_config_path=session / "empty_load_safety.json",
            trajectory_id=str(reference["trajectory_id"].iloc[0]),
            reference_path_override=session / "hold_diagnostic_reference.csv",
            enable_motion=True,
            operator_confirmation=OPERATOR_CONFIRMATION,
            live_wrench_config=load_live_wrench_config(config_path),
            wrench_hz=acquisition["wrench_hz"],
            adapter_factory=factory,
            collision_source=robot["collision_source"],
            collision_event_max_age_s=robot["collision_event_max_age_s"],
            prepare_realtime=True,
            execution_guard=guard,
            dispatch_observer=dispatched,
        )
        failure = RuntimeError(
            "hold diagnostic unexpectedly completed the full reference"
        )
    except RuntimeError as exc:
        expected_stop = "hold_diagnostic_complete" in str(exc)
        if not expected_stop:
            failure = exc

    verification_adapter = factory(robot["ip"])
    verification_adapter.connect()
    try:
        controller = verification_adapter.native_robot._realtime_prepared_state()
    finally:
        verification_adapter.disconnect()
    if controller.get("power_state") != "PowerState.off":
        failure = RuntimeError(f"controller power-down unconfirmed: {controller}")
    if controller.get("operation_state") != "OperationState.idle":
        failure = RuntimeError(f"controller idle state unconfirmed: {controller}")
    if controller.get("rt_loop_active") is not False:
        failure = RuntimeError(f"realtime loop still active: {controller}")

    dispatches = state["dispatches"]
    assert isinstance(dispatches, list)
    print(
        {
            "expected_stop": expected_stop,
            "dispatch_count": len(dispatches),
            "trajectory_times_s": [item[0] for item in dispatches],
            "controller": controller,
        }
    )
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
