"""Host-timed, fail-closed cache for xCoreSDK Event.safety callbacks.

The vendor does not promise an initial event or a heartbeat. A subscription
alone is not evidence of clearance; silence eventually invalidates clearance.
No SDK calls, controller writes, or motion operations belong in this cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Callable


def validate_collision_source(source: str, max_age_s: float | None) -> None:
    if source not in ("query", "events"):
        raise ValueError("collision_source must be query or events")
    if source == "events":
        if (isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float))
                or not math.isfinite(max_age_s) or max_age_s <= 0):
            raise ValueError("events requires an explicit finite positive collision_event_max_age_s")
    elif max_age_s is not None:
        raise ValueError("collision_event_max_age_s requires collision_source=events")


@dataclass(frozen=True)
class SafetyEventStatus:
    source: str
    generation: int
    registered: bool
    sequence: int
    collision_state: bool | None
    collision_latched: bool
    received_host_time_s: float | None
    age_s: float | None
    max_age_s: float
    valid: bool
    invalid_reason: str
    timestamp_source: str = "host_callback_receive_time_no_robot_device_timestamp"

    def to_dict(self) -> dict:
        return asdict(self)


class SafetyEventMonitor:
    def __init__(self, max_age_s: float, *, clock: Callable[[], float] = time.perf_counter):
        validate_collision_source("events", max_age_s)
        self.max_age_s = float(max_age_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._generation = 0
        self._active = False
        self._registered = False
        self._sequence = 0
        self._state: bool | None = None
        self._received: float | None = None
        self._fault: str | None = "safety_events_not_subscribed"
        self._collision_latched = False

    def begin(self) -> int:
        with self._lock:
            self._generation += 1
            self._active = True
            self._registered = False
            self._sequence = 0
            self._state = None
            self._received = None
            self._fault = None
            # Collision evidence survives re-subscription and reconnection.
            return self._generation

    def registered(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation and self._active:
                self._registered = True

    def invalidate(self, reason: str) -> None:
        with self._lock:
            self._generation += 1
            self._active = False
            self._registered = False
            self._state = None
            self._fault = reason

    def receive(self, generation: int, payload: object) -> None:
        with self._lock:
            if generation != self._generation or not self._active:
                return
            now = self._clock()
            # Do not let a late event renew already expired clearance.
            if self._received is not None and (now < self._received or now - self._received > self.max_age_s):
                self._fault = self._fault or "safety_event_expired_or_clock_invalid"
            if not isinstance(payload, dict) or type(payload.get("collided")) is not bool:
                self._fault = self._fault or "safety_event_invalid_collided_boolean"
                self._state = None
                return
            self._sequence += 1
            self._received = now
            self._state = payload["collided"]
            self._collision_latched = self._collision_latched or self._state

    def snapshot(self) -> SafetyEventStatus:
        with self._lock:
            age = None if self._received is None else self._clock() - self._received
            if age is not None and (not math.isfinite(age) or age < 0 or age > self.max_age_s):
                self._fault = self._fault or "safety_event_expired_or_clock_invalid"
            reason = self._fault
            if not reason and not self._registered:
                reason = "safety_event_registration_unconfirmed"
            if not reason and self._state is None:
                reason = "safety_event_initial_state_unknown"
            valid = self._active and self._registered and not reason
            state = True if self._collision_latched else (self._state if valid else None)
            return SafetyEventStatus(
                source="events", generation=self._generation,
                registered=self._registered, sequence=self._sequence,
                collision_state=state, collision_latched=self._collision_latched,
                received_host_time_s=self._received, age_s=age,
                max_age_s=self.max_age_s, valid=bool(valid), invalid_reason=reason or "",
            )


def safety_event_block_reason(adapter: object) -> str | None:
    """Read only a local event cache; legacy query-mode adapters return None."""
    reader = getattr(adapter, "get_safety_event_status", None)
    if not callable(reader):
        return None
    try:
        status = reader()
    except Exception as exc:
        return f"safety_event_status_error:{type(exc).__name__}"
    if status is None:
        return None
    if not isinstance(status, dict) or status.get("valid") is not True:
        return (str(status.get("invalid_reason") or "safety_event_unavailable")
                if isinstance(status, dict) else "safety_event_status_invalid")
    if status.get("collision_state") is not False:
        return "robot_collision" if status.get("collision_state") is True else "safety_event_state_unknown"
    return None
