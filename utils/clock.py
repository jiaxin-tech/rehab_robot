"""One high-resolution host-monotonic clock for experiment data.

The robot SDK does not guarantee a device timestamp for every source.  State,
wrench, command, alignment, and logger timestamps therefore share the host
``time.perf_counter_ns`` time base.  Wall-clock time is intentionally not mixed
into timing or freshness calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol, runtime_checkable


NANOSECONDS_PER_SECOND = 1_000_000_000
TIMESTAMP_SOURCE = "time.perf_counter_ns"


@runtime_checkable
class MonotonicClock(Protocol):
    """Minimal injectable clock contract used by the experiment data layer."""

    def now_ns(self) -> int:
        """Return the current host-monotonic timestamp in integer nanoseconds."""

    def now_s(self) -> float:
        """Return the same clock value expressed in seconds."""


@dataclass(frozen=True)
class PerfCounterClock:
    """High-resolution monotonic clock backed only by ``perf_counter_ns``."""

    timestamp_source: str = TIMESTAMP_SOURCE

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def now_s(self) -> float:
        return self.now_ns() / NANOSECONDS_PER_SECOND


SYSTEM_CLOCK = PerfCounterClock()


def host_time_ns() -> int:
    """Return the project-wide host-monotonic timestamp in nanoseconds."""

    return SYSTEM_CLOCK.now_ns()


def host_time_s() -> float:
    """Return the project-wide host-monotonic timestamp in seconds."""

    return SYSTEM_CLOCK.now_s()


__all__ = [
    "MonotonicClock",
    "NANOSECONDS_PER_SECOND",
    "PerfCounterClock",
    "SYSTEM_CLOCK",
    "TIMESTAMP_SOURCE",
    "host_time_ns",
    "host_time_s",
]
