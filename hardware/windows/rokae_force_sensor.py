"""Deprecated compatibility import for the robot-internal wrench source.

There is no external six-axis force sensor in this project.  Keep the old name
only so downstream scripts fail neither silently nor abruptly during migration;
new code must import :class:`RokaeInternalWrenchSource`.
"""

from .rokae_internal_wrench import RokaeInternalWrenchSource


class RokaeForceSensor(RokaeInternalWrenchSource):
    """Compatibility alias; it is a controller query source, not a sensor device."""


__all__ = ["RokaeForceSensor", "RokaeInternalWrenchSource"]
