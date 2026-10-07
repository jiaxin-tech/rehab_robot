"""SDK-free controller reachability checks for supervised commissioning.

These checks answer "is the robot controller reachable on the network at all?"
before any xCoreSDK session is opened.  They are deliberately independent of
the vendor extension so an operator can separate a network fault from an SDK,
version, mode, or controller fault.

Nothing here powers, moves, or configures the robot.  ICMP echo may be
disabled on the controller; a failed ping alone is not evidence that the
controller is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
import re
import socket
import subprocess
import time
from typing import Sequence

DEFAULT_PROBE_PORTS = (8055,)

_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_IPV6 = re.compile(r"^[0-9A-Fa-f:]+$")


def validate_ipv4(address: str) -> str:
    """Return a normalized dotted-quad string, or raise ``ValueError``."""
    text = str(address).strip()
    if not _IPV4.match(text):
        raise ValueError(f"not an IPv4 address: {address!r}")
    octets = [int(part) for part in text.split(".")]
    if any(part > 255 for part in octets):
        raise ValueError(f"not an IPv4 address: {address!r}")
    return ".".join(str(part) for part in octets)


def resolve_local_interface(remote_ip: str, *, port: int = 8055) -> str | None:
    """Return the local IPv4 address a route to ``remote_ip`` would use."""
    target = validate_ipv4(remote_ip)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.settimeout(0.5)
        probe.connect((target, int(port)))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def tcp_reachable(remote_ip: str, port: int, *, timeout_s: float = 0.5) -> bool:
    target = validate_ipv4(remote_ip)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(float(timeout_s))
    try:
        return probe.connect_ex((target, int(port))) == 0
    except OSError:
        return False
    finally:
        probe.close()


def _ping_arguments(system: str, target: str, timeout_s: float) -> list[str] | None:
    if system == "Windows":
        return ["ping", "-n", "1", "-w", str(int(timeout_s * 1000)), target]
    if system == "Darwin":
        return ["ping", "-c", "1", "-W", str(int(timeout_s * 1000)), target]
    if system == "Linux":
        return ["ping", "-c", "1", "-W", str(max(1, int(round(timeout_s)))), target]
    return None


def icmp_ping(remote_ip: str, *, timeout_s: float = 1.0) -> dict[str, object]:
    """Run one ICMP echo.  Returns ``reachable=None`` when the tool is absent."""
    target = validate_ipv4(remote_ip)
    arguments = _ping_arguments(platform.system(), target, timeout_s)
    if arguments is None:
        return {"attempted": False, "reachable": None, "reason": "unsupported_platform"}
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=float(timeout_s) + 3.0,
        )
    except FileNotFoundError:
        return {"attempted": False, "reachable": None, "reason": "ping_not_found"}
    except subprocess.TimeoutExpired:
        return {
            "attempted": True,
            "reachable": False,
            "duration_ms": (time.perf_counter() - started) * 1e3,
            "reason": "ping_timeout",
            "command": arguments,
        }
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "attempted": True,
        "reachable": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "duration_ms": (time.perf_counter() - started) * 1e3,
        "output_excerpt": output.strip().splitlines()[:4],
        "command": arguments,
    }


@dataclass(frozen=True)
class NetworkProbeResult:
    remote_ip: str
    ports: tuple[int, ...]
    icmp: dict[str, object]
    tcp: dict[int, bool]
    local_interface: str | None

    def to_dict(self) -> dict[str, object]:
        any_tcp = any(self.tcp.values())
        pinged = self.icmp.get("reachable")
        return {
            "schema_version": 1,
            "probe": "rokae_controller_network_reachability",
            "remote_ip": self.remote_ip,
            "resolved_local_interface": self.local_interface,
            "icmp": dict(self.icmp),
            "tcp": {str(port): open_ for port, open_ in sorted(self.tcp.items())},
            "reachable": bool(any_tcp) if self.tcp else pinged is True,
            "operator_hint": _hint(any_tcp, pinged),
            "robot_commands_sent": False,
        }


def _hint(any_tcp: bool, pinged: object) -> str:
    if any_tcp:
        return "controller answered a TCP connection; xCoreSDK session may be attempted"
    if pinged is True:
        return (
            "controller answers ICMP but no probed TCP port accepted a connection; "
            "check the vendor service port and controller firewall before blaming the SDK"
        )
    if pinged is False:
        return (
            "no ICMP reply; if the controller blocks ping this is inconclusive, "
            "verify robot subnet, cable, and the local interface address"
        )
    return "reachability is inconclusive; verify the robot subnet and local interface"


def probe_controller(
    remote_ip: str,
    *,
    ports: Sequence[int] = DEFAULT_PROBE_PORTS,
    timeout_s: float = 0.5,
) -> NetworkProbeResult:
    target = validate_ipv4(remote_ip)
    if float(timeout_s) <= 0.0:
        raise ValueError("timeout_s must be positive")
    parsed_ports = []
    for port in ports:
        value = int(port)
        if not 1 <= value <= 65535:
            raise ValueError(f"port out of range: {port}")
        parsed_ports.append(value)
    return NetworkProbeResult(
        remote_ip=target,
        ports=tuple(parsed_ports),
        icmp=icmp_ping(target, timeout_s=float(timeout_s) + 0.5),
        tcp={port: tcp_reachable(target, port, timeout_s=float(timeout_s)) for port in parsed_ports},
        local_interface=resolve_local_interface(target, port=parsed_ports[0] if parsed_ports else 8055),
    )


__all__ = [
    "DEFAULT_PROBE_PORTS",
    "NetworkProbeResult",
    "icmp_ping",
    "probe_controller",
    "resolve_local_interface",
    "tcp_reachable",
    "validate_ipv4",
]