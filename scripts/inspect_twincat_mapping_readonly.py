"""Read-only online ADS mapping diagnostic; never writes or changes TwinCAT state."""

from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from collection.twincat_leg import AmsAddr, AmsNetId, decode_image


class ReadOnlyAdsDiagnostic:
    """Minimal ADS reader exposing only port, timeout, read, and close calls."""

    def __init__(self, config: dict, timeout_ms: int) -> None:
        dll_path = Path(config["ads_dll_path"]).resolve(strict=True)
        self.dll = ct.WinDLL(str(dll_path))
        self.dll.AdsPortOpenEx.argtypes = []
        self.dll.AdsPortOpenEx.restype = ct.c_long
        self.dll.AdsPortCloseEx.argtypes = [ct.c_long]
        self.dll.AdsPortCloseEx.restype = ct.c_long
        self.dll.AdsSyncSetTimeoutEx.argtypes = [ct.c_long, ct.c_uint32]
        self.dll.AdsSyncSetTimeoutEx.restype = ct.c_long
        self.dll.AdsSyncReadReqEx2.argtypes = [
            ct.c_long,
            ct.POINTER(AmsAddr),
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_void_p,
            ct.POINTER(ct.c_uint32),
        ]
        self.dll.AdsSyncReadReqEx2.restype = ct.c_long
        self.port = self.dll.AdsPortOpenEx()
        if self.port <= 0:
            raise OSError("AdsPortOpenEx failed")
        net_id = (ct.c_ubyte * 6)(*map(int, config["ams_net_id"].split(".")))
        self.address = AmsAddr(AmsNetId(net_id), int(config["ads_port"]))
        status = self.dll.AdsSyncSetTimeoutEx(self.port, int(timeout_ms))
        if status:
            self.close()
            raise OSError(f"ADS timeout configuration failed: {status}")

    def read(self, config: dict) -> tuple[bytes, int]:
        buffer = (ct.c_ubyte * int(config["image_bytes"]))()
        actual = ct.c_uint32()
        status = self.dll.AdsSyncReadReqEx2(
            self.port,
            ct.byref(self.address),
            int(config["index_group"]),
            int(config["index_offset"]),
            len(buffer),
            buffer,
            ct.byref(actual),
        )
        if status:
            raise OSError(f"ADS input read failed: code={status}, bytes={actual.value}")
        return bytes(buffer[: actual.value]), int(actual.value)

    def close(self) -> None:
        if getattr(self, "port", 0):
            port, self.port = self.port, 0
            status = self.dll.AdsPortCloseEx(port)
            if status:
                raise OSError(f"AdsPortCloseEx failed: {status}")


def inspect(config: dict, *, samples: int, poll_s: float, timeout_ms: int) -> dict:
    if samples < 2 or not math.isfinite(poll_s) or poll_s < 0:
        raise ValueError("samples must be >=2 and poll_s must be finite and nonnegative")
    required = {
        "ams_net_id",
        "ads_port",
        "index_group",
        "index_offset",
        "image_bytes",
        "axis_offsets",
        "toggle_byte",
        "toggle_bit",
        "ads_dll_path",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"mapping diagnostic missing fields: {missing}")

    images: list[bytes] = []
    decoded: list[dict] = []
    query_ms: list[float] = []
    byte_counts: list[int] = []
    reader = ReadOnlyAdsDiagnostic(config, timeout_ms)
    try:
        for _ in range(samples):
            started = time.perf_counter_ns()
            image, byte_count = reader.read(config)
            finished = time.perf_counter_ns()
            if byte_count != int(config["image_bytes"]):
                raise OSError(
                    f"ADS read returned {byte_count}/{config['image_bytes']} bytes"
                )
            images.append(image)
            byte_counts.append(byte_count)
            decoded.append(decode_image(image, config))
            query_ms.append((finished - started) / 1e6)
            if poll_s:
                time.sleep(poll_s)
    finally:
        reader.close()

    changed_offsets: set[int] = set()
    for previous, current in zip(images, images[1:]):
        changed_offsets.update(
            index for index, (left, right) in enumerate(zip(previous, current)) if left != right
        )
    toggles = [int(row["input_toggle"]) for row in decoded]
    transitions = sum(left != right for left, right in zip(toggles, toggles[1:]))
    axis_snapshots = {}
    for axis in range(1, 5):
        axis_snapshots[f"axis{axis}"] = {
            key.removeprefix(f"axis{axis}_"): value
            for key, value in decoded[-1].items()
            if key.startswith(f"axis{axis}_")
        }

    return {
        "schema_version": 1,
        "diagnostic": "twincat_ads_mapping_read_only",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "write_operations_exposed": False,
        "mapping_reviewed_changed": False,
        "target": {
            "ams_net_id": config["ams_net_id"],
            "ads_port": config["ads_port"],
            "index_group": config["index_group"],
            "index_offset": config["index_offset"],
            "requested_bytes": config["image_bytes"],
        },
        "reads": {
            "attempted": samples,
            "successful": len(images),
            "unique_byte_counts": sorted(set(byte_counts)),
            "query_ms": {
                "mean": statistics.fmean(query_ms),
                "p95": sorted(query_ms)[math.ceil(0.95 * len(query_ms)) - 1],
                "max": max(query_ms),
            },
        },
        "toggle": {
            "byte": config["toggle_byte"],
            "bit": config["toggle_bit"],
            "values_seen": sorted(set(toggles)),
            "transitions_observed": transitions,
            "toggle_byte_changed": int(config["toggle_byte"]) in changed_offsets,
        },
        "axis_offsets": list(config["axis_offsets"]),
        "axis_snapshots_last_read": axis_snapshots,
        "changed_byte_offsets": sorted(changed_offsets),
        "findings": {
            "exact_read_span_confirmed": all(
                count == int(config["image_bytes"]) for count in byte_counts
            ),
            "four_axis_decode_succeeded": len(axis_snapshots) == 4,
            "toggle_location_observed_live": transitions > 0
            and int(config["toggle_byte"]) in changed_offsets,
            "torque_6077_signedness_confirmed": False,
            "human_mapping_review_complete": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--poll-ms", type=float, default=1.0)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    report = inspect(
        config,
        samples=args.samples,
        poll_s=args.poll_ms / 1000.0,
        timeout_ms=args.timeout_ms,
    )
    report["source_config"] = str(args.config.resolve())
    report["source_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["findings"], ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
