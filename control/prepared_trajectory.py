"""Rebuild frozen execution input through existing trajectory converters."""
import hashlib
from pathlib import Path
import pandas as pd

from control.independent_tcp_trajectory import build_tcp_trajectory
from control.start_anchored_relative_trajectory import build_start_anchored_relative_trajectory


def assert_frozen_trajectory(expected, path):
    """Compare CSV content without treating empty audit text as numeric NaN."""
    text_types = {name: column.dtype for name, column in expected.items()
                  if pd.api.types.is_string_dtype(column.dtype)
                  or pd.api.types.is_object_dtype(column.dtype)}
    frozen = pd.read_csv(path, dtype=text_types, keep_default_na=False)
    pd.testing.assert_frame_equal(expected, frozen,
                                  check_exact=False, atol=1e-12, rtol=0)


def build_prepared_trajectory(plan, *, current_tcp_start_pose, rehab_frame):
    if plan.get("mode") == "original_candidate":
        source = Path(plan["reference_path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != plan["reference_sha256"]:
            raise ValueError("frozen original candidate changed")
        return build_start_anchored_relative_trajectory(source,
            current_tcp_start_pose=current_tcp_start_pose, rehab_frame=rehab_frame)
    return build_tcp_trajectory(plan["tcp_settings"],
        current_tcp_start_pose=current_tcp_start_pose, rehab_frame=rehab_frame)
