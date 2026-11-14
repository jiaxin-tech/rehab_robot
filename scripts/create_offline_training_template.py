"""Write a new repository-version training template without editing live config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_algorithm.binding import current_binding
from measurement_validation.robot_leg_history import _digest, _text, _verified_file
from lower_limb_sim.config import L1, L2


def build_template(catalog_path, *, setup_id, setup_sha256):
    catalog_path = Path(catalog_path).resolve(strict=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    candidates = catalog["candidates"]
    for item in candidates:
        item["path"] = str(_verified_file(catalog_path.parent, item["path"], item["sha256"]))
    reference_id = _text(catalog["reference_candidate_id"], "reference_candidate_id")
    reference = next(item for item in candidates if item["candidate_id"] == reference_id)
    return dict(**current_binding(), setup_id=_text(setup_id, "setup_id"),
                setup_sha256=_digest(setup_sha256, "setup_sha256"), geometry_m=[L1, L2],
                candidates=candidates, reference_candidate_id=reference_id,
                reference_sha256=reference["sha256"], history=[],
                motion_approved=False, template_scope="OFFLINE_RESEARCH_V2_NOT_A_LIVE_REVIEW")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--setup-id", required=True)
    parser.add_argument("--setup-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_template(args.catalog, setup_id=args.setup_id, setup_sha256=args.setup_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
