import hashlib
import json
from pathlib import Path
import zipfile

import scripts.package_lab_visit as delivery


def test_package_includes_new_source_modules_and_today_checklist(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    payloads = {
        "scripts/new_helper.py": b"# current new launcher helper\n",
        "measurement_validation/torque_endpoints.py": b"# current new pure endpoint module\n",
        "measurement_validation/robot_leg_history.py": b"# current new adapter\n",
        "experiment_algorithm/binding.py": b"# current binding\n",
        "experiment_algorithm/README.md": b"Versioned algorithm contents\n",
        "experiment_algorithm/v2/SOURCE_LOCK.json": b"{}\n",
        "experiment_algorithm/v2/UPSTREAM_PROVENANCE.json": b"{}\n",
        "experiment_algorithm/.gitattributes": b"*.py -text\n",
        "config/lab_next_visit.json": b"{}\n",
        ".gitattributes": b"*.json -text\n",
        ".gitignore": b".cache/\n",
        "requirements.txt": b"numpy>=2.0\n",
        "config/templates/experiment_safety.json": b'{"reviewed": false}\n',
        "config/templates/static_validation_logging.json": b'{"enabled": false}\n',
        "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/frozen_inputs/formal_experiment_manifest.json": b'{"scope": "historical"}\n',
        "docs/CURRENT_WORKFLOW.md": b"Current workflow\n",
        "pytest-core.ini": b"[pytest]\n",
        "docs/TODAY_LAB_CHECKLIST_2026-11-14.md": "今日实验清单".encode(),
        "site_evidence/COLLISION_EVENT_INTERFACE_2026-10-19.md": b"existing interface evidence\n",
        "site_evidence/OFFLINE_EMPTY_LOAD_REVIEW_2026-10-19.md": b"existing failure evidence\n",
        "results/lab_bundle/frozen/config.py": b"# frozen source\n",
        # These must not become part of a source delivery by broad recursion.
        "results/private_raw/leg.csv": b"private lab rows\n",
        "measurement_validation/__pycache__/unwanted.py": b"ignored\n",
    }
    for name, content in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    initialization = root / "initialization.json"
    initialization.write_text(json.dumps(dict(status="ALGORITHM_INITIALIZED", source_hashes_checked=True,
        public_prior=dict(asset_files=[], asset_root=str(root / ".cache")))), encoding="utf-8")
    monkeypatch.setattr(delivery, "ROOT", root)
    def git_output(command, **kwargs):
        if command[1] == "ls-files":
            return b"scripts/new_helper.py\0"
        return b"fixture-commit\n"
    monkeypatch.setattr(delivery.subprocess, "check_output", git_output)
    output = tmp_path / "delivery.zip"
    report = delivery.package(root / "results/lab_bundle", initialization, output)
    assert report["zip_verified"] is True
    with zipfile.ZipFile(output) as archive:
        for name in ("measurement_validation/torque_endpoints.py", "measurement_validation/robot_leg_history.py",
                     "experiment_algorithm/binding.py", "experiment_algorithm/v2/SOURCE_LOCK.json",
                     "experiment_algorithm/README.md",
                     "experiment_algorithm/v2/UPSTREAM_PROVENANCE.json", "experiment_algorithm/.gitattributes",
                     "config/templates/experiment_safety.json", "config/templates/static_validation_logging.json",
                     "external_simulation/myoleg_v3_trajectory_parameterization_design_v1/frozen_inputs/formal_experiment_manifest.json",
                     "docs/CURRENT_WORKFLOW.md",
                     ".gitattributes", ".gitignore", "requirements.txt",
                     "docs/TODAY_LAB_CHECKLIST_2026-11-14.md"):
            assert archive.read(name) == payloads[name]
        assert "results/private_raw/leg.csv" not in archive.namelist()
        assert "measurement_validation/__pycache__/unwanted.py" not in archive.namelist()
        manifest = json.loads(archive.read("lab_delivery_manifest.json"))
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
