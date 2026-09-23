"""Frozen-source checks and physical portability checks for MyoLeg loading."""

import hashlib
import json

import numpy as np
import pytest

from lower_limb_sim.myoleg_benchmark import simulation
from lower_limb_sim.myoleg_benchmark.portable_model import DEFAULT_ASSET_ROOT, load_model


def test_text_identity_accepts_only_newline_equivalence(tmp_path):
    original = b'<option gravity="0 0 -9.81"/>\n'
    expected = hashlib.sha256(original).hexdigest()
    path = tmp_path / "model.xml"
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    identity = simulation._frozen_file_identity(path, expected, text_file=True)
    assert identity["match"] == "newline_equivalent"
    assert identity["raw_sha256"] != expected
    assert identity["lf_sha256"] == expected
    path.write_bytes(original.replace(b"-9.81", b"-9.80"))
    with pytest.raises(ValueError, match="FROZEN_SOURCE_CONTENT_MISMATCH"):
        simulation._frozen_file_identity(path, expected, text_file=True)


def test_delta_checks_before_and_assigns_exact_frozen_after():
    target = np.array([1. + 2e-16, 2.])
    changes = {}
    simulation._set_delta_field(target, 0, {"before": 1., "after": 1.25}, "mass", changes)
    assert target[0] == 1.25
    assert 0. < changes["mass"] < 1e-13
    with pytest.raises(ValueError, match="FROZEN_DELTA_BASE_FIELD_MISMATCH"):
        simulation._set_delta_field(target, 1, {"before": 2.01, "after": 3.}, "mass", changes)
    assert target[1] == 2.


def test_held_out_subject_is_rejected_before_model_or_subject_files(monkeypatch):
    monkeypatch.setattr(simulation, "development_ids", lambda: ("MYOLEG_VP_001",))

    def forbidden_load():
        pytest.fail("Model loading happened before subject access rejection")

    monkeypatch.setattr(simulation, "load_model_with_provenance", forbidden_load)
    with pytest.raises(ValueError, match="ONLY_NOMINAL_AND_FROZEN_DEVELOPMENT_SUBJECTS_ALLOWED"):
        simulation.subject_model("MYOLEG_VP_004")


@pytest.mark.skipif(not DEFAULT_ASSET_ROOT.is_dir(), reason="Official local MyoLeg assets are required")
def test_reference_equivalence_detects_gravity_change_omitted_by_old_fingerprint():
    model = load_model()
    record = json.loads(simulation.COHORT_MANIFEST.read_text(encoding="utf-8"))["nominal_control"]
    original_fingerprint = simulation.model_fingerprint(model)
    model.opt.gravity[2] *= .99
    assert simulation.model_fingerprint(model) == original_fingerprint
    with pytest.raises(ValueError, match="FROZEN_REFERENCE_PHYSICS_MISMATCH"):
        simulation._reference_equivalence(model, record)
