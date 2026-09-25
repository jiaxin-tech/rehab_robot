import json

import numpy as np
import pytest

from lower_limb_sim.myoleg_benchmark.controlled_resistance_cohort import (
    DEFAULT_MANIFEST, REFERENCE_FEATURES, ResistanceProfile, build_profiles,
    load_profiles, manifest, write_manifest,
)


def test_profiles_have_common_null_and_divergent_mechanism_arms():
    profiles = build_profiles()
    assert len(profiles) == 12
    assert {p.scenario for p in profiles} == {"COMMON_RESISTANCE", "DIVERGENT_RESISTANCE"}
    assert {p.split for p in profiles} == {"DEVELOPMENT", "CONFIRMATORY"}
    positive = [p for p in profiles if p.scenario == "DIVERGENT_RESISTANCE"]
    assert len({(p.resistance_onset_phase, p.velocity_sensitivity, p.hip_resistance_share) for p in positive}) >= 4
    assert all(np.isclose(p.value(REFERENCE_FEATURES), 1.0) for p in profiles)


def test_mechanism_is_smooth_and_profiles_can_have_different_oracles():
    profiles = build_profiles()
    early = next(p for p in profiles if p.arm == "EARLY" and p.split == "DEVELOPMENT")
    late = next(p for p in profiles if p.arm == "LATE" and p.split == "DEVELOPMENT")
    features = [(d, t, h) for d in (0.9, 1.0, 1.1) for t in (0.25, 0.5, 0.75) for h in (0.25, 0.5, 0.75)]
    early_oracle = min(features, key=early.value)
    late_oracle = min(features, key=late.value)
    assert early_oracle != late_oracle
    assert all(np.isfinite([early.value(x), late.value(x)]).all() for x in features)


def test_manifest_is_hashed_and_confirmatory_can_be_kept_sealed(tmp_path):
    target = tmp_path / "manifest.json"
    write_manifest(target)
    loaded = load_profiles(path=target)
    assert len(loaded) == 12
    assert len(load_profiles("DEVELOPMENT", path=target)) == 6
    assert len(load_profiles("CONFIRMATORY", path=target)) == 6
    document = json.loads(target.read_text())
    document["profiles"][0]["arm"] = "TAMPERED"
    target.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="FINGERPRINT"):
        load_profiles(path=target)


def test_invalid_profile_scope_fails_closed():
    with pytest.raises(ValueError, match="COMMON_RESISTANCE"):
        ResistanceProfile("RESIST_BAD", "COMMON_RESISTANCE", "DEVELOPMENT", "x", 1, .5, .5, .5, 0.1)
    with pytest.raises(ValueError, match="UNDECLARED"):
        build_profiles()[0].mechanism_cost((0.95, 0.5, 0.5))
