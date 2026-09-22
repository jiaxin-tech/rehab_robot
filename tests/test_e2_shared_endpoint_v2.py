"""Shared E2 equivalence with every frozen five-leg feature row; no regeneration."""
import json
from pathlib import Path
import numpy as np
import pytest
from lower_limb_sim.mechanical_endpoints import (
    MechanicalReferenceContext, BranchRMSReference, branch_balanced_reference_normalized_rms,
    branch_rms_components,
)
from lower_limb_sim.five_leg_mujoco_v1.endpoint_design import _endpoint_values, _response_metrics_from_torque

ROOT = Path(__file__).resolve().parents[1] / 'lower_limb_sim/five_leg_mujoco_v1'
FIELDS = ('flexion_hip_rms_nm', 'extension_hip_rms_nm', 'flexion_knee_rms_nm', 'extension_knee_rms_nm')


def test_all_frozen_e2_feature_rows_unchanged():
    study = json.loads((ROOT / 'results_endpoint_design_v1/study_summary.json').read_text())
    rows = study['feature_rows']
    assert len(rows) == 3125
    refs = {r['leg_id']: r for r in rows if r['beta_flex'] == 0 and r['beta_extend'] == 0}
    for row in rows:
        ref = refs[row['leg_id']]
        context = MechanicalReferenceContext(row['leg_id'], 1, row['leg_id'], 'frozen-study', ())
        new = branch_balanced_reference_normalized_rms([row[f] for f in FIELDS],
                  BranchRMSReference(context, tuple(ref[f] for f in FIELDS)), context=context)
        old = _endpoint_values({'full_rms_nm': row['endpoint_E0']}, row)['E2']
        assert new == pytest.approx(old, abs=1e-14)
        assert new == pytest.approx(row['endpoint_E2'], abs=1e-14)
    necessity = json.loads((ROOT / 'results_e2_necessity_v1/study_summary.json').read_text())
    assert necessity['E2_PERSONALIZATION_NECESSITY'] == 'NOT_SUPPORTED'
    assert necessity['READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON'] is False


def test_time_series_branch_rms_matches_original_formula():
    time = np.array([0., .1, .4, .7, 1.1, 1.9])
    torque = np.array([[1.,2.],[3.,4.],[2.,1.],[4.,5.],[7.,2.],[1.,3.]])
    branches = np.array(['flexion']*3 + ['extension']*3)
    old = _response_metrics_from_torque(torque, time, branches)
    new = branch_rms_components(torque[:,0], torque[:,1], time, branches)
    assert new == pytest.approx(tuple(old[f] for f in FIELDS), abs=1e-14)
