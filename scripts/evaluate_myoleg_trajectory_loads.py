"""Evaluate the existing native-ROM 625 V3 trajectories with frozen MyoLeg P0.

No optimizer, muscle recruitment solver, or physical cuff sensor is involved.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco

from external_simulation.myoleg_reference_trajectory_replay_v1.build_and_replay import prescribed_truth
from lower_limb_sim.visualization.myoleg_robot_scene import MODEL_PATH, native_domain
from lower_limb_sim.mechanical_endpoints import (
    MechanicalReferenceContext, BranchRMSReference, branch_rms_components,
    branch_balanced_reference_normalized_rms, full_cycle_dual_joint_rms,
)

OUT = Path('outputs/myoleg_trajectory_loads')
COMPONENTS = ('hip_flexion', 'hip_extension', 'knee_flexion', 'knee_extension')


def plot_results():
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rehab_benchmark_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(OUT/'trajectory_loads.csv')
    a = np.load(OUT/'trajectory_torques_and_muscle_loads.npz')
    selected = [0, int(frame.E0_Nm.idxmin()), int(frame.E2.idxmin())]
    labels = ['Reference', 'E0 minimum', 'E2 minimum']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, key in zip(axes[0], ('E0_Nm', 'E2')):
        grid = frame.pivot(index='beta_extend', columns='beta_flex', values=key)
        im = ax.imshow(grid.to_numpy(), origin='lower', extent=[-.03125,.03125,-.03125,.03125],
                       aspect='auto', cmap='viridis')
        ax.set(title=key, xlabel='beta flexion', ylabel='beta extension')
        fig.colorbar(im, ax=ax)
        k = int(frame[key].idxmin())
        ax.plot(frame.iloc[k].beta_flex, frame.iloc[k].beta_extend, 'rx', ms=10)
        ax.plot(0, 0, 'wo', ms=5)
    for j, ax in enumerate(axes[1]):
        for k, label in zip(selected, labels):
            ax.plot(a['time_s'], a['tau_required_Nm'][k,:,j], label=label,
                    alpha=.8, linewidth=1.3)
        ax.set(title=('Hip' if j == 0 else 'Knee')+' required external torque',
               xlabel='Time (s)', ylabel='N m')
        ax.legend()
    fig.suptitle('MyoLeg native ROM | zero muscle activation | 625 prescribed trajectories')
    fig.savefig(OUT/'loads_and_landscapes.png', dpi=160)
    plt.close(fig)
    # One row per trajectory and muscle, in N; no cuff-force interpretation.
    n = len(frame)
    names = a['actuator_names']
    pd.DataFrame(dict(candidate_id=np.repeat(a['candidate_id'],len(names)),
                      muscle=np.tile(names,n),
                      passive_condition_force_RMS_N=a['actuator_force_RMS_N'].reshape(-1),
                      passive_condition_force_peak_abs_N=a['actuator_force_peak_abs_N'].reshape(-1)
                      )).to_csv(OUT/'muscle_loads.csv', index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    domain = native_domain()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    r, p = domain.subject_reference, domain.profile
    context = MechanicalReferenceContext(p.profile_id, p.version, p.fingerprint,
                                         r.reference_version, ())
    candidates = [domain.reference] + [c for c in domain if c != domain.reference]
    rows, torques, muscle_rms, muscle_peak = [], [], [], []
    reference = None
    for i, c in enumerate(candidates):
        t = c.trajectory
        a, _ = prescribed_truth(model, dict(time_s=r.time_s, q=t.q, dq=t.dq,
                                          ddq=t.ddq, phases=r.phases))
        tau = a['tau_truth_nm']
        assert np.isfinite(tau).all()
        assert np.max(np.abs(a['actuator_activation'])) == 0
        residual = float(np.max(np.abs(a['decomposition_residual_nm'])))
        assert residual < 1e-8, residual
        comp = branch_rms_components(tau[:, 0], tau[:, 1], r.time_s, r.phases)
        if reference is None:
            reference = BranchRMSReference(context, comp)
        row = dict(candidate_id=c.candidate_id, beta_flex=c.beta_flex,
                   beta_extend=c.beta_extend,
                   E0_Nm=full_cycle_dual_joint_rms(tau[:, 0], tau[:, 1], r.time_s),
                   E2=branch_balanced_reference_normalized_rms(comp, reference, context=context),
                   hip_peak_abs_Nm=float(np.max(np.abs(tau[:, 0]))),
                   knee_peak_abs_Nm=float(np.max(np.abs(tau[:, 1]))),
                   decomposition_residual_max_Nm=residual,
                   warnings=int(np.max(a['warning_count'])))
        row.update({name+'_RMS_Nm': v for name, v in zip(COMPONENTS, comp)})
        row.update({name+'_relative': v / ref for name, v, ref in
                    zip(COMPONENTS, comp, reference.components)})
        for term in ('mass_term_nm', 'bias_term_nm', 'passive_internal_nm',
                     'actuator_internal_nm', 'constraint_internal_nm',
                     'constraint_joint_limit_internal_nm', 'constraint_contact_internal_nm'):
            row[term+'_peak_abs'] = float(np.max(np.abs(a[term])))
        rows.append(row)
        torques.append(tau)
        force = a['actuator_force_n']
        muscle_rms.append(np.sqrt(np.trapezoid(force**2, r.time_s, axis=0) /
                                  (r.time_s[-1] - r.time_s[0])))
        muscle_peak.append(np.max(np.abs(force), axis=0))
        if i == 0:
            print('Reference smoke passed:', row, flush=True)
        if (i + 1) % 50 == 0:
            print(f'{i+1}/625 trajectories complete', flush=True)
    assert len(rows) == 625
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'trajectory_loads.csv', index=False)
    np.savez_compressed(OUT/'trajectory_torques_and_muscle_loads.npz',
                        time_s=r.time_s, phases=r.phases,
                        candidate_id=frame.candidate_id.to_numpy(dtype=str),
                        beta=frame[['beta_flex', 'beta_extend']].to_numpy(),
                        tau_required_Nm=np.asarray(torques),
                        actuator_names=a['actuator_names'],
                        actuator_force_RMS_N=np.asarray(muscle_rms),
                        actuator_force_peak_abs_N=np.asarray(muscle_peak))
    result = dict(model=str(MODEL_PATH), condition='P0 zero control and zero activation',
                  scope='Single existing MyoLeg native-ROM model; not five mechanical legs',
                  hip_ROM_deg=np.rad2deg([r.q[:,0].min(),r.q[:,0].max()]).tolist(),
                  knee_ROM_deg=np.rad2deg([r.q[:,1].min(),r.q[:,1].max()]).tolist(),
                  count=len(frame), reference=rows[0],
                  oracle_E0=frame.loc[frame.E0_Nm.idxmin()].to_dict(),
                  oracle_E2=frame.loc[frame.E2.idxmin()].to_dict(),
                  below_reference_E2=int((frame.E2 < 1).sum()),
                  max_decomposition_residual_Nm=float(frame.decomposition_residual_max_Nm.max()),
                  max_warning_count=int(frame.warnings.max()))
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    plot_results()


if __name__ == '__main__':
    main()
