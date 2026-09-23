# REEVALUATE_PERSONALIZATION_NECESSITY_WITH_E2_V1

## 1. Scope and evidence boundary

This stage is a pure offline reevaluation using the frozen five-leg MuJoCo replay data. It preserves the five leg parameter sets, five `SubjectROMProfile`s, the 625-point V3 candidate grid, the historical E0 values, and the frozen E2 formula. It does not modify or run any personalization algorithm, BO method, PINN training, robot interface, controller, or safety path.

The task label `BRANCH_BALANCED_REFERENCE_NORMALIZED_RMS` and the source-study name `BRANCH_BALANCED_RELATIVE_RMS` refer here to the same frozen endpoint:

\[
E2_i(\beta)=\max\{r_{hip,flex},r_{hip,extend},r_{knee,flex},r_{knee,extend}\},
\]

with every component normalized by the same leg's own \(\beta=(0,0)\) replay. No endpoint redesign or reweighting was performed.

All conclusions are limited to model-derived offline synthetic MuJoCo evidence. They are not real-robot, human-subject, clinical, comfort, safety, or effectiveness validation.

## 2. Complete E2 landscapes and oracle characterization (Q1–Q2)

All \(5\times625=3125\) candidate rows were evaluated. E0 is retained alongside E2 in `e2_full_landscapes.csv`.

| Leg | E2 oracle \((\beta_{flex},\beta_{extend})\) | Oracle E2 | Reference-to-oracle improvement | Relative range | CV | Near-oracle count at 0.1% / 0.5% / 1% / 2% / 5% |
|---|---:|---:|---:|---:|---:|---:|
| LEG_0_NOMINAL | (0, 0) | 1.000000000 | 0 (0%) | 1.597731% | 0.478663% | 182 / 304 / 500 / 625 / 625 |
| LEG_1_HEAVY_HIP_STIFF | (0, 0) | 1.000000000 | 0 (0%) | 1.458882% | 0.452818% | 182 / 323 / 525 / 625 / 625 |
| LEG_2_KNEE_DOMINANT | (0, 0) | 1.000000000 | 0 (0%) | 1.420780% | 0.339780% | 10 / 247 / 525 / 625 / 625 |
| LEG_3_NONLINEAR_COUPLED | (0, 0) | 1.000000000 | 0 (0%) | 1.355624% | 0.416330% | 182 / 323 / 525 / 625 / 625 |
| LEG_4_STRONG_STRUCTURAL_MISMATCH | (-0.03, 0.0225) | 0.999293659 | 0.000706341 (0.070684%) | 1.120720% | 0.334512% | 169 / 340 / 575 / 625 / 625 |

Q1: Four legs have the exact reference oracle \((0,0)\); only Leg 4 has a different oracle, \((-0.03,0.0225)\).

Q2: E2 increases the full-range spread to 1.121–1.598%, but the landscapes remain broad around their optima. Four legs have 304–340 candidates within 0.5%; the more selective Leg 2 still has 247 (39.5% of the grid), and every candidate is within 2% for every leg. Thus E2 is less compressed than E0 but still decision-flat at practically relevant tolerances.

## 3. Oracle diversity and cross-leg transfer (Q4)

There are two exact oracle structures: Legs 0–3 share \((0,0)\), while Leg 4 selects \((-0.03,0.0225)\). This diversity is real but is not, by itself, evidence that personalization is necessary.

Each cell below is `absolute regret / relative regret (%)` for applying the row leg's oracle to the column leg:

| Source oracle → target | Leg 0 | Leg 1 | Leg 2 | Leg 3 | Leg 4 |
|---|---:|---:|---:|---:|---:|
| Leg 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.000706341 / 0.070684 |
| Leg 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.000706341 / 0.070684 |
| Leg 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.000706341 / 0.070684 |
| Leg 3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.000706341 / 0.070684 |
| Leg 4 | 0.000576825 / 0.057683 | 0.000192878 / 0.019288 | 0.005285687 / 0.528569 | 0.000545851 / 0.054585 | 0 / 0 |

Q4: Only 8 of 25 cells have nonzero regret. The largest transfer loss is 0.528569%, caused by applying the exceptional Leg 4 oracle to Leg 2. All transfers among the four reference-oracle legs have zero regret. Conversely, transferring their shared oracle to Leg 4 costs only 0.070684%.

## 4. Universal and common trajectory analysis (Q5–Q6)

The cohort candidate minimizing mean E2 is the reference candidate, index 312 with \(\beta_{common}=(0,0)\).

| Leg | Individual oracle E2 | Common E2 | Absolute common regret | Relative common regret |
|---|---:|---:|---:|---:|
| Leg 0 | 1.000000000 | 1.000000000 | 0 | 0% |
| Leg 1 | 1.000000000 | 1.000000000 | 0 | 0% |
| Leg 2 | 1.000000000 | 1.000000000 | 0 | 0% |
| Leg 3 | 1.000000000 | 1.000000000 | 0 | 0% |
| Leg 4 | 0.999293659 | 1.000000000 | 0.000706341 | 0.070684% |
| Cohort summary | — | mean E2 = 1.000000000 | mean = 0.000141268 | mean = 0.014137%; max = 0.070684% |

Q5: Universal near-oracle candidates exist at every requested threshold: 8 at 0.1%, 228 at 0.5%, 500 at 1%, and all 625 at both 2% and 5%.

Q6: The common reference incurs exactly zero regret on four legs and 0.070684% on Leg 4. The cohort mean relative regret is only 0.014137%. Therefore a common trajectory remains an excellent solution on this benchmark.

## 5. Landscape ordering and E0 comparison (Q3)

| Leg pair | E0 Spearman | E2 Spearman | E0 gradient cosine | E2 gradient cosine |
|---|---:|---:|---:|---:|
| 0–1 | 0.999403 | 0.993239 | 0.997050 | 0.980556 |
| 0–2 | 0.992647 | 0.874288 | 0.993220 | 0.885251 |
| 0–3 | 0.993812 | 0.986788 | 0.993681 | 0.969575 |
| 0–4 | 0.995108 | 0.967609 | 0.995682 | 0.989555 |
| 1–2 | 0.991982 | 0.885116 | 0.993695 | 0.885572 |
| 1–3 | 0.993261 | 0.997678 | 0.994995 | 0.996430 |
| 1–4 | 0.993979 | 0.969620 | 0.993974 | 0.993911 |
| 2–3 | 0.999858 | 0.883495 | 0.999887 | 0.892040 |
| 2–4 | 0.999448 | 0.814659 | 0.999464 | 0.874585 |
| 3–4 | 0.999596 | 0.967037 | 0.999200 | 0.983740 |
| Minimum | 0.991982 | 0.814659 | 0.993220 | 0.874585 |
| Median | 0.994544 | 0.967323 | 0.995338 | 0.975066 |

Q3: The minimum pairwise Spearman falls by 0.177323 (0.991982 to 0.814659), while the median falls by 0.027221 (0.994544 to 0.967323). The minimum/median normalized-gradient cosine similarly fall by 0.118635/0.020273. The largest ordering change involves Leg 2, especially the Leg 2–Leg 4 pair; most other pairs remain strongly aligned.

## 6. Subject × trajectory interaction and direct endpoint comparison (Q9)

| Metric | E0 | E2 |
|---|---:|---:|
| Unique exact oracle count | 1 | 2 |
| Minimum / median Spearman | 0.991982 / 0.994544 | 0.814659 / 0.967323 |
| Minimum / median gradient cosine | 0.993220 / 0.995338 | 0.874585 / 0.975066 |
| Normalized interaction energy | 0.359991% | 5.228442% |
| Normalized interaction RMS | 0.012781 | 0.067660 |
| Raw two-way leg main effect | 99.996366% | 4.660002% |
| Raw two-way trajectory main effect | 0.001720% | 90.567199% |
| Raw two-way interaction | 0.001914% | 4.772799% |
| Maximum cross-leg transfer regret | 0% | 0.528569% |
| Nonzero transfer cells | 0 / 25 | 8 / 25 |
| Mean / max common regret | 0% / 0% | 0.014137% / 0.070684% |
| Universal candidates within 0.5% / 1% | 75 / 272 | 228 / 500 |

E2 does change the simulated decision structure: its normalized interaction energy is about 14.5 times E0, its minimum rank correlation is lower, and it produces one distinct oracle. However, the trajectory main effect still dominates the raw E2 decomposition (90.57%), the common reference remains exact for four legs, and E2 has more—not fewer—universal candidates than E0 at 0.5% and 1% because of its broad minimax near-optimal region.

Q9: The subject×trajectory interaction is numerically real and affects one cross-leg transfer direction, but it is not sufficiently decision-relevant to establish a need for sequential personalization in the current frozen five-leg benchmark. The decisive evidence is the tiny common-candidate penalty, not the interaction-energy increase alone.

## 7. Branch driver analysis (Q7)

| Leg | Exact-oracle active/worst branch | 0.1%-near candidates | Tie-aware active counts in that region: hip flex / hip extend / knee flex / knee extend |
|---|---|---:|---:|
| Leg 0 | all four tied at 1.0 | 182 | 31 / 140 / 13 / 1 |
| Leg 1 | all four tied at 1.0 | 182 | 45 / 125 / 14 / 1 |
| Leg 2 | all four tied at 1.0 | 10 | 6 / 3 / 3 / 1 |
| Leg 3 | all four tied at 1.0 | 182 | 64 / 106 / 14 / 1 |
| Leg 4 | hip flexion | 169 | 108 / 62 / 1 / 1 |

Counts are tie-aware, so a candidate can contribute to more than one branch count. Legs 0–3 select the reference because all four normalized branch components meet at exactly 1.0 there; their nearby non-reference landscape is most often limited by hip extension, except for the small Leg 2 near set where hip flexion is most frequent. At the Leg 4 oracle the components are hip flexion 0.999293659, hip extension 0.999271499, knee flexion 0.993168279, and knee extension 0.992524971. Thus all four branches improve relative to reference, but hip flexion limits E2 and sets the distinct Leg 4 preference.

Q7: A mechanically interpretable limiting-branch change exists for Leg 4, while the other four legs retain an exactly balanced reference optimum. This explains the one exceptional oracle but not a cohort-wide need for distinct trajectories.

## 8. Robustness-aware necessity (Q8)

The reused E2 perturbation check consists of two deterministic torque-wave perturbations at 0.25% of each reference joint peak and two deterministic half-rate sample grids. It is a fixed sensitivity check, not a validated sensor-noise model.

| Quantity | Result |
|---|---:|
| Minimum perturbed-vs-baseline E2 rank Spearman | 0.999250 |
| Exact E2 oracle stability | 100% |
| Maximum relative E2 endpoint variability | 0.018515% |
| Maximum transfer regret / maximum variability | 28.548× |
| Maximum common regret / maximum variability | 3.818× |
| Mean common regret / maximum variability | 0.764× |
| Leg 4 common regret / Leg 4 variability | 5.402× |

Q8: The worst cross-leg transfer penalty is clearly above the fixed perturbation variation, and Leg 4's individual-vs-common gain is about 5.4 times its own fixed-perturbation variation. Nevertheless, personalization's cohort-average gain (0.014137%) is below the worst observed endpoint variability (0.018515%), four legs gain exactly nothing, and even Leg 4 gains only 0.070684%. Robustness therefore confirms that the exceptional effect is reproducible in this replay; it does not make the aggregate personalization target strong.

## 9. Decision across criteria A–E and Q10

| Criterion | Finding |
|---|---|
| A. Oracle diversity | Two oracle structures, but four of five legs share reference. |
| B. Transfer penalty | Nonzero in 8/25 cells; maximum 0.528569%, concentrated in applying the Leg 4 oracle to Leg 2. |
| C. Common candidate penalty | Mean 0.014137%, max 0.070684%; zero for four legs. |
| D. Interaction magnitude | E2 normalized interaction rises to 5.228442%, but trajectory main effect and a large common near-optimal region dominate the decision. |
| E. Robustness | Exceptional transfer effects exceed fixed perturbation variation; average common-vs-individual benefit does not. |

The evidence shows a stable exceptional Leg 4 preference, but the frozen cohort still primarily shares a common optimum. Running sequential personalization algorithms would compare algorithms on a target whose aggregate benefit is too small and too concentrated to justify that next stage.

```text
REEVALUATE_PERSONALIZATION_NECESSITY_WITH_E2_V1 = COMPLETE
E2_PERSONALIZATION_NECESSITY = NOT_SUPPORTED
READY_FOR_E2_FROZEN_ALGORITHM_COMPARISON = false
```

## 10. Outputs and tests

- Machine-readable summary: `lower_limb_sim/five_leg_mujoco_v1/results_e2_necessity_v1/study_summary.json`
- Full E0/E2 landscapes: `lower_limb_sim/five_leg_mujoco_v1/results_e2_necessity_v1/e2_full_landscapes.csv`
- Oracle, 5×5 transfer, common-regret, branch-driver, E0/E2 comparison, perturbation, and all-pairs similarity tables: `lower_limb_sim/five_leg_mujoco_v1/results_e2_necessity_v1/table_*.csv`
- Analysis implementation: `lower_limb_sim/five_leg_mujoco_v1/e2_necessity.py`
- Offline runner: `lower_limb_sim/five_leg_mujoco_v1/run_e2_necessity.py`
- Targeted test: `tests/test_e2_personalization_necessity_v1.py`

Validation command:

```text
PYTHONPATH=/private/tmp/rehab_robot_mujoco_deps:. pytest -q tests/test_e2_personalization_necessity_v1.py
```

Result: `8 passed in 0.62s`.

No commit was created. No next-stage algorithm comparison was started.
