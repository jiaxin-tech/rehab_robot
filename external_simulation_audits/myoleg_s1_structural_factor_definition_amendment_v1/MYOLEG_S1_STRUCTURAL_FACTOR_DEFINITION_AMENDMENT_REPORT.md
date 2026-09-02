# MyoLeg S1 Structural Factor Definition Amendment V1

## Formal outcome

**S1_STRUCTURAL_FACTOR_DEFINITION_AMENDED_READY_WITH_LIMITATIONS**

Amended identity: `S1_STRUCTURAL_DEFINITION_AMENDED_V1`  
Old S1 preserved SHA: `47ebf27c43ccca9621e315c1322946bb7b8687e098b243ee8d92f0f66d578394`  
Amended definition SHA: `3faf531f127bce1a26dd13b01dae07bc332bb107ac9d562c314f296780921763`

This stage used 401 frozen reference states only for mechanics-based projected moment-arm classification. It did not compute torque truth, J, oracle, rank, personalization, or held-out results. No pilot, subject, cohort, landscape, learner or optimizer was executed.

## Q1. Exact groups

- Biarticular lmax: `bflh_r, grac_r, recfem_r, sart_r, semimem_r, semiten_r, tfl_r`.
- `RECTUS_GROUP`: `recfem_r`.
- `HAMSTRING_GROUP`: `bflh_r, semimem_r, semiten_r`.
- Balance-unchanged biarticular: `grac_r, sart_r, tfl_r`.
- Hip monoarticular antagonist F0: `addmagDist_r, addmagIsch_r, addmagMid_r, glmax2_r, glmax3_r, glmed3_r, piri_r`.
- Knee monoarticular antagonist F0: `vasint_r, vaslat_r, vasmed_r`.

## Q2. Membership rule

Membership came from the projected actuator moment matrix in project hip/knee coordinates. Near-zero values below 1e-5 m were excluded from sign counts; target coverage had to reach 80%, sign consistency 95%, and a monoarticular non-target joint maximum had to remain at or below 1e-6 m. MuJoCo positive activation produces negative scalar muscle force, so unit positive muscle-tension effect is `-projected actuator moment`. Anatomy was reviewed afterward and did not override mechanical measurements.

`grac_r` is mechanically hamstring-like and `tfl_r` rectus-like, but neither belongs to the named anatomical family; both remain explicitly unchanged. `sart_r` has neither reciprocal family sign pattern. Hip members `glmed3_r` and `piri_r` remain mechanically clear but anatomically multi-action/ambiguous; this is a declared limitation rather than a hidden exclusion.

## Q3. Biarticular lmax operator

For every seven-muscle member, `lmax_i(z)=lmax_i0*exp(z)` and both gain/bias index 5 are assigned the same value. Each muscle uses its own nominal. `lmin`, `range`, `actuator_lengthrange`, geometry and transmission remain unchanged. This is a normalized MuJoCo curve parameter, not optimal fiber length or tendon slack length.

## Q4. Rectus-hamstring fpmax balance

`recfem_r` uses `fpmax_i(z)=fpmax_i0*exp(z)`; `bflh_r`, `semimem_r`, and `semiten_r` use `fpmax_i(z)=fpmax_i0*exp(-z)`. The family log-centers move symmetrically; `grac_r`, `sart_r`, and `tfl_r` remain unchanged. The V1 global biarticular fpmax factor is removed from a future personalization-focused scheme to avoid duplicated magnitude/balance degrees of freedom.

## Q5. Hip/knee F0 operators

Each included group member uses `F0_i(z)=F0_i0*exp(z)` relative to its own nominal, synchronized at gain/bias index 2. No muscle-specific weights are used. F0 scales the built-in active-plus-passive force expression; it is not labeled pure active strength.

## Q6. Nominal identity

All four operators recover the base targeted arrays bitwise at `z=0`.

## Q7. Field synchronization and integrity

At the unit-only `z=1e-8` check, only declared members/fields changed, gain/bias stayed synchronized, positivity and lmin/lmax domains passed, topology remained exact, forward state was finite, and warnings were zero. Integrity rows passed: **4/4**.

## Q8. V1 relationship

Femur, tibia/patella and foot mass/inertia are retained as background. Hip-only and knee-only common fpmax become secondary only. Common biarticular fpmax is removed from a future personalization-focused cohort. The amended factors replace those old factors' primary structural role, not their physiological meaning. Final Cohort V2 composition is not frozen here.

## Q9. Ambiguity and outcome independence

All four factors now have exact members, fields, mathematical operators, nominal identity, domain invariants and V1 relationships. Selection used field semantics, mechanics, invertibility and consistency only; no personalization outcome was read. Remaining limitations are range calibration, indirect physiological mapping, and two anatomically multi-action hip members.

## Q10. Pilot-design V2 readiness

**Yes, with limitations.** The amended SHA may be used as the authoritative input to `MYOLEG_STRUCTURAL_HETEROGENEITY_PILOT_DESIGN_V2`. That stage may freeze pilot diagnostic levels, trajectory subset and numeric scientific gates. It was not executed automatically.

## Stop state

- Scientific pilot/replay: **0 / 0**.
- New subjects/cohort/landscape: **none**.
- Held-out scientific access: **0**.
- Population range or pilot diagnostic level frozen: **no**.
- V1/V3/objective/normalization: **unchanged**.
- Robot/hardware: **untouched**.
