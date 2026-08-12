# Manuscript writing rules

1. Write concise IEEE-style engineering English.
2. State the evidence level whenever a result could be mistaken for physical or
   clinical evidence: analytical, simulation, software validation, offline
   synthetic experiment, physical dummy, real robot, or human.
3. Every quantitative manuscript statement must cite a FORMAL result ID from
   `results_registry.md` before it enters Section VI or the abstract.
4. Unit tests verify software behavior only. They must not be described as model,
   robot, or experimental validation.
5. Do not promote `/tmp`, scratch, debug, or temporary files.
6. Preserve `theta_shank = q_hip - q_knee` in every formula and figure.
7. Preserve the implemented hip upper bound of 120 deg. Keep the knee ROM
   explicitly configured until the 5–130 vs 5–145 pipeline split is resolved.
8. Describe `L2` as knee-to-strap-equivalent-traction-point distance.
9. A closed reference needs `q_ref(0) = q_ref(1)`; do not impose time-reversal
   symmetry. The active reference is explicitly asymmetric.
10. Use “equivalent” for estimated mass scale, passive stiffness, and passive
    damping. Do not imply unique physiology.
11. Keep the Method sections free of numerical results. Numerical constants that
    define the implemented model, solver, or constraint are allowed.
12. Add `% TODO: citation` rather than inventing a literature citation.
13. Keep timing and model mismatch as supporting validation, not independent
    headline contributions.
14. Keep tactile sensing as framework/TODO until calibrated data exists.
15. Do not describe the fixed C0–C8 Pareto comparison as a finalized
    subject-specific optimizer.
16. Do not describe static SDK signatures, fake adapters, previews, or offline
    preflight as real-robot validation.
17. Do not use Stage labels as manuscript section titles; retain them only in
    internal registries and source maps.

## Traceability convention

Draft paragraphs and displayed equations in Sections III and IV carry `% Trace:`
comments. The matching source-map entry must identify implementation files,
functions/classes, configuration, results if any, confidence, and assumptions.

## Result-status convention

- **FORMAL:** Reproducible repository experiment with primary result artifacts
  and a generating implementation; interpretation remains limited to its stated
  evidence level.
- **PRELIMINARY:** Useful evidence with a known unresolved input, calibration,
  gate, or completeness issue.
- **DEBUG:** Software/debug evidence only; never a scientific result.
- **TEMPORARY:** Scratch or transient result; excluded from manuscript claims.
- **SUPERSEDED:** Replaced by a newer formal result; retained only for provenance.
