# Current initial-identification stop-rule audit

## Scope

This audit covers the temporary comparator in
`SAFEGUARDED_SEQUENTIAL_INITIAL_IDENTIFICATION_V1`. It does not approve a
real-subject or real-robot stopping rule.

| Existing item | Value | Classification | Finding |
|---|---:|---|---|
| Five-parameter numerical rank | 5 | FROZEN | Structural requirement: all five equivalent parameters must be supported. Numerical rank alone is not model adequacy. |
| Minimum singular value | 20 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; no frozen statistical, physical, or prospective source was found. |
| Maximum condition number | 50 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; not a reviewed release limit. |
| Maximum absolute parameter correlation | 0.30 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; not a reviewed release limit. |
| Maximum uncertainty proxy | 0.05 | RESEARCH_ONLY / UNJUSTIFIED | Dimensionless design-matrix proxy; not a confidence interval and not formally frozen. |
| Minimum per-parameter sensitivity | 20 | RESEARCH_ONLY / UNJUSTIFIED | Virtual comparison value; not a reviewed release limit. |
| Parameter stability limit | not defined | NOT_DEFINED | The previous comparator had no independent accumulated-estimate stability threshold. |
| Validation residual | 0.20 N·m | RESEARCH_ONLY / UNJUSTIFIED | A convenient virtual software comparator declared in code; no clinical, hardware, statistical, or preregistered source exists. It must not be inherited as a formal threshold. |
| Mechanical-objective validation error limit | not defined | NOT_DEFINED | The previous comparator did not gate on validation `e_J`. |
| NRMSE validation limit | not defined | NOT_DEFINED | The formal metric exists, but no acceptance limit is frozen. |

## 0.20 N·m conclusion

The value is retained only as an explicitly labeled comparison line. The
observed TRAIN+VALIDATION distribution cannot establish whether it is
scientifically too strict or too loose because the repository has no
independent definition of an acceptable mismatched model. It is therefore not
promoted, loosened to 0.45 N·m, or used to freeze this protocol.
