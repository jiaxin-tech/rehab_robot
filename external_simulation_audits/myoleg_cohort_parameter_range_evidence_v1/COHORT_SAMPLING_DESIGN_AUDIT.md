# Cohort sampling design audit

| design | advantage | limitation | decision |
|---|---|---|---|
| deterministic predefined profiles | interpretable | weak six-dimensional coverage | retain for ablations, not primary cohort |
| Latin hypercube | one point per marginal stratum and works at n=32 | does not create a physiological joint distribution | selected with deterministic centered maximin construction |
| Sobol / low discrepancy | strong rectangular-space coverage | awkward exact nominal anchor and split semantics; still assumes a box | acceptable alternative, not selected |
| factorial extremes + nominal | clear corners | extreme-heavy and combinatorial | use only the five preregistered integrity corners |
| hybrid | can mix anchors and coverage | more design degrees of freedom and post-hoc discretion | reject for V1 |

Freeze a centered maximin Latin hypercube: six dimensions, 32 heterogeneous
profiles, NumPy PCG64 seed `20260830`, 512 permutation restarts and a
lexicographic tie-break.  The existing nominal base model is evaluated as a
separate control.  Held-out indices are every fourth generated row starting at
index 3 (`3,7,...,31`) and are frozen before learner performance is revealed.

The design fills the rectangular *model-parameter* ranges.  It does not assert
that factors are statistically independent in humans.  Anthropometric factors
are plausibly correlated and passive-group factors may be correlated, but no
quantitative covariance is defensible for these exact MyoLeg scales.  Therefore
V1 does not invent a covariance matrix and must report the independence-like
marginal design as a synthetic limitation.
