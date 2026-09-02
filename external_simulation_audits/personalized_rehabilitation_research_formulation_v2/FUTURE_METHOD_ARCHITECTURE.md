# Future Method Architecture

The diagram is conceptual. It does not authorize robot motion or a human trial.

```mermaid
flowchart LR
  subgraph OFF[Offline and fixed]
    REF[Measured reference and fixed task]
    V3[V3 P4 family: beta_flex, beta_extend]
    PHY[Analytical or gray-box physics prior]
    MYO[MyoLeg stress tests and feasibility support]
    SAFE[Independent reviewed safety and domain gate]
    REF --> V3
    MYO -. offline checks only .-> PHY
  end

  subgraph ON[Online per subject]
    SEL[Trajectory selector]
    EXEC[One complete robot trial]
    MEAS[Robot state and validated wrench; optional tactile or direct feedback]
    FEAT[Episode feature extraction and quality gates]
    OBS[Mechanical endpoint or direct preference observation model]
    SUBJ[Subject-specific gray-box posterior; optional gated residual]
    SEL -->|beta_k| SAFE
    SAFE -->|only if independently approved| EXEC
    EXEC --> MEAS --> FEAT
    FEAT --> OBS
    FEAT --> SUBJ
    SUBJ --> SEL
    OBS --> SEL
  end

  V3 --> SEL
  PHY --> SUBJ
```

## State ownership

- Fixed physics prior: analytical/gray-box structure; MyoLeg is offline-only support.
- Subject-specific state: valid episode ledger, effective parameter/posterior state, predictive uncertainty and optional residual state.
- Online selector state: evaluated beta set and mechanical/preference surrogate based only on causal observations.
- Safety layer: independent of the optimizer and unable to be relaxed by predicted reward.

`ALGORITHM_FORMULATION_READY != ROBOT_EXECUTION_READY`.
