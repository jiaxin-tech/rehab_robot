# Readiness Dependency Graph

```mermaid
flowchart LR
  M[Measurement semantics] --> F[Frame, sign, point and bias validation]
  F --> T[Task-direction validation]
  T --> S[Synchronization and delay validation]
  S --> R[Repeated identical-trial repeatability]
  R --> E[Endpoint sensitivity and validation]
  E --> D[Repeated safe measured episodes]
  D --> G[Gray-box identification]
  G --> X[Residual analysis]
  X --> P{PINN stop/go gate}
  E --> B{BO stop/go gate}
  G --> B
```

Current stop is before frame/task-direction validation. The exact next stage is `WRENCH_FRAME_AND_TASK_DIRECTION_RESOLUTION_V1`. It must resolve physical wrench frame/sign/reference-point semantics and validate a task line of action before endpoint repeatability, identification, PINN or BO. This stage does not execute it.
