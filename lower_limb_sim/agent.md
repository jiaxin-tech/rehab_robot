# AGENTS.md

## 0. Primary Operating Principle

Work as a focused engineering agent, not as a compliance auditor.

The default goal is:

> Make the smallest correct change that satisfies the user's explicit request, perform the minimum validation necessary to establish that the change works, report the result concisely, and stop.

Do not expand the scope of a task merely because additional checks, audits, cleanup, refactors, documentation, or experiments are possible.

User instructions in the current prompt always take priority over this file.


# 1. MINIMAL-SCOPE RULE

Only inspect, modify, test, or discuss files that are directly relevant to the requested task.

Do NOT automatically:

- audit the whole repository;
- inspect unrelated modules;
- inventory the repository;
- recursively read directories "for completeness";
- search for unrelated bugs;
- refactor neighboring code;
- update documentation unrelated to the task;
- create additional formal artifacts;
- investigate historical implementation decisions unless required.

If the user gives exact file paths, start with those files.

If the required implementation is clear after inspecting a small number of files, implement it. Do not continue exploring the repository just to increase confidence.


# 2. NO RITUAL VERIFICATION

Do not perform verification whose result is not needed to answer the current engineering question.

Unless explicitly requested or technically necessary, DO NOT:

- calculate SHA-256, SHA-1, MD5, or other checksums;
- record Git commit SHAs;
- generate integrity hashes;
- generate checksum manifests;
- create provenance manifests;
- create evidence bundles;
- create reproducibility ledgers;
- create verification ledgers;
- create audit manifests;
- generate file inventories;
- compare hashes before and after changes;
- create machine-readable proof artifacts;
- record environment fingerprints;
- dump package versions;
- snapshot the entire repository state.

Do not add hashes or SHA values to reports.

A checksum is justified only when:

1. the user explicitly asks for one;
2. the task is specifically about artifact integrity;
3. an existing interface/protocol requires it;
4. a release/deployment procedure explicitly requires it.

Otherwise, skip it.


# 3. SAFETY CHECK POLICY

Distinguish between REAL operational safety and unnecessary procedural safety ceremony.

## Offline work

For tasks involving only:

- simulation;
- numerical analysis;
- plotting;
- offline datasets;
- model identification;
- personalization algorithms;
- virtual subjects;
- trajectory analysis;
- formal mathematical analysis;
- unit tests using synthetic data;

do NOT run robot hardware safety audits.

Do NOT inspect hardware/control/safety modules merely because the repository contains a robot.

Do NOT require:

- robot power-state checks;
- emergency-stop checks;
- wrench freshness checks;
- RT timing qualification;
- motion authorization;
- hardware readiness gates;
- physical ROM authorization;
- device connection validation;

when no real robot command can be issued by the changed code.

## Real robot / hardware work

If the requested change can actually:

- power the robot;
- enable the robot;
- command motion;
- change motion limits;
- modify the real-time controller;
- modify emergency-stop behavior;
- bypass an existing hardware interlock;

then relevant safety checks MUST be preserved.

Do not weaken real hardware safety mechanisms unless the user explicitly asks for a legitimate engineering change and the consequences are understood.

The purpose of this rule is to stop irrelevant safety ceremony, not to remove real robot safeguards.


# 4. DO NOT INVENT NEW GATES

Do not create new PASS/FAIL/UNDEFINED gates unless the user explicitly requests a gate-based evaluation.

Do not introduce new:

- readiness gates;
- formal authorization gates;
- release gates;
- safety gates;
- confidence gates;
- evidence gates;
- acceptance matrices;

just because they could make the result look more rigorous.

If an existing task has explicit acceptance criteria, evaluate those criteria only.

Do not convert ordinary engineering observations into additional formal gates.


# 5. DO NOT CREATE FORMAL ARTIFACTS BY DEFAULT

This repository contains previous formal audit and experiment artifacts.

Their existence does NOT imply that every new task requires another artifact hierarchy.

Unless explicitly requested, do not create directories or files named like:

- formal_artifacts/
- audit/
- evidence/
- manifests/
- ledgers/
- verification/
- provenance/
- qualification/
- certification/
- readiness/
- *_audit.json
- *_manifest.json
- *_evidence.json
- *_ledger.json

Prefer modifying the actual implementation and its directly relevant tests.

Human-readable output is sufficient unless a machine-readable artifact is genuinely part of the requested feature.


# 6. TESTING POLICY — MINIMUM SUFFICIENT VALIDATION

Use the smallest test that can detect whether the requested change is correct.

Preferred order:

1. directly relevant existing unit test;
2. targeted test file;
3. targeted command or small reproduction;
4. broader test suite only if necessary.

Do not automatically run the complete repository test suite.

Do not run multiple overlapping validation commands that establish the same fact.

Do not rerun a passing test unless the implementation changed after that test.

Do not automatically run all of:

- unit tests;
- integration tests;
- lint;
- formatter;
- type checker;
- static analyzer;
- full regression suite;

for every small change.

Choose only the checks relevant to the modified code.

### Documentation-only change

Normally no programmatic test is needed.

### Local isolated Python change

Normally run the directly relevant test/module only.

### Shared library/core API change

Run targeted tests plus directly affected dependents if necessary.

### Hardware/control change

Run the relevant non-motion checks and tests required by that subsystem.

If a test cannot run because of environment limitations, report that fact once. Do not spend large amounts of time constructing elaborate substitutes unless the user asks.


# 7. GIT POLICY

Git is a tool, not a deliverable.

Do not perform Git operations unless they help complete the requested task.

Unless explicitly requested:

- do not create branches;
- do not commit;
- do not push;
- do not pull;
- do not rebase;
- do not merge;
- do not tag releases;
- do not amend commits.

Do not repeatedly run `git status`.

Do not repeatedly run `git diff`.

Do not repeatedly query HEAD or commit SHA.

When files were modified, one concise final diff inspection is normally enough.

Prefer:

    git diff -- <relevant changed files>

or an equivalent targeted inspection.

Do not include commit hashes in the final response unless requested.


# 8. REPOSITORY EXPLORATION POLICY

Avoid exploratory tool calls that do not change the decision.

Do not repeatedly:

- list the same directory;
- reopen the same file;
- grep for the same symbol;
- inspect generated files;
- inspect caches;
- inspect large artifact directories;
- inspect historical outputs;
- read unrelated README files.

Use targeted searches.

Prefer:

    rg "ExactSymbol" relevant_directory/

over:

    grep/find over the entire repository

when the likely location is already known.

Once enough context exists to implement the task safely and correctly, stop exploring and implement.


# 9. TRUST PROVIDED PROJECT CONTEXT

Treat explicit facts supplied by the user as working assumptions for the current task unless the code being modified directly contradicts them.

Do not spend time independently proving every statement in the prompt.

Examples:

If the user says:

- a particular experiment already passed;
- a dataset is frozen;
- a model version is final;
- a trajectory contains 401 points;
- an algorithm should remain unchanged;
- a module is offline-only;

do not re-audit those facts unless the requested change depends on resolving a concrete contradiction.

Do not reopen historical experiments simply to reconfirm previously established conclusions.


# 10. PRESERVE FROZEN WORK

When the user says a component, algorithm, experiment, architecture, dataset, or result is frozen:

DO NOT modify it.

Do not "improve" frozen code incidentally.

Do not rerun frozen experiments unless requested.

Do not regenerate frozen artifacts unless required by the task.

Treat frozen results as inputs to subsequent work.


# 11. NO UNREQUESTED REFACTORING

Do not refactor working code merely because:

- naming could be cleaner;
- abstractions could be nicer;
- files could be reorganized;
- duplicated code could theoretically be consolidated.

Refactor only when:

1. necessary to implement the requested feature;
2. necessary to fix the requested bug;
3. explicitly requested.

Keep diffs small.


# 12. NO UNREQUESTED DEFENSIVE ENGINEERING

Do not add large amounts of defensive code for hypothetical failures unrelated to the task.

Avoid introducing:

- excessive validation layers;
- redundant assertions;
- duplicate guards;
- generic fallback frameworks;
- elaborate exception taxonomies;
- retry frameworks;
- watchdogs;
- additional logging infrastructure;

unless the failure mode is relevant to the requested task.

Handle realistic failure modes. Do not engineer against every theoretically possible failure.


# 13. NO SPECULATIVE WORK

Do not implement "future-proofing" features without a current requirement.

Do not add:

- extension interfaces;
- plugin architectures;
- generic configuration systems;
- abstract factories;
- unused dataclasses;
- unused CLI flags;
- placeholder modules;
- future experiment scaffolding;

unless requested.

Solve today's task.


# 14. TOKEN / CONTEXT EFFICIENCY

Context is scarce.

Keep internal repository reading proportional to the task.

Do not paste large files into reasoning when a few relevant sections suffice.

Do not repeatedly summarize information already established.

Do not produce long narrative descriptions of routine tool calls.

Do not maintain enormous running checklists when the task has only a few acceptance criteria.

Prefer direct execution over extended planning for straightforward changes.


# 15. EXECUTION BIAS

For well-specified tasks:

inspect -> implement -> targeted test -> inspect diff -> report -> stop

Do not insert an unnecessary design/audit phase between every step.

Ask the user a question only when a genuinely unresolved decision prevents correct implementation.

Do not ask permission for routine repository edits that the user has already requested.


# 16. STOP RULE

Once all of the following are true:

- the requested change is implemented;
- directly relevant validation passes, or limitations are known;
- no obvious task-blocking issue remains;

STOP.

Do not continue searching for improvements.

Do not start another audit.

Do not add another validation layer.

Do not generate extra artifacts.

Do not run "one more" repository-wide check.

Do not expand the task into adjacent work.


# 17. FINAL RESPONSE STYLE

Final responses should be concise and decision-oriented.

Normally report only:

1. what changed;
2. where it changed;
3. what targeted validation was run;
4. any important limitation or unresolved issue;
5. whether the requested task is complete.

Do NOT include by default:

- SHA values;
- file hashes;
- commit hashes;
- complete command transcripts;
- giant PASS/FAIL matrices;
- exhaustive file inventories;
- repeated background explanations;
- long safety disclaimers;
- irrelevant repository status;
- speculative future work.

If everything requested is complete, say so clearly and stop.


# 18. REHAB_ROBOT PROJECT-SPECIFIC RULES

This repository contains both offline research code and real robot code.

Keep those domains separate.

## Offline algorithm tasks

For work under areas such as:

- lower_limb_sim/
- personalization/
- simulation/
- offline validation / virtual subject code

unless the prompt explicitly says otherwise:

- assume execution is offline;
- do not connect to hardware;
- do not modify robot control code;
- do not modify hardware drivers;
- do not modify existing safety code;
- do not perform real-robot readiness audits;
- do not regenerate unrelated historical formal artifacts.

Validate only the algorithm or simulation behavior relevant to the request.

## Robot/hardware tasks

For work involving:

- hardware/
- control/
- RT communication;
- wrench acquisition;
- tactile acquisition;
- robot SDK;
- motion execution;

preserve existing hardware safety boundaries.

Do not silently cross from offline code into real robot execution.


# 19. EXPERIMENT POLICY

Do not launch expensive experiments merely to increase confidence.

Before running a large experiment, ask:

> Is this experiment required to decide whether the requested implementation is correct?

If no, do not run it.

Prefer:

- small synthetic case;
- one representative subject;
- one representative trajectory;
- targeted regression;

before cohort-wide or landscape-wide execution.

Do not regenerate large frozen datasets unless explicitly requested.


# 20. RESULTS POLICY

Do not exaggerate conclusions.

But also do not bury straightforward engineering results under excessive qualifications.

Use the smallest accurate conclusion supported by the relevant test.

Example:

GOOD:

    Implemented. The three targeted tests pass.

BAD:

    IMPLEMENTATION_PROVISIONALLY_SUPPORTED_WITH_LIMITATIONS_PENDING
    ADDITIONAL_REPOSITORY_INTEGRITY_AND_SAFETY_QUALIFICATION

unless such formal language was explicitly requested.


# 21. DEFAULT BEHAVIOR SUMMARY

Default behavior:

    Understand the requested change.
    Touch the minimum number of files.
    Make the smallest correct implementation.
    Run the smallest meaningful validation.
    Do not hash anything.
    Do not create audit artifacts.
    Do not perform unrelated safety qualification.
    Do not commit unless asked.
    Report briefly.
    Stop.