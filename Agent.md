# AGENTS.md

## Project instructions for Codex and other coding agents

This repository contains a research implementation related to **Private-DE**.

You must read this file before making changes.

You must also read:

1. `SPEC.md`
2. the current repository structure
3. any existing docs relevant to implementation status

---

## 1. Mission

Your job is to help build a **clean, reproducible, paper-aligned implementation** of Private-DE.

The current repository contains legacy and experimental code.
Do not assume the existing implementation is correct.

The goal is **not** to preserve accidental historical behavior.
The goal is to produce a reliable implementation aligned with `SPEC.md`.

---

## 2. Source of truth

Priority order:

1. `SPEC.md`
2. explicit resolutions written into `ASSUMPTIONS.md`
3. current repository code

If code and `SPEC.md` disagree:

- follow `SPEC.md`
- report the mismatch explicitly
- do not silently preserve legacy behavior

If `SPEC.md` appears ambiguous:

- stop
- document the ambiguity
- propose a resolution in `ASSUMPTIONS.md`
- proceed only with the simplest defensible interpretation

---

## 3. First step: analyze before rewriting

Before making major edits, produce:

- `REPO_MAP.md`
- `SPEC_GAP_REPORT.md`
- `REFACTOR_PLAN.md`
- `ASSUMPTIONS.md` (initial version if needed)

Do not begin with a giant uncontrolled rewrite.

You should first identify:

- major entrypoints
- duplicate logic
- dead / experimental code
- hard-coded parameters
- paper-code mismatches
- modules that should be isolated as legacy

---

## 4. Refactoring strategy

Implement the new code incrementally in a **new path** such as:

`private_de_v2/`

Do **not** destroy the legacy codebase.

Legacy code should either:
- stay untouched, or
- be moved/marked clearly as `legacy/`, `baseline/`, or `experimental/`

Prefer additive refactoring over destructive rewriting.

---

## 5. Required implementation order

Work in this order unless there is a strong reason not to:

1. config and CLI
2. dataset abstraction and preprocessing
3. query and workload abstraction
4. privacy accountant
5. orthogonal query-vector construction
6. private selection
7. private measurement
8. record-level fitness
9. directed mutation
10. crossover / population handling
11. evaluation hooks
12. tests
13. README reproducibility instructions

Do not jump straight into optimizing the full pipeline before the foundations are clean.

---

## 6. Non-negotiable correctness rules

### 6.1 Signed error convention

The implementation must use:

`Err(q) = noisy_answer(q) - synthetic_answer(q)`

Interpretation:

- `Err(q) > 0`: synthetic underestimates target
- `Err(q) < 0`: synthetic overestimates target

If legacy code or comments conflict with this, fix the implementation to match the spec.

### 6.2 Privacy accounting

Privacy must be explicit and monotonic.

Never hide privacy spending inside helper functions without exposing it.

Always make it possible to inspect:
- spend per round
- cumulative spend
- total configured budget

### 6.3 Inspectability

Important intermediate values must be inspectable, especially:
- selected query vector
- noisy measurements
- synthetic answers
- signed errors
- per-record fitness
- mutation decisions

### 6.4 Reproducibility

Every experiment path must support:
- fixed seeds
- config-driven settings
- exact CLI reproducibility

Remove hard-coded `parse_args(args=[...])` style experiment stubs from the new code path.

---

## 7. Testing philosophy

Add tests early.

At minimum, tests must cover:

- signed error convention
- donor/recipient mutation direction
- orthogonal grouping behavior
- privacy budget monotonicity
- deterministic behavior under fixed seeds
- numerical stability of exponential mechanism
- workload construction correctness

Use small toy datasets whenever possible.

Prefer simple deterministic tests over fragile large end-to-end tests.

---

## 8. Documentation requirements

Whenever you make a meaningful structural decision, update the relevant docs.

Required docs to maintain:

- `SPEC_GAP_REPORT.md`
- `ASSUMPTIONS.md`
- `README.md`
- `TEST_PLAN.md` if added

If you resolve an ambiguity, record it.
Do not bury important behavior only in code.

---

## 9. Handling ambiguities

If you encounter any of the following, do not guess silently:

- unclear count vs frequency conventions
- unclear initialization rule
- unclear mutation count per round
- unclear meaning of "closest"
- approximate rather than strict orthogonality
- incomplete projection-related code
- unclear measurement-combination rule
- unclear crossover parent selection

Instead:

1. describe the ambiguity
2. describe current code behavior
3. describe the spec intent
4. propose the simplest defensible implementation
5. record it in `ASSUMPTIONS.md`

---

## 10. Handling legacy code

The legacy code is useful as:
- a reference
- a baseline
- a source of utility functions

But it is **not** automatically correct.

Do not:
- propagate poor structure into the new implementation
- keep duplicated logic just for compatibility
- preserve hidden behavior without documenting it

Do:
- isolate reusable pieces cleanly
- copy small utilities only when justified
- clearly label deprecated paths

---

## 11. Coding style expectations

Prefer:

- small focused modules
- explicit data flow
- typed function signatures where practical
- config-driven behavior
- pure functions for algorithmic logic where possible
- separation of algorithm logic from device/framework glue

Avoid:

- monolithic `main.py`
- hidden global state
- mixed experiment + library logic
- silent magic constants
- framework-specific hacks leaking into core algorithm code

---

## 12. Logging expectations

For iterative runs, log at least:

- round index
- selected vector id
- privacy spent this round
- cumulative privacy spent
- vector loss before update
- vector loss after update if available
- runtime

Logs should help debug algorithmic behavior, not just final outputs.

---

## 13. Ablation support

The new code path must make it easy to disable major components.

Provide flags for at least:

- `no_directed_mutation`
- `no_crossover`
- `no_orthogonal_grouping`
- `no_inverse_variance_weighting`

Optional but useful:
- random mutation baseline
- fixed vector selection baseline
- single-population baseline

---

## 14. Safe implementation behavior

Before making a risky structural change:

- explain briefly what you are about to change
- explain why it is needed
- mention affected files/modules

After each major step:

- summarize what changed
- summarize what remains
- mention any unresolved ambiguity

If a contradiction in `SPEC.md` blocks correct implementation, stop and report it.

---

## 15. What success looks like

A successful result has all of the following:

- a clean new implementation path
- legacy code preserved
- no silent mismatch with `SPEC.md`
- tests for the most failure-prone logic
- config-driven reproducible runs
- minimal reproducible example in README
- explicit documentation of assumptions and remaining limitations

---

## 16. Practical instruction

Do not try to be clever by preserving undocumented behavior.

When in doubt, choose:
- the simpler implementation
- the more inspectable implementation
- the more testable implementation
- the more reproducible implementation

If a behavior cannot be defended clearly, document it rather than hiding it.