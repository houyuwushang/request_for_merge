# Private-DE v2 Specification

## 1. Purpose

This document defines the intended algorithmic behavior for a clean, reproducible implementation of **Private-DE**.

This spec is the source of truth for the new implementation.

- The existing repository is treated as a **legacy / partial / experimental implementation**
- The paper is treated as the conceptual reference
- If the legacy code conflicts with this spec, the new implementation should follow **this spec**
- Any unresolved ambiguity must be recorded explicitly in `ASSUMPTIONS.md`

---

## 2. High-level goal

Private-DE is an iterative differentially private synthetic tabular data generation framework under a **select-measure-generate** paradigm.

At a high level, the algorithm:

1. constructs candidate query vectors from a workload
2. selects a query vector privately
3. measures the selected vector privately
4. converts query-level signed errors into record-level fitness
5. updates synthetic data through directed evolutionary operators
6. repeats until the privacy budget is exhausted

The design goal is to combine:

- privacy-aware query selection
- efficient noisy measurement
- record-level directed mutation
- population-level exploration via crossover

---

## 3. Scope of the new implementation

The new implementation should support:

- tabular datasets only
- discrete / discretized attributes
- configurable query workloads
- reproducible runs under fixed seeds
- explicit privacy accounting under zCDP
- ablations for key algorithmic components

The new implementation does **not** need to preserve all legacy APIs.

---

## 4. Source of truth and conflict resolution

Priority order:

1. `SPEC.md`
2. explicitly documented resolutions in `ASSUMPTIONS.md`
3. paper intent
4. legacy code

If the legacy code behaves differently from this spec:

- do **not** silently preserve the old behavior
- record the mismatch in `SPEC_GAP_REPORT.md`
- implement the behavior defined here

---

## 5. Data model

Let:

- `D` be the true tabular dataset
- `|D| = N` be the number of records
- `D_hat_t` be the synthetic dataset at iteration `t`

Each record is represented as a vector of attribute values.

Attributes may be:

- categorical
- ordinal / numeric after discretization into bins

Continuous attributes must be discretized before workload construction.

The implementation must make discretization explicit and configurable.

---

## 6. Query model

A query `q_phi` is defined by a predicate `phi(x)` over a record `x`.

The query answer on dataset `D` is:

`q_phi(D) = sum_{x in D} phi(x)`

or its normalized equivalent if the implementation chooses to work with frequencies.

The implementation must be explicit and consistent about whether answers are:

- counts, or
- normalized frequencies

This choice must be documented and used consistently in:
- selection
- measurement
- loss computation
- mutation logic
- evaluation

---

## 7. Workload construction

The implementation must support explicit workload construction.

Supported workload families:

1. **1-way marginals**
2. **2-way marginals**
3. **3-way marginals**
4. **range queries**
5. **prefix queries**
6. **conditional prefix queries**
7. **halfspace-style paired queries** where applicable

The workload configuration must be user-controlled through config / CLI.

The implementation must not hard-code a single workload choice in the main entrypoint.

---

## 8. Orthogonal query vectors

Candidate queries are grouped into **query vectors** `V`.

A query vector is intended to contain mutually exclusive / orthogonal predicates so that:
- each record matches at most one predicate in the vector, or
- the grouping satisfies the intended orthogonality property used by the mutation logic

Examples include:
- categories from the same categorical marginal
- complementary prefixes / ranges
- paired halfspaces where such grouping is well-defined

### Required behavior

- Query grouping must be explicit and inspectable
- The implementation must record how many queries are grouped into each vector
- The implementation must support a no-orthogonal-grouping ablation
- If strict orthogonality cannot be guaranteed for a query family, the implementation must either:
  - reject that grouping, or
  - mark it as approximate and document the behavior explicitly

### Important note

The implementation must not assume orthogonality silently.
Any approximation must be documented.

---

## 9. Initialization

Initialization must follow the paper-level intent:

1. privately measure all 1-way marginals first
2. use these measurements to construct an initial synthetic dataset `D_hat_0`

The implementation may choose a simple defensible initializer, for example:
- sample records from the product of measured 1-way marginals
- or another clearly documented initializer

The exact initializer must be documented in `ASSUMPTIONS.md` and `README.md`.

---

## 10. Privacy model

The implementation must use **zCDP** as the primary accounting framework.

Privacy must be tracked explicitly over all rounds.

The privacy accountant must separately account for at least:

- initialization measurements
- query-vector selection
- per-round Gaussian measurement

### Requirements

- privacy spent must be monotonic
- total privacy spent must never exceed the configured budget
- the code must log privacy spent per round
- the code must expose final total privacy spent

If a conversion to `(epsilon, delta)` is reported for interpretability, it must be clearly marked as derived from zCDP.

---

## 11. Iterative select-measure-generate loop

At each round `t`, the algorithm performs:

1. build / maintain a candidate set of query vectors
2. privately select one query vector `V_t`
3. privately measure answers for `V_t`
4. combine new and previous measurements if needed
5. compute query-level errors on the selected vector
6. convert them into record-level fitness
7. update the synthetic dataset / population
8. record metrics and continue until the budget is exhausted

This structure must be clear in code.

---

## 12. Selection score

Selection is performed over candidate query vectors.

The implementation should expose the quality / utility score used for selection.

If the current method uses a score of the form:

`q_s(V, D) = c_V^3 * (||V(D)||_2 / |D|) * ||V(D) - V(D_hat_t)||_1`

or a closely related form, the exact expression must be:

- implemented in a dedicated module
- logged / inspectable
- configurable where appropriate

### Requirements

- selection must use a numerically stable exponential mechanism implementation
- sensitivity assumptions must be explicit
- score normalization / clipping must be documented if applied
- large candidate sets must not cause silent numerical overflow

---

## 13. Measurement

The selected query vector is measured using a Gaussian mechanism under zCDP.

### Requirements

- Gaussian noise scale must be derived explicitly from the chosen zCDP allocation
- each measurement step must record:
  - selected vector
  - allocated privacy
  - noise scale
  - measured noisy answers

The implementation must keep measurement logic separate from selection and update logic.

---

## 14. Repeated measurements and inverse-variance weighting

If a query or a derived marginal has been measured more than once, the implementation may combine estimates using inverse-variance weighting.

### Requirements

- the combination rule must be explicit
- variances must be tracked
- repeated measurements must not be merged silently
- the no-inverse-variance-weighting ablation must be supported

If this feature is only partially supported, that must be documented explicitly.

---

## 15. Signed error convention

This section is **non-negotiable**.

The new implementation must use:

`Err(q) = noisy_answer(q) - synthetic_answer(q)`

Interpretation:

- `Err(q) > 0` means the synthetic data **underestimates** the noisy target
- `Err(q) < 0` means the synthetic data **overestimates** the noisy target

This sign convention must be consistent in:
- fitness computation
- donor/recipient selection
- mutation direction
- diagnostics
- tests
- comments and documentation

### Important correction

If any legacy code or paper text interprets the sign the opposite way, the new implementation must follow **this corrected convention**.

---

## 16. Record-level fitness

For a selected query vector `V`, record-level fitness is defined as:

`f_V(x) = sum_{q_phi in V} Err(q_phi) * phi(x)`

Interpretation:

- records with positive fitness are associated with underrepresented regions
- records with negative fitness are associated with overrepresented regions

This fitness is used to guide directed mutation.

### Requirements

- fitness computation must live in a dedicated module
- per-record fitness must be inspectable for debugging
- tests must verify fitness sign behavior on small toy datasets

---

## 17. Directed mutation

Directed mutation operates in record space rather than parameter space.

The update should identify:

- donor-like states / records associated with overrepresented query regions
- recipient-like states / records associated with underrepresented query regions

Then it should mutate records in a direction intended to reduce the selected-vector loss.

### Required behavior

The implementation must make the mutation logic explicit:

1. determine candidate donor and recipient states
2. choose the record(s) to modify
3. define how attributes are changed:
   - direct replacement
   - nearest / closest record transition
   - nudge-style update
   - resampling among valid target states

### Requirements

- "closest" must be formally defined if used
  - e.g. Hamming distance over discrete attributes
- mutation count per round must be explicit and configurable
- mutation order must be explicit
- the code must not hide multiple mutation steps behind a vague helper

### Guarantee language

The implementation must **not** claim unconditional loss decrease unless the exact conditions are satisfied.
If the implementation only supports a local / conditional decrease interpretation, that must be documented.

---

## 18. Crossover and population-level exploration

The implementation may maintain multiple synthetic datasets / islands.

Crossover is intended to improve exploration beyond local mutation.

### Requirements

- crossover must be implemented as a separate operator
- the implementation must support disabling crossover entirely
- if context-aware crossover is used, the context and selection rule must be explicit
- population size must be configurable
- diversity-related diagnostics are desirable

The code must not merge crossover logic into mutation logic.

---

## 19. Loss and optimization diagnostics

The implementation must define clearly what loss is being monitored.

At minimum, for the selected vector `V_t`, the code should track:

- noisy target answers
- synthetic answers
- signed errors
- an aggregate vector loss

The code should also track global run-level diagnostics where feasible.

### Required logs per round

- round index
- selected vector id
- privacy spent this round
- cumulative privacy spent
- vector loss before update
- vector loss after update if available
- runtime for the round

---

## 20. Complexity and runtime transparency

The implementation must make key runtime costs visible.

At minimum, document or estimate the complexity of:

- workload construction
- query-vector grouping
- selection scoring
- exponential mechanism selection
- fitness computation
- mutation
- crossover

Wall-clock timing must be logged for major phases.

---

## 21. Ablation support

The implementation must provide flags for the following ablations:

- `no_directed_mutation`
- `no_crossover`
- `no_orthogonal_grouping`
- `no_inverse_variance_weighting`

Optional additional ablations:

- random mutation instead of directed mutation
- single-population instead of island model
- fixed vector choice instead of private selection

Ablations should be easy to run from the CLI.

---

## 22. Evaluation support

The implementation should support at least the following evaluation hooks:

- MSE
- TVD
- downstream ML utility
- privacy-related risk metric(s)

### Important requirement

Any privacy-related risk metric must be formally named and defined.

Do not use vague labels like "disclosure risk" without:
- attacker model
- attack procedure
- parameters
- interpretation

If the implementation includes such a metric, the code and README must document it explicitly.

---

## 23. Reproducibility requirements

The implementation must be reproducible.

### Required controls

- global random seed
- framework-specific seeds
- deterministic mode where feasible
- config file support
- exact CLI command reproducibility

The README must include at least one minimal reproducible example.

---

## 24. Code structure requirements

The new implementation should live in a clean path, for example:

private_de_v2/
- `__init__.py`
- `config.py`
- `data.py`
- `queries.py`
- `selection.py`
- `measurement.py`
- `privacy.py`
- `fitness.py`
- `mutation.py`
- `crossover.py`
- `generator.py`
- `evaluation.py`
- `cli.py`
- `tests/`

Legacy code should be preserved and clearly marked as:
- legacy
- baseline
- experimental
- deprecated

---

## 25. Required tests

At minimum, add tests for:

1. signed error convention correctness
2. donor/recipient mutation direction
3. orthogonal query-vector construction correctness
4. privacy budget monotonicity
5. deterministic behavior under fixed seeds
6. numerical stability of exponential mechanism
7. workload construction correctness
8. inverse-variance weighting correctness if enabled

Small toy-dataset tests are preferred.

---

## 26. Known ambiguities to document in ASSUMPTIONS.md

The following must be resolved explicitly during implementation:

- count vs frequency query answers
- exact initialization method from 1-way marginals
- exact form of the selection score and its constants
- exact mutation count per round
- exact definition of "closest" record / state
- how incomplete or approximate orthogonality is handled
- how repeated measurements are stored and merged
- whether and how halfspace query pairing is implemented
- how crossover parents are chosen
- whether projection-based components are fully implemented or removed

---

## 27. Non-goals

The new implementation does not need to:

- preserve undocumented legacy behavior
- optimize for maximum speed before correctness
- support every historical experimental script
- claim stronger theory than the code actually implements

---

## 28. Final principle

Prefer:

- correctness
- transparency
- inspectability
- reproducibility

over:

- backward compatibility
- hidden heuristics
- undocumented shortcuts
- paper-style overclaiming