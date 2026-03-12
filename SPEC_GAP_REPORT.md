# Spec Gap Report

## Scope

This report compares the repository after the refactor against the intended Private-DE specification in `SPEC.md`.

Important distinction:

- `private_de_v2/` is the active implementation path.
- top-level PT scripts and `src/genetic_sd/` remain in the repo as legacy or baseline references.
- unresolved gaps below refer to the new v2 path unless stated otherwise.

## Executive Summary

`private_de_v2/` now implements the requested architecture:

- explicit configuration and CLI
- explicit discrete data model
- explicit workload and orthogonal query-vector construction
- explicit zCDP accountant
- stable exponential mechanism selection
- Gaussian measurement
- inverse-variance weighting for repeated direct measurements
- exact signed error convention
- record-level fitness in a separate module
- directed mutation and separate crossover
- per-round metrics and deterministic tests

The main remaining gaps are not hidden mismatches; they are documented simplifications:

1. selection uses a clipped L1 frequency-gap utility rather than a paper-specific constant-heavy score
2. halfspace queries are implemented as a simplified paired-threshold family
3. inverse-variance weighting is only implemented for repeated direct query measurements, not derived marginals
4. `no_orthogonal_grouping` mutation uses a documented complement fallback
5. downstream ML utility is only an optional hook when `scikit-learn` is available

## Summary Table

| Spec area | Legacy repo before refactor | `private_de_v2` status | Remaining gap |
| --- | --- | --- | --- |
| Iterative select-measure-generate under zCDP | missing in active PT path | Implemented in `private_de_v2/generator.py` | None |
| Orthogonal query-vector construction | missing | Implemented in `private_de_v2/queries.py` | Singleton ablation is intentionally non-grouped |
| Exponential mechanism query-vector selection | heuristic or unused | Implemented in `private_de_v2/selection.py` | Utility is a simplified clipped L1 score |
| Gaussian measurement and privacy accounting | fragmented/inconsistent | Implemented in `private_de_v2/measurement.py` and `private_de_v2/privacy.py` | None for direct measurements |
| Inverse-variance weighting | missing | Implemented for repeated direct query measurements | No derived-marginal fusion |
| Record-level fitness | heuristic dominance score | Implemented exactly in `private_de_v2/fitness.py` | None |
| Signed error convention | partially aligned | Implemented consistently as `noisy - synthetic` | None |
| Directed mutation | heuristic query-by-query transfer | Implemented explicitly in `private_de_v2/mutation.py` | Singleton ablation uses complement fallback |
| Crossover / population exploration | heuristic row transplant | Separate operator in `private_de_v2/crossover.py` | Context is limited to active-vector fitness and measured-loss best population |
| Initialization behavior | random / inconsistent | Private 1-way product-of-marginals initializer | Simpler than more advanced structured initializers |
| Evaluation metrics and reproducibility | weak / hard-coded | Implemented in `private_de_v2/evaluation.py`, `README.md`, `TEST_PLAN.md` | Downstream ML utility depends on optional dependency |

## Detailed Status

### 1. Iterative select-measure-generate loop under zCDP

Resolved in v2:

- `private_de_v2/generator.py` now owns the full outer loop:
  1. initialization measurement
  2. private vector selection
  3. private vector measurement
  4. repeated-measurement aggregation
  5. signed error computation
  6. record-level fitness
  7. directed mutation
  8. crossover
  9. round logging

Legacy mismatch remains documented but isolated:

- `main.py`, `primary.py`, and `mygenerator.py` still reflect the old one-shot or heuristic workflow.

### 2. Orthogonal query-vector construction

Resolved in v2:

- `private_de_v2/queries.py` introduces:
  - `Query`
  - `QueryVector`
  - `QueryWorkload`
- Supported workload families:
  - 1-way marginals
  - 2-way marginals
  - 3-way marginals
  - range partitions
  - prefix pairs
  - conditional-prefix triples
  - halfspace-style pairs

Documented simplification:

- Halfspace vectors are a simplified two-feature thresholded-sum family, not a general halfspace engine.

### 3. Exponential mechanism query-vector selection

Resolved in v2:

- `private_de_v2/selection.py` implements:
  - explicit utility scoring
  - explicit sensitivity
  - log-sum-exp style probability stabilization
  - sampling via a seeded RNG

Documented simplification:

- The score is a clipped L1 frequency gap, not a literal implementation of the paper-style `c_V^3 * ...` form.
- This was an explicit ambiguity resolution recorded in `ASSUMPTIONS.md`.

### 4. Gaussian measurement and privacy accounting

Resolved in v2:

- `private_de_v2/privacy.py` provides a monotonic zCDP accountant.
- `private_de_v2/measurement.py` uses:
  - singleton sensitivity `1 / N`
  - grouped-vector sensitivity `sqrt(2) / N`
  - `sigma = sensitivity / sqrt(2 * rho)`
- Per-round metrics include:
  - selected vector id
  - rho spent
  - cumulative rho
  - noise scale
  - loss and runtime

### 5. Repeated measurements and inverse-variance weighting

Resolved in v2 for direct repeated measurements:

- `MeasurementStore` tracks:
  - current estimate
  - variance
  - measurement count
  - history
- repeated direct measurements of the same query id are combined by inverse-variance weighting
- `--no-inverse-variance-weighting` switches to latest-measurement behavior explicitly

Remaining gap:

- derived marginal inference and fusion is not implemented

### 6. Record-level fitness

Resolved in v2:

- `private_de_v2/fitness.py` implements:

```text
f_V(x) = sum_{q in V} Err(q) * phi_q(x)
```

- For orthogonal vectors, each record receives the signed error of its matched state.
- Unit tests verify sign correctness and row-level behavior.

### 7. Signed error convention

Resolved in v2:

- `private_de_v2.fitness.compute_signed_errors(...)` uses:

```text
Err(q) = noisy_answer - synthetic_answer
```

- Tests verify positive means synthetic underestimates and negative means synthetic overestimates.

### 8. Directed mutation using donor/recipient states

Resolved in v2 for grouped vectors:

- donor states are negative-error query states
- recipient states are positive-error query states
- mutation projects a donor row into the recipient query state explicitly

Documented approximation:

- Under `no_orthogonal_grouping`, singleton vectors use a complement-based fallback because grouped donor/recipient states are intentionally absent in that ablation.

### 9. Crossover / population-level exploration

Resolved in v2 at the module/API level:

- `private_de_v2/crossover.py` is a separate operator
- population size is configurable
- crossover can be disabled fully

Remaining simplification:

- crossover chooses the donor population via lowest measured-query MSE and uses active-vector fitness to identify donor and recipient rows
- this is explicit and inspectable, but still simpler than richer diversity-aware population management

### 10. Initialization behavior

Resolved in v2:

- all 1-way marginals are measured first
- the initial synthetic populations are sampled from the product of those private marginals

Remaining simplification:

- no alternative initializer family has been added yet

### 11. Evaluation metrics and reproducibility

Resolved in v2:

- `README.md` includes an exact minimal run command
- `configs/private_de_v2_minimal.yaml` provides a minimal config
- `TEST_PLAN.md` documents local test execution
- `private_de_v2/tests/` covers the required core behaviors
- seeding covers:
  - Python `random`
  - NumPy
  - Torch if installed

Remaining limitation:

- downstream ML utility is an optional hook and depends on `scikit-learn`

## Explicitly Removed or Replaced Legacy Mismatches

- Hard-coded `parse_args(args=[...])` behavior was not carried into v2.
- The legacy consistency-projection placeholder was not carried into v2.
- Legacy heuristic dominance fitness was replaced with the exact signed row fitness.
- Legacy monolithic mutation/crossover logic was split into separate modules.
- Framework-specific JAX/Torch glue was not carried into the new algorithm core.

## Residual Risks

These are not silent mismatches, but they remain the main areas worth future extension:

1. richer selection-score variants for closer paper matching
2. broader halfspace-family support
3. derived-measurement fusion beyond repeated direct queries
4. stronger population-diversity diagnostics and crossover policies
5. larger integration benchmarks against the preserved legacy baselines
