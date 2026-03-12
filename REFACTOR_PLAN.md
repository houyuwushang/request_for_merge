# Refactor Plan

## Goals

- Preserve all legacy code as reference.
- Stop extending the current monolithic PT scripts.
- Build a clean `private_de_v2/` implementation path beside the existing code.
- Separate:
  - algorithm logic,
  - privacy accounting,
  - query construction,
  - framework/device glue,
  - experiment scripts,
  - evaluation utilities.
- Prioritize correctness, reproducibility, and clarity over backward compatibility.

## Proposed Package Layout

```text
private_de_v2/
  __init__.py
  config.py
  data.py
  queries.py
  selection.py
  measurement.py
  privacy.py
  fitness.py
  mutation.py
  crossover.py
  generator.py
  evaluation.py
  cli.py
  tests/
```

## Module Responsibilities

### `config.py`

- Define typed experiment and algorithm configs.
- Hold only declarative parameters:
  - seeds,
  - budget schedule,
  - population size,
  - iteration count,
  - workload/query-vector settings,
  - device policy.
- No hidden defaults tied to one machine or one dataset.

### `data.py`

- Centralize:
  - dataset loading,
  - schema/domain representation,
  - transform/raw-space conversions,
  - deterministic preprocessing.
- Explicitly separate:
  - internal algorithm space,
  - output/raw data space.
- Replace ad hoc transformer access from generator code.

### `queries.py`

- Define the Private-DE query abstraction, not just marginals.
- Introduce explicit types such as:
  - `Query`
  - `QueryVector`
  - `QueryState`
  - `QueryWorkload`
- Implement orthogonal query-vector construction.
- Provide workload correctness checks.

### `selection.py`

- Score candidate query vectors against the current synthetic population.
- Implement a numerically stable exponential mechanism.
- Keep sensitivity explicit.
- Return:
  - selected vector,
  - selection score,
  - selection probability metadata,
  - privacy cost.

### `measurement.py`

- Measure selected query vectors with Gaussian noise.
- Store measurement records with:
  - query/vector ID,
  - noisy answer,
  - variance,
  - rho spent,
  - round index.
- Implement inverse-variance weighting over repeated measurements.

### `privacy.py`

- Single source of truth for:
  - `epsilon, delta -> rho`,
  - Gaussian sigma under zCDP,
  - round-by-round budget schedule,
  - monotonic budget consumption,
  - audit logs.
- Remove privacy math from scripts and utility fragments.

### `fitness.py`

- Implement exact row fitness for the active query vector:

```text
f_V(x) = sum_{q in V} Err(q) * phi_q(x)
```

- Keep sign explicit.
- Provide:
  - row-level contributions,
  - vector-level aggregates,
  - population summaries for mutation/crossover.

### `mutation.py`

- Implement directed mutation around donor/recipient state transitions inside one orthogonal query vector.
- Inputs should be:
  - current synthetic population,
  - active query vector,
  - signed residuals,
  - row fitness decomposition.
- Outputs should include structured mutation diagnostics, not just a mutated tensor.

### `crossover.py`

- Implement context-aware crossover informed by:
  - active query vector,
  - row states,
  - population diversity,
  - donor/recipient mismatch structure.
- Keep generic diversity injection separate from true crossover logic.

### `generator.py`

- Own the full iterative Private-DE loop:
  - initialize synthetic population,
  - select query vector,
  - measure under zCDP,
  - aggregate measurements,
  - score rows,
  - mutate/crossover,
  - update synthetic state,
  - stop when budget or iteration limit is reached.
- Expose a library API independent of CLI scripts.

### `evaluation.py`

- Collect:
  - train-time diagnostics,
  - workload/query-vector error metrics,
  - privacy ledger summary,
  - reproducibility metadata,
  - output artifact metadata.
- Keep experiment reporting out of algorithm modules.

### `cli.py`

- Provide a thin script wrapper over the library API.
- No hard-coded local paths.
- No framework setup logic embedded into algorithm code.

## Legacy Preservation Strategy

- Keep these as read-only references:
  - `src/genetic_sd/`
  - `primary.py`
  - `mygenerator.py`
  - `Pi.py`
  - `backup_code/`

- Do not delete or overwrite them during the initial refactor.

- Add a short migration note later:
  - "legacy baseline"
  - "experimental PT branch"
  - "new implementation"

## Suggested Implementation Sequence

### Phase 1. Freeze reference behavior

- Add analysis docs only.
- Add targeted smoke tests around current behavior that will later become regression references.
- Record known broken behavior instead of preserving it silently.

### Phase 2. Build foundations

- Implement `config.py`, `data.py`, and `privacy.py`.
- Resolve internal-vs-raw data representation explicitly.
- Create deterministic RNG plumbing for:
  - Python,
  - NumPy,
  - Torch,
  - JAX if still needed.

### Phase 3. Query model

- Implement `queries.py`.
- Define orthogonal query-vector construction and validation.
- Add tests for workload/query correctness before generator work starts.

### Phase 4. Measurement and selection

- Implement `selection.py` and `measurement.py`.
- Make query-vector selection and Gaussian measurement independently testable.
- Add inverse-variance aggregation before generator integration.

### Phase 5. Fitness and generator core

- Implement `fitness.py` and `generator.py`.
- Start with the exact signed row fitness formula.
- Keep mutation and crossover out until row scoring is validated.

### Phase 6. Directed search operators

- Implement `mutation.py` and `crossover.py`.
- Use typed row/vector state objects rather than raw tensors plus ad hoc comments.

### Phase 7. Evaluation and CLI

- Implement `evaluation.py` and `cli.py`.
- Add structured outputs for experiments.

### Phase 8. Legacy adapters

- If needed, add small adapter scripts so existing experiments can call `private_de_v2` without reusing the old monoliths.

## Testing Plan

The new implementation must include tests for at least the following:

### 1. Signed error convention correctness

- Given fixed noisy and synthetic answers, verify:
  - `Err(q) = noisy_answer - synthetic_answer`
- Ensure positive error means "synthetic undercounts" and negative error means "synthetic overcounts".

### 2. Donor/recipient mutation direction

- Build a tiny synthetic population with known row states.
- Verify mutation moves mass from donor-compatible rows toward recipient-required states, not the reverse.

### 3. Orthogonal vector mutation effect

- On a small query vector with orthogonal states, mutate one direction and verify:
  - intended query states move,
  - unrelated orthogonal states are preserved within tolerance.

### 4. Privacy budget monotonic usage

- Verify each round only increases cumulative rho.
- Verify the run halts or errors once configured budget is exhausted.

### 5. Deterministic behavior under fixed seeds

- Same config + same seed must produce identical:
  - selected query vectors,
  - noisy measurements,
  - mutation choices,
  - final synthetic data.

### 6. Numerical stability of exponential mechanism

- Stress-test large positive and negative scores.
- Verify no overflow/underflow crashes and valid normalized sampling probabilities.

### 7. Query workload construction correctness

- Verify:
  - vector orthogonality,
  - state coverage,
  - query bounds,
  - and `phi_q(x)` evaluation on hand-built records.

### 8. Inverse-variance weighting

- Repeated measurements with known variances must combine to the expected weighted estimate.

### 9. Measurement variance calculation

- For fixed sensitivity and rho, verify sigma matches the zCDP Gaussian formula used by the privacy module.

### 10. Representation round-trip

- Verify internal-space rows and output/raw-space rows round-trip without double decoding.

## Non-Negotiable Design Rules for `private_de_v2`

- No hard-coded experiment paths.
- No `parse_args(args=[...])` in reusable code.
- No framework setup side effects in algorithm modules.
- No mutation/crossover logic in CLI files.
- No privacy math duplicated across modules.
- No silent preservation of behavior that conflicts with the intended algorithm.
- Every intentional deviation from the current experimental branch must be documented.

