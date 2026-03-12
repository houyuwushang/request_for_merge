# Repository Map

## Executive Summary

The repository now has a clean implementation path and preserved legacy references:

1. `private_de_v2/`
   - active refactored implementation
   - config-driven CLI and library API
   - explicit discrete data model, workload construction, zCDP accounting, selection, measurement, fitness, mutation, crossover, evaluation, and tests

2. Legacy top-level scripts
   - `main.py`, `primary.py`, `mygenerator.py`, `Pi.py`, `backup_code/`
   - preserved as baseline or experimental references
   - not treated as source of truth

3. Packaged JAX baseline
   - `src/genetic_sd/`
   - older coherent Genetic-SD implementation family
   - useful as historical baseline and utility reference

## Current Directory Structure

```text
.
|- README.md
|- Agent.md
|- SPEC.md
|- REPO_MAP.md
|- SPEC_GAP_REPORT.md
|- REFACTOR_PLAN.md
|- ASSUMPTIONS.md
|- TEST_PLAN.md
|- configs/
|  |- private_de_v2_minimal.yaml
|- private_de_v2/
|  |- __init__.py
|  |- __main__.py
|  |- cli.py
|  |- config.py
|  |- data.py
|  |- queries.py
|  |- privacy.py
|  |- selection.py
|  |- measurement.py
|  |- fitness.py
|  |- mutation.py
|  |- crossover.py
|  |- generator.py
|  |- evaluation.py
|  |- tests/
|- main.py
|- mygenerator.py
|- primary.py
|- Pi.py
|- backup_code/
|- src/
|  |- genetic_sd/
```

## `private_de_v2/` roles

- `config.py`
  - typed run configuration and YAML loading
  - explicit privacy, workload, algorithm, ablation, and output settings

- `cli.py`
  - new CLI entrypoint
  - supports config-file loading plus CLI overrides

- `data.py`
  - CSV loading
  - explicit discrete/discretized schema construction
  - encoded internal representation and decoding for output

- `queries.py`
  - query, query-vector, and workload abstractions
  - 1-way, 2-way, 3-way, range, prefix, conditional-prefix, and halfspace-style workload construction
  - singleton-vector ablation for `no_orthogonal_grouping`

- `privacy.py`
  - zCDP budget conversions and monotonic accountant

- `selection.py`
  - stable exponential mechanism
  - clipped L1 frequency-gap utility scoring

- `measurement.py`
  - Gaussian query-vector measurement
  - repeated-measurement store
  - inverse-variance weighting

- `fitness.py`
  - exact signed error and per-record fitness

- `mutation.py`
  - directed donor/recipient mutation in record space

- `crossover.py`
  - separate population-level crossover operator

- `generator.py`
  - iterative select-measure-generate loop
  - initialization from private 1-way marginals
  - per-round logging
  - output writing and summary generation

- `evaluation.py`
  - workload-query MSE
  - average vector TVD
  - exact-record-match-share risk hook
  - optional downstream ML utility hook if `scikit-learn` is available

- `tests/`
  - local `unittest` coverage for the new path only

## Legacy and reference code

### Top-level experimental branch

- `main.py`
  - legacy experimental entry script
  - still contains hard-coded argument injection and placeholder logic
  - preserved for reference only

- `mygenerator.py`
  - large PyTorch experimental generator
  - heuristic mutation and crossover logic
  - duplicated and not paper-aligned

- `primary.py`
  - near-duplicate of `mygenerator.py`
  - preserves alternative experimental branch state

- `Pi.py`
  - early prototype

- `backup_code/`
  - archived snapshots of earlier PT experiments

Classification:

- `main.py`, `mygenerator.py`, `primary.py`, `Pi.py`, `backup_code/`: experimental / legacy

### `src/genetic_sd/`

- coherent older JAX baseline
- includes adaptive workload selection, marginal measurement, and JAX generator code
- still not the source of truth for Private-DE v2

Classification:

- baseline / legacy reference

## Duplicated, placeholder, and dead code

Duplicated:

- `primary.py` vs `mygenerator.py`
- `backup_code/*` vs top-level PT scripts
- `src/genetic_sd/adaptive_statistics/*` vs `src/genetic_sd/fast_statistics/*`

Placeholders:

- old `README.md` placeholder has been replaced
- legacy `main.py:project_2way_to_consistent_1way(...)` remains a no-op in the old path
- `src/genetic_sd/test/test_statistics.py`

Dead or effectively unused:

- `src/genetic_sd/fast_statistics/*`
- `src/genetic_sd/diffevo/*` for Private-DE
- unreachable stagnation fallback blocks in `primary.py` and `mygenerator.py`

## Hard-coded or structurally risky legacy behavior

- hard-coded `parse_args(args=[...])` in legacy scripts
- mixed JAX/Torch/framework setup inside legacy entrypoints
- placeholder projection in legacy `main.py`
- duplicated mutation/crossover implementations across PT scripts
- inconsistent privacy helper usage inside the old baseline code

## What should be used going forward

Use:

- `python -m private_de_v2`
- `configs/private_de_v2_minimal.yaml`
- `private_de_v2/tests/`

Treat as reference only:

- top-level PT scripts
- `backup_code/`
- `src/genetic_sd/`
