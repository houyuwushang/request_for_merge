# Private-DE

This repository now contains two code paths:

- `private_de_v2/`
  - clean, reproducible, spec-aligned implementation
  - config-driven CLI
  - explicit zCDP accounting
  - explicit workload/query-vector construction
  - deterministic unit tests
- legacy top-level scripts and `src/genetic_sd/`
  - preserved as baseline and experimental references
  - not treated as source of truth for the refactor

## Minimal run

Use the new package entrypoint:

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --output-dir outputs/run_example
```

Optional ablations:

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-directed-mutation
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-crossover
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-orthogonal-grouping
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-inverse-variance-weighting
```

Outputs are written to the configured `output_dir`:

- `synthetic.csv`
- `metrics.jsonl`
- `summary.yaml`

## CLI notes

Key overrides are available directly on the CLI:

- dataset and output paths
- `epsilon`, `delta`, or `total_rho`
- seed and device metadata
- rounds, population size, synthetic size
- workload families and threshold settings
- ablation flags

The new entrypoint does not use hard-coded `parse_args(args=[...])` experiment stubs.

## Implementation choices

- Internal algorithm space uses discrete integer-coded records.
- Query answers are normalized frequencies, not raw counts.
- Initialization privately measures all 1-way marginals, clips them to valid probability vectors, and samples the initial synthetic data from the product of those marginals.
- Per-round selection uses a stable exponential mechanism over clipped L1 frequency gaps.
- Measurement uses the Gaussian mechanism under zCDP.
- Repeated direct measurements are merged by inverse-variance weighting unless `--no-inverse-variance-weighting` is set.
- Signed error is fixed everywhere as `Err(q) = noisy_answer - synthetic_answer`.

## Tests

Run the local v2 test suite with:

```bash
python -m unittest discover -s private_de_v2/tests -v
```

See `TEST_PLAN.md` for the exact coverage and manual smoke checks.

## Legacy code

These paths remain available for reference and baseline comparisons:

- `main.py`
- `primary.py`
- `mygenerator.py`
- `Pi.py`
- `backup_code/`
- `src/genetic_sd/`

They are preserved intentionally and may still contain historical mismatches, placeholders, or experimental behavior documented in `REPO_MAP.md`, `SPEC_GAP_REPORT.md`, and `ASSUMPTIONS.md`.
