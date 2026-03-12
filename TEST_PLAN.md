# Test Plan

## Implemented local tests

Run all tests with:

```bash
python -m unittest discover -s private_de_v2/tests -v
```

Implemented coverage:

1. Signed error convention
   - `private_de_v2/tests/test_fitness.py`
   - Verifies `Err(q) = noisy_answer - synthetic_answer`.

2. Record-level fitness
   - `private_de_v2/tests/test_fitness.py`
   - Verifies records inherit the signed error of the matched query state.

3. Donor/recipient mutation direction
   - `private_de_v2/tests/test_mutation.py`
   - Verifies directed mutation moves a row from an overrepresented state to an underrepresented state.

4. Orthogonal vector mutation effect
   - `private_de_v2/tests/test_mutation.py`
   - Verifies selected-vector L1 loss decreases on a toy orthogonal vector.

5. Privacy budget monotonic usage
   - `private_de_v2/tests/test_privacy_selection_measurement.py`
   - Verifies cumulative rho is monotonic and over-spend raises an error.

6. Deterministic behavior under fixed seeds
   - `private_de_v2/tests/test_generator_smoke.py`
   - Runs the full v2 generator twice and checks identical outputs.

7. Numerical stability of exponential mechanism
   - `private_de_v2/tests/test_privacy_selection_measurement.py`
   - Verifies finite normalized probabilities under large utilities.

8. Query workload construction correctness
   - `private_de_v2/tests/test_queries.py`
   - Verifies 1-way, 2-way, range, prefix, and singleton grouping behavior.

9. Inverse-variance weighting
   - `private_de_v2/tests/test_privacy_selection_measurement.py`
   - Verifies repeated measurements combine to the expected weighted mean and variance.

## Manual smoke checks

1. Minimal run

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --output-dir outputs/run_example
```

Expected outputs:

- `synthetic.csv`
- `metrics.jsonl`
- `summary.yaml`

2. Ablation examples

```bash
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-directed-mutation
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-crossover
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-orthogonal-grouping
python -m private_de_v2 --config configs/private_de_v2_minimal.yaml --dataset-path path/to/data.csv --no-inverse-variance-weighting
```

3. CLI reproducibility check

Run the same command twice and confirm:

- the selected vector sequence in `metrics.jsonl` is identical
- the final `synthetic.csv` is identical
- the privacy ledger summary in `summary.yaml` is identical

## Remaining useful tests

- Add a direct Gaussian sigma test around the measurement helper.
- Add a conditional-prefix and halfspace mutation projection test.
- Add a CLI parsing test around override precedence.
- Add a larger integration test with all workload families enabled on a toy mixed-type dataset.
