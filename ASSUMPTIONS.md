# Assumptions and Final Resolutions

This file records the implementation choices that were ambiguous in the paper-level spec and therefore had to be resolved explicitly for `private_de_v2/`.

For each item:

- current code behavior
- paper-intended behavior
- final implemented resolution

## 1. Query answers use normalized frequencies, not counts

### Current code behavior

```python
answer = float(np.count_nonzero(mask))
if normalize:
    answer /= float(data.shape[0])
```

from `private_de_v2/queries.py`.

### Paper-intended behavior

The spec allowed either counts or normalized frequencies, but required one convention to be explicit and consistent everywhere.

### Final resolution

- `private_de_v2` uses normalized frequencies throughout:
  - selection
  - measurement
  - signed errors
  - loss
  - mutation
  - evaluation
- This keeps the sensitivity expressions explicit:
  - singleton query sensitivity `1 / N`
  - grouped orthogonal vector L1 sensitivity conservatively `2 / N`
  - grouped orthogonal vector L2 sensitivity `sqrt(2) / N`

## 2. Internal algorithm space is fully discrete integer-coded data

### Current code behavior

Legacy code mixed transformed-space and decoded-space logic:

```python
real_dataset = temp_synthesizer._get_data(...)
```

and later:

```python
output_df = temp_synthesizer._transformer.inverse_transform(data_list)
```

from `main.py`.

### Paper-intended behavior

The algorithm should operate in one explicit internal space and only decode at the boundary.

### Final resolution

- `private_de_v2.data` loads CSVs and converts them into discrete integer-coded arrays.
- Numeric columns are discretized explicitly.
- All query evaluation and mutation operate on those discrete codes.
- Final output decoding maps codes back to category labels or interval labels exactly once.

## 3. Initialization uses private 1-way marginals and a product distribution

### Current code behavior

Legacy PT initialization was effectively random:

```python
initial_population = torch.rand(P, num_records, self.num_columns) * 2 - 1
```

from `Pi.py` and related PT variants.

### Paper-intended behavior

Initialization must privately measure all 1-way marginals first and build `D_hat_0` from them.

### Final resolution

- `private_de_v2.generator` privately measures every 1-way vector first.
- The resulting noisy 1-way answers are clipped to nonnegative values and renormalized.
- Each feature is then sampled independently from its private 1-way distribution.
- The initial synthetic dataset is therefore a product-of-marginals sample.

## 4. The exact selection score is a clipped L1 gap, not the more complex paper-like constant form

### Current code behavior

The implemented score is:

```python
raw_gap = float(np.sum(np.abs(real_answers - synthetic_answers)))
size_weight = float(len(vector.queries) ** config.vector_size_weight_power)
clipped_gap = min(raw_gap * size_weight, float(config.score_clip))
```

from `private_de_v2/selection.py`.

### Paper-intended behavior

The spec allowed the paper form or a closely related selection score, as long as it was explicit, inspectable, and used a stable exponential mechanism with explicit sensitivity.

### Final resolution

- The implementation uses a clipped L1 frequency-gap utility.
- Sensitivity is set conservatively to:
  - `min(score_clip, 2 * size_weight / N)`
- The exponential mechanism uses:
  - `epsilon = sqrt(2 * rho_selection)`
- This is simpler and easier to test than the paper-like cubic constant variant.

## 5. zCDP budget scheduling uses one initialization block and equal per-round spending thereafter

### Current code behavior

```python
init_rho_total = accountant.total_rho * self.config.privacy.initialization_rho_fraction
per_round_rho = (remaining_rho / float(self.config.algorithm.rounds))
selection_rho = per_round_rho * self.config.privacy.selection_rho_fraction
measurement_rho = per_round_rho - selection_rho
```

from `private_de_v2/generator.py`.

### Paper-intended behavior

The spec required explicit monotonic zCDP accounting and per-round reporting, but did not mandate a single schedule.

### Final resolution

- Total rho is either:
  - supplied directly, or
  - derived from `(epsilon, delta)` via the standard zCDP interpretation formula.
- A configurable initialization fraction is spent on all 1-way measurements.
- Remaining rho is divided evenly across the configured rounds.
- Each round splits its rho into:
  - selection
  - measurement

## 6. Gaussian measurement uses grouped-vector sensitivity `sqrt(2) / N`

### Current code behavior

```python
if len(vector.queries) <= 1:
    return 1.0 / float(dataset_size)
return math.sqrt(2.0) / float(dataset_size)
```

from `private_de_v2/measurement.py`.

### Paper-intended behavior

The spec required a single explicit Gaussian zCDP calibration rule.

### Final resolution

- Singleton vectors use L2 sensitivity `1 / N`.
- Multi-query orthogonal grouped vectors use conservative L2 sensitivity `sqrt(2) / N`.
- Noise scale is:
  - `sigma = sensitivity / sqrt(2 * rho)`

## 7. Repeated direct query measurements use inverse-variance weighting only

### Current code behavior

```python
old_weight = 1.0 / current.variance
new_weight = 1.0 / variance
current.value = ((current.value * old_weight) + (value * new_weight)) / combined_weight
current.variance = 1.0 / combined_weight
```

from `private_de_v2/measurement.py`.

### Paper-intended behavior

The spec required explicit repeated-measurement fusion if used.

### Final resolution

- `private_de_v2` tracks repeated direct measurements of the same query id.
- When IVW is enabled, repeated direct measurements are merged by inverse-variance weighting.
- When `--no-inverse-variance-weighting` is used, the store keeps the latest estimate instead.
- Derived-marginal fusion is not implemented in v2.

## 8. The legacy projection placeholder is removed from the new path

### Current code behavior

Legacy placeholder:

```python
def project_2way_to_consistent_1way(...):
    return noised_answers, 0.0
```

from `main.py`.

### Paper-intended behavior

The spec explicitly forbade carrying a fake placeholder silently.

### Final resolution

- `private_de_v2` does not contain a projection step.
- The new path measures and uses query vectors directly.
- The placeholder remains only in the legacy script and is documented as legacy behavior.

## 9. Mutation count per round is estimated from signed-error mass and capped explicitly

### Current code behavior

```python
mass = max(sum(positive_errors), sum(abs(negative_errors)))
estimated = int(round(mass * dataset_size))
return min(max_mutations_per_round, dataset_size, estimated)
```

from `private_de_v2/mutation.py`.

### Paper-intended behavior

The spec required explicit mutation counts but did not define the exact formula.

### Final resolution

- The requested mutation count is:
  - rounded signed-error mass in record units
  - clipped by `max_mutations_per_round`
  - clipped by dataset size
- If nonzero error exists but the rounded count is zero, one mutation is requested.

## 10. "Closest" record/state is implemented as deterministic discrete projection

### Current code behavior

Examples from `private_de_v2/queries.py`:

```python
projected[column_index] = int(np.clip(projected[column_index], lower, upper))
```

and

```python
return min(alternatives, key=lambda value: abs(value - current_value))
```

### Paper-intended behavior

If "closest" is used, it must be defined explicitly.

### Final resolution

- For equality-style states, projection sets the required feature values directly.
- For ordered predicates, projection uses clamping or the nearest valid discrete boundary.
- For categorical "leave this state" projections, the nearest alternative integer code is used.
- This makes mutation deterministic and inspectable.

## 11. `no_orthogonal_grouping` uses singleton vectors with explicit complement fallback in mutation

### Current code behavior

```python
queries=(query,)
grouping_mode="singleton"
```

from `private_de_v2/queries.py`.

Singleton mutation fallback:

```python
if signed_error > 0:
    donor_rows = np.where(matched_indices < 0)[0]
...
if signed_error < 0:
    mutated[row_index] = project_row_out_of_query(...)
```

from `private_de_v2/mutation.py`.

### Paper-intended behavior

The spec required a no-orthogonal-grouping ablation, but singleton vectors remove the natural donor/recipient state partition.

### Final resolution

- With `no_orthogonal_grouping`, each query becomes a singleton vector.
- Positive singleton error pulls rows from the complement into the query.
- Negative singleton error pushes rows out of the query into the complement.
- This is an explicit ablation behavior, not claimed to have the same guarantees as grouped orthogonal mutation.

## 12. Halfspace queries are implemented as simple paired thresholded sums over two ordered features

### Current code behavior

```python
label=f"{first_name}+{second_name} <= {threshold}"
```

and

```python
"weights": [1, 1]
```

from `private_de_v2/queries.py`.

### Paper-intended behavior

The spec required support for halfspace-style paired queries where applicable, but left the exact family under-specified.

### Final resolution

- v2 implements a simple inspectable variant:
  - two ordered features
  - unit weights
  - paired complementary thresholds on their discrete sum
- This is documented as a simplified halfspace-style family, not a general linear separator implementation.

## 13. Crossover uses the best measured-loss population as donor source

### Current code behavior

```python
best_population_index = int(np.argmin(self._population_global_losses(...)))
```

and then:

```python
populations[target_population_index, target_row_index] = populations[best_population_index, donor_row_index]
```

from `private_de_v2/generator.py` and `private_de_v2/crossover.py`.

### Paper-intended behavior

The spec required a separate crossover operator with explicit parent selection.

### Final resolution

- The donor population is the one with the lowest mean squared error against all measured queries currently stored.
- Donor rows preferentially come from query states with positive signed error on the active vector.
- Recipient rows in other populations are the lowest-fitness rows under the active vector.

## 14. Device handling is metadata only in v2

### Current code behavior

```python
device: str = "cpu"
```

and

```python
"device": self.config.algorithm.device,
```

from `private_de_v2/config.py` and `private_de_v2/generator.py`.

### Paper-intended behavior

The refactor requirement was to separate framework glue from algorithm logic.

### Final resolution

- The new algorithm core is NumPy/Pandas based.
- `device` is still exposed and logged for reproducibility metadata and future extension.
- Torch seeding is applied if Torch is installed, but the algorithm does not depend on Torch or JAX at runtime.
