from __future__ import annotations

from dataclasses import dataclass, field

import math
import numpy as np

from .data import DiscreteSchema
from .queries import Query, QueryVector, evaluate_vector_answers


@dataclass(frozen=True)
class MeasurementEvent:
    vector_id: str
    rho: float
    sigma: float
    noisy_answers: np.ndarray
    true_answers: np.ndarray
    variances: np.ndarray


@dataclass
class QueryEstimate:
    value: float
    variance: float
    count: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)


class MeasurementStore:
    def __init__(self, *, use_inverse_variance_weighting: bool) -> None:
        self.use_inverse_variance_weighting = use_inverse_variance_weighting
        self._estimates: dict[str, QueryEstimate] = {}
        self._registry: dict[str, Query] = {}

    def register_queries(self, queries: dict[str, Query]) -> None:
        self._registry.update(queries)

    def record_vector_measurement(
        self,
        vector: QueryVector,
        noisy_answers: np.ndarray,
        variances: np.ndarray,
    ) -> None:
        for query, value, variance in zip(vector.queries, noisy_answers, variances):
            self._registry[query.query_id] = query
            self._record_single(query.query_id, float(value), float(variance))

    def _record_single(self, query_id: str, value: float, variance: float) -> None:
        if variance <= 0.0:
            raise ValueError("variance must be positive")

        current = self._estimates.get(query_id)
        if current is None:
            self._estimates[query_id] = QueryEstimate(
                value=value,
                variance=variance,
                count=1,
                history=[(value, variance)],
            )
            return

        current.history.append((value, variance))
        current.count += 1
        if not self.use_inverse_variance_weighting:
            current.value = value
            current.variance = variance
            return

        old_weight = 1.0 / current.variance
        new_weight = 1.0 / variance
        combined_weight = old_weight + new_weight
        current.value = ((current.value * old_weight) + (value * new_weight)) / combined_weight
        current.variance = 1.0 / combined_weight

    def get_query_estimate(self, query_id: str) -> QueryEstimate:
        return self._estimates[query_id]

    def get_vector_estimates(self, vector: QueryVector) -> tuple[np.ndarray, np.ndarray]:
        means = []
        variances = []
        for query in vector.queries:
            estimate = self._estimates[query.query_id]
            means.append(estimate.value)
            variances.append(estimate.variance)
        return np.asarray(means, dtype=float), np.asarray(variances, dtype=float)

    def measured_query_ids(self) -> tuple[str, ...]:
        return tuple(self._estimates.keys())

    def query_registry(self) -> dict[str, Query]:
        return dict(self._registry)


def gaussian_measure_vector(
    real_data: np.ndarray,
    vector: QueryVector,
    schema: DiscreteSchema,
    rho: float,
    rng: np.random.Generator,
) -> MeasurementEvent:
    if rho <= 0.0:
        raise ValueError("rho must be positive for Gaussian measurement")
    true_answers = evaluate_vector_answers(real_data, vector, schema, normalize=True)
    sensitivity = _vector_l2_sensitivity(vector, real_data.shape[0])
    sigma = sensitivity / math.sqrt(2.0 * rho)
    noise = rng.normal(loc=0.0, scale=sigma, size=len(vector.queries))
    noisy_answers = true_answers + noise
    variances = np.full(len(vector.queries), sigma * sigma, dtype=float)
    return MeasurementEvent(
        vector_id=vector.vector_id,
        rho=rho,
        sigma=sigma,
        noisy_answers=np.asarray(noisy_answers, dtype=float),
        true_answers=true_answers,
        variances=variances,
    )


def _vector_l2_sensitivity(vector: QueryVector, dataset_size: int) -> float:
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if len(vector.queries) <= 1:
        return 1.0 / float(dataset_size)
    return math.sqrt(2.0) / float(dataset_size)
