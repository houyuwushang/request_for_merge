from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DiscreteSchema
from .queries import QueryVector, assign_records_to_vector


@dataclass(frozen=True)
class FitnessResult:
    signed_errors: np.ndarray
    record_fitness: np.ndarray
    matched_query_indices: np.ndarray


def compute_signed_errors(noisy_answers: np.ndarray, synthetic_answers: np.ndarray) -> np.ndarray:
    noisy_answers = np.asarray(noisy_answers, dtype=float)
    synthetic_answers = np.asarray(synthetic_answers, dtype=float)
    if noisy_answers.shape != synthetic_answers.shape:
        raise ValueError("noisy_answers and synthetic_answers must have the same shape")
    return noisy_answers - synthetic_answers


def compute_record_fitness(
    data: np.ndarray,
    vector: QueryVector,
    signed_errors: np.ndarray,
    schema: DiscreteSchema,
) -> FitnessResult:
    signed_errors = np.asarray(signed_errors, dtype=float)
    if signed_errors.shape != (len(vector.queries),):
        raise ValueError("signed_errors must align with the selected vector")

    matched_query_indices = assign_records_to_vector(data, vector, schema)
    record_fitness = np.zeros(data.shape[0], dtype=float)
    matched_mask = matched_query_indices >= 0
    record_fitness[matched_mask] = signed_errors[matched_query_indices[matched_mask]]
    return FitnessResult(
        signed_errors=signed_errors,
        record_fitness=record_fitness,
        matched_query_indices=matched_query_indices,
    )
