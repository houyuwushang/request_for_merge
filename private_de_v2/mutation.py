from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DiscreteSchema
from .fitness import FitnessResult
from .queries import QueryVector, project_row_out_of_query, project_row_to_query


@dataclass(frozen=True)
class MutationEvent:
    row_index: int
    donor_query_index: int | None
    recipient_query_index: int | None
    before: tuple[int, ...]
    after: tuple[int, ...]


@dataclass(frozen=True)
class MutationResult:
    mutated_data: np.ndarray
    events: tuple[MutationEvent, ...]
    requested_mutations: int
    applied_mutations: int


def mutation_count_from_errors(
    signed_errors: np.ndarray,
    dataset_size: int,
    max_mutations_per_round: int,
) -> int:
    signed_errors = np.asarray(signed_errors, dtype=float)
    mass = max(float(np.sum(np.clip(signed_errors, 0.0, None))), float(np.sum(np.clip(-signed_errors, 0.0, None))))
    estimated = int(round(mass * float(dataset_size)))
    if estimated == 0 and np.any(np.abs(signed_errors) > 1e-12):
        estimated = 1
    return min(max_mutations_per_round, dataset_size, estimated)


def apply_directed_mutation(
    data: np.ndarray,
    vector: QueryVector,
    fitness: FitnessResult,
    schema: DiscreteSchema,
    rng: np.random.Generator,
    max_mutations_per_round: int,
) -> MutationResult:
    mutated = np.asarray(data, dtype=np.int64).copy()
    requested_mutations = mutation_count_from_errors(fitness.signed_errors, mutated.shape[0], max_mutations_per_round)
    if requested_mutations <= 0:
        return MutationResult(mutated, tuple(), requested_mutations, 0)

    events: list[MutationEvent] = []
    matched_indices = fitness.matched_query_indices.copy()
    signed_errors = fitness.signed_errors

    positive_queries = np.where(signed_errors > 1e-12)[0]
    negative_queries = np.where(signed_errors < -1e-12)[0]

    if len(vector.queries) == 1:
        events.extend(_apply_singleton_mutation(mutated, vector, matched_indices, signed_errors[0], schema, rng, requested_mutations))
    else:
        if positive_queries.size == 0 or negative_queries.size == 0:
            return MutationResult(mutated, tuple(), requested_mutations, 0)
        recipient_weights = signed_errors[positive_queries]
        donor_weights = np.abs(signed_errors[negative_queries])
        recipient_weights = recipient_weights / np.sum(recipient_weights)
        donor_weights = donor_weights / np.sum(donor_weights)

        for _ in range(requested_mutations):
            donor_query_index = int(rng.choice(negative_queries, p=donor_weights))
            recipient_query_index = int(rng.choice(positive_queries, p=recipient_weights))
            donor_rows = np.where(matched_indices == donor_query_index)[0]
            if donor_rows.size == 0:
                continue
            row_index = int(rng.choice(donor_rows))
            before = tuple(int(value) for value in mutated[row_index].tolist())
            mutated[row_index] = project_row_to_query(mutated[row_index], vector.queries[recipient_query_index], schema)
            matched_indices[row_index] = recipient_query_index
            events.append(
                MutationEvent(
                    row_index=row_index,
                    donor_query_index=donor_query_index,
                    recipient_query_index=recipient_query_index,
                    before=before,
                    after=tuple(int(value) for value in mutated[row_index].tolist()),
                )
            )

    return MutationResult(
        mutated_data=mutated,
        events=tuple(events),
        requested_mutations=requested_mutations,
        applied_mutations=len(events),
    )


def _apply_singleton_mutation(
    mutated: np.ndarray,
    vector: QueryVector,
    matched_indices: np.ndarray,
    signed_error: float,
    schema: DiscreteSchema,
    rng: np.random.Generator,
    requested_mutations: int,
) -> list[MutationEvent]:
    events: list[MutationEvent] = []
    query = vector.queries[0]
    if signed_error > 1e-12:
        donor_rows = np.where(matched_indices < 0)[0]
        for _ in range(requested_mutations):
            if donor_rows.size == 0:
                break
            row_index = int(rng.choice(donor_rows))
            before = tuple(int(value) for value in mutated[row_index].tolist())
            mutated[row_index] = project_row_to_query(mutated[row_index], query, schema)
            matched_indices[row_index] = 0
            donor_rows = np.where(matched_indices < 0)[0]
            events.append(
                MutationEvent(
                    row_index=row_index,
                    donor_query_index=None,
                    recipient_query_index=0,
                    before=before,
                    after=tuple(int(value) for value in mutated[row_index].tolist()),
                )
            )
        return events

    if signed_error < -1e-12:
        donor_rows = np.where(matched_indices == 0)[0]
        for _ in range(requested_mutations):
            if donor_rows.size == 0:
                break
            row_index = int(rng.choice(donor_rows))
            before = tuple(int(value) for value in mutated[row_index].tolist())
            mutated[row_index] = project_row_out_of_query(mutated[row_index], query, schema)
            matched_indices[row_index] = -1
            donor_rows = np.where(matched_indices == 0)[0]
            events.append(
                MutationEvent(
                    row_index=row_index,
                    donor_query_index=0,
                    recipient_query_index=None,
                    before=before,
                    after=tuple(int(value) for value in mutated[row_index].tolist()),
                )
            )
    return events
