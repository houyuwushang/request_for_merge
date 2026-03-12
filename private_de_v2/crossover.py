from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitness import FitnessResult


@dataclass(frozen=True)
class CrossoverEvent:
    target_population_index: int
    donor_population_index: int
    target_row_index: int
    donor_row_index: int


@dataclass(frozen=True)
class CrossoverResult:
    populations: np.ndarray
    events: tuple[CrossoverEvent, ...]


def apply_crossover(
    populations: np.ndarray,
    fitness_results: list[FitnessResult],
    best_population_index: int,
    signed_errors: np.ndarray,
    rng: np.random.Generator,
    crossover_rate: float,
    crossover_rows: int,
) -> CrossoverResult:
    populations = np.asarray(populations, dtype=np.int64).copy()
    if populations.shape[0] <= 1 or crossover_rows <= 0 or crossover_rate <= 0.0:
        return CrossoverResult(populations=populations, events=tuple())

    events: list[CrossoverEvent] = []
    donor_fitness = fitness_results[best_population_index]
    positive_queries = np.where(np.asarray(signed_errors, dtype=float) > 1e-12)[0]
    if positive_queries.size > 0:
        donor_candidates = np.where(np.isin(donor_fitness.matched_query_indices, positive_queries))[0]
    else:
        donor_candidates = np.arange(populations.shape[1])
    if donor_candidates.size == 0:
        donor_candidates = np.arange(populations.shape[1])

    for population_index in range(populations.shape[0]):
        if population_index == best_population_index:
            continue
        if float(rng.random()) > crossover_rate:
            continue

        target_fitness = fitness_results[population_index]
        ranking = np.argsort(target_fitness.record_fitness)
        target_rows = ranking[: min(crossover_rows, len(ranking))]
        donor_rows = rng.choice(donor_candidates, size=len(target_rows), replace=True)
        for target_row_index, donor_row_index in zip(target_rows, donor_rows):
            populations[population_index, int(target_row_index)] = populations[best_population_index, int(donor_row_index)]
            events.append(
                CrossoverEvent(
                    target_population_index=population_index,
                    donor_population_index=best_population_index,
                    target_row_index=int(target_row_index),
                    donor_row_index=int(donor_row_index),
                )
            )

    return CrossoverResult(populations=populations, events=tuple(events))
