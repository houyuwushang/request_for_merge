from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import yaml

from .config import RunConfig, config_to_dict
from .crossover import CrossoverEvent, apply_crossover
from .data import DiscreteDataset, DiscreteSchema, load_discrete_dataset
from .evaluation import EvaluationResult, evaluate_synthetic_data
from .fitness import FitnessResult, compute_record_fitness, compute_signed_errors
from .measurement import MeasurementStore, gaussian_measure_vector
from .mutation import MutationEvent, MutationResult, apply_directed_mutation
from .privacy import ZCDPAccountant, rho_from_epsilon_delta
from .queries import (
    QueryVector,
    QueryWorkload,
    build_initialization_vectors,
    build_workload,
    evaluate_query_answer,
    evaluate_vector_answers,
)
from .selection import SelectionCandidate, score_vector, select_query_vector


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    selected_vector_id: str
    selected_family: str
    selected_query_ids: tuple[str, ...]
    selection_rho: float
    measurement_rho: float
    cumulative_rho: float
    selection_utility: float
    vector_loss_before: float
    vector_loss_after: float
    best_global_loss: float
    measurement_sigma: float
    runtime_seconds: float
    mutations_applied: tuple[int, ...]
    crossovers_applied: int


@dataclass(frozen=True)
class RunResult:
    config: dict[str, Any]
    summary: dict[str, Any]
    rounds: tuple[RoundRecord, ...]
    synthetic_data_path: str
    metrics_path: str
    summary_path: str


class PrivateDEGeneratorV2:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.config.validate()
        self.config.resolve_output_paths()

    def run(self) -> RunResult:
        dataset = load_discrete_dataset(self.config.data)
        return self.run_on_dataset(dataset)

    def run_on_dataset(self, dataset: DiscreteDataset) -> RunResult:
        _seed_everything(self.config.algorithm.seed, self.config.algorithm.deterministic)
        rng = np.random.default_rng(self.config.algorithm.seed)

        init_vectors = build_initialization_vectors(dataset.schema)
        workload = build_workload(
            dataset.schema,
            self.config.workload,
            no_orthogonal_grouping=self.config.ablations.no_orthogonal_grouping,
        )
        if not workload.vectors:
            raise ValueError("Configured workload is empty after applying filters")

        total_rho = (
            self.config.privacy.total_rho
            if self.config.privacy.total_rho is not None
            else rho_from_epsilon_delta(self.config.privacy.epsilon, self.config.privacy.delta)
        )
        accountant = ZCDPAccountant(total_rho)
        store = MeasurementStore(
            use_inverse_variance_weighting=not self.config.ablations.no_inverse_variance_weighting
        )
        store.register_queries(workload.query_registry())
        store.register_queries(_query_registry_from_vectors(init_vectors))

        real_answers_by_vector = {
            vector.vector_id: evaluate_vector_answers(dataset.encoded, vector, dataset.schema, normalize=True)
            for vector in workload.vectors
        }

        init_rho_total = accountant.total_rho * self.config.privacy.initialization_rho_fraction
        rho_per_init_vector = init_rho_total / float(len(init_vectors))
        initialization_probabilities = self._measure_initialization_vectors(
            dataset.encoded,
            dataset.schema,
            init_vectors,
            rho_per_init_vector,
            store,
            accountant,
            rng,
        )

        synthetic_size = int(self.config.algorithm.synthetic_size or dataset.num_records)
        populations = np.stack(
            [
                _sample_initial_population(dataset.schema, initialization_probabilities, synthetic_size, rng)
                for _ in range(self.config.algorithm.population_size)
            ],
            axis=0,
        )
        global_losses = self._population_global_losses(populations, store, dataset.schema)

        round_records: list[RoundRecord] = []
        remaining_rho = accountant.remaining_rho
        per_round_rho = (remaining_rho / float(self.config.algorithm.rounds)) if self.config.algorithm.rounds > 0 else 0.0

        metrics_path = Path(self.config.output.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for round_index in range(self.config.algorithm.rounds):
                if accountant.remaining_rho <= 1e-12 or per_round_rho <= 0.0:
                    break

                round_start = time.perf_counter()
                reference_population_index = int(np.argmin(global_losses))
                reference_population = populations[reference_population_index]

                selection_rho = per_round_rho * self.config.privacy.selection_rho_fraction
                measurement_rho = per_round_rho - selection_rho
                if measurement_rho <= 0.0:
                    break

                selection_candidates = self._selection_candidates(
                    workload,
                    real_answers_by_vector,
                    reference_population,
                    dataset.schema,
                    dataset.num_records,
                )
                selection_result = select_query_vector(selection_candidates, selection_rho, rng)
                selected_vector = workload.vectors[selection_result.vector_index]
                accountant.spend(
                    f"round_{round_index}:selection",
                    selection_rho,
                    metadata={"vector_id": selected_vector.vector_id},
                )

                measurement_event = gaussian_measure_vector(
                    dataset.encoded,
                    selected_vector,
                    dataset.schema,
                    measurement_rho,
                    rng,
                )
                store.record_vector_measurement(selected_vector, measurement_event.noisy_answers, measurement_event.variances)
                accountant.spend(
                    f"round_{round_index}:measurement",
                    measurement_rho,
                    metadata={"vector_id": selected_vector.vector_id, "sigma": measurement_event.sigma},
                )
                target_answers, _ = store.get_vector_estimates(selected_vector)

                vector_losses_before: list[float] = []
                vector_losses_after_mutation: list[float] = []
                post_mutation_fitness_results: list[FitnessResult] = []
                mutation_results: list[MutationResult] = []
                mutated_populations = populations.copy()
                for population_index in range(populations.shape[0]):
                    synthetic_answers_before = evaluate_vector_answers(
                        populations[population_index], selected_vector, dataset.schema, normalize=True
                    )
                    signed_errors_before = compute_signed_errors(target_answers, synthetic_answers_before)
                    vector_losses_before.append(float(np.sum(np.abs(signed_errors_before))))
                    fitness_before = compute_record_fitness(
                        populations[population_index],
                        selected_vector,
                        signed_errors_before,
                        dataset.schema,
                    )
                    if self.config.ablations.no_directed_mutation:
                        mutation_result = MutationResult(
                            mutated_data=populations[population_index].copy(),
                            events=tuple(),
                            requested_mutations=0,
                            applied_mutations=0,
                        )
                    else:
                        mutation_result = apply_directed_mutation(
                            populations[population_index],
                            selected_vector,
                            fitness_before,
                            dataset.schema,
                            rng,
                            self.config.algorithm.max_mutations_per_round,
                        )
                    mutated_populations[population_index] = mutation_result.mutated_data
                    mutation_results.append(mutation_result)

                    synthetic_answers_after_mutation = evaluate_vector_answers(
                        mutation_result.mutated_data,
                        selected_vector,
                        dataset.schema,
                        normalize=True,
                    )
                    signed_errors_after_mutation = compute_signed_errors(target_answers, synthetic_answers_after_mutation)
                    vector_losses_after_mutation.append(float(np.sum(np.abs(signed_errors_after_mutation))))
                    post_mutation_fitness_results.append(
                        compute_record_fitness(
                            mutation_result.mutated_data,
                            selected_vector,
                            signed_errors_after_mutation,
                            dataset.schema,
                        )
                    )

                best_population_index = int(
                    np.argmin(self._population_global_losses(mutated_populations, store, dataset.schema))
                )
                if self.config.ablations.no_crossover:
                    crossover_events: tuple[CrossoverEvent, ...] = tuple()
                    populations = mutated_populations
                else:
                    crossover_result = apply_crossover(
                        mutated_populations,
                        post_mutation_fitness_results,
                        best_population_index,
                        post_mutation_fitness_results[best_population_index].signed_errors,
                        rng,
                        self.config.algorithm.crossover_rate,
                        self.config.algorithm.crossover_rows,
                    )
                    populations = crossover_result.populations
                    crossover_events = crossover_result.events

                global_losses = self._population_global_losses(populations, store, dataset.schema)
                reference_loss_after = float(
                    np.sum(
                        np.abs(
                            compute_signed_errors(
                                target_answers,
                                evaluate_vector_answers(
                                    populations[reference_population_index],
                                    selected_vector,
                                    dataset.schema,
                                    normalize=True,
                                ),
                            )
                        )
                    )
                )

                round_record = RoundRecord(
                    round_index=round_index,
                    selected_vector_id=selected_vector.vector_id,
                    selected_family=selected_vector.family,
                    selected_query_ids=tuple(query.query_id for query in selected_vector.queries),
                    selection_rho=selection_rho,
                    measurement_rho=measurement_rho,
                    cumulative_rho=accountant.spent_rho,
                    selection_utility=selection_result.utility,
                    vector_loss_before=float(vector_losses_before[reference_population_index]),
                    vector_loss_after=reference_loss_after,
                    best_global_loss=float(np.min(global_losses)),
                    measurement_sigma=float(measurement_event.sigma),
                    runtime_seconds=float(time.perf_counter() - round_start),
                    mutations_applied=tuple(result.applied_mutations for result in mutation_results),
                    crossovers_applied=len(crossover_events),
                )
                round_records.append(round_record)
                if self.config.algorithm.log_every_round:
                    metrics_handle.write(json.dumps(_json_ready(asdict(round_record))) + "\n")

            if not self.config.algorithm.log_every_round:
                for round_record in round_records:
                    metrics_handle.write(json.dumps(_json_ready(asdict(round_record))) + "\n")

        best_population_index = int(np.argmin(global_losses))
        best_population = populations[best_population_index]
        synthetic_frame = dataset.decode(best_population)
        synthetic_path = Path(self.config.output.synthetic_data_path)
        synthetic_frame.to_csv(synthetic_path, index=False)

        evaluation = evaluate_synthetic_data(dataset.encoded, best_population, dataset.schema, workload)
        summary = self._build_summary(
            dataset,
            workload,
            accountant,
            best_population_index,
            evaluation,
            round_records,
            total_rho,
        )
        summary_path = Path(self.config.output.summary_path)
        with summary_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_json_ready(summary), handle, sort_keys=False)

        return RunResult(
            config=config_to_dict(self.config),
            summary=summary,
            rounds=tuple(round_records),
            synthetic_data_path=str(synthetic_path),
            metrics_path=str(metrics_path),
            summary_path=str(summary_path),
        )

    def _measure_initialization_vectors(
        self,
        real_data: np.ndarray,
        schema: DiscreteSchema,
        init_vectors: tuple[QueryVector, ...],
        rho_per_init_vector: float,
        store: MeasurementStore,
        accountant: ZCDPAccountant,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        probabilities: dict[str, np.ndarray] = {}
        for vector in init_vectors:
            measurement_event = gaussian_measure_vector(real_data, vector, schema, rho_per_init_vector, rng)
            store.record_vector_measurement(vector, measurement_event.noisy_answers, measurement_event.variances)
            accountant.spend(
                f"init:{vector.vector_id}",
                rho_per_init_vector,
                metadata={"vector_id": vector.vector_id, "sigma": measurement_event.sigma},
            )
            means, _ = store.get_vector_estimates(vector)
            column_name = vector.metadata["columns"][0]
            probabilities[column_name] = _sanitize_probability_vector(means)
        return probabilities

    def _selection_candidates(
        self,
        workload: QueryWorkload,
        real_answers_by_vector: dict[str, np.ndarray],
        synthetic_population: np.ndarray,
        schema: DiscreteSchema,
        dataset_size: int,
    ) -> list[SelectionCandidate]:
        candidates: list[SelectionCandidate] = []
        for vector in workload.vectors:
            synthetic_answers = evaluate_vector_answers(synthetic_population, vector, schema, normalize=True)
            candidates.append(
                score_vector(
                    real_answers_by_vector[vector.vector_id],
                    synthetic_answers,
                    vector,
                    dataset_size,
                    self.config.selection,
                )
            )
        return candidates

    def _population_global_losses(
        self,
        populations: np.ndarray,
        store: MeasurementStore,
        schema: DiscreteSchema,
    ) -> np.ndarray:
        measured_query_ids = store.measured_query_ids()
        query_registry = store.query_registry()
        losses = np.zeros(populations.shape[0], dtype=float)
        if not measured_query_ids:
            return losses

        for population_index, population in enumerate(populations):
            squared_errors: list[float] = []
            for query_id in measured_query_ids:
                estimate = store.get_query_estimate(query_id)
                synthetic_answer = evaluate_query_answer(population, query_registry[query_id], schema, normalize=True)
                squared_errors.append((estimate.value - synthetic_answer) ** 2)
            losses[population_index] = float(np.mean(squared_errors))
        return losses

    def _build_summary(
        self,
        dataset: DiscreteDataset,
        workload: QueryWorkload,
        accountant: ZCDPAccountant,
        best_population_index: int,
        evaluation: EvaluationResult,
        round_records: list[RoundRecord],
        total_rho: float,
    ) -> dict[str, Any]:
        return {
            "dataset_path": dataset.path,
            "num_records": dataset.num_records,
            "num_columns": dataset.num_columns,
            "device": self.config.algorithm.device,
            "total_rho_configured": total_rho,
            "total_rho_spent": accountant.spent_rho,
            "epsilon_spent_at_delta": accountant.epsilon_delta(self.config.privacy.delta),
            "delta": self.config.privacy.delta,
            "rounds_completed": len(round_records),
            "population_size": self.config.algorithm.population_size,
            "best_population_index": best_population_index,
            "workload_vectors": len(workload.vectors),
            "evaluation": _json_ready(asdict(evaluation)),
        }


def _sample_initial_population(
    schema: DiscreteSchema,
    probabilities_by_column: dict[str, np.ndarray],
    synthetic_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    population = np.zeros((synthetic_size, len(schema.columns)), dtype=np.int64)
    for column_index, column in enumerate(schema.columns):
        probabilities = probabilities_by_column[column.name]
        population[:, column_index] = rng.choice(column.cardinality, size=synthetic_size, p=probabilities)
    return population


def _sanitize_probability_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    clipped = np.clip(values, 0.0, None)
    total = float(np.sum(clipped))
    if total <= 0.0:
        return np.full(values.shape, 1.0 / len(values), dtype=float)
    return clipped / total


def _query_registry_from_vectors(vectors: tuple[QueryVector, ...]) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for vector in vectors:
        for query in vector.queries:
            registry[query.query_id] = query
    return registry


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value
