from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any

import numpy as np

from .config import WorkloadConfig
from .data import DiscreteColumn, DiscreteSchema


@dataclass(frozen=True)
class Query:
    query_id: str
    family: str
    label: str
    payload: dict[str, Any]


@dataclass
class QueryVector:
    vector_id: str
    family: str
    queries: tuple[Query, ...]
    orthogonal: bool
    exhaustive: bool
    approximate: bool = False
    grouping_mode: str = "orthogonal"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryWorkload:
    vectors: tuple[QueryVector, ...]

    @property
    def vector_ids(self) -> tuple[str, ...]:
        return tuple(vector.vector_id for vector in self.vectors)

    def get_vector(self, vector_id: str) -> QueryVector:
        for vector in self.vectors:
            if vector.vector_id == vector_id:
                return vector
        raise KeyError(f"Unknown vector id: {vector_id}")

    def query_registry(self) -> dict[str, Query]:
        registry: dict[str, Query] = {}
        for vector in self.vectors:
            for query in vector.queries:
                registry[query.query_id] = query
        return registry


def build_initialization_vectors(schema: DiscreteSchema) -> tuple[QueryVector, ...]:
    return tuple(_build_oneway_vectors(schema, max_vector_size=None))


def build_workload(
    schema: DiscreteSchema,
    config: WorkloadConfig,
    *,
    no_orthogonal_grouping: bool,
) -> QueryWorkload:
    vectors: list[QueryVector] = []
    families = list(dict.fromkeys(config.families))
    for family in families:
        if family == "1way":
            vectors.extend(_build_oneway_vectors(schema, config.max_vector_size))
        elif family == "2way":
            vectors.extend(_build_kway_vectors(schema, 2, config.max_vector_size))
        elif family == "3way":
            vectors.extend(_build_kway_vectors(schema, 3, config.max_vector_size))
        elif family == "range":
            vectors.extend(_build_range_vectors(schema, config.range_widths))
        elif family == "prefix":
            vectors.extend(_build_prefix_vectors(schema, config.prefix_thresholds_per_feature))
        elif family == "conditional_prefix":
            vectors.extend(
                _build_conditional_prefix_vectors(
                    schema,
                    config.conditional_prefix_thresholds_per_feature,
                    config.conditional_prefix_max_condition_values,
                )
            )
        elif family == "halfspace":
            vectors.extend(
                _build_halfspace_vectors(
                    schema,
                    config.halfspace_thresholds_per_pair,
                    config.halfspace_pairs,
                )
            )
        else:
            raise ValueError(f"Unsupported workload family: {family}")

    if no_orthogonal_grouping:
        singleton_vectors: list[QueryVector] = []
        for vector in vectors:
            for index, query in enumerate(vector.queries):
                singleton_vectors.append(
                    QueryVector(
                        vector_id=f"{vector.vector_id}::singleton::{index}",
                        family=vector.family,
                        queries=(query,),
                        orthogonal=False,
                        exhaustive=False,
                        approximate=vector.approximate,
                        grouping_mode="singleton",
                        metadata={**vector.metadata, "base_vector_id": vector.vector_id},
                    )
                )
        vectors = singleton_vectors

    return QueryWorkload(tuple(vectors))


def evaluate_query_mask(data: np.ndarray, query: Query, schema: DiscreteSchema) -> np.ndarray:
    payload = query.payload
    if query.family in {"1way", "2way", "3way"}:
        mask = np.ones(data.shape[0], dtype=bool)
        for column_name, value in payload["assignments"].items():
            mask &= data[:, schema.index_of(column_name)] == int(value)
        return mask
    if query.family == "range":
        column_index = schema.index_of(payload["column"])
        return (data[:, column_index] >= int(payload["lower"])) & (data[:, column_index] <= int(payload["upper"]))
    if query.family == "prefix":
        column_index = schema.index_of(payload["column"])
        threshold = int(payload["threshold"])
        if payload["direction"] == "le":
            return data[:, column_index] <= threshold
        return data[:, column_index] > threshold
    if query.family == "conditional_prefix":
        condition_index = schema.index_of(payload["condition_column"])
        target_index = schema.index_of(payload["target_column"])
        condition_value = int(payload["condition_value"])
        threshold = int(payload["threshold"])
        condition_match = data[:, condition_index] == condition_value
        state = payload["state"]
        if state == "condition_off":
            return ~condition_match
        if state == "le":
            return condition_match & (data[:, target_index] <= threshold)
        return condition_match & (data[:, target_index] > threshold)
    if query.family == "halfspace":
        first_index = schema.index_of(payload["columns"][0])
        second_index = schema.index_of(payload["columns"][1])
        score = (int(payload["weights"][0]) * data[:, first_index]) + (int(payload["weights"][1]) * data[:, second_index])
        if payload["direction"] == "le":
            return score <= int(payload["threshold"])
        return score > int(payload["threshold"])
    raise ValueError(f"Unsupported query family: {query.family}")


def evaluate_query_answer(
    data: np.ndarray,
    query: Query,
    schema: DiscreteSchema,
    *,
    normalize: bool = True,
) -> float:
    mask = evaluate_query_mask(data, query, schema)
    answer = float(np.count_nonzero(mask))
    if normalize:
        answer /= float(data.shape[0])
    return answer


def evaluate_vector_answers(
    data: np.ndarray,
    vector: QueryVector,
    schema: DiscreteSchema,
    *,
    normalize: bool = True,
) -> np.ndarray:
    assignments = assign_records_to_vector(data, vector, schema)
    if vector.orthogonal and np.all(assignments >= 0):
        counts = np.bincount(assignments, minlength=len(vector.queries)).astype(float)
        if normalize:
            counts /= float(data.shape[0])
        return counts

    answers = [evaluate_query_answer(data, query, schema, normalize=normalize) for query in vector.queries]
    return np.asarray(answers, dtype=float)


def assign_records_to_vector(data: np.ndarray, vector: QueryVector, schema: DiscreteSchema) -> np.ndarray:
    assignments = np.full(data.shape[0], -1, dtype=np.int64)
    for query_index, query in enumerate(vector.queries):
        mask = evaluate_query_mask(data, query, schema)
        if vector.orthogonal and np.any(mask & (assignments >= 0)):
            raise ValueError(f"Vector {vector.vector_id} is marked orthogonal but has overlapping queries")
        assignments[mask] = query_index
    return assignments


def project_row_to_query(row: np.ndarray, query: Query, schema: DiscreteSchema) -> np.ndarray:
    projected = np.asarray(row, dtype=np.int64).copy()
    payload = query.payload
    if query.family in {"1way", "2way", "3way"}:
        for column_name, value in payload["assignments"].items():
            projected[schema.index_of(column_name)] = int(value)
        return projected
    if query.family == "range":
        column_index = schema.index_of(payload["column"])
        projected[column_index] = int(np.clip(projected[column_index], int(payload["lower"]), int(payload["upper"])))
        return projected
    if query.family == "prefix":
        return _project_row_to_prefix(projected, payload, schema)
    if query.family == "conditional_prefix":
        return _project_row_to_conditional_prefix(projected, payload, schema)
    if query.family == "halfspace":
        return _project_row_to_halfspace(projected, payload, schema)
    raise ValueError(f"Unsupported query family: {query.family}")


def project_row_out_of_query(row: np.ndarray, query: Query, schema: DiscreteSchema) -> np.ndarray:
    projected = np.asarray(row, dtype=np.int64).copy()
    payload = query.payload
    if query.family in {"1way", "2way", "3way"}:
        first_column, first_value = next(iter(payload["assignments"].items()))
        column_index = schema.index_of(first_column)
        projected[column_index] = _nearest_alternative_value(schema.column(first_column), projected[column_index], int(first_value))
        return projected
    if query.family == "range":
        column_name = payload["column"]
        column_index = schema.index_of(column_name)
        lower = int(payload["lower"])
        upper = int(payload["upper"])
        if lower > 0:
            projected[column_index] = lower - 1
        else:
            projected[column_index] = min(schema.column(column_name).cardinality - 1, upper + 1)
        return projected
    if query.family == "prefix":
        column_name = payload["column"]
        column_index = schema.index_of(column_name)
        threshold = int(payload["threshold"])
        column = schema.column(column_name)
        if payload["direction"] == "le":
            projected[column_index] = min(column.cardinality - 1, threshold + 1)
        else:
            projected[column_index] = max(0, threshold)
        return projected
    if query.family == "conditional_prefix":
        condition_column = payload["condition_column"]
        condition_index = schema.index_of(condition_column)
        condition_value = int(payload["condition_value"])
        target_column = payload["target_column"]
        target_index = schema.index_of(target_column)
        threshold = int(payload["threshold"])
        if payload["state"] == "condition_off":
            projected[condition_index] = condition_value
            projected[target_index] = min(schema.column(target_column).cardinality - 1, threshold)
            return projected
        if payload["state"] == "le":
            projected[target_index] = min(schema.column(target_column).cardinality - 1, threshold + 1)
            return projected
        projected[target_index] = max(0, threshold)
        return projected
    if query.family == "halfspace":
        return _project_row_out_of_halfspace(projected, payload, schema)
    raise ValueError(f"Unsupported query family: {query.family}")


def _project_row_to_prefix(row: np.ndarray, payload: dict[str, Any], schema: DiscreteSchema) -> np.ndarray:
    column_name = payload["column"]
    column_index = schema.index_of(column_name)
    column = schema.column(column_name)
    threshold = int(payload["threshold"])
    if payload["direction"] == "le":
        row[column_index] = min(row[column_index], threshold)
    else:
        row[column_index] = max(row[column_index], min(column.cardinality - 1, threshold + 1))
    return row


def _project_row_to_conditional_prefix(row: np.ndarray, payload: dict[str, Any], schema: DiscreteSchema) -> np.ndarray:
    condition_index = schema.index_of(payload["condition_column"])
    target_index = schema.index_of(payload["target_column"])
    condition_value = int(payload["condition_value"])
    threshold = int(payload["threshold"])
    target_column = schema.column(payload["target_column"])
    state = payload["state"]

    if state == "condition_off":
        row[condition_index] = _nearest_alternative_value(
            schema.column(payload["condition_column"]),
            row[condition_index],
            condition_value,
        )
        return row

    row[condition_index] = condition_value
    if state == "le":
        row[target_index] = min(row[target_index], threshold)
    else:
        row[target_index] = max(row[target_index], min(target_column.cardinality - 1, threshold + 1))
    return row


def _project_row_to_halfspace(row: np.ndarray, payload: dict[str, Any], schema: DiscreteSchema) -> np.ndarray:
    first_name, second_name = payload["columns"]
    first_index = schema.index_of(first_name)
    second_index = schema.index_of(second_name)
    first_column = schema.column(first_name)
    second_column = schema.column(second_name)
    threshold = int(payload["threshold"])
    score = row[first_index] + row[second_index]
    if payload["direction"] == "le" and score <= threshold:
        return row
    if payload["direction"] == "gt" and score > threshold:
        return row

    if payload["direction"] == "le":
        max_second = max(0, threshold - row[first_index])
        row[second_index] = min(row[second_index], min(second_column.cardinality - 1, max_second))
        if row[first_index] + row[second_index] > threshold:
            max_first = max(0, threshold - row[second_index])
            row[first_index] = min(row[first_index], min(first_column.cardinality - 1, max_first))
    else:
        min_second = max(0, threshold + 1 - row[first_index])
        row[second_index] = max(row[second_index], min(second_column.cardinality - 1, min_second))
        if row[first_index] + row[second_index] <= threshold:
            min_first = max(0, threshold + 1 - row[second_index])
            row[first_index] = max(row[first_index], min(first_column.cardinality - 1, min_first))
    return row


def _project_row_out_of_halfspace(row: np.ndarray, payload: dict[str, Any], schema: DiscreteSchema) -> np.ndarray:
    flipped_payload = dict(payload)
    flipped_payload["direction"] = "gt" if payload["direction"] == "le" else "le"
    return _project_row_to_halfspace(row, flipped_payload, schema)


def _nearest_alternative_value(column: DiscreteColumn, current_value: int, forbidden_value: int) -> int:
    alternatives = [value for value in range(column.cardinality) if value != forbidden_value]
    if not alternatives:
        return current_value
    return min(alternatives, key=lambda value: abs(value - current_value))


def _build_oneway_vectors(schema: DiscreteSchema, max_vector_size: int | None) -> list[QueryVector]:
    vectors: list[QueryVector] = []
    for column in schema.columns:
        if max_vector_size is not None and column.cardinality > max_vector_size:
            continue
        queries = []
        for value in range(column.cardinality):
            queries.append(
                Query(
                    query_id=f"1way::{column.name}::{value}",
                    family="1way",
                    label=f"{column.name}={column.labels[value]}",
                    payload={"assignments": {column.name: value}},
                )
            )
        vectors.append(
            QueryVector(
                vector_id=f"1way::{column.name}",
                family="1way",
                queries=tuple(queries),
                orthogonal=True,
                exhaustive=True,
                metadata={"columns": [column.name]},
            )
        )
    return vectors


def _build_kway_vectors(schema: DiscreteSchema, k: int, max_vector_size: int) -> list[QueryVector]:
    vectors: list[QueryVector] = []
    for column_group in combinations(schema.columns, k):
        vector_size = int(np.prod([column.cardinality for column in column_group], dtype=np.int64))
        if vector_size > max_vector_size:
            continue
        queries: list[Query] = []
        assignments_template = [range(column.cardinality) for column in column_group]
        for values in product(*assignments_template):
            assignments = {column.name: int(value) for column, value in zip(column_group, values)}
            query_id = "::".join([f"{column.name}={int(value)}" for column, value in zip(column_group, values)])
            queries.append(
                Query(
                    query_id=f"{k}way::{query_id}",
                    family=f"{k}way",
                    label=", ".join(f"{column.name}={column.labels[value]}" for column, value in zip(column_group, values)),
                    payload={"assignments": assignments},
                )
            )
        vectors.append(
            QueryVector(
                vector_id=f"{k}way::{'__'.join(column.name for column in column_group)}",
                family=f"{k}way",
                queries=tuple(queries),
                orthogonal=True,
                exhaustive=True,
                metadata={"columns": [column.name for column in column_group]},
            )
        )
    return vectors


def _build_range_vectors(schema: DiscreteSchema, widths: list[int]) -> list[QueryVector]:
    vectors: list[QueryVector] = []
    ordered_columns = [column for column in schema.columns if column.ordered]
    for column in ordered_columns:
        for width in widths:
            if width < 1 or width >= column.cardinality:
                continue
            queries: list[Query] = []
            start = 0
            while start < column.cardinality:
                end = min(column.cardinality - 1, start + width - 1)
                queries.append(
                    Query(
                        query_id=f"range::{column.name}::{start}::{end}",
                        family="range",
                        label=f"{column.name} in [{column.labels[start]}, {column.labels[end]}]",
                        payload={"column": column.name, "lower": start, "upper": end},
                    )
                )
                start = end + 1
            if len(queries) > 1:
                vectors.append(
                    QueryVector(
                        vector_id=f"range::{column.name}::width={width}",
                        family="range",
                        queries=tuple(queries),
                        orthogonal=True,
                        exhaustive=True,
                        metadata={"columns": [column.name], "width": width},
                    )
                )
    return vectors


def _build_prefix_vectors(schema: DiscreteSchema, thresholds_per_feature: int) -> list[QueryVector]:
    vectors: list[QueryVector] = []
    for column in schema.columns:
        if not column.ordered or column.cardinality < 2:
            continue
        for threshold in _interior_thresholds(column.cardinality, thresholds_per_feature):
            queries = (
                Query(
                    query_id=f"prefix::{column.name}::le::{threshold}",
                    family="prefix",
                    label=f"{column.name} <= {column.labels[threshold]}",
                    payload={"column": column.name, "threshold": threshold, "direction": "le"},
                ),
                Query(
                    query_id=f"prefix::{column.name}::gt::{threshold}",
                    family="prefix",
                    label=f"{column.name} > {column.labels[threshold]}",
                    payload={"column": column.name, "threshold": threshold, "direction": "gt"},
                ),
            )
            vectors.append(
                QueryVector(
                    vector_id=f"prefix::{column.name}::threshold={threshold}",
                    family="prefix",
                    queries=queries,
                    orthogonal=True,
                    exhaustive=True,
                    metadata={"columns": [column.name], "threshold": threshold},
                )
            )
    return vectors


def _build_conditional_prefix_vectors(
    schema: DiscreteSchema,
    thresholds_per_feature: int,
    max_condition_values: int,
) -> list[QueryVector]:
    vectors: list[QueryVector] = []
    ordered_columns = [column for column in schema.columns if column.ordered and column.cardinality >= 2]
    for condition_column in schema.columns:
        condition_values = _representative_values(condition_column.cardinality, max_condition_values)
        for target_column in ordered_columns:
            if target_column.name == condition_column.name:
                continue
            for condition_value in condition_values:
                for threshold in _interior_thresholds(target_column.cardinality, thresholds_per_feature):
                    queries = (
                        Query(
                            query_id=f"conditional_prefix::{condition_column.name}::{condition_value}::off::{target_column.name}::{threshold}",
                            family="conditional_prefix",
                            label=f"{condition_column.name}!={condition_column.labels[condition_value]}",
                            payload={
                                "condition_column": condition_column.name,
                                "condition_value": condition_value,
                                "target_column": target_column.name,
                                "threshold": threshold,
                                "state": "condition_off",
                            },
                        ),
                        Query(
                            query_id=f"conditional_prefix::{condition_column.name}::{condition_value}::le::{target_column.name}::{threshold}",
                            family="conditional_prefix",
                            label=f"{condition_column.name}={condition_column.labels[condition_value]} and {target_column.name}<={target_column.labels[threshold]}",
                            payload={
                                "condition_column": condition_column.name,
                                "condition_value": condition_value,
                                "target_column": target_column.name,
                                "threshold": threshold,
                                "state": "le",
                            },
                        ),
                        Query(
                            query_id=f"conditional_prefix::{condition_column.name}::{condition_value}::gt::{target_column.name}::{threshold}",
                            family="conditional_prefix",
                            label=f"{condition_column.name}={condition_column.labels[condition_value]} and {target_column.name}>{target_column.labels[threshold]}",
                            payload={
                                "condition_column": condition_column.name,
                                "condition_value": condition_value,
                                "target_column": target_column.name,
                                "threshold": threshold,
                                "state": "gt",
                            },
                        ),
                    )
                    vectors.append(
                        QueryVector(
                            vector_id=(
                                f"conditional_prefix::{condition_column.name}::{condition_value}"
                                f"::{target_column.name}::threshold={threshold}"
                            ),
                            family="conditional_prefix",
                            queries=queries,
                            orthogonal=True,
                            exhaustive=True,
                            metadata={
                                "columns": [condition_column.name, target_column.name],
                                "condition_value": condition_value,
                                "threshold": threshold,
                            },
                        )
                    )
    return vectors


def _build_halfspace_vectors(
    schema: DiscreteSchema,
    thresholds_per_pair: int,
    configured_pairs: list[list[str]],
) -> list[QueryVector]:
    vectors: list[QueryVector] = []
    ordered_columns = [column.name for column in schema.columns if column.ordered]
    if configured_pairs:
        pairs = [tuple(pair) for pair in configured_pairs]
    else:
        pairs = list(combinations(ordered_columns, 2))

    for first_name, second_name in pairs:
        first_column = schema.column(first_name)
        second_column = schema.column(second_name)
        max_score = (first_column.cardinality - 1) + (second_column.cardinality - 1)
        for threshold in _interior_thresholds(max_score + 1, thresholds_per_pair):
            queries = (
                Query(
                    query_id=f"halfspace::{first_name}::{second_name}::le::{threshold}",
                    family="halfspace",
                    label=f"{first_name}+{second_name} <= {threshold}",
                    payload={
                        "columns": [first_name, second_name],
                        "weights": [1, 1],
                        "threshold": threshold,
                        "direction": "le",
                    },
                ),
                Query(
                    query_id=f"halfspace::{first_name}::{second_name}::gt::{threshold}",
                    family="halfspace",
                    label=f"{first_name}+{second_name} > {threshold}",
                    payload={
                        "columns": [first_name, second_name],
                        "weights": [1, 1],
                        "threshold": threshold,
                        "direction": "gt",
                    },
                ),
            )
            vectors.append(
                QueryVector(
                    vector_id=f"halfspace::{first_name}::{second_name}::threshold={threshold}",
                    family="halfspace",
                    queries=queries,
                    orthogonal=True,
                    exhaustive=True,
                    metadata={"columns": [first_name, second_name], "threshold": threshold},
                )
            )
    return vectors


def _interior_thresholds(cardinality: int, count: int) -> list[int]:
    if cardinality < 2 or count < 1:
        return []
    candidates = np.linspace(0, cardinality - 2, num=min(count, cardinality - 1), dtype=int)
    return sorted(set(int(value) for value in candidates.tolist()))


def _representative_values(cardinality: int, count: int) -> list[int]:
    if count < 1:
        return []
    if count >= cardinality:
        return list(range(cardinality))
    values = np.linspace(0, cardinality - 1, num=count, dtype=int)
    return sorted(set(int(value) for value in values.tolist()))
