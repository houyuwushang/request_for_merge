from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ColumnConfig, DataConfig


@dataclass(frozen=True)
class DiscreteColumn:
    name: str
    kind: str
    labels: tuple[str, ...]
    ordered: bool
    bin_edges: tuple[float, ...] | None = None
    source_values: tuple[Any, ...] | None = None

    @property
    def cardinality(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class DiscreteSchema:
    columns: tuple[DiscreteColumn, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def domain_sizes(self) -> tuple[int, ...]:
        return tuple(column.cardinality for column in self.columns)

    def index_of(self, column_name: str) -> int:
        try:
            return self.column_names.index(column_name)
        except ValueError as exc:
            raise KeyError(f"Unknown column: {column_name}") from exc

    def column(self, column_name: str) -> DiscreteColumn:
        return self.columns[self.index_of(column_name)]

    def decode_array(self, encoded: np.ndarray) -> pd.DataFrame:
        if encoded.ndim != 2 or encoded.shape[1] != len(self.columns):
            raise ValueError("Encoded array shape does not match schema")
        data: dict[str, list[str]] = {}
        for index, column in enumerate(self.columns):
            values = np.asarray(encoded[:, index], dtype=np.int64)
            data[column.name] = [column.labels[int(value)] for value in values]
        return pd.DataFrame(data)


@dataclass(frozen=True)
class DiscreteDataset:
    path: str
    schema: DiscreteSchema
    encoded: np.ndarray
    original: pd.DataFrame

    @property
    def num_records(self) -> int:
        return int(self.encoded.shape[0])

    @property
    def num_columns(self) -> int:
        return int(self.encoded.shape[1])

    def decode(self, encoded: np.ndarray) -> pd.DataFrame:
        return self.schema.decode_array(encoded)


def load_discrete_dataset(config: DataConfig) -> DiscreteDataset:
    frame = pd.read_csv(config.dataset_path)
    if config.drop_missing:
        frame = frame.dropna(axis=0).reset_index(drop=True)
    elif frame.isnull().any().any():
        raise ValueError("Missing values are present and data.drop_missing is False")

    encoded_columns: list[np.ndarray] = []
    schema_columns: list[DiscreteColumn] = []
    for column_name in frame.columns:
        column_config = config.columns.get(column_name, ColumnConfig())
        encoded, schema_column = _encode_column(
            frame[column_name],
            column_config,
            config.default_numeric_bins,
            config.discretization_strategy,
        )
        encoded_columns.append(encoded)
        schema_columns.append(schema_column)

    encoded_dataset = np.column_stack(encoded_columns).astype(np.int64, copy=False)
    return DiscreteDataset(
        path=str(Path(config.dataset_path).resolve()),
        schema=DiscreteSchema(tuple(schema_columns)),
        encoded=encoded_dataset,
        original=frame,
    )


def _encode_column(
    series: pd.Series,
    column_config: ColumnConfig,
    default_numeric_bins: int,
    discretization_strategy: str,
) -> tuple[np.ndarray, DiscreteColumn]:
    kind = column_config.kind
    if kind == "auto":
        kind = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"

    if kind in {"numeric", "ordinal"}:
        return _encode_numeric_column(series, kind, column_config, default_numeric_bins, discretization_strategy)
    if kind == "categorical":
        return _encode_categorical_column(series, column_config)
    raise ValueError(f"Unsupported column kind: {kind}")


def _encode_categorical_column(series: pd.Series, column_config: ColumnConfig) -> tuple[np.ndarray, DiscreteColumn]:
    values = series.astype(str)
    categories = list(column_config.categories or pd.unique(values))
    category_to_index = {category: index for index, category in enumerate(categories)}
    unseen = sorted(set(values) - set(categories))
    if unseen:
        raise ValueError(f"Configured categories for column {series.name!r} miss values: {unseen}")
    encoded = values.map(category_to_index).to_numpy(dtype=np.int64)
    column = DiscreteColumn(
        name=str(series.name),
        kind="categorical",
        labels=tuple(str(category) for category in categories),
        ordered=False,
        source_values=tuple(categories),
    )
    return encoded, column


def _encode_numeric_column(
    series: pd.Series,
    kind: str,
    column_config: ColumnConfig,
    default_numeric_bins: int,
    discretization_strategy: str,
) -> tuple[np.ndarray, DiscreteColumn]:
    numeric = pd.to_numeric(series, errors="raise")
    requested_bins = column_config.bins or default_numeric_bins
    unique_values = np.sort(numeric.dropna().unique())

    if kind == "ordinal" and column_config.bin_edges is None and len(unique_values) <= requested_bins:
        value_to_index = {value: index for index, value in enumerate(unique_values)}
        encoded = np.array([value_to_index[value] for value in numeric.to_numpy()], dtype=np.int64)
        labels = tuple(str(value) for value in unique_values)
        column = DiscreteColumn(
            name=str(series.name),
            kind="ordinal",
            labels=labels,
            ordered=True,
            source_values=tuple(float(value) for value in unique_values),
        )
        return encoded, column

    if column_config.bin_edges:
        edges = np.asarray(column_config.bin_edges, dtype=float)
    else:
        edges = _make_bin_edges(numeric.to_numpy(dtype=float), requested_bins, discretization_strategy)

    encoded = pd.cut(
        numeric,
        bins=edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    ).to_numpy()
    if np.isnan(encoded).any():
        raise ValueError(f"Column {series.name!r} could not be discretized with edges {edges.tolist()}")

    edges = np.asarray(edges, dtype=float)
    labels = tuple(_format_interval(edges[index], edges[index + 1]) for index in range(len(edges) - 1))
    column = DiscreteColumn(
        name=str(series.name),
        kind=kind,
        labels=labels,
        ordered=True,
        bin_edges=tuple(float(edge) for edge in edges),
    )
    return encoded.astype(np.int64), column


def _make_bin_edges(values: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    if bins < 1:
        raise ValueError("The number of bins must be at least 1")

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        return np.array([minimum - 0.5, maximum + 0.5], dtype=float)

    if strategy == "equal_width":
        return np.linspace(minimum, maximum, bins + 1, dtype=float)
    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, bins + 1, dtype=float)
        edges = np.quantile(values, quantiles)
        edges[0] = minimum
        edges[-1] = maximum
        edges = np.unique(edges)
        if len(edges) < 2:
            return np.array([minimum - 0.5, maximum + 0.5], dtype=float)
        return edges
    raise ValueError(f"Unsupported discretization strategy: {strategy}")


def _format_interval(lower: float, upper: float) -> str:
    return f"[{lower:.6g}, {upper:.6g}]"
