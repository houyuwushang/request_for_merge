from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SelectionConfig
from .queries import QueryVector


@dataclass(frozen=True)
class SelectionCandidate:
    vector_id: str
    utility: float
    sensitivity: float


@dataclass(frozen=True)
class SelectionResult:
    vector_index: int
    vector_id: str
    utility: float
    sensitivity: float
    epsilon: float
    probabilities: np.ndarray
    utilities: np.ndarray


def score_vector(
    real_answers: np.ndarray,
    synthetic_answers: np.ndarray,
    vector: QueryVector,
    dataset_size: int,
    config: SelectionConfig,
) -> SelectionCandidate:
    if config.score_mode != "l1_gap":
        raise ValueError(f"Unsupported selection score mode: {config.score_mode}")

    raw_gap = float(np.sum(np.abs(real_answers - synthetic_answers)))
    size_weight = float(len(vector.queries) ** config.vector_size_weight_power)
    clipped_gap = min(raw_gap * size_weight, float(config.score_clip))
    sensitivity = min((2.0 / float(dataset_size)) * max(size_weight, 1.0), float(config.score_clip))
    return SelectionCandidate(vector_id=vector.vector_id, utility=clipped_gap, sensitivity=sensitivity)


def exponential_mechanism_probabilities(
    utilities: np.ndarray,
    epsilon: float,
    sensitivities: np.ndarray,
) -> np.ndarray:
    utilities = np.asarray(utilities, dtype=float)
    sensitivities = np.asarray(sensitivities, dtype=float)
    if utilities.ndim != 1 or sensitivities.ndim != 1 or utilities.shape != sensitivities.shape:
        raise ValueError("utilities and sensitivities must be one-dimensional arrays with the same shape")
    if len(utilities) == 0:
        raise ValueError("At least one utility is required")

    if epsilon <= 0.0:
        return np.full_like(utilities, 1.0 / len(utilities), dtype=float)

    safe_sensitivity = np.maximum(sensitivities, 1e-12)
    logits = (epsilon * utilities) / (2.0 * safe_sensitivity)
    logits -= np.max(logits)
    weights = np.exp(logits)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.full_like(utilities, 1.0 / len(utilities), dtype=float)
    return weights / weight_sum


def select_query_vector(
    candidates: list[SelectionCandidate],
    rho: float,
    rng: np.random.Generator,
) -> SelectionResult:
    utilities = np.asarray([candidate.utility for candidate in candidates], dtype=float)
    sensitivities = np.asarray([candidate.sensitivity for candidate in candidates], dtype=float)
    epsilon = float(np.sqrt(max(0.0, 2.0 * rho)))
    probabilities = exponential_mechanism_probabilities(utilities, epsilon, sensitivities)
    vector_index = int(rng.choice(len(candidates), p=probabilities))
    selected = candidates[vector_index]
    return SelectionResult(
        vector_index=vector_index,
        vector_id=selected.vector_id,
        utility=selected.utility,
        sensitivity=selected.sensitivity,
        epsilon=epsilon,
        probabilities=probabilities,
        utilities=utilities,
    )
