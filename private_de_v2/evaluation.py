from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DiscreteSchema
from .queries import QueryWorkload, evaluate_query_answer, evaluate_vector_answers


@dataclass(frozen=True)
class EvaluationResult:
    query_mse: float
    average_vector_tvd: float
    exact_match_share: float
    downstream_accuracy_gap: float | None = None


def evaluate_synthetic_data(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    schema: DiscreteSchema,
    workload: QueryWorkload,
) -> EvaluationResult:
    query_errors: list[float] = []
    vector_tvds: list[float] = []
    for vector in workload.vectors:
        real_answers = evaluate_vector_answers(real_data, vector, schema, normalize=True)
        synthetic_answers = evaluate_vector_answers(synthetic_data, vector, schema, normalize=True)
        query_errors.extend((real_answers - synthetic_answers) ** 2)
        vector_tvds.append(0.5 * float(np.sum(np.abs(real_answers - synthetic_answers))))

    return EvaluationResult(
        query_mse=float(np.mean(query_errors)) if query_errors else 0.0,
        average_vector_tvd=float(np.mean(vector_tvds)) if vector_tvds else 0.0,
        exact_match_share=exact_match_share(real_data, synthetic_data),
    )


def exact_match_share(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    real_rows = {tuple(int(value) for value in row) for row in np.asarray(real_data, dtype=np.int64)}
    synthetic_rows = [tuple(int(value) for value in row) for row in np.asarray(synthetic_data, dtype=np.int64)]
    if not synthetic_rows:
        return 0.0
    matches = sum(1 for row in synthetic_rows if row in real_rows)
    return float(matches) / float(len(synthetic_rows))


def downstream_accuracy_gap(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    label_column: str,
) -> float | None:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        return None

    if label_column not in real_frame.columns or label_column not in synthetic_frame.columns:
        raise KeyError(f"Unknown label column: {label_column}")

    feature_columns = [column for column in real_frame.columns if column != label_column]
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_columns),
        ]
    )
    model = Pipeline(
        steps=[
            ("transformer", transformer),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )

    train_x = synthetic_frame[feature_columns]
    train_y = synthetic_frame[label_column]
    test_x = real_frame[feature_columns]
    test_y = real_frame[label_column]
    model.fit(train_x, train_y)
    synthetic_trained_accuracy = accuracy_score(test_y, model.predict(test_x))

    model.fit(test_x, test_y)
    real_trained_accuracy = accuracy_score(test_y, model.predict(test_x))
    return float(real_trained_accuracy - synthetic_trained_accuracy)
