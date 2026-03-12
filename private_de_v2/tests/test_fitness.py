from __future__ import annotations

import unittest

import numpy as np

from private_de_v2.data import DiscreteColumn, DiscreteSchema
from private_de_v2.fitness import compute_record_fitness, compute_signed_errors
from private_de_v2.queries import Query, QueryVector


class FitnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = DiscreteSchema(columns=(DiscreteColumn(name="x", kind="categorical", labels=("a", "b"), ordered=False),))
        self.vector = QueryVector(
            vector_id="1way::x",
            family="1way",
            queries=(
                Query("1way::x::0", "1way", "x=a", {"assignments": {"x": 0}}),
                Query("1way::x::1", "1way", "x=b", {"assignments": {"x": 1}}),
            ),
            orthogonal=True,
            exhaustive=True,
        )

    def test_signed_error_convention(self) -> None:
        noisy = np.array([0.75, 0.25], dtype=float)
        synthetic = np.array([0.50, 0.50], dtype=float)
        errors = compute_signed_errors(noisy, synthetic)
        self.assertTrue(np.allclose(errors, np.array([0.25, -0.25])))
        self.assertGreater(errors[0], 0.0)
        self.assertLess(errors[1], 0.0)

    def test_record_fitness_uses_selected_query_error(self) -> None:
        data = np.array([[0], [1], [1]], dtype=np.int64)
        errors = np.array([0.2, -0.3], dtype=float)
        result = compute_record_fitness(data, self.vector, errors, self.schema)
        self.assertTrue(np.allclose(result.record_fitness, np.array([0.2, -0.3, -0.3])))


if __name__ == "__main__":
    unittest.main()
