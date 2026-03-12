from __future__ import annotations

import unittest

import numpy as np

from private_de_v2.data import DiscreteColumn, DiscreteSchema
from private_de_v2.fitness import compute_record_fitness, compute_signed_errors
from private_de_v2.mutation import apply_directed_mutation
from private_de_v2.queries import Query, QueryVector, evaluate_vector_answers


class MutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = DiscreteSchema(columns=(DiscreteColumn(name="x", kind="categorical", labels=("a", "b", "c"), ordered=False),))
        self.vector = QueryVector(
            vector_id="1way::x",
            family="1way",
            queries=(
                Query("1way::x::0", "1way", "x=a", {"assignments": {"x": 0}}),
                Query("1way::x::1", "1way", "x=b", {"assignments": {"x": 1}}),
                Query("1way::x::2", "1way", "x=c", {"assignments": {"x": 2}}),
            ),
            orthogonal=True,
            exhaustive=True,
        )

    def test_donor_recipient_direction(self) -> None:
        data = np.array([[0], [0], [0], [0]], dtype=np.int64)
        noisy = np.array([0.25, 0.75, 0.0], dtype=float)
        synthetic = evaluate_vector_answers(data, self.vector, self.schema, normalize=True)
        fitness = compute_record_fitness(data, self.vector, compute_signed_errors(noisy, synthetic), self.schema)
        result = apply_directed_mutation(
            data,
            self.vector,
            fitness,
            self.schema,
            np.random.default_rng(5),
            max_mutations_per_round=1,
        )
        self.assertEqual(result.applied_mutations, 1)
        self.assertEqual(result.events[0].donor_query_index, 0)
        self.assertEqual(result.events[0].recipient_query_index, 1)
        self.assertEqual(int(result.mutated_data[:, 0].sum()), 1)

    def test_orthogonal_mutation_reduces_selected_vector_loss(self) -> None:
        data = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.int64)
        noisy = np.array([0.5, 0.5, 0.0], dtype=float)
        synthetic_before = evaluate_vector_answers(data, self.vector, self.schema, normalize=True)
        fitness = compute_record_fitness(data, self.vector, compute_signed_errors(noisy, synthetic_before), self.schema)
        before_loss = float(np.sum(np.abs(noisy - synthetic_before)))
        result = apply_directed_mutation(
            data,
            self.vector,
            fitness,
            self.schema,
            np.random.default_rng(2),
            max_mutations_per_round=2,
        )
        synthetic_after = evaluate_vector_answers(result.mutated_data, self.vector, self.schema, normalize=True)
        after_loss = float(np.sum(np.abs(noisy - synthetic_after)))
        self.assertLess(after_loss, before_loss)


if __name__ == "__main__":
    unittest.main()
