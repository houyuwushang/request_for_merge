from __future__ import annotations

import unittest

import numpy as np

from private_de_v2.config import WorkloadConfig
from private_de_v2.data import DiscreteColumn, DiscreteSchema
from private_de_v2.queries import build_workload, evaluate_vector_answers


class QueryConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = DiscreteSchema(
            columns=(
                DiscreteColumn(name="a", kind="categorical", labels=("x", "y"), ordered=False),
                DiscreteColumn(name="b", kind="ordinal", labels=("0", "1", "2"), ordered=True),
            )
        )
        self.data = np.array([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=np.int64)

    def test_workload_construction_and_answers(self) -> None:
        workload = build_workload(
            self.schema,
            WorkloadConfig(
                families=["1way", "2way", "range", "prefix"],
                max_vector_size=8,
                range_widths=[2],
                prefix_thresholds_per_feature=1,
            ),
            no_orthogonal_grouping=False,
        )
        vector_ids = set(workload.vector_ids)
        self.assertIn("1way::a", vector_ids)
        self.assertIn("2way::a__b", vector_ids)
        self.assertIn("prefix::b::threshold=0", vector_ids)
        self.assertIn("range::b::width=2", vector_ids)

        two_way = workload.get_vector("2way::a__b")
        self.assertTrue(two_way.orthogonal)
        self.assertEqual(len(two_way.queries), 6)
        answers = evaluate_vector_answers(self.data, two_way, self.schema, normalize=True)
        self.assertAlmostEqual(float(np.sum(answers)), 1.0)

    def test_no_orthogonal_grouping_creates_singletons(self) -> None:
        workload = build_workload(
            self.schema,
            WorkloadConfig(families=["1way"], max_vector_size=8),
            no_orthogonal_grouping=True,
        )
        self.assertTrue(all(len(vector.queries) == 1 for vector in workload.vectors))
        self.assertTrue(all(vector.grouping_mode == "singleton" for vector in workload.vectors))


if __name__ == "__main__":
    unittest.main()
