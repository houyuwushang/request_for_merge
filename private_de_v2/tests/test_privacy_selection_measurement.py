from __future__ import annotations

import unittest

import numpy as np

from private_de_v2.data import DiscreteColumn, DiscreteSchema
from private_de_v2.measurement import MeasurementStore
from private_de_v2.privacy import ZCDPAccountant
from private_de_v2.queries import Query, QueryVector
from private_de_v2.selection import exponential_mechanism_probabilities


class PrivacySelectionMeasurementTests(unittest.TestCase):
    def test_privacy_budget_monotonicity(self) -> None:
        accountant = ZCDPAccountant(1.0)
        first = accountant.spend("init", 0.2)
        second = accountant.spend("round", 0.3)
        self.assertLess(first.cumulative_rho, second.cumulative_rho)
        self.assertAlmostEqual(accountant.remaining_rho, 0.5)
        with self.assertRaises(ValueError):
            accountant.spend("too_much", 0.6)

    def test_exponential_mechanism_is_numerically_stable(self) -> None:
        probabilities = exponential_mechanism_probabilities(
            np.array([1000.0, 999.0, 998.0], dtype=float),
            epsilon=8.0,
            sensitivities=np.array([0.1, 0.1, 0.1], dtype=float),
        )
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertEqual(int(np.argmax(probabilities)), 0)

    def test_inverse_variance_weighting(self) -> None:
        schema = DiscreteSchema(columns=(DiscreteColumn(name="x", kind="categorical", labels=("a", "b"), ordered=False),))
        vector = QueryVector(
            vector_id="1way::x",
            family="1way",
            queries=(Query("1way::x::0", "1way", "x=a", {"assignments": {"x": 0}}),),
            orthogonal=False,
            exhaustive=False,
        )
        store = MeasurementStore(use_inverse_variance_weighting=True)
        store.record_vector_measurement(vector, np.array([0.4], dtype=float), np.array([0.04], dtype=float))
        store.record_vector_measurement(vector, np.array([0.6], dtype=float), np.array([0.01], dtype=float))
        estimate = store.get_query_estimate("1way::x::0")
        self.assertAlmostEqual(estimate.value, 0.56, places=6)
        self.assertAlmostEqual(estimate.variance, 0.008, places=6)


if __name__ == "__main__":
    unittest.main()
