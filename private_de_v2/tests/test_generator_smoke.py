from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from private_de_v2.config import AlgorithmConfig, DataConfig, OutputConfig, PrivacyConfig, RunConfig, WorkloadConfig
from private_de_v2.generator import PrivateDEGeneratorV2


class GeneratorSmokeTests(unittest.TestCase):
    def test_fixed_seed_is_deterministic(self) -> None:
        rows = [
            {"a": "x", "b": 0},
            {"a": "x", "b": 1},
            {"a": "y", "b": 0},
            {"a": "y", "b": 1},
            {"a": "y", "b": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "toy.csv"
            pd.DataFrame(rows).to_csv(dataset_path, index=False)

            first = self._run_once(dataset_path, tmp_path / "run1")
            second = self._run_once(dataset_path, tmp_path / "run2")

            self.assertEqual(Path(first.synthetic_data_path).read_text(encoding="utf-8"), Path(second.synthetic_data_path).read_text(encoding="utf-8"))
            self.assertEqual(first.summary["evaluation"], second.summary["evaluation"])

    def _run_once(self, dataset_path: Path, output_dir: Path):
        config = RunConfig(
            data=DataConfig(dataset_path=str(dataset_path), output_dir=str(output_dir), default_numeric_bins=2),
            workload=WorkloadConfig(families=["1way", "prefix"], prefix_thresholds_per_feature=1),
            privacy=PrivacyConfig(epsilon=1.0, delta=1e-6, initialization_rho_fraction=0.5, selection_rho_fraction=0.5),
            algorithm=AlgorithmConfig(rounds=2, population_size=2, synthetic_size=5, max_mutations_per_round=2, seed=11),
            output=OutputConfig(),
        )
        return PrivateDEGeneratorV2(config).run()


if __name__ == "__main__":
    unittest.main()
