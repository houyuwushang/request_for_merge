from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_WORKLOAD_FAMILIES = {
    "1way",
    "2way",
    "3way",
    "range",
    "prefix",
    "conditional_prefix",
    "halfspace",
}


@dataclass
class ColumnConfig:
    kind: str = "auto"
    bins: int | None = None
    bin_edges: list[float] | None = None
    categories: list[str] | None = None


@dataclass
class DataConfig:
    dataset_path: str = ""
    output_dir: str = "outputs/private_de_v2"
    default_numeric_bins: int = 8
    discretization_strategy: str = "equal_width"
    drop_missing: bool = True
    columns: dict[str, ColumnConfig] = field(default_factory=dict)


@dataclass
class WorkloadConfig:
    families: list[str] = field(default_factory=lambda: ["1way", "2way", "prefix"])
    max_vector_size: int = 128
    range_widths: list[int] = field(default_factory=lambda: [2])
    prefix_thresholds_per_feature: int = 2
    conditional_prefix_thresholds_per_feature: int = 1
    conditional_prefix_max_condition_values: int = 4
    halfspace_thresholds_per_pair: int = 1
    halfspace_pairs: list[list[str]] = field(default_factory=list)


@dataclass
class PrivacyConfig:
    epsilon: float = 3.0
    delta: float = 1e-6
    total_rho: float | None = None
    initialization_rho_fraction: float = 0.2
    selection_rho_fraction: float = 0.25


@dataclass
class SelectionConfig:
    score_mode: str = "l1_gap"
    score_clip: float = 1.0
    vector_size_weight_power: float = 0.0


@dataclass
class AlgorithmConfig:
    rounds: int = 10
    population_size: int = 4
    synthetic_size: int | None = None
    max_mutations_per_round: int = 16
    crossover_rate: float = 0.5
    crossover_rows: int = 8
    seed: int = 7
    device: str = "cpu"
    deterministic: bool = True
    log_every_round: bool = True


@dataclass
class AblationConfig:
    no_directed_mutation: bool = False
    no_crossover: bool = False
    no_orthogonal_grouping: bool = False
    no_inverse_variance_weighting: bool = False


@dataclass
class OutputConfig:
    synthetic_data_path: str = "synthetic.csv"
    metrics_path: str = "metrics.jsonl"
    summary_path: str = "summary.yaml"


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolve_output_paths(self) -> None:
        out_dir = Path(self.data.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.output.synthetic_data_path = str(out_dir / Path(self.output.synthetic_data_path).name)
        self.output.metrics_path = str(out_dir / Path(self.output.metrics_path).name)
        self.output.summary_path = str(out_dir / Path(self.output.summary_path).name)

    def validate(self) -> None:
        families = set(self.workload.families)
        unsupported = families - SUPPORTED_WORKLOAD_FAMILIES
        if unsupported:
            raise ValueError(f"Unsupported workload families: {sorted(unsupported)}")
        if not 0.0 < self.privacy.initialization_rho_fraction <= 1.0:
            raise ValueError("privacy.initialization_rho_fraction must be in (0, 1]")
        if not 0.0 <= self.privacy.selection_rho_fraction <= 1.0:
            raise ValueError("privacy.selection_rho_fraction must be in [0, 1]")
        if self.algorithm.rounds < 0:
            raise ValueError("algorithm.rounds must be non-negative")
        if self.algorithm.population_size < 1:
            raise ValueError("algorithm.population_size must be at least 1")
        if self.algorithm.max_mutations_per_round < 0:
            raise ValueError("algorithm.max_mutations_per_round must be non-negative")
        if self.algorithm.crossover_rows < 0:
            raise ValueError("algorithm.crossover_rows must be non-negative")
        if self.workload.max_vector_size < 1:
            raise ValueError("workload.max_vector_size must be at least 1")
        if not self.data.dataset_path:
            raise ValueError("data.dataset_path must be set")


def _merge_dict(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def _build_column_configs(raw_columns: dict[str, Any] | None) -> dict[str, ColumnConfig]:
    configs: dict[str, ColumnConfig] = {}
    for name, payload in (raw_columns or {}).items():
        if isinstance(payload, ColumnConfig):
            configs[name] = payload
        else:
            configs[name] = ColumnConfig(**(payload or {}))
    return configs


def _build_data_config(payload: dict[str, Any]) -> DataConfig:
    values = dict(payload)
    values["columns"] = _build_column_configs(values.get("columns"))
    return DataConfig(**values)


def _build_section(section_type: type, payload: dict[str, Any]) -> Any:
    return section_type(**dict(payload))


def load_run_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> RunConfig:
    merged = asdict(RunConfig())
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        _merge_dict(merged, loaded)
    if overrides:
        _merge_dict(merged, overrides)

    config = RunConfig(
        data=_build_data_config(merged.get("data", {})),
        workload=_build_section(WorkloadConfig, merged.get("workload", {})),
        privacy=_build_section(PrivacyConfig, merged.get("privacy", {})),
        selection=_build_section(SelectionConfig, merged.get("selection", {})),
        algorithm=_build_section(AlgorithmConfig, merged.get("algorithm", {})),
        ablations=_build_section(AblationConfig, merged.get("ablations", {})),
        output=_build_section(OutputConfig, merged.get("output", {})),
    )
    config.validate()
    config.resolve_output_paths()
    return config


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    return asdict(config)
