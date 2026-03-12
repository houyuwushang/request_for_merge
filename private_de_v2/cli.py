from __future__ import annotations

import argparse
import json
from typing import Any

from .config import RunConfig, load_run_config
from .generator import PrivateDEGeneratorV2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Private-DE v2 reference implementation.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the input CSV dataset.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for synthetic data and logs.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--device", type=str, default=None, help="Logged device string for reproducibility metadata.")
    parser.add_argument("--rounds", type=int, default=None, help="Number of Private-DE rounds.")
    parser.add_argument("--population-size", type=int, default=None, help="Number of synthetic populations.")
    parser.add_argument("--synthetic-size", type=int, default=None, help="Number of synthetic records to generate.")
    parser.add_argument("--max-mutations-per-round", type=int, default=None, help="Maximum directed mutations per round.")
    parser.add_argument("--crossover-rate", type=float, default=None, help="Probability of crossover per non-best population.")
    parser.add_argument("--crossover-rows", type=int, default=None, help="Rows copied during crossover.")
    parser.add_argument("--epsilon", type=float, default=None, help="Interpretability epsilon if total rho is not provided.")
    parser.add_argument("--delta", type=float, default=None, help="Interpretability delta if total rho is not provided.")
    parser.add_argument("--total-rho", type=float, default=None, help="Explicit total zCDP rho budget.")
    parser.add_argument("--initialization-rho-fraction", type=float, default=None, help="Share of rho spent on initialization.")
    parser.add_argument("--selection-rho-fraction", type=float, default=None, help="Per-round share of rho spent on selection.")
    parser.add_argument("--families", nargs="+", default=None, help="Workload families to include.")
    parser.add_argument("--max-vector-size", type=int, default=None, help="Maximum number of queries in a grouped vector.")
    parser.add_argument("--range-widths", nargs="+", type=int, default=None, help="Range partition widths for ordered columns.")
    parser.add_argument("--prefix-thresholds-per-feature", type=int, default=None, help="Prefix thresholds per ordered feature.")
    parser.add_argument(
        "--conditional-prefix-thresholds-per-feature",
        type=int,
        default=None,
        help="Conditional-prefix thresholds per ordered feature.",
    )
    parser.add_argument(
        "--conditional-prefix-max-condition-values",
        type=int,
        default=None,
        help="Maximum number of condition values per conditioning feature.",
    )
    parser.add_argument("--halfspace-thresholds-per-pair", type=int, default=None, help="Halfspace thresholds per feature pair.")
    parser.add_argument(
        "--halfspace-pair",
        action="append",
        default=None,
        help="Repeatable feature-pair override in the form 'col_a,col_b'.",
    )
    parser.add_argument("--no-directed-mutation", action="store_true", help="Disable directed mutation.")
    parser.add_argument("--no-crossover", action="store_true", help="Disable crossover.")
    parser.add_argument("--no-orthogonal-grouping", action="store_true", help="Use singleton query vectors.")
    parser.add_argument("--no-inverse-variance-weighting", action="store_true", help="Use latest measurement instead of IVW.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_run_config(args.config, _build_overrides(args))
    result = PrivateDEGeneratorV2(config).run()
    print(json.dumps(result.summary, indent=2))
    return 0


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    _set_override(overrides, ["data", "dataset_path"], args.dataset_path)
    _set_override(overrides, ["data", "output_dir"], args.output_dir)
    _set_override(overrides, ["privacy", "epsilon"], args.epsilon)
    _set_override(overrides, ["privacy", "delta"], args.delta)
    _set_override(overrides, ["privacy", "total_rho"], args.total_rho)
    _set_override(overrides, ["privacy", "initialization_rho_fraction"], args.initialization_rho_fraction)
    _set_override(overrides, ["privacy", "selection_rho_fraction"], args.selection_rho_fraction)
    _set_override(overrides, ["algorithm", "seed"], args.seed)
    _set_override(overrides, ["algorithm", "device"], args.device)
    _set_override(overrides, ["algorithm", "rounds"], args.rounds)
    _set_override(overrides, ["algorithm", "population_size"], args.population_size)
    _set_override(overrides, ["algorithm", "synthetic_size"], args.synthetic_size)
    _set_override(overrides, ["algorithm", "max_mutations_per_round"], args.max_mutations_per_round)
    _set_override(overrides, ["algorithm", "crossover_rate"], args.crossover_rate)
    _set_override(overrides, ["algorithm", "crossover_rows"], args.crossover_rows)
    _set_override(overrides, ["workload", "families"], args.families)
    _set_override(overrides, ["workload", "max_vector_size"], args.max_vector_size)
    _set_override(overrides, ["workload", "range_widths"], args.range_widths)
    _set_override(overrides, ["workload", "prefix_thresholds_per_feature"], args.prefix_thresholds_per_feature)
    _set_override(
        overrides,
        ["workload", "conditional_prefix_thresholds_per_feature"],
        args.conditional_prefix_thresholds_per_feature,
    )
    _set_override(
        overrides,
        ["workload", "conditional_prefix_max_condition_values"],
        args.conditional_prefix_max_condition_values,
    )
    _set_override(overrides, ["workload", "halfspace_thresholds_per_pair"], args.halfspace_thresholds_per_pair)
    if args.halfspace_pair:
        _set_override(
            overrides,
            ["workload", "halfspace_pairs"],
            [pair.split(",") for pair in args.halfspace_pair],
        )
    if args.no_directed_mutation:
        _set_override(overrides, ["ablations", "no_directed_mutation"], True)
    if args.no_crossover:
        _set_override(overrides, ["ablations", "no_crossover"], True)
    if args.no_orthogonal_grouping:
        _set_override(overrides, ["ablations", "no_orthogonal_grouping"], True)
    if args.no_inverse_variance_weighting:
        _set_override(overrides, ["ablations", "no_inverse_variance_weighting"], True)
    return overrides


def _set_override(payload: dict[str, Any], path: list[str], value: Any) -> None:
    if value is None:
        return
    cursor = payload
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value
