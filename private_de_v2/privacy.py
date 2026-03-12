from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math


def rho_from_epsilon_delta(epsilon: float, delta: float) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    log_term = math.log(1.0 / delta)
    root_rho = math.sqrt(log_term + epsilon) - math.sqrt(log_term)
    return max(root_rho * root_rho, 0.0)


def epsilon_from_rho_delta(rho: float, delta: float) -> float:
    if rho < 0.0:
        raise ValueError("rho must be non-negative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


@dataclass(frozen=True)
class PrivacyLedgerEntry:
    label: str
    rho: float
    cumulative_rho: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ZCDPAccountant:
    def __init__(self, total_rho: float) -> None:
        if total_rho <= 0.0:
            raise ValueError("total_rho must be positive")
        self.total_rho = float(total_rho)
        self.spent_rho = 0.0
        self.ledger: list[PrivacyLedgerEntry] = []

    @property
    def remaining_rho(self) -> float:
        return max(self.total_rho - self.spent_rho, 0.0)

    def spend(self, label: str, rho: float, metadata: dict[str, Any] | None = None) -> PrivacyLedgerEntry:
        rho = float(rho)
        if rho < 0.0:
            raise ValueError("rho spend must be non-negative")
        if self.spent_rho + rho > self.total_rho + 1e-12:
            raise ValueError(
                f"Privacy budget exceeded: attempted to spend {self.spent_rho + rho:.8f} > {self.total_rho:.8f}"
            )
        self.spent_rho += rho
        entry = PrivacyLedgerEntry(
            label=label,
            rho=rho,
            cumulative_rho=self.spent_rho,
            metadata=dict(metadata or {}),
        )
        self.ledger.append(entry)
        return entry

    def epsilon_delta(self, delta: float) -> float:
        return epsilon_from_rho_delta(self.spent_rho, delta)
