"""StageRunner abstraction: the contract every execution backend implements.

A stage runner executes one pipeline stage somewhere (local subprocess, a
Hugging Face Job, a remote SSH GPU box) and reports progress back through the
StageContext callbacks. Orchestration (what a stage *means*) stays in
``pipeline.DigitalTwinStudioRunner``; only execution moves behind this seam.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class StageCancelledError(RuntimeError):
    """Raised when a stage run is cancelled by the user."""


class CostDeniedError(RuntimeError):
    """Raised when the user declines the cost of a paid remote run."""


class CancelToken:
    """Thread-safe cancellation flag shared between the GUI and a running stage."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


# Approximate pay-per-hour rates (USD) for Hugging Face Jobs flavors.
# These are estimates for the cost-confirmation dialog, not billing truth —
# the authoritative price is whatever the HF invoice says.
FLAVOR_RATES_USD_PER_HOUR: dict[str, float] = {
    "cpu-basic": 0.05,
    "cpu-upgrade": 0.10,
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "l4x4": 3.20,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
    "a10g-largex2": 3.00,
    "a10g-largex4": 6.00,
    "a100-large": 4.00,
}
RATE_TABLE_SOURCE = "approximate public rates, recorded 2026-06"


@dataclass(frozen=True)
class CostEstimate:
    flavor: str
    est_minutes: float
    rate_usd_per_hour: float
    est_usd: float
    source: str = RATE_TABLE_SOURCE

    def summary(self) -> str:
        return (
            f"flavor={self.flavor} ~{self.est_minutes:.0f} min "
            f"@ ${self.rate_usd_per_hour:.2f}/h => ~${self.est_usd:.2f}"
        )


def estimate_cost(flavor: str, est_minutes: float) -> CostEstimate:
    rate = FLAVOR_RATES_USD_PER_HOUR.get(flavor, 5.0)
    return CostEstimate(
        flavor=flavor,
        est_minutes=est_minutes,
        rate_usd_per_hour=rate,
        est_usd=round(rate * est_minutes / 60.0, 2),
    )


@dataclass
class StageContext:
    """Everything a runner needs to execute one stage."""

    job_dir: Path
    job_id: str
    stage_key: str
    params: dict = field(default_factory=dict)
    inputs: list[Path] = field(default_factory=list)
    expected_outputs: list[Path] = field(default_factory=list)
    log: Callable[[str], None] = lambda _msg: None
    progress: Callable[[float, str], None] = lambda _pct, _msg: None
    cancel: CancelToken = field(default_factory=CancelToken)
    # When True the runner skips uploading ctx.inputs and assumes the HF
    # artifact dataset already has the correct files at jobs/<job_id>/<stage>/in/.
    # Used by --resume-job to reuse a prior run's already-uploaded frames.zip.
    skip_inputs_upload: bool = False


@dataclass
class StageResult:
    status: str  # "complete" | "failed"
    artifacts: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class StageRunner(ABC):
    """Executes one stage. Implementations: local subprocess, HF Jobs, SSH GPU."""

    name: str = "base"

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        """Run the stage to completion (or raise). Must honor ctx.cancel."""
