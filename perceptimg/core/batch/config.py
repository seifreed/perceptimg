"""Configuration types for batch processing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..report import OptimizationReport

from ..optimizer import OptimizationResult
from ..policy import Policy
from ..rate_limiter import RateLimiter


@dataclass(slots=True)
class BatchProgress:
    """Progress information for batch operations."""

    total: int
    completed: int
    failed: int
    skipped: int = 0
    current_file: str | None = None
    errors: list[str] = field(default_factory=list)
    sequence: int = 0

    def snapshot(self) -> BatchProgress:
        """Return a copy safe to hand to progress callbacks."""
        return BatchProgress(
            total=self.total,
            completed=self.completed,
            failed=self.failed,
            skipped=self.skipped,
            current_file=self.current_file,
            errors=list(self.errors),
            sequence=self.sequence,
        )

    @property
    def success_rate(self) -> float:
        processed = self.completed + self.failed + self.skipped
        if processed == 0:
            return 0.0
        return self.completed / processed


@dataclass(slots=True)
class BatchResult:
    """Result of a batch optimization operation."""

    successful: list[tuple[Path, OptimizationResult]]
    failed: list[tuple[Path, Exception]]
    skipped: list[Path] = field(default_factory=list)
    successful_input_indices: list[int] = field(default_factory=list)
    failed_input_indices: list[int] = field(default_factory=list)
    skipped_input_indices: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.successful) + len(self.failed) + len(self.skipped)

    @property
    def success_rate(self) -> float:
        processed = len(self.successful) + len(self.failed) + len(self.skipped)
        if processed == 0:
            return 0.0
        return len(self.successful) / processed

    def get_reports(self) -> list[OptimizationReport]:
        return [result.report for _, result in self.successful]


OnProgressCallback = Callable[[BatchProgress], None]
OnImageCallback = Callable[[Path], None]
OnResultCallback = Callable[[Path, OptimizationResult], None]
OnErrorCallback = Callable[[Path, Exception], None]


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Encapsulates all batch options following Clean Code principles.
    """

    policy: Policy
    max_workers: int = 4
    continue_on_error: bool = True
    cache_analysis: bool = True
    cache_maxsize: int = 128


@dataclass
class BatchHooks:
    """Hooks for customizing batch processing behavior.

    Follows Open/Closed Principle - open for extension via hooks,
    closed for modification of core logic.
    """

    on_image_start: OnImageCallback | None = None
    on_image_complete: OnResultCallback | None = None
    on_image_error: OnErrorCallback | None = None
    on_progress: OnProgressCallback | None = None
    should_checkpoint: Callable[[], bool] | None = None
    on_checkpoint: Callable[[], None] | None = None
    rate_limiter: RateLimiter | None = None
