"""Application-level ports for infrastructure-facing services."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from ..core.rate_limiter import RateLimitConfig
from ..core.retry import RetryConfig

if TYPE_CHECKING:
    from ..core.batch.cache import AnalysisCache
    from ..core.batch.config import BatchConfig, BatchHooks, BatchResult, OnProgressCallback
    from ..core.optimizer import OptimizationResult
    from ..core.policy import Policy


CheckpointStatus = Literal["pending", "in_progress", "completed", "failed", "skipped"]
JOB_STATUS_PENDING: Final[CheckpointStatus] = "pending"
JOB_STATUS_IN_PROGRESS: Final[CheckpointStatus] = "in_progress"
JOB_STATUS_COMPLETED: Final[CheckpointStatus] = "completed"
JOB_STATUS_FAILED: Final[CheckpointStatus] = "failed"
JOB_STATUS_SKIPPED: Final[CheckpointStatus] = "skipped"


@dataclass(slots=True)
class CheckpointJob:
    """Portable checkpoint row used by application orchestration."""

    path: str
    status: CheckpointStatus
    error: str | None
    size_before_kb: float | None
    size_after_kb: float | None
    ssim: float | None
    format: str | None
    quality: int | None
    psnr: float | None
    perceptual_score: float | None
    reasons: list[str] | None
    artifact_base64: str | None


class BatchProcessorPort(Protocol):
    """Port for batch processor services."""

    def execute(
        self,
        image_paths: Sequence[str | Path],
        config: BatchConfig,
        hooks: BatchHooks | None = None,
    ) -> BatchResult: ...

    def process_single(
        self,
        image_path: Path,
        policy: Policy,
        cache: AnalysisCache | None = None,
    ) -> tuple[Path, OptimizationResult | Exception]: ...


class CheckpointJobPort(Protocol):
    """Protocol for checkpoint row payload used to rebuild batch results."""

    path: str
    status: CheckpointStatus
    error: str | None
    size_before_kb: float | None
    size_after_kb: float | None
    ssim: float | None
    format: str | None
    quality: int | None
    psnr: float | None
    perceptual_score: float | None
    reasons: list[str] | None
    artifact_base64: str | None


class CheckpointPort(Protocol):
    """Port for checkpoint lifecycle operations."""

    def start(self, image_paths: Sequence[str | Path], job_id: str | None = None) -> None: ...

    def load(self) -> bool: ...

    def save(self) -> None: ...

    def save_if_needed(self, interval: int = 10) -> bool: ...

    def should_checkpoint(self, interval: int = 10) -> bool: ...

    def mark_completed(
        self,
        path: str,
        result: CheckpointJob,
        *,
        checkpoint_interval: int | None = None,
    ) -> None: ...

    def mark_failed(
        self, path: str, error: str, *, checkpoint_interval: int | None = None
    ) -> None: ...

    def merge_paths(self, image_paths: Sequence[str | Path]) -> list[str]: ...

    def get_pending_for(self, image_paths: Sequence[str | Path]) -> list[str]: ...

    def get_results(self) -> list[CheckpointJobPort]: ...

    def get_results_for(self, image_paths: Sequence[str | Path]) -> list[CheckpointJobPort]:
        """Get processed results limited to requested input paths."""
        ...

    def get_metric_weights(self) -> tuple[float, float] | None: ...

    def set_metric_weights(self, ssim_weight: float, size_weight: float) -> None: ...

    def is_complete(self) -> bool: ...

    def get_stats(self) -> dict[str, int]: ...


class RetryPort(Protocol):
    """Port for retry calculations used by batch orchestration."""

    def should_retry(self, error: Exception) -> bool: ...

    def calculate_delay(self, attempt: int) -> float: ...


class RateLimiterPort(Protocol):
    """Port for rate limiter acquisition services."""

    def acquire(self, timeout_ms: int | None = None) -> bool: ...


@dataclass(frozen=True)
class BatchRuntimeServices:
    """Concrete set of runtime factories used by batch orchestration."""

    batch_processor_factory: Callable[..., BatchProcessorPort]
    checkpoint_manager_factory: Callable[[Path | str | None], CheckpointPort]
    retry_policy_factory: Callable[[RetryConfig | None], RetryPort]
    rate_limiter_factory: Callable[[RateLimitConfig | None], RateLimiterPort]


__all__ = [
    "BatchProcessorPort",
    "CheckpointJobPort",
    "CheckpointJob",
    "CheckpointPort",
    "OnProgressCallback",
    "RetryPort",
    "RateLimiterPort",
    "CheckpointStatus",
    "JOB_STATUS_COMPLETED",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_IN_PROGRESS",
    "JOB_STATUS_PENDING",
    "JOB_STATUS_SKIPPED",
    "BatchRuntimeServices",
]
