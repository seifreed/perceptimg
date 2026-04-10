"""Concrete adapters wiring application ports to core implementations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ..core.batch.cache import AnalysisCache
from ..core.batch.config import BatchConfig, BatchHooks, BatchResult
from ..core.batch.processor import BatchProcessor
from ..core.checkpoint import CheckpointManager, JobResult, JobStatus
from ..core.optimizer import OptimizationResult, Optimizer
from ..core.policy import Policy
from ..core.rate_limiter import RateLimitConfig, RateLimiter
from ..core.retry import RetryConfig, RetryPolicy
from .ports import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_IN_PROGRESS,
    JOB_STATUS_PENDING,
    JOB_STATUS_SKIPPED,
    CheckpointJob,
    CheckpointJobPort,
    CheckpointPort,
    CheckpointStatus,
    RateLimiterPort,
    RetryPort,
)


class CoreBatchProcessorAdapter:
    """Adapter for ``core.batch.processor.BatchProcessor``."""

    def __init__(self, optimizer: Optimizer | None = None, cache: AnalysisCache | None = None):
        self._processor = BatchProcessor(optimizer=optimizer, cache=cache)

    def execute(
        self,
        image_paths: Sequence[str | Path],
        config: BatchConfig,
        hooks: BatchHooks | None = None,
    ) -> BatchResult:
        return self._processor.execute(image_paths, config, hooks)

    def process_single(
        self,
        image_path: Path,
        policy: Policy,
        cache: AnalysisCache | None = None,
    ) -> tuple[Path, OptimizationResult | Exception]:
        return self._processor.process_single(image_path, policy, cache)


class CoreCheckpointAdapter(CheckpointPort):
    """Adapter for ``core.checkpoint.CheckpointManager``."""

    def __init__(self, checkpoint_path: Path | str | None):
        self._manager = CheckpointManager(checkpoint_path)

    def start(self, image_paths: Sequence[str | Path], job_id: str | None = None) -> None:
        self._manager.start(image_paths, job_id)

    def load(self) -> bool:
        return self._manager.load()

    def save(self) -> None:
        self._manager.save()

    def save_if_needed(self, interval: int = 10) -> bool:
        return self._manager.save_if_needed(interval=interval)

    def should_checkpoint(self, interval: int = 10) -> bool:
        return self._manager.should_checkpoint(interval)

    def mark_completed(
        self,
        path: str,
        result: CheckpointJob,
        *,
        checkpoint_interval: int | None = None,
    ) -> None:
        status_map = {
            JOB_STATUS_PENDING: JobStatus.PENDING,
            JOB_STATUS_IN_PROGRESS: JobStatus.IN_PROGRESS,
            JOB_STATUS_COMPLETED: JobStatus.COMPLETED,
            JOB_STATUS_FAILED: JobStatus.FAILED,
            JOB_STATUS_SKIPPED: JobStatus.SKIPPED,
        }
        self._manager.mark_completed(
            path,
            JobResult(
                path=result.path,
                status=status_map[result.status],
                error=result.error,
                output_path=None,
                size_before_kb=result.size_before_kb,
                size_after_kb=result.size_after_kb,
                ssim=result.ssim,
                format=result.format,
                quality=result.quality,
                psnr=result.psnr,
                perceptual_score=result.perceptual_score,
                reasons=result.reasons,
                artifact_base64=result.artifact_base64,
            ),
            checkpoint_interval=checkpoint_interval,
        )

    def mark_failed(self, path: str, error: str, *, checkpoint_interval: int | None = None) -> None:
        self._manager.mark_failed(path, error, checkpoint_interval=checkpoint_interval)

    def merge_paths(self, image_paths: Sequence[str | Path]) -> list[str]:
        return self._manager.merge_paths(image_paths)

    def get_pending_for(self, image_paths: Sequence[str | Path]) -> list[str]:
        return self._manager.get_pending_for(image_paths)

    def get_results(self) -> list[CheckpointJobPort]:
        return cast(
            list[CheckpointJobPort],
            [self._to_checkpoint_job(job) for job in self._manager.get_results()],
        )

    def get_results_for(self, image_paths: Sequence[str | Path]) -> list[CheckpointJobPort]:
        return cast(
            list[CheckpointJobPort],
            [self._to_checkpoint_job(job) for job in self._manager.get_results_for(image_paths)],
        )

    @staticmethod
    def _to_checkpoint_job(job: JobResult) -> CheckpointJob:
        """Translate a core checkpoint row into application checkpoint contract."""
        status: CheckpointStatus = {
            JobStatus.PENDING.value: JOB_STATUS_PENDING,
            JobStatus.IN_PROGRESS.value: JOB_STATUS_IN_PROGRESS,
            JobStatus.COMPLETED.value: JOB_STATUS_COMPLETED,
            JobStatus.FAILED.value: JOB_STATUS_FAILED,
            JobStatus.SKIPPED.value: JOB_STATUS_SKIPPED,
        }[job.status.value]
        return CheckpointJob(
            path=job.path,
            status=status,
            error=job.error,
            size_before_kb=job.size_before_kb,
            size_after_kb=job.size_after_kb,
            ssim=job.ssim,
            format=job.format,
            quality=job.quality,
            psnr=job.psnr,
            perceptual_score=job.perceptual_score,
            reasons=job.reasons,
            artifact_base64=job.artifact_base64,
        )

    def get_metric_weights(self) -> tuple[float, float] | None:
        return self._manager.get_metric_weights()

    def set_metric_weights(self, ssim_weight: float, size_weight: float) -> None:
        self._manager.set_metric_weights(ssim_weight=ssim_weight, size_weight=size_weight)


class CoreRetryAdapter(RetryPort):
    """Adapter for ``core.retry.RetryPolicy``."""

    def __init__(self, config: RetryConfig | None = None):
        self._policy = RetryPolicy(config)

    def should_retry(self, error: Exception) -> bool:
        return self._policy.should_retry(error)

    def calculate_delay(self, attempt: int) -> float:
        return self._policy.calculate_delay(attempt)


class CoreRateLimiterAdapter(RateLimiterPort):
    """Adapter for ``core.rate_limiter.RateLimiter``."""

    def __init__(self, config: RateLimitConfig | None = None):
        self._rate_limiter = RateLimiter(config)

    def acquire(self, timeout_ms: int | None = None) -> bool:
        return self._rate_limiter.acquire(timeout_ms)
