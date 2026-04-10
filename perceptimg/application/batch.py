from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from ..core.batch.cache import AnalysisCache
from ..core.batch.config import (
    BatchConfig,
    BatchHooks,
    BatchProgress,
    BatchResult,
    OnProgressCallback,
)
from ..core.batch.processor import BatchProcessor as _CoreBatchProcessor
from ..core.metrics import MetricCalculator
from ..core.metrics_exporter import MetricsCollector
from ..core.optimizer import OptimizationResult, Optimizer
from ..core.policy import Policy
from ..core.rate_limiter import RateLimitConfig
from ..core.report import OptimizationReport
from ..core.retry import RetryConfig
from ..exceptions import ImageLoadError, OptimizationError
from ..utils.image_io import bytes_to_image, load_image
from .ports import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_SKIPPED,
    BatchProcessorPort,
    BatchRuntimeServices,
    CheckpointJob,
    CheckpointJobPort,
    CheckpointPort,
)
from .runtime import get_default_batch_services

# Public compatibility symbol expected by legacy tests. Keep as alias to the core
# batch processor.
BatchProcessor = _CoreBatchProcessor

__all__ = [
    "AnalysisCache",
    "BatchConfig",
    "BatchHooks",
    "BatchProgress",
    "BatchResult",
    "BatchProcessor",
    "OnProgressCallback",
    "optimize_batch",
    "optimize_batch_async",
    "optimize_lazy",
    "estimate_batch_size",
    "optimize_batch_with_checkpoint",
    "optimize_batch_with_retry",
    "optimize_batch_with_rate_limit",
    "optimize_batch_with_metrics",
]

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, OSError)


def _normalize_batch_path(path: str | Path) -> str:
    """Return a stable absolute path representation used for checkpoint keys."""
    return str(Path(path).expanduser().resolve(strict=False))


def _checkpoint_fallback_image(path: Path) -> Image.Image:
    try:
        return load_image(path)
    except (ImageLoadError, OSError, ValueError):
        return Image.new("RGB", (1, 1))


def _result_from_checkpoint(
    job: CheckpointJobPort,
    metric_calculator: MetricCalculator | None = None,
) -> OptimizationResult | None:
    if job.status != JOB_STATUS_COMPLETED:
        return None

    image_bytes = b""
    artifact_degraded = False
    if job.artifact_base64:
        image_bytes = base64.b64decode(job.artifact_base64.encode("ascii"))

    checkpoint_path = Path(_normalize_batch_path(job.path))
    if image_bytes:
        try:
            image = bytes_to_image(image_bytes)
        except Exception:
            logging.getLogger(__name__).warning(
                "Corrupt checkpoint artifact for %s; falling back to reload",
                checkpoint_path,
            )
            image = _checkpoint_fallback_image(checkpoint_path)
            image_bytes = b""
            artifact_degraded = True
    else:
        image = _checkpoint_fallback_image(checkpoint_path)
        artifact_degraded = True

    reasons = list(job.reasons) if job.reasons else ["checkpoint_restored"]
    if artifact_degraded:
        reasons.append("checkpoint_artifact_corrupt_reloaded_from_disk")

    # Use provided MetricCalculator or create one with default weights
    calc = metric_calculator or MetricCalculator()

    report = OptimizationReport(
        chosen_format=job.format or checkpoint_path.suffix.lstrip(".") or "unknown",
        quality=job.quality,
        size_before_kb=job.size_before_kb if job.size_before_kb is not None else 0.0,
        size_after_kb=job.size_after_kb if job.size_after_kb is not None else 0.0,
        ssim=0.0 if artifact_degraded else (job.ssim if job.ssim is not None else 0.0),
        psnr=0.0 if artifact_degraded else (job.psnr if job.psnr is not None else 0.0),
        perceptual_score=(
            0.0
            if artifact_degraded
            else (
                job.perceptual_score
                if job.perceptual_score is not None
                else (
                    calc._perceptual_score(job.ssim, job.size_before_kb, job.size_after_kb)
                    if job.ssim is not None
                    and job.size_before_kb is not None
                    and job.size_after_kb is not None
                    else 0.0
                )
            )
        ),
        reasons=reasons,
    )
    return OptimizationResult(image_bytes=image_bytes, image=image, report=report)


def _batch_result_from_checkpoint(
    manager: CheckpointPort,
    requested_paths: Sequence[str | Path],
    metric_calculator: MetricCalculator | None = None,
) -> BatchResult:
    """Reconstruct batch result from checkpoint for the requested paths.

    Args:
        manager: Checkpoint manager with saved results.
        requested_paths: List of paths that were requested for processing.
        metric_calculator: Optional MetricCalculator with custom weights for
            recalculating perceptual scores if needed.

    Returns:
        BatchResult with successful, failed, and skipped results.
    """
    successful: list[tuple[Path, OptimizationResult]] = []
    failed: list[tuple[Path, Exception]] = []
    skipped: list[Path] = []
    successful_input_indices: list[int] = []
    failed_input_indices: list[int] = []
    skipped_input_indices: list[int] = []

    normalized_requested_paths = [_normalize_batch_path(path) for path in requested_paths]
    requested_counts = Counter(normalized_requested_paths)
    requested_results = manager.get_results_for(requested_paths)
    available_results_by_path: dict[str, list[CheckpointJobPort]] = {}
    for job in requested_results:
        available_results_by_path.setdefault(_normalize_batch_path(job.path), []).append(job)

    selected_counts = Counter(
        path_key for path_key, jobs in available_results_by_path.items() for _ in jobs
    )
    if selected_counts != requested_counts:
        logging.getLogger(__name__).warning(
            "Checkpoint result counts differ from requested paths. "
            "Reconciling by input position. "
            "Checkpoint: %s, Requested: %s",
            dict(selected_counts),
            requested_counts,
        )

    for request_index, request_path in enumerate(requested_paths):
        request_key = _normalize_batch_path(request_path)
        pending_jobs = available_results_by_path.get(request_key, [])
        if not pending_jobs:
            failed.append(
                (Path(request_path), Exception("Missing checkpoint result for input item"))
            )
            failed_input_indices.append(request_index)
            continue

        job = pending_jobs.pop(0)
        path = Path(request_path)

        if job.status == JOB_STATUS_FAILED:
            failed.append((path, Exception(job.error or "Unknown error")))
            failed_input_indices.append(request_index)
            continue
        if job.status == JOB_STATUS_SKIPPED:
            skipped.append(path)
            skipped_input_indices.append(request_index)
            continue

        restored = _result_from_checkpoint(job, metric_calculator=metric_calculator)
        if restored is not None:
            successful.append((path, restored))
            successful_input_indices.append(request_index)
        else:
            failed.append((path, Exception(f"Incomplete job (status: {job.status})")))
            failed_input_indices.append(request_index)

    remaining_results = [job for jobs in available_results_by_path.values() for job in jobs]
    if remaining_results:
        logging.getLogger(__name__).warning(
            "Ignoring %d unmatched checkpoint results for requested paths.",
            len(remaining_results),
        )

    return BatchResult(
        successful=successful,
        failed=failed,
        skipped=skipped,
        successful_input_indices=successful_input_indices,
        failed_input_indices=failed_input_indices,
        skipped_input_indices=skipped_input_indices,
    )


def _resubmit_with_retry(
    image_path: Path,
    input_index: int,
    *,
    executor: ThreadPoolExecutor,
    process_fn: Callable[[Path], tuple[Path, OptimizationResult | Exception]],
    futures: dict[Future[tuple[Path, OptimizationResult | Exception]], tuple[int, Path]],
    progress_lock: Lock,
    abort_requested: threading.Event,
    resubmit_event: threading.Event,
) -> None:
    """Retry callback that can be safely executed from a timer thread."""

    with progress_lock:
        if abort_requested.is_set():
            return
        future = executor.submit(process_fn, image_path)
        futures[future] = (input_index, image_path)
    resubmit_event.set()


def optimize_batch(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
    cache_analysis: bool = True,
    cache_maxsize: int = 128,
    optimizer: Optimizer | None = None,
    processor: BatchProcessorPort | None = None,
    services: BatchRuntimeServices | None = None,
) -> BatchResult:
    """Optimize multiple images in parallel using thread pool.

    Args:
        image_paths: List of image file paths to optimize.
        policy: Policy to apply to all images.
        max_workers: Maximum number of parallel workers. Defaults to CPU count.
        on_progress: Optional callback for progress updates.
        continue_on_error: If True, continue processing on errors.
        cache_analysis: If True, cache analysis results for repeated images.
        cache_maxsize: Maximum number of cached analysis results.

    Returns:
        BatchResult with successful and failed results.

    Example:
        >>> from perceptimg import Policy
        >>> from perceptimg.core.batch import optimize_batch
        >>> policy = Policy(max_size_kb=100, min_ssim=0.9)
        >>> result = optimize_batch(["img1.png", "img2.png"], policy, max_workers=4)
        >>> print(f"Success: {len(result.successful)}, Failed: {len(result.failed)}")
    """
    config = BatchConfig(
        policy=policy,
        max_workers=max_workers or min(os.cpu_count() or 4, 8),
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
        cache_maxsize=cache_maxsize,
    )
    hooks = BatchHooks(on_progress=on_progress)
    services = services or get_default_batch_services()
    processor = processor or services.batch_processor_factory(optimizer=optimizer)
    return processor.execute(image_paths, config, hooks)


async def optimize_batch_async(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
    cache_analysis: bool = True,
    cache_maxsize: int = 128,
) -> BatchResult:
    """Optimize multiple images asynchronously.

    This is an async wrapper around optimize_batch for use in async contexts.
    Runs optimization in a thread pool executor to avoid blocking the event loop.

    Args:
        image_paths: List of image file paths to optimize.
        policy: Policy to apply to all images.
        on_progress: Optional callback for progress updates.
        continue_on_error: If True, continue processing on errors.
        cache_analysis: If True, cache analysis results for repeated images.
        cache_maxsize: Maximum number of cached analysis results.

    Returns:
        BatchResult with successful and failed results.

    Example:
        >>> import asyncio
        >>> from perceptimg import Policy
        >>> from perceptimg.core.batch import optimize_batch_async
        >>> async def main():
        ...     policy = Policy(max_size_kb=100)
        ...     result = await optimize_batch_async(["img1.png", "img2.png"], policy)
        ...     print(f"Success: {result.success_rate:.0%}")
        >>> asyncio.run(main())
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            optimize_batch,
            list(image_paths),
            policy,
            on_progress=on_progress,
            continue_on_error=continue_on_error,
            cache_analysis=cache_analysis,
            cache_maxsize=cache_maxsize,
        ),
    )


def optimize_lazy(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    cache_analysis: bool = True,
    cache_maxsize: int = 128,
) -> Iterator[tuple[Path, OptimizationResult | Exception]]:
    """Lazy generator for memory-efficient batch processing.

    Yields results one at a time instead of collecting all results.
    Useful for very large batches or streaming pipelines.

    Args:
        image_paths: Sequence of image file paths.
        policy: Policy to apply to all images.
        cache_analysis: If True, cache analysis results for repeated images.
        cache_maxsize: Maximum number of cached analysis results.

    Yields:
        Tuples of (path, result_or_exception).

    Example:
        >>> from perceptimg import Policy
        >>> from perceptimg.core.batch import optimize_lazy
        >>> policy = Policy(max_size_kb=100)
        >>> for path, result in optimize_lazy(["img1.png", "img2.png"], policy):
        ...     if isinstance(result, Exception):
        ...         print(f"Error {path}: {result}")
        ...     else:
        ...         print(f"OK {path}: {result.report.size_after_kb}KB")
    """
    optimizer = Optimizer()
    cache = AnalysisCache(maxsize=cache_maxsize) if cache_analysis else None

    for image_path in image_paths:
        path = Path(image_path)
        try:
            if cache:
                image = load_image(path)
                original_bytes = path.read_bytes()
                cached_analysis = cache.get(image, path)
                if cached_analysis:
                    result = optimizer.optimize_from_analysis(
                        image, cached_analysis, policy, original_bytes=original_bytes
                    )
                else:
                    analysis = optimizer.analyzer.analyze(image)
                    cache.set(image, analysis, path)
                    result = optimizer.optimize_from_analysis(
                        image, analysis, policy, original_bytes=original_bytes
                    )
            else:
                result = optimizer.optimize(path, policy)
            yield (path, result)
        except (OptimizationError, ImageLoadError, OSError, ValueError) as exc:
            yield (path, exc)


def estimate_batch_size(
    image_paths: Sequence[str | Path],
    policy: Policy,
    sample_size: int = 3,
) -> dict[str, float]:
    """Estimate total size reduction for a batch without full processing.

    Processes a sample to estimate compression ratio, then extrapolates.

    Args:
        image_paths: List of image file paths.
        policy: Policy to apply.
        sample_size: Number of images to sample for estimation.

    Returns:
        Dict with 'estimated_total_kb_before', 'estimated_total_kb_after',
        "estimated_reduction_percent", "sample_size".
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}")

    paths = [Path(p) for p in image_paths]

    if len(paths) <= sample_size:
        sample = paths
    else:
        if sample_size == 1:
            sample = [paths[len(paths) // 2]]
        else:
            indices = [round(i * (len(paths) - 1) / (sample_size - 1)) for i in range(sample_size)]
            sample = [paths[i] for i in indices]

    optimizer = Optimizer()
    total_before = 0.0
    total_after = 0.0

    for path in sample:
        try:
            result = optimizer.optimize(path, policy)
            total_before += result.report.size_before_kb
            total_after += result.report.size_after_kb
        except (OptimizationError, ImageLoadError, OSError):
            try:
                file_size = path.stat().st_size / 1024
            except OSError:
                file_size = 0.0
            total_before += file_size
            total_after += file_size

    def _safe_file_size_kb(path: Path) -> float:
        try:
            return path.stat().st_size / 1024
        except OSError:
            return 0.0

    all_sizes = sum(_safe_file_size_kb(p) for p in paths)

    if total_before <= 0:
        return {
            "estimated_total_kb_before": all_sizes,
            "estimated_total_kb_after": all_sizes,
            "estimated_reduction_percent": 0.0,
            "sample_size": len(sample),
            "all_samples_failed": True,
        }

    ratio = total_after / total_before

    return {
        "estimated_total_kb_before": all_sizes,
        "estimated_total_kb_after": all_sizes * ratio,
        "estimated_reduction_percent": (1 - ratio) * 100,
        "sample_size": len(sample),
    }


def optimize_batch_with_checkpoint(
    image_paths: Sequence[str | Path],
    policy: Policy,
    checkpoint_path: Path | str,
    *,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
    cache_analysis: bool = True,
    checkpoint_interval: int = 10,
    metric_calculator: MetricCalculator | None = None,
    services: BatchRuntimeServices | None = None,
) -> BatchResult:
    """Optimize batch with checkpoint/resume support.

    Saves progress periodically and can resume after interruption.

    Args:
        image_paths: List of image file paths to optimize.
        policy: Policy to apply to all images.
        checkpoint_path: Path to save checkpoint file.
        max_workers: Maximum number of parallel workers.
        on_progress: Optional callback for progress updates.
        continue_on_error: If True, continue processing on errors.
        cache_analysis: If True, cache analysis results for repeated images.
        checkpoint_interval: Save checkpoint every N completed images.
        metric_calculator: Optional MetricCalculator with custom weights for
            perceptual score calculation. If not provided and checkpoint has
            saved weights, uses checkpoint weights; otherwise uses defaults.

    Returns:
        BatchResult with successful and failed results.

    Example:
        >>> # First run - interrupted
        >>> optimize_batch_with_checkpoint(images, policy, "checkpoint.json")
        >>> # Resume after crash
        >>> optimize_batch_with_checkpoint(images, policy, "checkpoint.json")
    """
    if checkpoint_interval < 1:
        raise ValueError(f"checkpoint_interval must be >= 1, got {checkpoint_interval}")

    services = services or get_default_batch_services()
    manager: CheckpointPort = services.checkpoint_manager_factory(checkpoint_path)
    processor = services.batch_processor_factory(optimizer=Optimizer())
    paths = [Path(p) for p in image_paths]

    # Determine MetricCalculator to use for perceptual score calculation
    effective_calc: MetricCalculator | None = metric_calculator
    pending_paths: list[Path] = []

    if manager.load():
        # Try to use checkpoint weights if no custom calculator provided
        if effective_calc is None:
            stored_weights = manager.get_metric_weights()
            if stored_weights is not None:
                effective_calc = MetricCalculator(
                    ssim_weight=stored_weights[0],
                    size_weight=stored_weights[1],
                )
        new_paths = manager.merge_paths(paths)
        if new_paths:
            manager.save()
        pending = manager.get_pending_for(paths)
        if not pending:
            return _batch_result_from_checkpoint(manager, paths, metric_calculator=effective_calc)
        pending_paths = [Path(path) for path in pending]
    else:
        # Save metric weights to checkpoint for reproducibility
        manager.start(paths)
        if effective_calc is not None:
            manager.set_metric_weights(
                effective_calc.ssim_weight,
                effective_calc.size_weight,
            )
        elif metric_calculator is None:
            # Use default weights for new checkpoint
            manager.set_metric_weights(0.7, 0.3)
        pending_paths = paths

    def on_image_complete_hook(path: Path, result: OptimizationResult) -> None:
        manager.mark_completed(
            _normalize_batch_path(path),
            CheckpointJob(
                path=str(path),
                status=JOB_STATUS_COMPLETED,
                error=None,
                size_before_kb=result.report.size_before_kb,
                size_after_kb=result.report.size_after_kb,
                ssim=result.report.ssim,
                format=result.report.chosen_format,
                quality=result.report.quality,
                psnr=result.report.psnr,
                perceptual_score=result.report.perceptual_score,
                reasons=list(result.report.reasons),
                artifact_base64=base64.b64encode(result.image_bytes).decode("ascii"),
            ),
            checkpoint_interval=checkpoint_interval,
        )

    def on_image_error_hook(path: Path, exc: Exception) -> None:
        manager.mark_failed(
            _normalize_batch_path(path),
            str(exc),
            checkpoint_interval=checkpoint_interval,
        )

    def should_checkpoint() -> bool:
        return manager.should_checkpoint(checkpoint_interval)

    def on_checkpoint() -> None:
        manager.save()

    def progress_callback(p: BatchProgress) -> None:
        if on_progress:
            on_progress(p)

    config = BatchConfig(
        policy=policy,
        max_workers=max_workers or min(os.cpu_count() or 4, 8),
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
    )
    hooks = BatchHooks(
        on_image_complete=on_image_complete_hook,
        on_image_error=on_image_error_hook,
        on_progress=progress_callback,
        should_checkpoint=should_checkpoint,
        on_checkpoint=on_checkpoint,
    )

    processor.execute(pending_paths, config, hooks)
    manager.save()
    reconstructed = _batch_result_from_checkpoint(manager, paths, metric_calculator=effective_calc)
    # Always return reconstructed: it includes both previous checkpoint
    # results and current run results (saved via on_image_complete_hook).
    return reconstructed


def _checkpoint_on_success(
    manager: CheckpointPort | None,
    path: Path,
    result: OptimizationResult,
    checkpoint_interval: int,
) -> None:
    """Persist a successful result to checkpoint if manager is active."""
    if manager is None:
        return
    manager.mark_completed(
        _normalize_batch_path(path),
        CheckpointJob(
            path=_normalize_batch_path(path),
            status=JOB_STATUS_COMPLETED,
            error=None,
            size_before_kb=result.report.size_before_kb,
            size_after_kb=result.report.size_after_kb,
            ssim=result.report.ssim,
            format=result.report.chosen_format,
            quality=result.report.quality,
            psnr=result.report.psnr,
            perceptual_score=result.report.perceptual_score,
            reasons=list(result.report.reasons),
            artifact_base64=base64.b64encode(result.image_bytes).decode("ascii"),
        ),
        checkpoint_interval=checkpoint_interval,
    )


def _checkpoint_on_failure(
    manager: CheckpointPort | None,
    path: Path,
    exc: Exception,
    checkpoint_interval: int,
) -> None:
    """Persist a failure to checkpoint if manager is active."""
    if manager is None:
        return
    manager.mark_failed(
        _normalize_batch_path(path),
        str(exc),
        checkpoint_interval=checkpoint_interval,
    )


def optimize_batch_with_retry(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    retry_config: RetryConfig | None = None,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    continue_on_error: bool = True,
    cache_analysis: bool = True,
    cache_maxsize: int = 128,
    checkpoint_path: Path | str | None = None,
    checkpoint_interval: int = 10,
    metric_calculator: MetricCalculator | None = None,
    services: BatchRuntimeServices | None = None,
) -> BatchResult:
    """Optimize batch with automatic retry on transient errors.

    Args:
        image_paths: List of image file paths to optimize.
        policy: Policy to apply to all images.
        retry_config: Retry configuration. Default: 3 retries with exponential backoff.
        max_workers: Maximum number of parallel workers.
        on_progress: Optional callback for progress updates.
        on_retry: Optional callback for retry events: (attempt, error, delay_ms).
        continue_on_error: If True, continue processing on errors.
        checkpoint_path: Optional path to save checkpoint file for crash recovery.
        checkpoint_interval: Save checkpoint every N completed images (default: 10).
        metric_calculator: Optional MetricCalculator with custom weights for
            perceptual score calculation.

    Returns:
        BatchResult with successful and failed results.

    Example:
        >>> config = RetryConfig(max_retries=3, base_delay_ms=100)
        >>> result = optimize_batch_with_retry(images, policy, retry_config=config)
        >>> # With checkpoint for crash recovery:
        >>> result = optimize_batch_with_retry(
        ...     images, policy, retry_config=config, checkpoint_path="retry_ckpt.json"
        ... )
    """
    config = BatchConfig(
        policy=policy,
        max_workers=max_workers or min(os.cpu_count() or 4, 8),
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
        cache_maxsize=cache_maxsize,
    )
    retry_cfg = retry_config or RetryConfig()
    if retry_cfg.retry_on is None:
        retry_cfg = RetryConfig(
            max_retries=retry_cfg.max_retries,
            base_delay_ms=retry_cfg.base_delay_ms,
            max_delay_ms=retry_cfg.max_delay_ms,
            exponential_base=retry_cfg.exponential_base,
            jitter_ms=retry_cfg.jitter_ms,
            retry_on=_TRANSIENT_ERRORS,
        )
    services = services or get_default_batch_services()
    retry_policy = services.retry_policy_factory(retry_cfg)

    processor = services.batch_processor_factory(optimizer=Optimizer())
    paths = [Path(p) for p in image_paths]
    skipped: list[Path] = []
    skipped_input_indices: list[int] = []

    # Set up checkpoint manager if path provided
    manager: CheckpointPort | None = None
    effective_calc: MetricCalculator | None = metric_calculator
    if checkpoint_path is not None:
        manager = services.checkpoint_manager_factory(checkpoint_path)
        if manager.load():
            # Try to use checkpoint weights if no custom calculator provided
            if effective_calc is None:
                stored_weights = manager.get_metric_weights()
                if stored_weights is not None:
                    effective_calc = MetricCalculator(
                        ssim_weight=stored_weights[0],
                        size_weight=stored_weights[1],
                    )
            new_paths = manager.merge_paths(paths)
            if new_paths:
                manager.save()
            pending_strs = manager.get_pending_for(paths)
            if not pending_strs:
                return _batch_result_from_checkpoint(
                    manager, paths, metric_calculator=effective_calc
                )
            pending_set = {_normalize_batch_path(path) for path in pending_strs}
            for idx, p in enumerate(paths):
                if _normalize_batch_path(p) not in pending_set:
                    skipped.append(p)
                    skipped_input_indices.append(idx)
            paths = [p for p in paths if _normalize_batch_path(p) in pending_set]
        else:
            manager.start(paths)
            # Save metric weights to checkpoint for reproducibility
            if effective_calc is not None:
                manager.set_metric_weights(
                    effective_calc.ssim_weight,
                    effective_calc.size_weight,
                )
            elif metric_calculator is None:
                manager.set_metric_weights(0.7, 0.3)

    original_total = len(paths) + len(skipped)
    cache = AnalysisCache(maxsize=config.cache_maxsize) if config.cache_analysis else None
    progress = BatchProgress(total=original_total, completed=len(skipped), failed=0)
    successful: list[tuple[Path, OptimizationResult]] = []
    failed: list[tuple[Path, Exception]] = []
    successful_input_indices: list[int] = []
    failed_input_indices: list[int] = []
    progress_lock = Lock()
    retry_counts: dict[tuple[int, str], int] = {}

    def process_single(image_path: Path) -> tuple[Path, OptimizationResult | Exception]:
        result = processor.process_single(image_path, policy, cache)
        if isinstance(result[1], Exception):
            return (image_path, result[1])
        return (image_path, result[1])

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures: dict[Future[tuple[Path, OptimizationResult | Exception]], tuple[int, Path]] = {
            executor.submit(process_single, p): (index, p) for index, p in enumerate(paths)
        }
        pending_timers: list[threading.Timer] = []
        resubmit_event = threading.Event()
        abort_requested = threading.Event()

        while futures:
            # Clean up expired timers to prevent memory leak
            pending_timers = [t for t in pending_timers if t.is_alive()]
            for future in as_completed(list(futures.keys())):
                input_index, path = futures.pop(future)
                snapshot = None
                try:
                    _, result_or_exc = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    result_or_exc = exc

                if isinstance(result_or_exc, Exception):
                    with progress_lock:
                        path_key = (input_index, str(path))
                        attempt = retry_counts.get(path_key, 0) + 1
                        retry_counts[path_key] = attempt
                        should_retry = (
                            attempt <= retry_cfg.max_retries
                            and retry_policy.should_retry(result_or_exc)
                            and not abort_requested.is_set()
                        )

                    if should_retry:
                        delay_ms = retry_policy.calculate_delay(attempt)
                        if on_retry:
                            on_retry(attempt, result_or_exc, delay_ms)

                        timer = threading.Timer(
                            delay_ms / 1000.0,
                            _resubmit_with_retry,
                            args=(
                                path,
                                input_index,
                            ),
                            kwargs={
                                "executor": executor,
                                "process_fn": process_single,
                                "futures": futures,
                                "progress_lock": progress_lock,
                                "abort_requested": abort_requested,
                                "resubmit_event": resubmit_event,
                            },
                        )
                        timer.daemon = True
                        timer.start()
                        pending_timers.append(timer)
                        continue

                    with progress_lock:
                        progress.current_file = str(path)
                        failed.append((path, result_or_exc))
                        failed_input_indices.append(input_index)
                        progress.failed += 1
                        progress.errors.append(f"{path}: {result_or_exc}")
                        _checkpoint_on_failure(manager, path, result_or_exc, checkpoint_interval)
                        if on_progress:
                            snapshot = progress.snapshot()
                else:
                    with progress_lock:
                        progress.current_file = str(path)
                        successful.append((path, result_or_exc))
                        successful_input_indices.append(input_index)
                        progress.completed += 1
                        _checkpoint_on_success(manager, path, result_or_exc, checkpoint_interval)
                        if on_progress:
                            snapshot = progress.snapshot()

                if snapshot and on_progress:
                    on_progress(snapshot)

                if manager is not None:
                    manager.save_if_needed(checkpoint_interval)

                if not continue_on_error and failed and not abort_requested.is_set():
                    abort_requested.set()
                    for t in pending_timers:
                        t.cancel()
                    break

            if not futures:
                # Wait for any pending timers that may add new futures
                active_timers = [t for t in pending_timers if t.is_alive()]
                if not active_timers:
                    break
                resubmit_event.clear()
                # Wait until a timer fires _resubmit and signals the event
                max_wait = (
                    max(
                        (t.interval for t in active_timers if hasattr(t, "interval")),
                        default=5.0,
                    )
                    + 1.0
                )
                resubmit_event.wait(timeout=max_wait)
                pending_timers = [t for t in pending_timers if t.is_alive()]
                if not futures:
                    # Timers finished but no new futures added (cancelled/aborted)
                    break

    if manager is not None:
        manager.save()

    return BatchResult(
        successful=successful,
        failed=failed,
        skipped=skipped,
        successful_input_indices=successful_input_indices,
        failed_input_indices=failed_input_indices,
        skipped_input_indices=skipped_input_indices,
    )


def optimize_batch_with_rate_limit(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    rate_limit: RateLimitConfig | None = None,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
    services: BatchRuntimeServices | None = None,
) -> BatchResult:
    """Optimize batch with rate limiting.

    Useful when processing many images to avoid overwhelming I/O or API limits.

    Args:
        image_paths: List of image file paths to optimize.
        policy: Policy to apply to all images.
        rate_limit: Rate limit configuration. Default: 10 requests/second.
        max_workers: Maximum number of parallel workers.
        on_progress: Optional callback for progress updates.
        continue_on_error: If True, continue processing on errors.

    Returns:
        BatchResult with successful and failed results.

    Example:
        >>> config = RateLimitConfig(requests_per_second=5)
        >>> result = optimize_batch_with_rate_limit(images, policy, rate_limit=config)
    """
    services = services or get_default_batch_services()
    limiter = services.rate_limiter_factory(rate_limit or RateLimitConfig())
    config = BatchConfig(
        policy=policy,
        max_workers=max_workers or min(os.cpu_count() or 4, 8),
        continue_on_error=continue_on_error,
    )
    hooks = BatchHooks(rate_limiter=limiter, on_progress=on_progress)
    processor = services.batch_processor_factory()
    return processor.execute(image_paths, config, hooks)


def optimize_batch_with_metrics(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    metrics: MetricsCollector | None = None,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
    cache_analysis: bool = True,
) -> tuple[BatchResult, dict[str, Any]]:
    """Optimize batch with Prometheus metrics collection.

    Args:
        image_paths: List of image file paths to optimize.
        policy: Policy to apply to all images.
        metrics: Optional metrics collector. Creates a new one if None.
        max_workers: Maximum number of parallel workers.
        on_progress: Optional callback for progress updates.
        continue_on_error: If True, continue processing on errors.
        cache_analysis: If True, cache analysis results for repeated images.

    Returns:
        Tuple of (BatchResult, metrics_dict).

    Example:
        >>> result, metrics = optimize_batch_with_metrics(images, policy)
        >>> print(f"Average SSIM: {metrics['average_ssim']:.2f}")
        >>> print(f"Compression: {metrics['average_compression_ratio']:.1%}")
    """
    collector = metrics or MetricsCollector()
    config = BatchConfig(
        policy=policy,
        max_workers=max_workers or min(os.cpu_count() or 4, 8),
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
    )

    paths = [Path(p) for p in image_paths]
    collector.start_job(len(paths))
    processing_times_ms: dict[str, list[float]] = {}
    processing_times_lock = Lock()

    class TimedBatchProcessor(BatchProcessor):
        def process_single(
            self,
            image_path: Path,
            policy: Policy,
            cache: AnalysisCache | None = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            started_at = time.monotonic()
            result = super().process_single(image_path, policy, cache)
            elapsed_ms = (time.monotonic() - started_at) * 1000
            with processing_times_lock:
                processing_times_ms.setdefault(str(image_path), []).append(elapsed_ms)
            return result

    def on_image_complete(path: Path, result: OptimizationResult) -> None:
        with processing_times_lock:
            timings = processing_times_ms.get(str(path), [])
            processing_time_ms = timings.pop(0) if timings else 0.0
            if not timings and str(path) in processing_times_ms:
                processing_times_ms.pop(str(path), None)
            collector.record_success(
                format=result.report.chosen_format,
                bytes_before=int(result.report.size_before_kb * 1024),
                bytes_after=int(result.report.size_after_kb * 1024),
                ssim=result.report.ssim,
                processing_time_ms=processing_time_ms,
            )

    def on_image_error(path: Path, exc: Exception) -> None:
        with processing_times_lock:
            timings = processing_times_ms.get(str(path), [])
            if timings:
                timings.pop(0)
            if not timings and str(path) in processing_times_ms:
                processing_times_ms.pop(str(path), None)
            collector.record_failure(type(exc).__name__)

    hooks = BatchHooks(
        on_image_complete=on_image_complete,
        on_image_error=on_image_error,
        on_progress=on_progress,
    )

    processor = TimedBatchProcessor()
    result = processor.execute(paths, config, hooks)
    processing_times_ms.clear()
    collector.end_job()

    return result, collector.collect()
