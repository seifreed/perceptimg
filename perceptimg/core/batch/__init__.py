from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ...exceptions import ImageLoadError, OptimizationError
from ...utils.image_io import load_image
from ..checkpoint import CheckpointManager, JobResult, JobStatus
from ..metrics_exporter import MetricsCollector
from ..optimizer import OptimizationResult, Optimizer
from ..policy import Policy
from ..rate_limiter import RateLimitConfig, RateLimiter
from ..retry import RetryConfig, RetryPolicy
from .cache import AnalysisCache
from .config import (
    BatchConfig,
    BatchHooks,
    BatchProgress,
    BatchResult,
    OnProgressCallback,
)
from .processor import BatchProcessor

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


def optimize_batch(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
    cache_analysis: bool = True,
    cache_maxsize: int = 128,
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
    processor = BatchProcessor()
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: optimize_batch(
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
                cached_analysis = cache.get(image, path)
                if cached_analysis:
                    result = optimizer.optimize_from_analysis(
                        image, cached_analysis, policy, original_bytes=path.read_bytes()
                    )
                else:
                    analysis = optimizer.analyzer.analyze(image)
                    cache.set(image, analysis, path)
                    result = optimizer.optimize(path, policy)
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
    paths = [Path(p) for p in image_paths]

    if len(paths) <= sample_size:
        sample = paths
    else:
        step = max(1, len(paths) // sample_size)
        sample = paths[::step][:sample_size]

    optimizer = Optimizer()
    total_before = 0.0
    total_after = 0.0

    for path in sample:
        try:
            result = optimizer.optimize(path, policy)
            total_before += result.report.size_before_kb
            total_after += result.report.size_after_kb
        except (OptimizationError, ImageLoadError, OSError):
            total_before += path.stat().st_size / 1024

    ratio = total_after / total_before if total_before > 0 else 1.0
    all_sizes = sum(p.stat().st_size / 1024 for p in paths)

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

    Returns:
        BatchResult with successful and failed results.

    Example:
        >>> # First run - interrupted
        >>> optimize_batch_with_checkpoint(images, policy, "checkpoint.json")
        >>> # Resume after crash
        >>> optimize_batch_with_checkpoint(images, policy, "checkpoint.json")
    """
    manager = CheckpointManager(checkpoint_path)
    paths = [Path(p) for p in image_paths]
    successful: list[tuple[Path, OptimizationResult]] = []
    failed: list[tuple[Path, Exception]] = []

    if manager.load() and not manager.is_complete():
        pending = manager.get_pending()
    else:
        manager.start(paths)
        pending = [str(p) for p in paths]

    if not pending:
        results = manager.get_results()
        for r in results:
            if r.status == JobStatus.FAILED:
                failed.append((Path(r.path), Exception(r.error or "Unknown error")))
        return BatchResult(successful=successful, failed=failed)

    processor = BatchProcessor()

    def on_image_complete_hook(path: Path, result: OptimizationResult) -> None:
        manager.mark_completed(
            str(path),
            JobResult(
                path=str(path),
                status=JobStatus.COMPLETED,
                error=None,
                size_before_kb=result.report.size_before_kb,
                size_after_kb=result.report.size_after_kb,
                ssim=result.report.ssim,
                format=result.report.chosen_format,
            ),
        )

    def on_image_error_hook(path: Path, exc: Exception) -> None:
        manager.mark_failed(str(path), str(exc))

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

    result = processor.execute(pending, config, hooks)
    manager.save()
    return result


def optimize_batch_with_retry(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    retry_config: RetryConfig | None = None,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    continue_on_error: bool = True,
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

    Returns:
        BatchResult with successful and failed results.

    Example:
        >>> config = RetryConfig(max_retries=3, base_delay_ms=100)
        >>> result = optimize_batch_with_retry(images, policy, retry_config=config)
    """
    config = BatchConfig(
        policy=policy,
        max_workers=max_workers or min(os.cpu_count() or 4, 8),
        continue_on_error=continue_on_error,
    )
    retry_policy = RetryPolicy(retry_config or RetryConfig())

    processor = BatchProcessor()
    paths = [Path(p) for p in image_paths]
    progress = BatchProgress(total=len(paths), completed=0, failed=0)
    successful: list[tuple[Path, OptimizationResult]] = []
    failed: list[tuple[Path, Exception]] = []

    def process_with_retry(image_path: Path) -> tuple[Path, OptimizationResult | Exception]:
        def operation() -> OptimizationResult:
            progress.current_file = str(image_path)
            result = processor.process_single(image_path, policy)
            if isinstance(result[1], Exception):
                raise result[1]
            return result[1]

        retry_result = retry_policy.execute(operation, on_retry=on_retry)

        if retry_result.success:
            return (image_path, retry_result.result)
        return (image_path, retry_result.error or Exception("Unknown error"))

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {executor.submit(process_with_retry, p): p for p in paths}

        for future in as_completed(futures):
            path = futures[future]
            try:
                _, result_or_exc = future.result()
                if isinstance(result_or_exc, Exception):
                    failed.append((path, result_or_exc))
                    progress.failed += 1
                    progress.errors.append(f"{path}: {result_or_exc}")
                else:
                    successful.append((path, result_or_exc))
                    progress.completed += 1
            except Exception as exc:
                failed.append((path, exc))
                progress.failed += 1
                progress.errors.append(f"{path}: {exc}")

            if on_progress:
                on_progress(progress)

            if not continue_on_error and failed:
                executor.shutdown(wait=False, cancel_futures=True)
                break

    return BatchResult(successful=successful, failed=failed)


def optimize_batch_with_rate_limit(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    rate_limit: RateLimitConfig | None = None,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
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
    limiter = RateLimiter(rate_limit or RateLimitConfig())
    config = BatchConfig(
        policy=policy,
        max_workers=max_workers or min(os.cpu_count() or 4, 8),
        continue_on_error=continue_on_error,
    )
    hooks = BatchHooks(rate_limiter=limiter, on_progress=on_progress)
    processor = BatchProcessor()
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

    def on_image_complete(path: Path, result: OptimizationResult) -> None:
        collector.record_success(
            format=result.report.chosen_format,
            bytes_before=int(result.report.size_before_kb * 1024),
            bytes_after=int(result.report.size_after_kb * 1024),
            ssim=result.report.ssim,
            processing_time_ms=0.0,
        )

    def on_image_error(path: Path, exc: Exception) -> None:
        collector.record_failure(type(exc).__name__)

    hooks = BatchHooks(
        on_image_complete=on_image_complete,
        on_image_error=on_image_error,
        on_progress=on_progress,
    )

    processor = BatchProcessor()
    result = processor.execute(paths, config, hooks)
    collector.end_job()

    return result, collector.collect()
