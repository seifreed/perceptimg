"""Public facade for the package API."""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .application import (
    UNSET,
    AnalysisCache,
    AnalysisResult,
    BatchConfig,
    BatchHooks,
    BatchProgress,
    BatchResult,
    ImageAdapter,
    ImageLoader,
    ImageWriter,
    OnProgressCallback,
    OptimizationResult,
    Optimizer,
    Policy,
    RateLimitConfig,
    RetryConfig,
    StrategyCandidate,
)
from .application import (
    estimate_batch_size as _app_estimate_batch_size,
)
from .application import (
    optimize as _app_optimize,
)
from .application import (
    optimize_batch as _app_optimize_batch,
)
from .application import (
    optimize_batch_async as _app_optimize_batch_async,
)
from .application import (
    optimize_batch_with_checkpoint as _app_optimize_batch_with_checkpoint,
)
from .application import (
    optimize_batch_with_metrics as _app_optimize_batch_with_metrics,
)
from .application import (
    optimize_batch_with_rate_limit as _app_optimize_batch_with_rate_limit,
)
from .application import (
    optimize_batch_with_retry as _app_optimize_batch_with_retry,
)
from .application import (
    optimize_bytes as _app_optimize_bytes,
)
from .application import (
    optimize_image as _app_optimize_image,
)
from .application import (
    optimize_lazy as _app_optimize_lazy,
)
from .application.presentation import (
    _parse_preferred_formats as _app_parse_preferred_formats,
)
from .application.presentation import (
    _resolve_output_extension as _app_resolve_output_extension,
)
from .application.presentation import (
    batch_report_data as _app_batch_report_data,
)
from .application.presentation import (
    batch_successful_report_rows as _app_batch_successful_report_rows,
)
from .application.presentation import (
    batch_summary_text as _app_batch_summary_text,
)
from .application.presentation import (
    build_optimizer as _app_build_optimizer,
)
from .application.presentation import (
    get_allowed_formats as _app_get_allowed_formats,
)
from .application.presentation import (
    load_policy as _app_load_policy,
)
from .application.presentation import (
    plan_batch_successful_outputs as _app_plan_batch_successful_outputs,
)
from .application.presentation import (
    policy_from_flags as _app_policy_from_flags,
)
from .application.presentation import (
    reserve_batch_output_path as _app_reserve_batch_output_path,
)
from .application.presentation import (
    with_collision_suffix as _app_with_collision_suffix,
)
from .application.presentation import (
    write_batch_report as _app_write_batch_report,
)
from .core.metrics import MetricCalculator
from .core.metrics_exporter import MetricsCollector


def _boot_if_needed() -> None:
    from ._composition import ensure_default_wiring

    ensure_default_wiring()


def _parse_preferred_formats(formats: str | None) -> tuple[str, ...] | None:
    """Facade wrapper for preferred-format parsing."""
    return _app_parse_preferred_formats(formats)


def _resolve_output_extension(format_name: str) -> str:
    """Facade wrapper for deterministic output extensions."""
    return _app_resolve_output_extension(format_name)


def _get_allowed_formats() -> tuple[str, ...]:
    return _app_get_allowed_formats()


def _batch_report_data(
    result: BatchResult,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> dict[str, object]:
    """Facade wrapper for batch report payload rendering."""
    return _app_batch_report_data(result, successful_outputs=successful_outputs)


def _batch_successful_report_rows(
    result: BatchResult,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> list[dict[str, object]]:
    """Facade wrapper for batch report rows."""
    return _app_batch_successful_report_rows(result, successful_outputs=successful_outputs)


def _batch_summary_text(result: BatchResult) -> str:
    """Facade wrapper for batch summary formatting."""
    return _app_batch_summary_text(result)


def _write_batch_report(
    report_path: Path,
    result: BatchResult,
    report_format: str,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> None:
    """Facade wrapper for writing batch reports."""
    _app_write_batch_report(
        report_path,
        result,
        report_format,
        successful_outputs=successful_outputs,
    )


def _load_policy(
    policy_path: Path,
) -> Policy:
    """Facade wrapper for loading policy files."""
    return _app_load_policy(policy_path)


def _policy_from_flags(
    args: object,
) -> Policy:
    """Facade wrapper for building policy from CLI flags."""
    return _app_policy_from_flags(args)


def _plan_batch_successful_outputs(
    input_paths: Sequence[Path],
    successful: Sequence[tuple[Path, OptimizationResult]],
    output_dir: Path,
    output_pattern: str,
    successful_input_indices: Sequence[int] | None = None,
) -> list[tuple[Path, OptimizationResult, Path]]:
    """Facade wrapper for batch successful output planning."""
    return _app_plan_batch_successful_outputs(
        input_paths,
        successful,
        output_dir,
        output_pattern,
        successful_input_indices=successful_input_indices,
    )


def _with_collision_suffix(path: Path, suffix_index: int) -> Path:
    """Facade wrapper for deterministic output collision suffixing."""
    return _app_with_collision_suffix(path, suffix_index)


def _reserve_batch_output_path(
    output_dir: Path,
    output_name: str,
    reserved: set[Path],
) -> Path:
    """Facade wrapper for reserving deduplicated output path."""
    return _app_reserve_batch_output_path(output_dir, output_name, reserved)


def _build_optimizer(
    *,
    ssim_weight: float,
    size_weight: float,
    prioritize_quality: bool,
    max_candidates: int,
) -> Optimizer:
    """Build optimizer using the application presentation factory."""
    from ._composition import build_optimizer as default_optimizer_builder

    return _app_build_optimizer(
        ssim_weight=ssim_weight,
        size_weight=size_weight,
        prioritize_quality=prioritize_quality,
        max_candidates=max_candidates,
        optimizer_factory=default_optimizer_builder,
    )


def optimize(image_path: str | Path, policy: Policy) -> OptimizationResult:
    """Optimize a file path using the default optimizer stack."""
    _boot_if_needed()
    return _app_optimize(image_path, policy)


def optimize_image(
    image: object,
    policy: Policy,
    *,
    optimizer: Optimizer | None = None,
    original_bytes: bytes | None = None,
) -> OptimizationResult:
    """Optimize a PIL image in-memory without touching disk."""
    _boot_if_needed()
    return _app_optimize_image(
        image,
        policy,
        optimizer=optimizer,
        original_bytes=original_bytes,
    )


def optimize_bytes(
    data: bytes,
    policy: Policy,
    *,
    optimizer: Optimizer | None = None,
    original_format: str | None = None,
) -> OptimizationResult:
    """Optimize raw image bytes."""
    _boot_if_needed()
    return _app_optimize_bytes(
        data,
        policy,
        optimizer=optimizer,
        original_format=original_format,
    )


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
) -> BatchResult:
    """Optimize multiple images in parallel using thread pool."""
    _boot_if_needed()
    return _app_optimize_batch(
        image_paths=image_paths,
        policy=policy,
        max_workers=max_workers,
        on_progress=on_progress,
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
        cache_maxsize=cache_maxsize,
        optimizer=optimizer,
    )


async def optimize_batch_async(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
    cache_analysis: bool = True,
    cache_maxsize: int = 128,
) -> BatchResult:
    """Async wrapper around batch optimization."""
    _boot_if_needed()
    return await _app_optimize_batch_async(
        image_paths=image_paths,
        policy=policy,
        on_progress=on_progress,
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
        cache_maxsize=cache_maxsize,
    )


def optimize_lazy(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    cache_analysis: bool = True,
    cache_maxsize: int = 128,
) -> Iterator[tuple[Path, OptimizationResult | Exception]]:
    """Yield optimization results lazily."""
    _boot_if_needed()
    return _app_optimize_lazy(
        image_paths=image_paths,
        policy=policy,
        cache_analysis=cache_analysis,
        cache_maxsize=cache_maxsize,
    )


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
) -> BatchResult:
    _boot_if_needed()
    return _app_optimize_batch_with_checkpoint(
        image_paths=image_paths,
        policy=policy,
        checkpoint_path=checkpoint_path,
        max_workers=max_workers,
        on_progress=on_progress,
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
        checkpoint_interval=checkpoint_interval,
        metric_calculator=metric_calculator,
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
) -> BatchResult:
    _boot_if_needed()
    return _app_optimize_batch_with_retry(
        image_paths=image_paths,
        policy=policy,
        retry_config=retry_config,
        max_workers=max_workers,
        on_progress=on_progress,
        on_retry=on_retry,
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
        cache_maxsize=cache_maxsize,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        metric_calculator=metric_calculator,
    )


def optimize_batch_with_rate_limit(
    image_paths: Sequence[str | Path],
    policy: Policy,
    *,
    rate_limit: RateLimitConfig | None = None,
    max_workers: int | None = None,
    on_progress: OnProgressCallback | None = None,
    continue_on_error: bool = True,
) -> BatchResult:
    _boot_if_needed()
    return _app_optimize_batch_with_rate_limit(
        image_paths=image_paths,
        policy=policy,
        rate_limit=rate_limit,
        max_workers=max_workers,
        on_progress=on_progress,
        continue_on_error=continue_on_error,
    )


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
    """Return batch results plus collected metrics."""
    _boot_if_needed()
    return _app_optimize_batch_with_metrics(
        image_paths=image_paths,
        policy=policy,
        metrics=metrics,
        max_workers=max_workers,
        on_progress=on_progress,
        continue_on_error=continue_on_error,
        cache_analysis=cache_analysis,
    )


def estimate_batch_size(
    image_paths: Sequence[str | Path],
    policy: Policy,
    sample_size: int = 3,
) -> dict[str, float]:
    _boot_if_needed()
    return _app_estimate_batch_size(
        image_paths=image_paths,
        policy=policy,
        sample_size=sample_size,
    )


PUBLIC_API: tuple[str, ...] = (
    "AnalysisCache",
    "AnalysisResult",
    "BatchConfig",
    "BatchHooks",
    "BatchProgress",
    "BatchResult",
    "ImageLoader",
    "OnProgressCallback",
    "ImageWriter",
    "ImageAdapter",
    "OptimizationResult",
    "Policy",
    "RateLimitConfig",
    "RetryConfig",
    "StrategyCandidate",
    "UNSET",
    "estimate_batch_size",
    "optimize",
    "optimize_batch",
    "optimize_batch_async",
    "optimize_batch_with_checkpoint",
    "optimize_batch_with_metrics",
    "optimize_batch_with_rate_limit",
    "optimize_batch_with_retry",
    "optimize_bytes",
    "optimize_image",
    "optimize_lazy",
)

AnalysisCache = AnalysisCache
AnalysisResult = AnalysisResult
BatchConfig = BatchConfig
BatchHooks = BatchHooks
BatchProgress = BatchProgress
ImageAdapter = ImageAdapter
ImageLoader = ImageLoader
ImageWriter = ImageWriter
OptimizationResult = OptimizationResult
Optimizer = Optimizer
Policy = Policy
RateLimitConfig = RateLimitConfig
RetryConfig = RetryConfig
StrategyCandidate = StrategyCandidate
UNSET = UNSET

__all__ = [
    "AnalysisCache",
    "AnalysisResult",
    "BatchConfig",
    "BatchHooks",
    "BatchProgress",
    "BatchResult",
    "ImageLoader",
    "OnProgressCallback",
    "ImageWriter",
    "ImageAdapter",
    "OptimizationResult",
    "Policy",
    "RateLimitConfig",
    "RetryConfig",
    "StrategyCandidate",
    "UNSET",
    "estimate_batch_size",
    "optimize",
    "optimize_batch",
    "optimize_batch_async",
    "optimize_batch_with_checkpoint",
    "optimize_batch_with_metrics",
    "optimize_batch_with_rate_limit",
    "optimize_batch_with_retry",
    "optimize_bytes",
    "optimize_image",
    "optimize_lazy",
]
