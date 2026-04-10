"""Top-level public API facade."""

from .api import (  # noqa: F401
    PUBLIC_API,
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
    Policy,
    RateLimitConfig,
    RetryConfig,
    StrategyCandidate,
    estimate_batch_size,
    optimize,
    optimize_batch,
    optimize_batch_async,
    optimize_batch_with_checkpoint,
    optimize_batch_with_metrics,
    optimize_batch_with_rate_limit,
    optimize_batch_with_retry,
    optimize_bytes,
    optimize_image,
    optimize_lazy,
)

# Deprecated compatibility re-exports:
# Keep these available for existing consumers while excluding them from ``__all__``
# to maintain a cleaner public API contract for wildcard imports.
__all__ = list(PUBLIC_API)
