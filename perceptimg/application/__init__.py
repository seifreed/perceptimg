"""Public application boundary exports."""

from __future__ import annotations

from ..core.analyzer import AnalysisResult
from ..core.interfaces import ImageAdapter, ImageLoader, ImageWriter
from ..core.optimizer import (
    OptimizationResult,
    Optimizer,
    optimize,
    optimize_bytes,
    optimize_image,
)
from ..core.policy import UNSET, Policy
from ..core.rate_limiter import RateLimitConfig
from ..core.retry import RetryConfig
from ..core.strategy import StrategyCandidate
from .batch import (
    AnalysisCache,
    BatchConfig,
    BatchHooks,
    BatchProcessor,
    BatchProgress,
    BatchResult,
    OnProgressCallback,
    estimate_batch_size,
    optimize_batch,
    optimize_batch_async,
    optimize_batch_with_checkpoint,
    optimize_batch_with_metrics,
    optimize_batch_with_rate_limit,
    optimize_batch_with_retry,
    optimize_lazy,
)
from .presentation import build_optimizer

__all__ = [
    "AnalysisCache",
    "AnalysisResult",
    "BatchConfig",
    "BatchHooks",
    "BatchProcessor",
    "BatchProgress",
    "BatchResult",
    "ImageAdapter",
    "ImageLoader",
    "ImageWriter",
    "OnProgressCallback",
    "OptimizationResult",
    "Optimizer",
    "Policy",
    "RateLimitConfig",
    "RetryConfig",
    "StrategyCandidate",
    "UNSET",
    "build_optimizer",
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
