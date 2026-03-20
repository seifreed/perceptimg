"""Batch processing - re-exports from modular components.

This module provides parallel batch processing with:
- Thread pool execution
- Progress callbacks
- Checkpoint/resume
- Rate limiting
- Retry support
- Metrics collection

For Clean Architecture compliance, internal modules are organized as:
- batch/config.py - Configuration classes (BatchConfig, BatchHooks, etc.)
- batch/processor.py - Core BatchProcessor class
- batch/cache.py - AnalysisCache for caching analysis results
- batch/__init__.py - Public API functions
"""

from __future__ import annotations

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

__all__ = [
    "AnalysisCache",
    "BatchConfig",
    "BatchHooks",
    "BatchProcessor",
    "BatchProgress",
    "BatchResult",
    "OnProgressCallback",
    "estimate_batch_size",
    "optimize_batch",
    "optimize_batch_async",
    "optimize_batch_with_checkpoint",
    "optimize_batch_with_metrics",
    "optimize_batch_with_rate_limit",
    "optimize_batch_with_retry",
    "optimize_lazy",
]
