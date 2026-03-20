"""perceptimg: perceptual image optimization engine.

Public API exposes `Policy` for declarative constraints and `optimize` for
executing the default optimization pipeline.

For in-memory optimization, use `optimize_image` and `optimize_bytes` from
`perceptimg.core.optimizer`.

For batch processing, use `optimize_batch`, `optimize_batch_async`, and
`optimize_lazy` from `perceptimg.core.batch`.

For enterprise features (checkpoint, retry, rate limiting, metrics):
- `optimize_batch_with_checkpoint` - Resume after interruption
- `optimize_batch_with_retry` - Automatic retry with backoff
- `optimize_batch_with_rate_limit` - Rate-limited processing
- `optimize_batch_with_metrics` - Prometheus metrics collection

Clean Architecture:
- core.interfaces - Protocol abstractions (ImageAdapter, ImageLoader, etc.)
- adapters.pil_adapter - Concrete PIL implementation
- BatchProcessor - Core batch processing logic (DRY)
- BatchConfig - Configuration dataclass
- BatchHooks - Extension points for customization

For domain interfaces:
- `AnalysisResult` - Immutable analysis result
- `StrategyCandidate` - Immutable strategy candidate
- `ImageAdapter` - Protocol for image operations
"""

from __future__ import annotations

from .adapters.pil_adapter import PILImageAdapter, bytes_to_image, image_to_bytes, load_image
from .core.analyzer import AnalysisResult
from .core.batch import (
    AnalysisCache,
    BatchConfig,
    BatchHooks,
    BatchProcessor,
    BatchProgress,
    BatchResult,
    estimate_batch_size,
    optimize_batch,
    optimize_batch_async,
    optimize_batch_with_checkpoint,
    optimize_batch_with_metrics,
    optimize_batch_with_rate_limit,
    optimize_batch_with_retry,
    optimize_lazy,
)
from .core.interfaces import ImageAdapter, ImageLoader, ImageWriter
from .core.optimizer import optimize, optimize_bytes, optimize_image
from .core.policy import Policy
from .core.strategy import StrategyCandidate

__all__ = [
    "optimize",
    "optimize_image",
    "optimize_bytes",
    "Policy",
    "optimize_batch",
    "optimize_batch_async",
    "optimize_lazy",
    "estimate_batch_size",
    "BatchResult",
    "BatchProgress",
    "BatchConfig",
    "BatchHooks",
    "BatchProcessor",
    "AnalysisCache",
    "optimize_batch_with_checkpoint",
    "optimize_batch_with_retry",
    "optimize_batch_with_rate_limit",
    "optimize_batch_with_metrics",
    "AnalysisResult",
    "StrategyCandidate",
    "ImageAdapter",
    "ImageLoader",
    "ImageWriter",
    "PILImageAdapter",
    "load_image",
    "bytes_to_image",
    "image_to_bytes",
]
