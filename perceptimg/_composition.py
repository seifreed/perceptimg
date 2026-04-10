"""Private composition root for package wiring.

This module owns the default object graph used by the public facades.
It is intentionally kept outside the application package so application
modules only depend on ports and use-case collaborators.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from .adapters.pil_adapter import PILImageIO
from .application.adapters import (
    CoreBatchProcessorAdapter,
    CoreCheckpointAdapter,
    CoreRateLimiterAdapter,
    CoreRetryAdapter,
)
from .application.ports import BatchRuntimeServices
from .application.runtime import set_default_batch_services_provider
from .core.metrics import MetricCalculator
from .core.optimizer import (
    Optimizer,
    set_default_engine_provider,
    set_default_image_io_provider,
)
from .engines.apng_engine import ApngEngine
from .engines.avif_engine import AvifEngine
from .engines.base import OptimizationEngine
from .engines.heif_engine import HeifEngine
from .engines.jxl_engine import JxlEngine
from .engines.pillow_engine import PillowEngine
from .engines.webp_engine import WebPEngine

_bootstrapped = False
_bootstrap_lock = threading.Lock()


def build_default_engines() -> list[OptimizationEngine]:
    """Return default engine instances used by the default optimizer."""
    return [
        JxlEngine(),
        AvifEngine(),
        WebPEngine(),
        HeifEngine(),
        ApngEngine(),
        PillowEngine(),
    ]


def build_default_engine_sequence() -> Sequence[OptimizationEngine]:
    """Expose default engines as an immutable sequence."""
    return tuple(build_default_engines())


def register_default_engine_provider() -> None:
    """Register the application engine provider in the core layer."""

    def _provider() -> Sequence[OptimizationEngine]:
        return build_default_engine_sequence()

    set_default_engine_provider(_provider)


def register_default_image_io_provider() -> None:
    """Register the default image I/O provider in the core layer."""

    def _provider() -> PILImageIO:
        return PILImageIO()

    set_default_image_io_provider(_provider)


def build_default_batch_services() -> BatchRuntimeServices:
    """Build and return the default batch service adapters."""
    return BatchRuntimeServices(
        batch_processor_factory=CoreBatchProcessorAdapter,
        checkpoint_manager_factory=CoreCheckpointAdapter,
        retry_policy_factory=CoreRetryAdapter,
        rate_limiter_factory=CoreRateLimiterAdapter,
    )


def build_optimizer(
    *,
    ssim_weight: float,
    size_weight: float,
    prioritize_quality: bool,
    max_candidates: int,
) -> Optimizer:
    """Build a CLI-compatible optimizer with configured strategy priorities."""
    metric_calculator = MetricCalculator(ssim_weight=ssim_weight, size_weight=size_weight)
    optimizer = Optimizer(
        metric_calculator=metric_calculator,
        prioritize_quality=prioritize_quality,
    )
    strategy_generator = getattr(optimizer, "strategy_generator", None)
    if strategy_generator is not None and hasattr(strategy_generator, "max_candidates"):
        strategy_generator.max_candidates = max_candidates
    return optimizer


def ensure_default_wiring() -> None:
    """Register core default providers exactly once."""
    global _bootstrapped
    if _bootstrapped:
        return

    with _bootstrap_lock:
        if _bootstrapped:
            return
        register_default_engine_provider()
        register_default_image_io_provider()
        set_default_batch_services_provider(build_default_batch_services)
        _bootstrapped = True


def reset_default_wiring_for_tests() -> None:
    """Reset bootstrap state for tests that need explicit reconfiguration."""
    global _bootstrapped
    with _bootstrap_lock:
        _bootstrapped = False


__all__ = [
    "build_default_batch_services",
    "build_default_engine_sequence",
    "build_default_engines",
    "build_optimizer",
    "ensure_default_wiring",
    "register_default_engine_provider",
    "register_default_image_io_provider",
    "reset_default_wiring_for_tests",
]
