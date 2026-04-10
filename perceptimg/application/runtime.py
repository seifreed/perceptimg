"""Application-local runtime wiring hooks.

This module only stores the default batch services provider. It does not know
how to build the defaults; that remains the responsibility of the outer
composition root.
"""

from __future__ import annotations

from collections.abc import Callable

from .ports import BatchRuntimeServices

_default_batch_services_provider: Callable[[], BatchRuntimeServices] | None = None


def set_default_batch_services_provider(
    provider: Callable[[], BatchRuntimeServices],
) -> None:
    """Register the provider used by batch orchestration when services are omitted."""
    global _default_batch_services_provider
    _default_batch_services_provider = provider


def get_default_batch_services() -> BatchRuntimeServices:
    """Return the registered default batch services provider."""
    if _default_batch_services_provider is None:
        raise RuntimeError(
            "No default batch services provider configured. "
            "Call ensure_default_wiring() from the composition root or pass "
            "BatchRuntimeServices explicitly."
        )
    return _default_batch_services_provider()
