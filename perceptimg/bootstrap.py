"""Compatibility shim for legacy bootstrap imports."""

from __future__ import annotations

from ._composition import (
    build_default_batch_services,
    build_default_engine_sequence,
    build_default_engines,
    build_optimizer,
    ensure_default_wiring,
    register_default_engine_provider,
    register_default_image_io_provider,
    reset_default_wiring_for_tests,
)

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
