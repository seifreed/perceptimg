"""Project-wide exceptions."""

from __future__ import annotations


class PerceptimgError(Exception):
    """Base exception for the library."""


class StrategyError(PerceptimgError):
    """Raised when strategy generation fails."""


class OptimizationError(PerceptimgError):
    """Raised when optimization cannot be completed."""


class ImageLoadError(PerceptimgError):
    """Raised when an image cannot be loaded."""


class ImageSaveError(PerceptimgError):
    """Raised when an image cannot be saved/encoded."""
