"""Validation helpers used across the codebase."""

from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    """Raised when validation fails."""


def ensure_positive(value: int | float | None, name: str) -> None:
    """Ensure a numeric value is positive when provided."""

    if value is None:
        return
    if value <= 0:
        raise ValidationError(f"{name} must be positive")


def ensure_between_0_1(value: float | None, name: str) -> None:
    """Ensure a float is within (0, 1]."""

    if value is None:
        return
    if not (0 < value <= 1):
        raise ValidationError(f"{name} must be within (0, 1]")


def ensure_non_empty(sequence: Any, name: str) -> None:
    """Ensure a sequence is not empty."""

    if sequence is None:
        return
    if not sequence:
        raise ValidationError(f"{name} must not be empty")
