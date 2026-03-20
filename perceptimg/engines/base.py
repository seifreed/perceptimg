"""Engine abstractions for format-specific optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from ..core.strategy import StrategyCandidate

DEFAULT_QUALITY = 85


@dataclass(slots=True)
class EngineResult:
    """Result of encoding an image with an engine."""

    data: bytes
    format: str
    quality: int | None
    metadata: dict[str, object]


class OptimizationEngine(ABC):
    """Base interface for specific format encoders."""

    format: str
    priority: int = 0

    @property
    def is_available(self) -> bool:
        return True

    @abstractmethod
    def can_handle(self, fmt: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        """Run optimization and return encoded bytes."""
        raise NotImplementedError
