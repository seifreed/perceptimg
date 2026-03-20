"""AVIF engine using Pillow when available."""

from __future__ import annotations

from PIL import Image, features

from ..core.strategy import StrategyCandidate
from ..exceptions import ImageSaveError, OptimizationError
from ..utils.image_io import image_to_bytes
from .base import DEFAULT_QUALITY, EngineResult, OptimizationEngine


class AvifEngine(OptimizationEngine):
    format = "avif"

    @property
    def is_available(self) -> bool:
        return bool(features.check("avif"))

    def can_handle(self, fmt: str) -> bool:
        return fmt.lower() == "avif"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        if not self.is_available:
            raise OptimizationError("AVIF support is not available in Pillow build")

        if strategy.lossless:
            save_kwargs: dict[str, object] = {
                "lossless": True,
            }
            actual_quality = None
        else:
            actual_quality = strategy.quality or DEFAULT_QUALITY
            save_kwargs = {
                "quality": actual_quality,
                "lossless": False,
            }
        try:
            data = image_to_bytes(image, format="AVIF", save_kwargs=save_kwargs)
        except ImageSaveError as exc:
            raise OptimizationError(f"AVIF encoding failed: {exc}") from exc
        return EngineResult(
            data=data,
            format="avif",
            quality=actual_quality if actual_quality is not None else strategy.quality,
            metadata=save_kwargs,
        )
