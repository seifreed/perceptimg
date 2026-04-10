"""WebP engine using Pillow bindings."""

from __future__ import annotations

from PIL import Image, features

from ..core.strategy import StrategyCandidate
from ..exceptions import ImageSaveError, OptimizationError
from ..utils.image_io import image_to_bytes
from .base import DEFAULT_QUALITY, EngineResult, OptimizationEngine


class WebPEngine(OptimizationEngine):
    format = "webp"

    @property
    def is_available(self) -> bool:
        return bool(features.check("webp"))

    def can_handle(self, fmt: str) -> bool:
        return fmt.lower() == "webp"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        if not self.is_available:
            raise OptimizationError("WebP support is not available in Pillow build")

        if strategy.lossless:
            save_kwargs: dict[str, object] = {
                "lossless": True,
                "method": 6,
            }
            actual_quality = None
        else:
            actual_quality = strategy.quality or DEFAULT_QUALITY
            save_kwargs = {
                "quality": actual_quality,
                "method": 6,
                "lossless": False,
            }
        try:
            data = image_to_bytes(image, format="WEBP", save_kwargs=save_kwargs)
        except ImageSaveError as exc:
            raise OptimizationError(f"WebP encoding failed: {exc}") from exc
        return EngineResult(
            data=data,
            format="webp",
            quality=None if strategy.lossless else actual_quality,
            metadata=save_kwargs,
        )
