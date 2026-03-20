"""HEIF/HEIC engine using Pillow when available."""

from __future__ import annotations

from PIL import Image

from ..core.strategy import StrategyCandidate
from ..exceptions import ImageSaveError, OptimizationError
from ..utils.image_io import image_to_bytes
from .base import DEFAULT_QUALITY, EngineResult, OptimizationEngine


class HeifEngine(OptimizationEngine):
    format = "heif"
    SUPPORTED = {"heif", "heic"}

    @property
    def is_available(self) -> bool:
        return "HEIF" in Image.SAVE or "HEIC" in Image.SAVE

    def can_handle(self, fmt: str) -> bool:
        return fmt.lower() in self.SUPPORTED

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        if not self.is_available:
            raise OptimizationError("HEIF/HEIC support is not available in Pillow build")

        fmt = strategy.format.lower()
        save_format = "HEIF" if fmt == "heif" else "HEIC"
        if strategy.lossless:
            save_kwargs: dict[str, object] = {
                "lossless": True,
            }
            actual_quality = None
        else:
            actual_quality = strategy.quality or DEFAULT_QUALITY
            save_kwargs = {
                "quality": actual_quality,
            }
        try:
            data = image_to_bytes(image, format=save_format, save_kwargs=save_kwargs)
        except ImageSaveError as exc:
            raise OptimizationError(f"HEIF/HEIC encoding failed: {exc}") from exc
        return EngineResult(
            data=data,
            format=fmt,
            quality=actual_quality if actual_quality is not None else strategy.quality,
            metadata=save_kwargs,
        )
