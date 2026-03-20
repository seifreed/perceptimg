"""Pillow-backed engine for JPEG, PNG, TIFF, and GIF formats.

Note: GIF and PNG are always lossless in this implementation. GIF animations
are supported but treated as lossless (no palette optimization). For animated
GIFs with reduced palettes, consider using a specialized GIF optimizer.
"""

from __future__ import annotations

from PIL import Image

from ..core.strategy import StrategyCandidate
from ..exceptions import ImageSaveError, OptimizationError
from ..utils.image_io import image_to_bytes
from .base import DEFAULT_QUALITY, EngineResult, OptimizationEngine


class PillowEngine(OptimizationEngine):
    format = "pillow"
    SUPPORTED = {"jpeg", "png", "tiff", "gif"}

    def can_handle(self, fmt: str) -> bool:
        return fmt.lower() in self.SUPPORTED

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        fmt = strategy.format.lower()
        save_kwargs: dict[str, object] = {}
        actual_quality: int | None = None
        if fmt == "jpeg":
            actual_quality = strategy.quality or DEFAULT_QUALITY
            save_kwargs.update(
                quality=actual_quality,
                subsampling=strategy.subsampling,
                progressive=strategy.progressive,
                optimize=True,
            )
        elif fmt == "png":
            save_kwargs.update(optimize=True, compress_level=6)
        elif fmt == "tiff":
            save_kwargs.update(compression="tiff_lzw")
        elif fmt == "gif":
            save_kwargs.update(optimize=True)
        else:
            raise OptimizationError(f"PillowEngine cannot handle format {fmt}")

        try:
            data = image_to_bytes(image, format=fmt.upper(), save_kwargs=save_kwargs)
        except ImageSaveError as exc:
            raise OptimizationError(f"PillowEngine failed for {fmt}: {exc}") from exc
        return EngineResult(data=data, format=fmt, quality=actual_quality, metadata=save_kwargs)
