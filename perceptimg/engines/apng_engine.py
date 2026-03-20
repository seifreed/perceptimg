"""APNG engine using Pillow when available.

Note: APNG is always lossless. The quality parameter is always None in the result
as APNG does not support quality-based compression like lossy formats.
"""

from __future__ import annotations

from PIL import Image

from ..core.strategy import StrategyCandidate
from ..exceptions import ImageSaveError, OptimizationError
from ..utils.image_io import image_to_bytes
from .base import EngineResult, OptimizationEngine


class ApngEngine(OptimizationEngine):
    format = "apng"

    @property
    def is_available(self) -> bool:
        return "APNG" in Image.SAVE

    def can_handle(self, fmt: str) -> bool:
        return fmt.lower() == "apng"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        if not self.is_available:
            raise OptimizationError("APNG support is not available in Pillow build")

        save_kwargs: dict[str, object] = {"optimize": True}
        try:
            data = image_to_bytes(image, format="APNG", save_kwargs=save_kwargs)
        except ImageSaveError as exc:
            raise OptimizationError(f"APNG encoding failed: {exc}") from exc
        return EngineResult(
            data=data,
            format="apng",
            quality=None,
            metadata=save_kwargs,
        )
