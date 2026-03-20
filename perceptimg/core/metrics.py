"""Perceptual metrics for evaluating optimization outputs.

Clean Architecture: Core domain only depends on abstractions (ImageAdapter Protocol).
PIL is imported only in adapters layer. This module uses duck typing to work
with both PIL.Image and ImageAdapter implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
from skimage.metrics import structural_similarity

if TYPE_CHECKING:
    from PIL import Image

    from ..adapters.pil_adapter import PILImageAdapter

from ..utils.image_io import size_kb

ImageLike = Union["Image.Image", "PILImageAdapter"]

PSNR_MAX_VALUE = 100.0
SSIM_MAX_DIMENSION = 2048


@dataclass(slots=True)
class MetricResult:
    """Computed metrics for an optimization candidate."""

    ssim: float
    psnr: float
    size_before_kb: float
    size_after_kb: float
    perceptual_score: float


@dataclass(slots=True)
class MetricConfig:
    """Configuration for metric calculation."""

    ssim_weight: float = 0.7
    size_weight: float = 0.3
    max_dimension_for_ssim: int = SSIM_MAX_DIMENSION
    downsample_method: Literal["none", "auto", "always"] = "auto"
    approximate_for_large_images: bool = True


class MetricCalculator:
    """Computes perceptual quality metrics.

    Clean Architecture: Accepts ImageAdapter implementations (PILImageAdapter)
    or PIL.Image for backwards compatibility. Core domain depends only
    on abstractions, not on PIL directly.
    """

    def __init__(
        self,
        *,
        ssim_weight: float = 0.7,
        size_weight: float = 0.3,
        config: MetricConfig | None = None,
    ) -> None:
        if math.isnan(ssim_weight) or math.isinf(ssim_weight):
            raise ValueError("ssim_weight must be a finite number")
        if math.isnan(size_weight) or math.isinf(size_weight):
            raise ValueError("size_weight must be a finite number")
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be non-negative")
        if size_weight < 0:
            raise ValueError("size_weight must be non-negative")
        total_weight = ssim_weight + size_weight
        if total_weight == 0:
            self.ssim_weight = 0.5
            self.size_weight = 0.5
        else:
            self.ssim_weight = ssim_weight / total_weight
            self.size_weight = size_weight / total_weight

        self._config = config or MetricConfig(
            ssim_weight=self.ssim_weight,
            size_weight=self.size_weight,
        )

    def compute(
        self,
        original: ImageLike,
        optimized: ImageLike,
        *,
        original_bytes: bytes,
        optimized_bytes: bytes,
    ) -> MetricResult:
        """Compute metrics for original vs optimized image.

        Args:
            original: Original image (ImageAdapter or PIL.Image).
            optimized: Optimized image (ImageAdapter or PIL.Image).
            original_bytes: Original image bytes.
            optimized_bytes: Optimized image bytes.

        Returns:
            MetricResult with computed metrics.
        """
        original_pil = self._get_pil_image(original)
        optimized_pil = self._get_pil_image(optimized)

        if original_pil.size != optimized_pil.size:
            raise ValueError(
                f"Image size mismatch: original {original_pil.size} "
                f"vs optimized {optimized_pil.size}"
            )

        ssim_value = self._ssim(original_pil, optimized_pil)
        psnr_value = self._psnr(original_pil, optimized_pil)
        before = size_kb(original_bytes)
        after = size_kb(optimized_bytes)
        score = self._perceptual_score(ssim_value, before, after)

        return MetricResult(
            ssim=ssim_value,
            psnr=psnr_value,
            size_before_kb=before,
            size_after_kb=after,
            perceptual_score=score,
        )

    def _get_pil_image(self, image: ImageLike) -> Image.Image:
        """Extract PIL image from adapter or use directly."""
        from perceptimg.adapters.pil_adapter import PILImageAdapter

        if isinstance(image, PILImageAdapter):
            return image.pil_image
        return image

    def _should_downsample(self, width: int, height: int) -> bool:
        if self._config.downsample_method == "none":
            return False
        if self._config.downsample_method == "always":
            return True
        max_dim = max(width, height)
        return max_dim > self._config.max_dimension_for_ssim

    def _downsample_image(self, image: Image.Image, max_dimension: int) -> Image.Image:
        """Downsample PIL image using LANCZOS resampling."""
        from PIL import Image

        width, height = image.size
        scale = min(max_dimension / width, max_dimension / height)
        if scale >= 1.0:
            return image
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _ssim(self, original: Image.Image, optimized: Image.Image) -> float:
        if original.mode in ("RGBA", "LA"):
            target_mode = "RGBA"
        else:
            target_mode = "RGB"

        width, height = original.size
        if self._should_downsample(width, height):
            max_dim = self._config.max_dimension_for_ssim
            original = self._downsample_image(original.convert(target_mode), max_dim)
            optimized = self._downsample_image(optimized.convert(target_mode), max_dim)

        original_arr = np.asarray(original.convert(target_mode), dtype=np.float32)
        optimized_arr = np.asarray(optimized.convert(target_mode), dtype=np.float32)

        ssim_result = structural_similarity(
            original_arr,
            optimized_arr,
            data_range=255,
            channel_axis=2 if target_mode in ("RGB", "RGBA") else None,
        )
        return float(ssim_result[0] if isinstance(ssim_result, tuple) else ssim_result)

    def _psnr(self, original: Image.Image, optimized: Image.Image) -> float:
        if original.mode in ("RGBA", "LA"):
            target_mode = "RGBA"
        else:
            target_mode = "RGB"

        width, height = original.size
        if self._should_downsample(width, height):
            max_dim = self._config.max_dimension_for_ssim
            original = self._downsample_image(original.convert(target_mode), max_dim)
            optimized = self._downsample_image(optimized.convert(target_mode), max_dim)

        original_arr = np.asarray(original.convert(target_mode), dtype=np.float32)
        optimized_arr = np.asarray(optimized.convert(target_mode), dtype=np.float32)
        mse = float(np.mean((original_arr - optimized_arr) ** 2))
        if mse == 0:
            return PSNR_MAX_VALUE
        return float(10 * np.log10((255.0**2) / mse))

    def _perceptual_score(self, ssim_value: float, before_kb: float, after_kb: float) -> float:
        if after_kb <= 0 or before_kb <= 0:
            return ssim_value
        compression_ratio = (before_kb - after_kb) / before_kb
        size_score = max(0.0, compression_ratio)
        return self.ssim_weight * ssim_value + self.size_weight * size_score
