"""Perceptual metrics for evaluating optimization outputs.

Clean Architecture: Core domain only depends on abstractions (ImageAdapter Protocol).
PIL is imported only in adapters layer. This module uses duck typing to work
with both PIL.Image and ImageAdapter implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from skimage.metrics import structural_similarity

if TYPE_CHECKING:
    from PIL import Image

ImageLike = Any

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
        if config is not None:
            ssim_weight = config.ssim_weight
            size_weight = config.size_weight
        if math.isnan(ssim_weight) or math.isinf(ssim_weight):
            raise ValueError("ssim_weight must be a finite number")
        if math.isnan(size_weight) or math.isinf(size_weight):
            raise ValueError("size_weight must be a finite number")
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be non-negative")
        if size_weight < 0:
            raise ValueError("size_weight must be non-negative")
        total_weight = ssim_weight + size_weight
        if total_weight < 1e-9:
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
        before = len(original_bytes) / 1024.0
        after = len(optimized_bytes) / 1024.0
        score = self._perceptual_score(ssim_value, before, after)

        return MetricResult(
            ssim=ssim_value,
            psnr=psnr_value,
            size_before_kb=before,
            size_after_kb=after,
            perceptual_score=score,
        )

    def _get_pil_image(self, image: ImageLike) -> Any:
        """Extract PIL image from adapter or use directly."""
        if hasattr(image, "pil_image"):
            adapter_image = getattr(image, "pil_image", None)
            if adapter_image is not None:
                return cast(Any, adapter_image)
        return cast(Any, image)

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
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def _ssim_win_size(width: int, height: int) -> int | None:
        """Return a valid SSIM window size for the given dimensions.

        skimage defaults to a 7x7 window, which fails for images smaller than 7px.
        We pick the largest odd window up to 7 that fits the image. For images
        smaller than 7px in either dimension, SSIM's sliding-window formulation is
        not defined reliably and we fall back to a stable similarity proxy.
        """
        min_dim = min(width, height)
        if min_dim < 7:
            return None
        win_size = min(min_dim, 7)
        if win_size % 2 == 0:
            win_size -= 1
        return win_size

    @staticmethod
    def _tiny_image_similarity(
        original_arr: np.ndarray,
        optimized_arr: np.ndarray,
    ) -> float:
        """Fallback similarity for images too small for windowed SSIM."""
        if np.array_equal(original_arr, optimized_arr):
            return 1.0
        mse = float(np.mean((original_arr - optimized_arr) ** 2))
        return float(np.clip(1.0 - (mse / (255.0**2)), 0.0, 1.0))

    @staticmethod
    def _resolve_target_mode(original: Image.Image, optimized: Image.Image) -> str:
        """Determine comparison mode considering both images.

        If both images have an alpha channel, compare in RGBA to account for
        transparency differences. When only one image has alpha (e.g. RGBA
        original optimized to JPEG/RGB), compare in RGB so the identical
        synthetic alpha channel does not dilute real RGB differences.

        For grayscale images, maintain L mode to avoid unnecessary conversion.

        Palette images (P mode) with transparency are treated as having alpha.
        """
        alpha_modes = {"RGBA", "LA", "PA"}
        gray_modes = {"L", "I", "F"}

        # Detect if images have alpha channels
        orig_has_alpha = original.mode in alpha_modes
        opt_has_alpha = optimized.mode in alpha_modes

        # Check for transparency in palette mode
        if original.mode == "P":
            trans = original.info.get("transparency")
            if trans is not None:
                orig_has_alpha = True
        if optimized.mode == "P":
            trans = optimized.info.get("transparency")
            if trans is not None:
                opt_has_alpha = True

        # If both have alpha, compare in RGBA
        if orig_has_alpha and opt_has_alpha:
            return "RGBA"

        # If both are grayscale (without alpha, since alpha case handled above)
        orig_is_gray = original.mode in gray_modes
        opt_is_gray = optimized.mode in gray_modes
        if orig_is_gray and opt_is_gray:
            return "L"

        return "RGB"

    def _ssim(self, original: Image.Image, optimized: Image.Image) -> float:
        target_mode = self._resolve_target_mode(original, optimized)

        width, height = original.size
        if self._should_downsample(width, height):
            max_dim = self._config.max_dimension_for_ssim
            original = self._downsample_image(original.convert(target_mode), max_dim)
            optimized = self._downsample_image(optimized.convert(target_mode), max_dim)
        else:
            original = original.convert(target_mode)
            optimized = optimized.convert(target_mode)

        original_arr = np.asarray(original, dtype=np.float32)
        optimized_arr = np.asarray(optimized, dtype=np.float32)
        win_size = self._ssim_win_size(*original.size)
        if win_size is None:
            return self._tiny_image_similarity(original_arr, optimized_arr)

        ssim_result = structural_similarity(
            original_arr,
            optimized_arr,
            data_range=255,
            channel_axis=2 if target_mode in ("RGB", "RGBA") else None,
            win_size=win_size,
        )
        # With channel_axis parameter, skimage always returns a float, not a tuple
        return float(ssim_result)

    def _psnr(self, original: Image.Image, optimized: Image.Image) -> float:
        target_mode = self._resolve_target_mode(original, optimized)

        width, height = original.size
        if self._should_downsample(width, height):
            max_dim = self._config.max_dimension_for_ssim
            original = self._downsample_image(original.convert(target_mode), max_dim)
            optimized = self._downsample_image(optimized.convert(target_mode), max_dim)
        else:
            original = original.convert(target_mode)
            optimized = optimized.convert(target_mode)

        original_arr = np.asarray(original, dtype=np.float32)
        optimized_arr = np.asarray(optimized, dtype=np.float32)
        mse = float(np.mean((original_arr - optimized_arr) ** 2))
        if mse == 0:
            return PSNR_MAX_VALUE
        return float(10 * np.log10((255.0**2) / mse))

    def _perceptual_score(self, ssim_value: float, before_kb: float, after_kb: float) -> float:
        if not math.isfinite(before_kb) or before_kb <= 0:
            return -float("inf")
        if not math.isfinite(after_kb) or after_kb <= 0:
            return -float("inf")  # Invalid 0-byte output should never win
        if not math.isfinite(ssim_value):
            # NaN or infinite SSIM indicates numerical issues; reject candidate
            return -float("inf")
        ssim_clamped = max(0.0, min(1.0, ssim_value))
        compression_ratio = (before_kb - after_kb) / before_kb
        size_score = max(-1.0, min(1.0, compression_ratio))
        return self.ssim_weight * ssim_clamped + self.size_weight * size_score
