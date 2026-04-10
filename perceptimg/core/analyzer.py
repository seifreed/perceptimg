"""Visual content analyzer for perceptual signals.

Clean Architecture: Core domain only depends on abstractions (ImageAdapter Protocol).
PIL is imported only in adapters layer. This module uses duck typing to work
with both PIL.Image and ImageAdapter implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..utils import heuristics

if TYPE_CHECKING:
    from PIL import Image

ImageLike = Any


class AnalysisResult:
    """Lightweight description of image signals.

    Immutable value object representing the results of image analysis.
    """

    __slots__ = (
        "edge_density",
        "color_variance",
        "probable_text",
        "probable_faces",
        "resolution",
        "aspect_ratio",
    )

    def __init__(
        self,
        edge_density: float,
        color_variance: float,
        probable_text: bool,
        probable_faces: bool,
        resolution: tuple[int, int],
        aspect_ratio: float,
    ) -> None:
        self.edge_density = edge_density
        self.color_variance = color_variance
        self.probable_text = probable_text
        self.probable_faces = probable_faces
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio

    def __repr__(self) -> str:
        return (
            f"AnalysisResult(edge_density={self.edge_density:.3f}, "
            f"color_variance={self.color_variance:.3f}, "
            f"probable_text={self.probable_text}, "
            f"probable_faces={self.probable_faces}, "
            f"resolution={self.resolution})"
        )


class Analyzer:
    """Extracts visual signals for optimization decisions.

    Clean Architecture: Accepts ImageAdapter implementations (PILImageAdapter)
    or PIL.Image for backwards compatibility. Core domain depends only
    on abstractions, not on PIL directly.
    """

    def __init__(self, config: heuristics.HeuristicConfig | None = None) -> None:
        self.config = config or heuristics.DEFAULT_CONFIG

    def analyze(self, image: ImageLike) -> AnalysisResult:
        """Analyze image content.

        Args:
            image: ImageAdapter implementation or PIL.Image for compatibility.

        Returns:
            AnalysisResult with extracted visual signals.
        """
        pil_image = self._get_pil_image(image)

        width, height = pil_image.size
        aspect_ratio = heuristics.compute_aspect_ratio(width, height)
        edge_density = heuristics.compute_edge_density(pil_image, config=self.config)
        rgb_array = heuristics.to_rgb_array(pil_image)
        color_variance = heuristics.compute_color_variance(rgb_array)
        probable_text = heuristics.detect_probable_text(
            edge_density,
            color_variance,
            aspect_ratio,
            config=self.config,
        )
        probable_faces = heuristics.detect_probable_faces(rgb_array, config=self.config)

        return AnalysisResult(
            edge_density=edge_density,
            color_variance=color_variance,
            probable_text=probable_text,
            probable_faces=probable_faces,
            resolution=(width, height),
            aspect_ratio=aspect_ratio,
        )

    def _get_pil_image(self, image: ImageLike) -> Image.Image:
        """Extract PIL image from adapter or use directly."""
        if hasattr(image, "pil_image"):
            adapter_image = getattr(image, "pil_image", None)
            if adapter_image is not None:
                return cast(Image.Image, adapter_image)
        return cast(Image.Image, image)
