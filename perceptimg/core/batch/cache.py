"""Analysis cache with LRU eviction.

Clean Architecture: Core domain uses ImageAdapter Protocol abstraction.
PIL is imported only in adapters layer.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from PIL import Image

    from ..adapters.pil_adapter import PILImageAdapter

from perceptimg.core.analyzer import AnalysisResult

ImageLike = Union["Image.Image", "PILImageAdapter"]


@dataclass(slots=True)
class CacheEntry:
    """Cache entry for analysis results."""

    image_hash: str
    analysis: AnalysisResult
    file_size: int


class AnalysisCache:
    """LRU cache for image analysis results.

    Clean Architecture: Accepts ImageAdapter implementations or PIL.Image
    for backwards compatibility. Core depends only on abstractions.
    """

    def __init__(self, maxsize: int = 128):
        self._cache: dict[str, CacheEntry] = {}
        self._maxsize = maxsize
        self._order: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def _get_pil_image(self, image: ImageLike) -> Image.Image:
        """Extract PIL image from adapter or use directly."""
        from perceptimg.adapters.pil_adapter import PILImageAdapter

        if isinstance(image, PILImageAdapter):
            return image.pil_image
        return image

    def _compute_hash(self, image: ImageLike, path: Path | None = None) -> str:
        if path and path.exists():
            stat = path.stat()
            return f"{stat.st_mtime_ns}_{stat.st_size}_{path.resolve()}"
        pil_image = self._get_pil_image(image)
        img_bytes = pil_image.tobytes()
        img_hash = hashlib.md5(img_bytes, usedforsecurity=False).hexdigest()
        return f"{img_hash}_{pil_image.size[0]}x{pil_image.size[1]}"

    def get(self, image: ImageLike, path: Path | None = None) -> AnalysisResult | None:
        key = self._compute_hash(image, path)
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                self._order.move_to_end(key)
                return entry.analysis
            return None

    def set(self, image: ImageLike, analysis: AnalysisResult, path: Path | None = None) -> None:
        key = self._compute_hash(image, path)
        file_size = path.stat().st_size if path and path.exists() else 0
        with self._lock:
            if len(self._cache) >= self._maxsize and key not in self._cache:
                if self._order:
                    oldest = next(iter(self._order))
                    del self._order[oldest]
                    self._cache.pop(oldest, None)
            self._cache[key] = CacheEntry(
                image_hash=key,
                analysis=analysis,
                file_size=file_size,
            )
            self._order[key] = None
            self._order.move_to_end(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._order.clear()
