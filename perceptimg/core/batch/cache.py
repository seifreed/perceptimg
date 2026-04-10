"""Analysis cache with LRU eviction.

Clean Architecture: Core domain uses ImageAdapter Protocol abstraction.
PIL is imported only in adapters layer.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from PIL import Image

from perceptimg.core.analyzer import AnalysisResult

ImageLike = Any


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
        if hasattr(image, "pil_image"):
            adapter_image = getattr(image, "pil_image", None)
            if adapter_image is not None:
                return cast(Image.Image, adapter_image)
        return cast(Image.Image, image)

    def _compute_efficient_hash(self, pil_image: Image.Image) -> str:
        """Compute hash efficiently for large images using downsampling.

        For images larger than 10MP, downsampling to 256x256 before hashing
        reduces memory usage significantly while maintaining collision resistance
        for cache purposes.
        """
        from PIL import Image

        thumb_size = (256, 256)
        thumb = pil_image.resize(thumb_size, Image.Resampling.LANCZOS)
        img_bytes = thumb.tobytes()
        return hashlib.md5(img_bytes, usedforsecurity=False).hexdigest()

    def _compute_hash(self, image: ImageLike, path: Path | None = None) -> str:
        """Compute a hash key for the image or file.

        Note: When a file path is provided, this method uses file metadata
        (mtime, size, path) as the cache key instead of hashing the image content.
        This is a performance optimization that avoids reading the full file content
        for cache lookups. The trade-offs are:

        - Cache misses may occur if file content changes without mtime/size changes
          (e.g., some editors preserve mtime when saving)
        - Cache hits may be incorrect if different files share the same mtime/size
          (extremely rare in practice)
        - Files moved or renamed will get new cache keys (path-dependent)

        For applications requiring strict content-based cache invalidation,
        consider computing a content hash instead of using this metadata-based approach.
        """
        if path and path.exists():
            stat = path.stat()
            return f"{stat.st_mtime_ns}_{stat.st_size}_{path.resolve()}"
        pil_image = self._get_pil_image(image)
        width, height = pil_image.size
        # For large images (>10MP), use efficient hashing to avoid memory issues
        if width * height > 10_000_000:
            img_hash = self._compute_efficient_hash(pil_image)
        else:
            img_bytes = pil_image.tobytes()
            img_hash = hashlib.md5(img_bytes, usedforsecurity=False).hexdigest()
        return f"{img_hash}_{width}x{height}"

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
        if path:
            try:
                file_size = path.stat().st_size
            except (FileNotFoundError, OSError):
                file_size = 0
        else:
            file_size = 0
        with self._lock:
            if len(self._cache) >= self._maxsize and key not in self._cache:
                if self._order:
                    oldest = next(iter(self._order))
                    del self._order[oldest]
                    self._cache.pop(oldest, None)
                elif self._cache:
                    # Inconsistent state - _order is empty but _cache has entries
                    # Rebuild _order from _cache keys to recover
                    logging.getLogger(__name__).warning(
                        "Cache _order empty but _cache has %d entries, rebuilding",
                        len(self._cache),
                    )
                    self._order = OrderedDict((k, None) for k in self._cache.keys())
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
