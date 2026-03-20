"""PIL adapter implementing the ImageAdapter Protocol.

Clean Architecture: This is the ONLY place where PIL is imported in the adapters layer.
The core domain uses ImageAdapter Protocol and never imports PIL directly.

Dependency Rule:
    core/analyzer.py → uses ImageAdapter (abstraction)
    adapters/pil_adapter.py → implements ImageAdapter (concrete)
    PIL → used only in adapters layer
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    pass


class PILImageAdapter:
    """Adapter that wraps PIL.Image to satisfy the ImageAdapter Protocol.

    This keeps PIL as an implementation detail in the adapters layer,
    following Clean Architecture's Dependency Rule.

    Attributes:
        _image: The underlying PIL Image object.

    Example:
        >>> adapter = PILImageAdapter.from_path("image.png")
        >>> print(adapter.size, adapter.mode)
        (800, 600) RGB
        >>> converted = adapter.convert("RGBA")
        >>> data = adapter.to_bytes("PNG")
    """

    def __init__(self, pil_image: Image.Image) -> None:
        """Initialize adapter with a PIL image.

        Args:
            pil_image: PIL Image object to wrap.
        """
        self._image = pil_image

    @property
    def size(self) -> tuple[int, int]:
        """Image dimensions as (width, height)."""
        w: int = self._image.width
        h: int = self._image.height
        return (w, h)

    @property
    def mode(self) -> str:
        """Image color mode (e.g., 'RGB', 'RGBA', 'L')."""
        mode: str = self._image.mode
        return mode

    @property
    def width(self) -> int:
        """Image width in pixels."""
        w: int = self._image.width
        return w

    @property
    def height(self) -> int:
        """Image height in pixels."""
        h: int = self._image.height
        return h

    def convert(self, mode: str) -> PILImageAdapter:
        """Convert image to a different color mode.

        Args:
            mode: Target color mode (e.g., 'RGB', 'RGBA', 'L').

        Returns:
            New adapter with converted image.
        """
        converted = self._image.convert(mode)
        return PILImageAdapter(converted)

    def to_bytes(self, format: str, **kwargs: Any) -> bytes:
        """Serialize image to bytes in specified format.

        Args:
            format: Image format (e.g., 'PNG', 'JPEG', 'WEBP').
            **kwargs: Format-specific save options.

        Returns:
            Serialized image bytes.
        """
        buffer = BytesIO()
        self._image.save(buffer, format=format, **kwargs)
        return buffer.getvalue()

    def save(self, path: str | Path, format: str | None = None, **kwargs: Any) -> None:
        """Save image to a file.

        Args:
            path: Destination path.
            format: Image format (defaults to extension-based detection).
            **kwargs: Format-specific save options.
        """
        self._image.save(Path(path), format=format, **kwargs)

    def to_array(self, dtype: str = "float32") -> Any:
        """Convert image to numpy array.

        Args:
            dtype: Data type for array.

        Returns:
            Numpy array representation.
        """
        import numpy as np

        return np.asarray(self._image, dtype=dtype)

    @property
    def pil_image(self) -> Image.Image:
        """Get the underlying PIL image for engine operations.

        Note: This is intentionally exposed for the engines layer
        which needs direct PIL access for optimization operations.
        """
        return self._image

    @classmethod
    def from_path(cls, path: str | Path) -> PILImageAdapter:
        """Load an image from a file path.

        Args:
            path: Path to image file.

        Returns:
            Adapter wrapping the loaded image.
        """
        pil_image = Image.open(Path(path))
        return cls(pil_image)

    @classmethod
    def from_bytes(cls, data: bytes) -> PILImageAdapter:
        """Create an adapter from raw image bytes.

        Args:
            data: Raw image bytes (PNG, JPEG, etc.).

        Returns:
            Adapter wrapping the decoded image.
        """
        pil_image = Image.open(BytesIO(data))
        return cls(pil_image)

    @classmethod
    def from_pil(cls, pil_image: Image.Image) -> PILImageAdapter:
        """Create adapter from existing PIL image.

        Args:
            pil_image: PIL Image object.

        Returns:
            Adapter wrapping the image.
        """
        return cls(pil_image)

    @classmethod
    def new(cls, mode: str, size: tuple[int, int], color: Any = None) -> PILImageAdapter:
        """Create a new blank image.

        Args:
            mode: Color mode (e.g., 'RGB', 'RGBA').
            size: Image dimensions as (width, height).
            color: Background color.

        Returns:
            Adapter wrapping the new image.
        """
        pil_image = Image.new(mode, size, color)
        return cls(pil_image)


def load_image(path: str | Path) -> PILImageAdapter:
    """Load an image from a file path.

    Args:
        path: Path to the image file.

    Returns:
        PILImageAdapter wrapping the loaded image.
    """
    return PILImageAdapter.from_path(path)


def bytes_to_image(data: bytes) -> PILImageAdapter:
    """Create an image adapter from bytes.

    Args:
        data: Raw image bytes.

    Returns:
        PILImageAdapter wrapping the decoded image.
    """
    return PILImageAdapter.from_bytes(data)


def image_to_bytes(image: PILImageAdapter, format: str, **kwargs: Any) -> bytes:
    """Serialize an image adapter to bytes.

    Args:
        image: Image adapter to serialize.
        format: Target format (PNG, JPEG, WEBP, etc.).
        **kwargs: Format-specific options.

    Returns:
        Serialized image bytes.
    """
    return image.to_bytes(format, **kwargs)
