"""Domain interfaces following Clean Architecture principles.

These protocols define the contracts between layers, allowing the core
domain to remain independent of external frameworks like PIL.

Dependency Rule:
    core → interfaces (depends on abstractions only)
    adapters → implements interfaces (depends on frameworks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class ImageAdapter(Protocol):
    """Protocol for image operations, abstracting any image framework.

    This allows the core domain to work with images without
    depending directly on PIL, enabling:
    - Testing with mock images
    - Potential framework replacements
    - Clean dependency inversion
    """

    @property
    def size(self) -> tuple[int, int]:
        """Image dimensions as (width, height)."""
        ...

    @property
    def mode(self) -> str:
        """Image color mode (e.g., 'RGB', 'RGBA', 'L')."""
        ...

    @property
    def width(self) -> int:
        """Image width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Image height in pixels."""
        ...

    def convert(self, mode: str) -> ImageAdapter:
        """Convert image to a different color mode.

        Args:
            mode: Target color mode (e.g., 'RGB', 'RGBA', 'L').

        Returns:
            Converted image adapter.
        """
        ...

    def to_bytes(self, format: str, **kwargs: object) -> bytes:
        """Serialize image to bytes in specified format.

        Args:
            format: Image format (e.g., 'PNG', 'JPEG', 'WEBP').
            **kwargs: Format-specific save options.

        Returns:
            Serialized image bytes.
        """
        ...


class ImageLoader(Protocol):
    """Protocol for loading images from various sources."""

    def load_from_path(self, path: str | Path) -> ImageAdapter:
        """Load an image from a file path.

        Args:
            path: Path to image file.

        Returns:
            Loaded image adapter.
        """
        ...

    def load_from_bytes(self, data: bytes) -> ImageAdapter:
        """Load an image from raw bytes.

        Args:
            data: Raw image bytes.

        Returns:
            Loaded image adapter.
        """
        ...


class ImageWriter(Protocol):
    """Protocol for writing images to various destinations."""

    def save(self, path: str | Path, format: str | None = None, **kwargs: object) -> None:
        """Save image to a file.

        Args:
            path: Destination path.
            format: Image format (defaults to extension-based detection).
            **kwargs: Format-specific save options.
        """
        ...


class ArrayAdapter(Protocol):
    """Protocol for converting images to arrays for computation."""

    def to_array(self, dtype: str = "float32") -> object:
        """Convert image to numpy array.

        Args:
            dtype: Data type for array.

        Returns:
            Numpy array representation.
        """
        ...
