"""Image I/O helpers isolated from business logic."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ..exceptions import ImageLoadError, ImageSaveError

__all__ = [
    "ImageIOError",
    "InvalidImageError",
    "ImageLoadError",
    "ImageSaveError",
    "load_image",
    "image_to_bytes",
    "bytes_to_image",
    "size_kb",
]


class ImageIOError(RuntimeError):
    """Raised when image I/O operations fail."""


class InvalidImageError(ImageIOError):
    """Raised when an image is invalid (e.g., zero dimensions)."""


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk, preserving transparency if present.

    Images with transparency (RGBA, LA, P with transparency) retain their alpha channel.
    Palette images (P mode) are converted to RGBA if they have transparency, otherwise RGB.
    All other modes are converted to RGB for consistent processing.

    Raises:
        ImageLoadError: If the image cannot be loaded.
        InvalidImageError: If the image has invalid dimensions (0x0).
    """
    image = Path(path)
    try:
        with Image.open(image) as raw_image:
            raw_image.load()
            loaded_image: Image.Image = raw_image.copy()
    except FileNotFoundError as exc:
        raise ImageLoadError(f"Image file not found: {path}") from exc
    except PermissionError as exc:
        raise ImageLoadError(f"Permission denied reading image: {path}") from exc
    except Exception as exc:
        if "UnidentifiedImageError" in type(exc).__name__:
            raise ImageLoadError(f"Cannot identify image file: {path}") from exc
        raise

    width, height = loaded_image.size
    if width == 0 or height == 0:
        raise InvalidImageError(f"Image has invalid dimensions: {width}x{height}")

    mode = loaded_image.mode
    if mode == "P":
        if "transparency" in loaded_image.info:
            loaded_image = loaded_image.convert("RGBA")
        else:
            loaded_image = loaded_image.convert("RGB")
    elif mode in ("RGBA", "LA"):
        pass
    elif mode != "RGB":
        loaded_image = loaded_image.convert("RGB")
    return loaded_image


def image_to_bytes(
    image: Image.Image, *, format: str, save_kwargs: Mapping[str, Any] | None = None
) -> bytes:
    """Serialize a PIL image into bytes for a given format.

    Raises:
        ImageSaveError: If the image cannot be encoded.
    """
    buffer = BytesIO()
    kwargs = dict(save_kwargs or {})
    try:
        image.save(buffer, format=format, **kwargs)
    except KeyError as exc:
        raise ImageSaveError(f"Unknown format: {format}") from exc
    except ValueError as exc:
        raise ImageSaveError(f"Invalid save parameters for {format}: {exc}") from exc

    return buffer.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    """Construct a PIL image from raw bytes, preserving transparency if present.

    Images with transparency (RGBA, LA, P with transparency) retain their alpha channel.
    Palette images (P mode) are converted to RGBA if they have transparency, otherwise RGB.
    All other modes are converted to RGB for consistent processing.

    Raises:
        ImageLoadError: If the image cannot be decoded.
        InvalidImageError: If the image has invalid dimensions (0x0).
    """
    try:
        with Image.open(BytesIO(data)) as raw_image:
            raw_image.load()
            loaded_image: Image.Image = raw_image.copy()
    except Exception as exc:
        if "UnidentifiedImageError" in type(exc).__name__:
            raise ImageLoadError(f"Cannot identify image from bytes: {exc}") from exc
        raise

    width, height = loaded_image.size
    if width == 0 or height == 0:
        raise InvalidImageError(f"Image has invalid dimensions: {width}x{height}")

    mode = loaded_image.mode
    if mode == "P":
        if "transparency" in loaded_image.info:
            loaded_image = loaded_image.convert("RGBA")
        else:
            loaded_image = loaded_image.convert("RGB")
    elif mode in ("RGBA", "LA"):
        pass
    elif mode != "RGB":
        loaded_image = loaded_image.convert("RGB")
    return loaded_image


def size_kb(data: bytes) -> float:
    """Return size in kilobytes (binary KB)."""
    return len(data) / 1024.0
