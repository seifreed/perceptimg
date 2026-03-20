"""Tests for image I/O error handling."""

from __future__ import annotations

import pytest

from perceptimg.exceptions import ImageLoadError
from perceptimg.utils.image_io import bytes_to_image, load_image


def test_bytes_to_image_with_invalid_bytes() -> None:
    """bytes_to_image should raise ImageLoadError for non-image bytes."""
    invalid_bytes = b"not an image at all"
    with pytest.raises(ImageLoadError) as exc_info:
        bytes_to_image(invalid_bytes)
    assert "Cannot identify" in str(exc_info.value)


def test_bytes_to_image_with_empty_bytes() -> None:
    """bytes_to_image should raise ImageLoadError for empty bytes."""
    empty_bytes = b""
    with pytest.raises(ImageLoadError) as exc_info:
        bytes_to_image(empty_bytes)
    assert "Cannot identify" in str(exc_info.value)


def test_bytes_to_image_with_truncated_png(tmp_path) -> None:
    """bytes_to_image should raise ImageLoadError for truncated image data."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), "red")
    img.save(tmp_path / "temp.png", "PNG")
    with open(tmp_path / "temp.png", "rb") as f:
        valid_bytes = f.read()
    truncated_bytes = valid_bytes[: len(valid_bytes) // 2]

    with pytest.raises(ImageLoadError):
        bytes_to_image(truncated_bytes)


def test_load_image_nonexistent_file(tmp_path) -> None:
    """load_image should raise ImageLoadError for nonexistent file."""
    nonexistent = tmp_path / "does_not_exist.png"
    with pytest.raises(ImageLoadError) as exc_info:
        load_image(nonexistent)
    assert "not found" in str(exc_info.value)


def test_load_image_corrupted_file(tmp_path) -> None:
    """load_image should raise ImageLoadError for corrupted image file."""
    corrupted = tmp_path / "corrupted.png"
    corrupted.write_bytes(b"not valid image data")

    with pytest.raises(ImageLoadError) as exc_info:
        load_image(corrupted)
    assert "Cannot identify" in str(exc_info.value)


def test_bytes_to_image_with_valid_png(tmp_path) -> None:
    """bytes_to_image should succeed with valid PNG bytes."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), "blue")
    img.save(tmp_path / "test.png", "PNG")

    with open(tmp_path / "test.png", "rb") as f:
        valid_bytes = f.read()

    result = bytes_to_image(valid_bytes)
    assert result.size == (10, 10)
    assert result.mode == "RGB"


def test_policy_validate_for_size_warnings(tmp_path) -> None:
    """Policy.validate_for_size should return warnings for aggressive policies."""
    from perceptimg import Policy

    policy = Policy(max_size_kb=1, min_ssim=0.99)
    warnings = policy.validate_for_size(100.0)
    assert len(warnings) > 0
    assert "impossible" in warnings[0].lower()


def test_policy_validate_for_size_lossless_warning(tmp_path) -> None:
    """Policy.validate_for_size should warn about lossless with small max_size."""
    from perceptimg import Policy

    policy = Policy(max_size_kb=10, allow_lossy=False)
    warnings = policy.validate_for_size(100.0)
    assert len(warnings) > 0
    assert "lossless" in warnings[0].lower()


def test_policy_validate_for_size_no_warnings_reasonable(tmp_path) -> None:
    """Policy.validate_for_size should return empty list for reasonable policies."""
    from perceptimg import Policy

    policy = Policy(max_size_kb=100, min_ssim=0.95)
    warnings = policy.validate_for_size(200.0)
    assert len(warnings) == 0
