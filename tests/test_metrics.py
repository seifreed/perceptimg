import pytest
from PIL import Image

from perceptimg.core.metrics import PSNR_MAX_VALUE, MetricCalculator, MetricConfig
from perceptimg.utils.image_io import image_to_bytes


def test_metrics_identical_images_perfect_scores() -> None:
    image = Image.new("RGB", (16, 16), "red")
    data = image_to_bytes(image, format="PNG")
    metrics = MetricCalculator().compute(image, image, original_bytes=data, optimized_bytes=data)
    assert metrics.ssim == 1.0
    assert metrics.psnr == PSNR_MAX_VALUE
    assert metrics.perceptual_score <= 1.0


def test_metrics_size_changes_reflected() -> None:
    image = Image.new("RGB", (16, 16), "blue")
    original = image_to_bytes(image, format="PNG")
    optimized = image_to_bytes(image, format="PNG", save_kwargs={"compress_level": 9})
    metrics = MetricCalculator().compute(
        image, image, original_bytes=original, optimized_bytes=optimized
    )
    assert metrics.perceptual_score <= 1.0
    assert metrics.ssim == 1.0


def test_metrics_small_image_uses_valid_ssim_window() -> None:
    image = Image.new("RGB", (4, 4), "green")
    data = image_to_bytes(image, format="PNG")
    metrics = MetricCalculator().compute(image, image, original_bytes=data, optimized_bytes=data)
    assert metrics.ssim == 1.0


def test_metrics_tiny_image_fallback_does_not_raise() -> None:
    image = Image.new("RGB", (2, 2), "yellow")
    data = image_to_bytes(image, format="PNG")
    metrics = MetricCalculator().compute(image, image, original_bytes=data, optimized_bytes=data)
    assert metrics.ssim == 1.0


def test_metrics_extreme_aspect_ratio_no_crash() -> None:
    """Downsampling a 1x3000 image must not produce 0-dimension images."""
    image = Image.new("RGB", (1, 3000), "red")
    data = image_to_bytes(image, format="PNG")
    config = MetricConfig(max_dimension_for_ssim=2048, downsample_method="auto")
    calc = MetricCalculator(config=config)
    metrics = calc.compute(image, image, original_bytes=data, optimized_bytes=data)
    assert metrics.ssim == 1.0


def test_metric_calculator_respects_config_weights() -> None:
    config = MetricConfig(ssim_weight=0.5, size_weight=0.5)
    calc = MetricCalculator(config=config)
    assert calc.ssim_weight == 0.5
    assert calc.size_weight == 0.5


def test_metric_calculator_config_overrides_kwargs() -> None:
    config = MetricConfig(ssim_weight=0.2, size_weight=0.8)
    calc = MetricCalculator(ssim_weight=0.9, size_weight=0.1, config=config)
    assert calc.ssim_weight == pytest.approx(0.2)
    assert calc.size_weight == pytest.approx(0.8)


def test_metrics_rgba_original_vs_rgb_optimized_compares_in_rgb() -> None:
    """When original is RGBA and optimized is RGB, compare in RGB to avoid
    inflating SSIM with a synthetic identical alpha channel."""
    # Create RGBA image with variable alpha (semi-transparent)
    rgba_image = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    # Create RGB version (same RGB color, alpha dropped)
    rgb_image = Image.new("RGB", (16, 16), (255, 0, 0))
    rgba_bytes = image_to_bytes(rgba_image, format="PNG")
    rgb_bytes = image_to_bytes(rgb_image, format="PNG")
    calc = MetricCalculator()
    metrics = calc.compute(
        rgba_image,
        rgb_image,
        original_bytes=rgba_bytes,
        optimized_bytes=rgb_bytes,
    )
    # RGB content is identical — alpha loss is expected when converting to RGB
    # and should not penalize the candidate
    assert metrics.ssim == pytest.approx(1.0)
    assert metrics.psnr == pytest.approx(PSNR_MAX_VALUE)


def test_metrics_rgba_vs_rgba_compares_in_rgba() -> None:
    """When both images are RGBA, SSIM should compare in RGBA including alpha."""
    rgba1 = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    rgba2 = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    bytes1 = image_to_bytes(rgba1, format="PNG")
    bytes2 = image_to_bytes(rgba2, format="PNG")
    calc = MetricCalculator()
    metrics = calc.compute(rgba1, rgba2, original_bytes=bytes1, optimized_bytes=bytes2)
    # Different alpha channels should lower SSIM
    assert metrics.ssim < 1.0


def test_perceptual_score_penalizes_file_growth() -> None:
    """File growth should actively penalize perceptual score below ssim contribution."""
    image = Image.new("RGB", (16, 16), "red")
    small_bytes = image_to_bytes(image, format="PNG", save_kwargs={"compress_level": 9})
    large_bytes = image_to_bytes(image, format="PNG", save_kwargs={"compress_level": 0})
    calc = MetricCalculator()
    # Use large_bytes as "original" and small_bytes as "optimized" (compression)
    compressed = calc.compute(image, image, original_bytes=large_bytes, optimized_bytes=small_bytes)
    # Use small_bytes as "original" and large_bytes as "optimized" (growth)
    grown = calc.compute(image, image, original_bytes=small_bytes, optimized_bytes=large_bytes)
    # File growth should yield a lower perceptual score than compression
    assert grown.perceptual_score < compressed.perceptual_score
    # File growth should produce a score at most equal to the pure ssim contribution
    # (size_score is clamped to 0 when the file grows, so no size bonus)
    assert grown.perceptual_score <= calc.ssim_weight * grown.ssim
