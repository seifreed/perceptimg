from PIL import Image

from perceptimg.core.metrics import MetricCalculator
from perceptimg.utils.image_io import image_to_bytes


def test_perceptual_score_branch_otherwise() -> None:
    image = Image.new("RGB", (8, 8), "white")
    original = image_to_bytes(image, format="PNG")
    # Make optimized artificially larger to hit size inflation penalty
    optimized = original + b"pad"
    calc = MetricCalculator()
    metrics = calc.compute(
        image, image, original_bytes=original, optimized_bytes=optimized
    )
    assert metrics.perceptual_score <= 1.0
    # Score should not exceed pure SSIM component when file grows (size_score clamped to 0)
    assert metrics.perceptual_score <= calc.ssim_weight * metrics.ssim + calc.size_weight * 0.0
