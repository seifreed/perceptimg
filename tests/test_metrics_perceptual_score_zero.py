from PIL import Image

from perceptimg.core.metrics import MetricCalculator
from perceptimg.utils.image_io import image_to_bytes


def test_perceptual_score_zero_after_bytes() -> None:
    image = Image.new("RGB", (8, 8), "black")
    original = image_to_bytes(image, format="PNG")
    optimized = b""
    calc = MetricCalculator()
    result = calc.compute(image, image, original_bytes=original, optimized_bytes=optimized)
    assert result.perceptual_score == -float("inf")  # 0-byte output is invalid and penalized
