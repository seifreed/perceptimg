from PIL import Image

from perceptimg.core.metrics import MetricCalculator
from perceptimg.utils.image_io import image_to_bytes


def test_perceptual_score_branch() -> None:
    image = Image.new("RGB", (16, 16), "gray")
    original = image_to_bytes(image, format="PNG")
    optimized = image_to_bytes(image, format="PNG", save_kwargs={"compress_level": 0})
    calc = MetricCalculator()
    result = calc.compute(image, image, original_bytes=original, optimized_bytes=optimized)
    assert result.perceptual_score <= 1.0
