from PIL import Image

from perceptimg.core.metrics import MetricCalculator
from perceptimg.utils.image_io import image_to_bytes


def test_perceptual_score_branch_otherwise() -> None:
    image = Image.new("RGB", (8, 8), "white")
    original = image_to_bytes(image, format="PNG")
    # Make optimized artificially larger to hit after_kb <=0 check bypass
    optimized = original + b"pad"
    metrics = MetricCalculator().compute(
        image, image, original_bytes=original, optimized_bytes=optimized
    )
    assert metrics.perceptual_score <= 1.0
