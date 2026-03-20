from PIL import Image

from perceptimg.core.metrics import PSNR_MAX_VALUE, MetricCalculator
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
    assert metrics.size_after_kb <= metrics.size_before_kb
    assert metrics.perceptual_score <= 1.0
