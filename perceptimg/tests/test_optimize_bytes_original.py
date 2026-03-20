from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import optimize_bytes
from perceptimg.utils.image_io import image_to_bytes


def test_optimize_bytes_uses_original_bytes_when_provided() -> None:
    image = Image.new("RGB", (16, 16), "orange")
    raw = image_to_bytes(image, format="PNG")
    policy = Policy(max_size_kb=200)
    result = optimize_bytes(raw, policy)
    assert result.image_bytes
    # Should not be empty and ssim should be computed
    assert result.report.ssim > 0.5
