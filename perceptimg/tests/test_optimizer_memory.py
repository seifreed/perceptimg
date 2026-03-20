from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer, optimize_bytes, optimize_image
from perceptimg.utils import heuristics
from perceptimg.utils.image_io import image_to_bytes


def test_optimize_bytes_in_memory_flow() -> None:
    image = Image.new("RGB", (32, 32), "green")
    buf = image_to_bytes(image, format="PNG")
    policy = Policy(max_size_kb=100, min_ssim=0.9)
    result = optimize_bytes(buf, policy)
    assert result.image_bytes
    assert result.report.ssim >= 0.9


def test_optimize_image_injects_optimizer() -> None:
    image = Image.new("RGB", (24, 24), "yellow")
    policy = Policy(max_size_kb=200)
    custom_optimizer = Optimizer(
        heuristic_config=heuristics.HeuristicConfig(edge_density_text_threshold=0.01)
    )
    result = optimize_image(image, policy, optimizer=custom_optimizer)
    assert result.report.size_after_kb <= 200
    assert result.report.chosen_format
