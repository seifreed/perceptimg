from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer
from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.utils.image_io import image_to_bytes


class SimpleEngine(OptimizationEngine):
    format = "jpeg"
    priority = 2

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        data = image_to_bytes(image, format="JPEG")
        return EngineResult(data=data, format="jpeg", quality=strategy.quality, metadata={})


def test_optimizer_accepts_when_policy_allows() -> None:
    optimizer = Optimizer(engines=[SimpleEngine()])
    image = Image.new("RGB", (64, 64), "white")
    policy = Policy(max_size_kb=None, min_ssim=None, preferred_formats=("jpeg",))
    # Use uncompressed PNG as original so JPEG output is smaller
    original_bytes = image_to_bytes(image, format="PNG", save_kwargs={"compress_level": 0})
    result = optimizer.optimize_from_analysis(
        image, optimizer.analyzer.analyze(image), policy, original_bytes=original_bytes
    )
    assert result.image_bytes
