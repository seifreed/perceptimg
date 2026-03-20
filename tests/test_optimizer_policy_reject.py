import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer
from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.exceptions import OptimizationError
from perceptimg.utils.image_io import image_to_bytes


class TooBigEngine(OptimizationEngine):
    format = "jpeg"
    priority = 1

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        data = image_to_bytes(image, format="PNG", save_kwargs={"compress_level": 0})
        return EngineResult(data=data, format="jpeg", quality=strategy.quality, metadata={})


def test_optimize_from_analysis_rejects_policy() -> None:
    optimizer = Optimizer(engines=[TooBigEngine()])
    image = Image.new("RGB", (8, 8), "white")
    policy = Policy(max_size_kb=1, min_ssim=1.0)
    with pytest.raises(OptimizationError):
        optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)
