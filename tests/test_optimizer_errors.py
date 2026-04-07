import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer, optimize_image
from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.exceptions import OptimizationError
from perceptimg.utils.image_io import image_to_bytes


class NoOpEngine(OptimizationEngine):
    format = "jpeg"
    priority = 1

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        data = image_to_bytes(image, format="PNG", save_kwargs={"compress_level": 0})
        return EngineResult(data=data, format="jpeg", quality=100, metadata={})


def test_optimize_raises_when_no_engines() -> None:
    optimizer = Optimizer()
    optimizer.engines = []
    optimizer.engine_registry = {}
    image = Image.new("RGB", (8, 8), "black")
    policy = Policy(max_size_kb=1)
    with pytest.raises(OptimizationError):
        optimize_image(image, policy, optimizer=optimizer)


def test_optimize_from_analysis_passthrough_when_all_candidates_larger() -> None:
    optimizer = Optimizer(engines=[NoOpEngine()])
    image = Image.new("RGB", (512, 512), "black")
    analysis = optimizer.analyzer.analyze(image)
    policy = Policy(max_size_kb=1, min_ssim=0.99)
    result = optimizer.optimize_from_analysis(image, analysis, policy)
    assert "already_optimal" in result.report.reasons
