from typing import Any

from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer, optimize_image
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.utils.image_io import image_to_bytes


class FailingEngine(OptimizationEngine):
    format = "webp"

    def can_handle(self, fmt: str) -> bool:
        return fmt == "webp"

    def optimize(self, image: Image.Image, strategy: Any) -> EngineResult:
        raise RuntimeError("fail")


class WorkingEngine(OptimizationEngine):
    format = "webp"

    def can_handle(self, fmt: str) -> bool:
        return fmt == "webp"

    def optimize(self, image: Image.Image, strategy: Any) -> EngineResult:
        # Return actual PNG bytes so the image can be decoded
        data = image_to_bytes(image, format="PNG")
        return EngineResult(data=data, format="webp", quality=80, metadata={})


def test_engine_fallback_when_first_fails() -> None:
    optimizer = Optimizer(engines=[FailingEngine(), WorkingEngine()])
    image = Image.new("RGB", (8, 8), "white")
    policy = Policy(max_size_kb=10)
    result = optimize_image(image, policy, optimizer=optimizer)
    assert len(result.image_bytes) > 0
