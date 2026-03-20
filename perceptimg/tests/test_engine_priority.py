from typing import Any

from PIL import Image

from perceptimg.core.optimizer import Optimizer, register_engine
from perceptimg.engines.base import EngineResult, OptimizationEngine


class FirstEngine(OptimizationEngine):
    format = "jpeg"
    priority = 10

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: Any) -> EngineResult:
        data = b"first"
        return EngineResult(data=data, format="jpeg", quality=90, metadata={})


class SecondEngine(OptimizationEngine):
    format = "jpeg"
    priority = 5

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: Any) -> EngineResult:
        data = b"second"
        return EngineResult(data=data, format="jpeg", quality=80, metadata={})


def test_engine_registration_order_respected() -> None:
    optimizer = Optimizer(engines=[FirstEngine()])
    register_engine(optimizer, SecondEngine())
    assert optimizer.engine_registry["jpeg"][0].__class__ is FirstEngine
    assert optimizer.engine_registry["jpeg"][1].__class__ is SecondEngine
