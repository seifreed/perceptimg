import logging

import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer
from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.utils.image_io import image_to_bytes


class NullEngine(OptimizationEngine):
    format = "jpeg"
    priority = 0

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        return EngineResult(
            data=image_to_bytes(image, format="PNG"),
            format="jpeg",
            quality=strategy.quality,
            metadata={},
        )


def test_evaluate_candidates_policy_rejects() -> None:
    optimizer = Optimizer(engines=[NullEngine()])
    image = Image.new("RGB", (8, 8), "white")
    strategy = StrategyCandidate(
        format="jpeg",
        quality=50,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    candidates = optimizer.evaluate_candidates_for_test(
        image,
        [strategy],
        Policy(allow_lossy=False),
    )
    assert candidates == []


def test_register_engine_logs_multiple(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    optimizer = Optimizer()

    class Dummy(OptimizationEngine):
        format = "jpeg"

        def can_handle(self, fmt: str) -> bool:
            return fmt == "jpeg"

        def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
            return EngineResult(data=b"x", format="jpeg", quality=80, metadata={})

    optimizer.engine_registry.setdefault("jpeg", []).append(Dummy())
    optimizer.engines.append(Dummy())
    optimizer._sort_registry()
    candidates = optimizer.engine_registry["jpeg"]
    assert candidates
