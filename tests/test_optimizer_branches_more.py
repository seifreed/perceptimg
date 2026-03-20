import logging

import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer, register_engine
from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.exceptions import OptimizationError


class FailingEngine(OptimizationEngine):
    format = "jpeg"
    priority = 5

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        raise RuntimeError("fail")


class PassingEngine(OptimizationEngine):
    format = "jpeg"
    priority = 1

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        from perceptimg.utils.image_io import image_to_bytes

        data = image_to_bytes(image, format="PNG")
        return EngineResult(
            data=data,
            format="jpeg",
            quality=strategy.quality,
            metadata={},
        )


def test_engine_failure_logs_and_fallback(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    optimizer = Optimizer(engines=[FailingEngine(), PassingEngine()])
    image = Image.new("RGB", (8, 8), "white")
    policy = Policy(max_size_kb=1000, preferred_formats=("jpeg",))
    result = optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)
    assert result.image_bytes
    assert any(
        "Engine" in rec.message and ("failed" in rec.message or "crashed" in rec.message)
        for rec in caplog.records
    )


def test_register_engine_logging(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    optimizer = Optimizer(engines=[PassingEngine()])
    register_engine(optimizer, PassingEngine())
    assert any("Multiple engines registered" in rec.message for rec in caplog.records)


def test_optimize_bytes_wrapper_uses_helper() -> None:
    optimizer = Optimizer(engines=[PassingEngine()])
    from perceptimg.utils.image_io import image_to_bytes

    data = image_to_bytes(Image.new("RGB", (8, 8), "white"), format="PNG")
    policy = Policy(max_size_kb=1000, preferred_formats=("jpeg",))
    from perceptimg.core.optimizer import optimize_bytes

    result = optimize_bytes(data, policy, optimizer=optimizer)
    assert result.image_bytes


def test_optimize_raises_when_no_candidates() -> None:
    optimizer = Optimizer(engines=[])
    image = Image.new("RGB", (8, 8), "white")
    policy = Policy(max_size_kb=1, preferred_formats=("jpeg",))
    # Force strategy generator to return no formats by clearing registry
    optimizer.engine_registry.clear()
    with pytest.raises(OptimizationError):
        optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)
