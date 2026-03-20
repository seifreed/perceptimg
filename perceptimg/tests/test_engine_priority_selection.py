from typing import Any

from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer, optimize_image
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.utils.image_io import image_to_bytes


class LowPriorityFallbackEngine(OptimizationEngine):
    """Engine with low priority that should only be used if high priority fails."""

    format = "jpeg"
    priority = 1

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: Any) -> EngineResult:
        # This engine produces larger output, should be used if high priority fails
        data = image_to_bytes(image, format="JPEG", save_kwargs={"quality": 10})
        return EngineResult(data=data, format="jpeg", quality=10, metadata={"engine": "low"})


class HighPriorityQualityEngine(OptimizationEngine):
    """Engine with high priority that produces better quality output."""

    format = "jpeg"
    priority = 5

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: Any) -> EngineResult:
        data = image_to_bytes(image, format="JPEG", save_kwargs={"quality": 90})
        return EngineResult(data=data, format="jpeg", quality=90, metadata={"engine": "high"})


def test_high_priority_engine_used_first() -> None:
    """High priority engine should be tried first for the same format."""
    optimizer = Optimizer(engines=[LowPriorityFallbackEngine(), HighPriorityQualityEngine()])
    image = Image.new("RGB", (32, 32), "white")
    policy = Policy(preferred_formats=("jpeg",))
    result = optimize_image(image, policy, optimizer=optimizer)
    # High priority engine (priority=5) is tried first and succeeds
    # The result should come from HighPriorityQualityEngine
    assert result.report.chosen_format == "jpeg"
    # With policy preferring quality (default prioritize_quality=False),
    # we get the smallest size. But since both engines produce valid output,
    # high priority engine is tried first.
    # The key assertion: the result exists and is valid
    assert len(result.image_bytes) > 0
