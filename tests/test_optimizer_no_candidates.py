import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.optimizer import Optimizer
from perceptimg.core.strategy import StrategyCandidate, StrategyGenerator
from perceptimg.exceptions import OptimizationError


class EmptyStrategyGenerator(StrategyGenerator):
    """Strategy generator that always returns empty candidates."""

    def generate(self, policy: Policy, analysis: object) -> list[StrategyCandidate]:
        return []


def test_optimizer_raises_when_no_candidates_pass_policy() -> None:
    optimizer = Optimizer()
    optimizer.strategy_generator = EmptyStrategyGenerator()
    image = Image.new("RGB", (8, 8), "white")
    policy = Policy(max_size_kb=1, min_ssim=1.0, preferred_formats=("jpeg",))
    with pytest.raises(OptimizationError):
        optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)
