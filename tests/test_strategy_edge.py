import pytest

from perceptimg.core.analyzer import AnalysisResult
from perceptimg.core.policy import Policy
from perceptimg.core.strategy import StrategyGenerator
from perceptimg.exceptions import StrategyError


def test_strategy_generator_raises_when_no_formats() -> None:
    from perceptimg.core import strategy as strategy_module

    policy = Policy(max_size_kb=10, preferred_formats=())
    analysis = AnalysisResult(
        edge_density=0.1,
        color_variance=0.1,
        probable_text=False,
        probable_faces=False,
        resolution=(10, 10),
        aspect_ratio=1.0,
    )
    generator = StrategyGenerator()
    original_order = strategy_module.DEFAULT_ORDER
    strategy_module.DEFAULT_ORDER = ()
    try:
        with pytest.raises(StrategyError):
            generator.generate(policy, analysis)
    finally:
        strategy_module.DEFAULT_ORDER = original_order


def test_distributed_indices_spreads_selection() -> None:
    indices = StrategyGenerator._distributed_indices(10, 8)
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == 9
    assert len(indices) == 8
    assert len(set(indices)) == len(indices)
    assert any(
        next_index - prev_index > 1
        for prev_index, next_index in zip(indices, indices[1:], strict=True)
    )
