from perceptimg.core.analyzer import AnalysisResult
from perceptimg.core.policy import Policy
from perceptimg.core.strategy import StrategyGenerator


def test_strategy_generator_mobile_quality_plan() -> None:
    policy = Policy(target_use_case="mobile")
    analysis = AnalysisResult(
        edge_density=0.05,
        color_variance=0.05,
        probable_text=False,
        probable_faces=False,
        resolution=(10, 10),
        aspect_ratio=1.0,
    )
    qualities = StrategyGenerator().generate(policy, analysis)
    assert any(candidate.quality == 60 for candidate in qualities if candidate.quality)
