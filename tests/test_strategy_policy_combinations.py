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
    qualities = StrategyGenerator(max_candidates=32).generate(policy, analysis)
    assert any(candidate.quality == 60 for candidate in qualities if candidate.quality)


def test_strategy_generator_spreads_candidates_across_formats() -> None:
    policy = Policy()
    analysis = AnalysisResult(
        edge_density=0.05,
        color_variance=0.05,
        probable_text=False,
        probable_faces=False,
        resolution=(10, 10),
        aspect_ratio=1.0,
    )

    candidates = StrategyGenerator(max_candidates=8).generate(policy, analysis)
    formats = [candidate.format for candidate in candidates]

    assert "jpeg" in formats
    assert "png" in formats
    assert "gif" in formats
    assert "apng" in formats


def test_strategy_generator_respects_max_candidates_hard_limit() -> None:
    policy = Policy()
    analysis = AnalysisResult(
        edge_density=0.05,
        color_variance=0.05,
        probable_text=False,
        probable_faces=False,
        resolution=(10, 10),
        aspect_ratio=1.0,
    )

    for limit in (1, 2, 8):
        candidates = StrategyGenerator(max_candidates=limit).generate(policy, analysis)
        assert len(candidates) == limit


def test_strategy_generator_prefers_available_formats_when_provided() -> None:
    policy = Policy()
    analysis = AnalysisResult(
        edge_density=0.05,
        color_variance=0.05,
        probable_text=False,
        probable_faces=False,
        resolution=(10, 10),
        aspect_ratio=1.0,
    )

    candidates = StrategyGenerator(max_candidates=2).generate(
        policy,
        analysis,
        available_formats={"jpeg", "png"},
    )

    assert [candidate.format for candidate in candidates] == ["jpeg", "png"]
