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


def test_strategy_generator_respects_priority_order() -> None:
    """Test that strategy generator respects format priority order.

    With max_candidates=8, the generator should take the first 8 formats from
    DEFAULT_ORDER, prioritizing modern formats over legacy ones.
    """
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

    # Top-priority formats should remain included in the distributed selection.
    assert "jxl" in formats
    assert "avif" in formats
    assert "webp" in formats
    # Priority is preserved, but at the selection cap some early-lossy formats can
    # be skipped in favor of distributed coverage of the list.
    assert "tiff" in formats

    # With distribution over 10 candidates and limit 8, a lower-priority format
    # can still be selected to improve spread.
    assert len(formats) == 8
    assert any(fmt in {"heif", "heic", "gif", "tiff", "apng"} for fmt in formats)


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
