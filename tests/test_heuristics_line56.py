from perceptimg.utils import heuristics


def test_heuristics_line56_extreme_aspect_inverse() -> None:
    cfg = heuristics.HeuristicConfig()
    assert heuristics.detect_probable_text(
        edge_density=0.2,
        color_variance=0.02,
        aspect_ratio=0.1,
        config=cfg,
    )
