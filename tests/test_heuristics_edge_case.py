import numpy as np

from perceptimg.utils import heuristics


def test_heuristics_empty_array_returns_false_for_faces() -> None:
    assert heuristics.detect_probable_faces(np.array([])) is False


def test_heuristics_detect_probable_text_extreme_aspect_inverse() -> None:
    cfg = heuristics.HeuristicConfig()
    assert heuristics.detect_probable_text(0.2, 0.01, 0.01, cfg)
