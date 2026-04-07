import pytest

from perceptimg.utils import heuristics, logging_config, validation


def test_compute_aspect_ratio_zero_height() -> None:
    assert heuristics.compute_aspect_ratio(10, 0) == 10_000.0


def test_compute_aspect_ratio_zero_width() -> None:
    assert heuristics.compute_aspect_ratio(0, 10) == pytest.approx(1.0 / 10_000.0)


def test_compute_aspect_ratio_both_zero() -> None:
    assert heuristics.compute_aspect_ratio(0, 0) == 1.0


def test_detect_probable_faces_empty_array() -> None:
    import numpy as np

    assert heuristics.detect_probable_faces(np.array([])) is False


def test_image_io_load_error() -> None:
    from perceptimg.exceptions import ImageLoadError
    from perceptimg.utils.image_io import load_image

    with pytest.raises(ImageLoadError):
        load_image("nonexistent_file.png")


def test_logging_config_json_output() -> None:
    logging_config.configure_logging(json_output=True, merge=False)


def test_validation_paths() -> None:
    validation.ensure_positive(1, "x")
    validation.ensure_between_0_1(0.5, "y")
    validation.ensure_non_empty([1], "z")
    with pytest.raises(validation.ValidationError):
        validation.ensure_non_empty([], "z")
