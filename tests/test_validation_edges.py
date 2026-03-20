import pytest

from perceptimg.utils import validation


def test_validation_errors() -> None:
    with pytest.raises(validation.ValidationError):
        validation.ensure_positive(-1, "x")
    with pytest.raises(validation.ValidationError):
        validation.ensure_between_0_1(2.0, "y")
