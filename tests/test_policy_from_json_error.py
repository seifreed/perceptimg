import pytest

from perceptimg.core.policy import Policy


def test_policy_from_json_requires_mapping() -> None:
    with pytest.raises(ValueError):
        Policy.from_json('["not", "a", "dict"]')
