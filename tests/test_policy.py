import json

import pytest

from perceptimg.core.policy import Policy


def test_policy_validation_limits():
    with pytest.raises(ValueError):
        Policy(max_size_kb=0)
    with pytest.raises(ValueError):
        Policy(min_ssim=1.5)
    with pytest.raises(ValueError):
        Policy(preferred_formats=("invalid",))


def test_policy_serialization_roundtrip():
    policy = Policy(
        max_size_kb=150,
        min_ssim=0.97,
        preserve_text=True,
        preserve_faces=False,
        allow_lossy=True,
        preferred_formats=("webp", "jpeg"),
        target_use_case="web",
    )
    encoded = policy.to_json()
    decoded = Policy.from_json(encoded)
    assert decoded == policy
    assert json.loads(encoded)["preferred_formats"] == ["webp", "jpeg"]


def test_policy_defaults_preferred_formats_fallback():
    policy = Policy(max_size_kb=100)
    # Should return allowed formats in default order when none provided
    order = policy.preferred_format_order(
        ("jxl", "avif", "webp", "heif", "heic", "jpeg", "png", "tiff", "gif", "apng")
    )
    assert order == (
        "jxl",
        "avif",
        "webp",
        "heif",
        "heic",
        "jpeg",
        "png",
        "tiff",
        "gif",
        "apng",
    )


def test_policy_override_and_merge():
    base = Policy(max_size_kb=100, allow_lossy=True)
    updated = base.override(min_ssim=0.95)
    assert updated.min_ssim == 0.95
    assert updated.max_size_kb == 100

    other = Policy(max_size_kb=50, preserve_text=True)
    merged = base.merge(other)
    assert merged.max_size_kb == 50
    assert merged.preserve_text is True
