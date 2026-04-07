"""Tests for Policy edge cases."""

from __future__ import annotations

import pytest

from perceptimg.core.policy import Policy


def test_policy_merge_with_empty_preferred_formats() -> None:
    """Policy.merge should respect explicitly passed empty tuple from Policy."""
    base = Policy(max_size_kb=100, preferred_formats=("webp", "jpeg"))
    other = Policy(max_size_kb=50, preferred_formats=())
    merged = base.merge(other)
    assert merged.max_size_kb == 50
    # Explicitly passed empty tuple via Policy should override base
    assert merged.preferred_formats == ()


def test_policy_merge_dict_with_empty_tuple_overrides_base() -> None:
    """Policy.merge with a plain dict should respect explicitly empty containers."""
    base = Policy(max_size_kb=100, preferred_formats=("webp", "jpeg"))
    merged = base.merge({"max_size_kb": 50, "preferred_formats": ()})
    assert merged.max_size_kb == 50
    # Empty tuple via dict should override base (user explicitly passed empty)
    assert merged.preferred_formats == ()


def test_policy_merge_preserves_base_values() -> None:
    """Policy.merge with a dict should only override keys present in the dict."""
    base = Policy(max_size_kb=100, min_ssim=0.95, preserve_text=True)
    merged = base.merge({"max_size_kb": 50})
    assert merged.max_size_kb == 50
    assert merged.min_ssim == 0.95
    assert merged.preserve_text is True


def test_policy_merge_empty_dict_keeps_base() -> None:
    """Policy.merge with an empty dict should not change anything."""
    base = Policy(preserve_text=True, preserve_faces=True)
    merged = base.merge({})
    assert merged.preserve_text is True
    assert merged.preserve_faces is True


def test_policy_merge_policy_overrides_explicitly_set_fields() -> None:
    """Policy.merge with a Policy should override when a field was explicitly passed,
    even if its value matches the default."""
    base = Policy(preserve_text=True, allow_lossy=False)
    other = Policy(allow_lossy=True)  # explicitly passed, should override
    merged = base.merge(other)
    assert merged.allow_lossy is True
    # preserve_text was NOT passed to other, so base's True is preserved
    assert merged.preserve_text is True


def test_policy_merge_bare_policy_is_noop() -> None:
    """Policy.merge with a bare Policy() should not override anything."""
    base = Policy(preserve_text=True, allow_lossy=False, max_size_kb=200)
    merged = base.merge(Policy())
    assert merged.preserve_text is True
    assert merged.allow_lossy is False
    assert merged.max_size_kb == 200


def test_policy_merge_can_reset_to_default() -> None:
    """Policy.merge should allow resetting a field to its default value."""
    base = Policy(preserve_text=True)
    other = Policy(preserve_text=False)  # explicitly set to default value
    merged = base.merge(other)
    assert merged.preserve_text is False


def test_policy_merge_policy_only_overrides_explicit_fields() -> None:
    """Policy.merge with a Policy should only override fields that were explicitly passed."""
    base = Policy(preserve_text=True, allow_lossy=False, max_size_kb=200)
    other = Policy(max_size_kb=100)
    merged = base.merge(other)
    assert merged.max_size_kb == 100
    # preserve_text was not passed to other, so base's True is preserved
    assert merged.preserve_text is True
    # allow_lossy was not passed to other, so base's False is preserved
    assert merged.allow_lossy is False


def test_policy_merge_complete_override() -> None:
    """Policy.merge should completely override when other has all values."""
    base = Policy()
    other = Policy(
        max_size_kb=100,
        min_ssim=0.98,
        preserve_text=True,
        preserve_faces=True,
        allow_lossy=False,
        preferred_formats=("avif", "webp"),
        target_use_case="mobile",
    )
    merged = base.merge(other)
    assert merged.max_size_kb == 100
    assert merged.min_ssim == 0.98
    assert merged.preserve_text is True
    assert merged.preserve_faces is True
    assert merged.allow_lossy is False
    assert merged.preferred_formats == ("avif", "webp")
    assert merged.target_use_case == "mobile"


def test_policy_validation_rejects_zero_max_size() -> None:
    """Policy should reject max_size_kb=0."""
    with pytest.raises(ValueError, match="max_size_kb must be positive"):
        Policy(max_size_kb=0)


def test_policy_validation_rejects_negative_max_size() -> None:
    """Policy should reject negative max_size_kb."""
    with pytest.raises(ValueError, match="max_size_kb must be positive"):
        Policy(max_size_kb=-10)


def test_policy_validation_rejects_invalid_ssim() -> None:
    """Policy should reject SSIM outside valid range."""
    with pytest.raises(ValueError, match="min_ssim must be within"):
        Policy(min_ssim=1.5)

    with pytest.raises(ValueError, match="min_ssim must be within"):
        Policy(min_ssim=-0.1)


def test_policy_validate_for_size_aggressive_compression_warning() -> None:
    """validate_for_size should warn about aggressive compression with high SSIM."""
    policy = Policy(max_size_kb=1, min_ssim=0.99)
    warnings = policy.validate_for_size(100.0)
    assert len(warnings) == 1
    assert "impossible" in warnings[0].lower()


def test_policy_validate_for_size_lossless_warning() -> None:
    """validate_for_size should warn about lossless with small max_size."""
    policy = Policy(max_size_kb=10, allow_lossy=False)
    warnings = policy.validate_for_size(100.0)
    assert len(warnings) == 1
    assert "lossless" in warnings[0].lower()


def test_policy_validate_for_size_reasonable_policy_no_warnings() -> None:
    """validate_for_size should not warn for reasonable policies."""
    policy = Policy(max_size_kb=100, min_ssim=0.95)
    warnings = policy.validate_for_size(200.0)
    assert len(warnings) == 0


def test_policy_validation_rejects_nan_max_size() -> None:
    """Policy should reject NaN for max_size_kb."""
    import math

    with pytest.raises(ValueError, match="finite number"):
        Policy.from_dict({"max_size_kb": math.nan})


def test_policy_validation_rejects_inf_max_size() -> None:
    """Policy should reject Inf for max_size_kb."""
    import math

    with pytest.raises(ValueError, match="finite number"):
        Policy.from_dict({"max_size_kb": math.inf})


def test_policy_validation_rejects_nan_ssim() -> None:
    """Policy should reject NaN for min_ssim."""
    import math

    with pytest.raises(ValueError, match="finite number"):
        Policy.from_dict({"min_ssim": math.nan})


def test_policy_validation_rejects_inf_ssim() -> None:
    """Policy should reject Inf for min_ssim."""
    import math

    with pytest.raises(ValueError, match="finite number"):
        Policy.from_dict({"min_ssim": math.inf})
