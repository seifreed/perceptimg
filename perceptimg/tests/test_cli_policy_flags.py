import argparse

from perceptimg.cli import _policy_from_flags


def test_policy_from_flags_builds_policy_without_file() -> None:
    args = argparse.Namespace(
        policy=None,
        max_size_kb=123,
        min_ssim=0.95,
        preserve_text=True,
        preserve_faces=False,
        allow_lossy=None,
        lossless=True,
        target_use_case="mobile",
        formats="webp,avif",
    )
    policy = _policy_from_flags(args)
    assert policy is not None
    assert policy.max_size_kb == 123
    assert policy.min_ssim == 0.95
    assert policy.preserve_text is True
    assert policy.allow_lossy is False  # lossless=True -> allow_lossy=False
    assert policy.preferred_formats == ("webp", "avif")
