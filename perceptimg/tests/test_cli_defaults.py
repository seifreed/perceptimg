import argparse

from perceptimg.cli import _policy_from_flags


def test_policy_from_flags_defaults_to_web_when_no_inputs() -> None:
    args = argparse.Namespace(
        policy=None,
        max_size_kb=None,
        min_ssim=None,
        preserve_text=False,
        preserve_faces=False,
        allow_lossy=None,
        target_use_case=None,
        formats=None,
        lossless=False,
    )
    policy = _policy_from_flags(args)
    assert policy is not None
    assert policy.target_use_case == "web"
    assert policy.allow_lossy is True


def test_policy_from_flags_respects_lossless_flag() -> None:
    args = argparse.Namespace(
        policy=None,
        max_size_kb=None,
        min_ssim=None,
        preserve_text=False,
        preserve_faces=False,
        allow_lossy=None,
        target_use_case=None,
        formats=None,
        lossless=True,
    )
    policy = _policy_from_flags(args)
    assert policy is not None and policy.allow_lossy is False
