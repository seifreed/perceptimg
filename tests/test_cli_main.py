import json
import sys
from pathlib import Path

from PIL import Image

from perceptimg.cli import _load_policy, _policy_from_flags, main


def test_policy_from_flags_default_lossless() -> None:
    args = type(
        "Args",
        (),
        {
            "policy": None,
            "max_size_kb": 10,
            "min_ssim": None,
            "preserve_text": False,
            "preserve_faces": False,
            "allow_lossy": None,
            "target_use_case": None,
            "formats": None,
            "lossless": True,
        },
    )()
    policy = _policy_from_flags(args)
    assert policy is not None and policy.allow_lossy is False


def test_load_policy_with_txt_extension(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text(json.dumps({"max_size_kb": 10}), encoding="utf-8")
    policy = _load_policy(path)
    assert policy.max_size_kb == 10


def test_cli_main_executes(tmp_path: Path, capsys: object) -> None:
    image_path = tmp_path / "in.png"
    Image.new("RGB", (16, 16), "red").save(image_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"max_size_kb": 500}), encoding="utf-8")
    out_path = tmp_path / "out.webp"

    argv = ["perceptimg", str(image_path), "--policy", str(policy_path), "--out", str(out_path)]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    assert out_path.exists()
    assert "chosen_format" in captured.out


def test_cli_main_default_policy_and_output(tmp_path: Path, capsys: object) -> None:
    image_path = tmp_path / "in2.png"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    argv = ["perceptimg", str(image_path)]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    default_out = image_path.with_stem(f"{image_path.stem}_optimized")
    suffix = ".jpg" if report["chosen_format"] == "jpeg" else f'.{report["chosen_format"]}'
    if report["chosen_format"] == "tiff":
        suffix = ".tif"
    default_out = default_out.with_suffix(suffix)
    assert default_out.exists()
    assert "chosen_format" in captured.out
