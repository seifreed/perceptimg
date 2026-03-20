import sys
from pathlib import Path

from perceptimg.cli import _policy_from_flags, main


def test_cli_formats_flag_sets_preferred_formats(tmp_path: Path, capsys: object) -> None:
    img = tmp_path / "img.png"
    img.write_bytes(b"not an image")  # will not be read because we exit before optimize
    args = _policy_from_flags(
        type(
            "Args",
            (),
            {
                "policy": None,
                "max_size_kb": 10,
                "min_ssim": 0.9,
                "preserve_text": False,
                "preserve_faces": False,
                "allow_lossy": None,
                "target_use_case": None,
                "formats": "webp,jpeg",
                "lossless": False,
            },
        )()
    )
    assert args is not None
    assert args.preferred_formats == ("webp", "jpeg")


def test_cli_main_default_output(tmp_path: Path, capsys: object) -> None:
    image_path = tmp_path / "input.png"
    # Simple valid image
    from PIL import Image

    Image.new("RGB", (8, 8), "blue").save(image_path)
    argv = ["perceptimg", str(image_path)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old
    captured = capsys.readouterr()
    out_path = image_path.with_stem(f"{image_path.stem}_optimized").with_suffix(".webp")
    assert out_path.exists()
    assert "chosen_format" in captured.out
