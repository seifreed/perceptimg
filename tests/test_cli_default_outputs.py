import json
import sys
from pathlib import Path

from perceptimg.cli import main


def test_cli_default_policy_and_output_path(tmp_path: Path, capsys: object) -> None:
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    argv = ["perceptimg", str(image_path)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    suffix = ".jpg" if report["chosen_format"] == "jpeg" else f'.{report["chosen_format"]}'
    if report["chosen_format"] == "tiff":
        suffix = ".tif"
    out_path = image_path.with_stem(f"{image_path.stem}_optimized").with_suffix(suffix)
    assert out_path.exists()
    assert "chosen_format" in captured.out
