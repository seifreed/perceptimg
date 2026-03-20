import sys
from pathlib import Path

from perceptimg.cli import main


def test_cli_main_runs_default_output(tmp_path: Path, capsys: object) -> None:
    from PIL import Image

    image_path = tmp_path / "img_cli.png"
    Image.new("RGB", (10, 10), "green").save(image_path)
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
