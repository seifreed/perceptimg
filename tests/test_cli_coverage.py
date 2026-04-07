import json
import sys
from pathlib import Path

import pytest

from perceptimg.cli import (
    _batch_report_data,
    _batch_summary_text,
    _collect_batch_inputs,
    _parse_args,
    _policy_from_flags,
    _write_batch_report,
    main,
)
from perceptimg.core.batch.config import BatchResult


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
    report = json.loads(captured.out)
    suffix = ".jpg" if report["chosen_format"] == "jpeg" else f'.{report["chosen_format"]}'
    if report["chosen_format"] == "tiff":
        suffix = ".tif"
    out_path = image_path.with_stem(f"{image_path.stem}_optimized").with_suffix(suffix)
    assert out_path.exists()
    assert "chosen_format" in captured.out


def test_cli_batch_defaults_to_continue_on_error() -> None:
    old = sys.argv
    sys.argv = ["perceptimg", "--batch", "inputdir"]
    try:
        args = _parse_args()
    finally:
        sys.argv = old
    assert args.continue_on_error is True


def test_cli_stop_on_error_disables_continue_on_error() -> None:
    old = sys.argv
    sys.argv = ["perceptimg", "--batch", "inputdir", "--stop-on-error"]
    try:
        args = _parse_args()
    finally:
        sys.argv = old
    assert args.continue_on_error is False


def test_cli_batch_accepts_multiple_inputs(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"")
    second.write_bytes(b"")

    old = sys.argv
    sys.argv = ["perceptimg", "--batch", str(first), str(second)]
    try:
        args = _parse_args()
    finally:
        sys.argv = old

    assert args.input == [str(first), str(second)]


def test_collect_batch_inputs_returns_multiple_files(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"")
    second.write_bytes(b"")
    args = type("Args", (), {"input": [str(first), str(second)], "input_dir": None})()

    paths = _collect_batch_inputs(args)

    assert paths == [first, second]


def test_collect_batch_inputs_rejects_missing_input_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_dir = tmp_path / "missing"
    args = type("Args", (), {"input": None, "input_dir": str(missing_dir)})()

    with pytest.raises(SystemExit) as exc:
        _collect_batch_inputs(args)

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert f"Batch input directory not found: {missing_dir}" in captured.err


def test_collect_batch_inputs_rejects_non_directory_input_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_dir = tmp_path / "file.png"
    not_a_dir.write_bytes(b"")
    args = type("Args", (), {"input": None, "input_dir": str(not_a_dir)})()

    with pytest.raises(SystemExit) as exc:
        _collect_batch_inputs(args)

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert f"--input-dir must be a directory: {not_a_dir}" in captured.err


def test_collect_batch_inputs_rejects_empty_input_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    args = type("Args", (), {"input": None, "input_dir": str(empty_dir)})()

    with pytest.raises(SystemExit) as exc:
        _collect_batch_inputs(args)

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "No image files found in directory:" in captured.err


def test_collect_batch_inputs_ignores_directories_with_image_extensions(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    valid_image = input_dir / "real.png"
    valid_image.write_bytes(b"")
    (input_dir / "fake.png").mkdir()
    args = type("Args", (), {"input": None, "input_dir": str(input_dir)})()

    paths = _collect_batch_inputs(args)

    assert paths == [valid_image]


def test_collect_batch_inputs_rejects_input_dir_with_only_image_named_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "fake.png").mkdir()
    args = type("Args", (), {"input": None, "input_dir": str(input_dir)})()

    with pytest.raises(SystemExit) as exc:
        _collect_batch_inputs(args)

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "No image files found in directory:" in captured.err


def test_cli_main_auto_batches_multiple_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from PIL import Image

    image_one = tmp_path / "img1.png"
    image_two = tmp_path / "img2.png"
    output_dir = tmp_path / "out"
    Image.new("RGB", (8, 8), "blue").save(image_one)
    Image.new("RGB", (8, 8), "green").save(image_two)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        str(image_one),
        str(image_two),
        "--output-dir",
        str(output_dir),
        "--report",
        str(tmp_path / "report.json"),
    ]
    try:
        main()
    finally:
        sys.argv = old

    captured = capsys.readouterr()
    assert not captured.out
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["successful"] == 2
    assert len(list(output_dir.iterdir())) == 2
    assert all("output_path" in row for row in report["results"])
    assert {Path(row["output_path"]).name for row in report["results"]} == {
        path.name for path in output_dir.iterdir()
    }


def test_cli_batch_same_basename_writes_unique_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image

    first_dir = tmp_path / "dir1"
    second_dir = tmp_path / "dir2"
    first_dir.mkdir()
    second_dir.mkdir()
    image_one = first_dir / "same.png"
    image_two = second_dir / "same.png"
    output_dir = tmp_path / "out"
    Image.new("RGB", (8, 8), "blue").save(image_one)
    Image.new("RGB", (8, 8), "green").save(image_two)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_one),
        str(image_two),
        "--output-dir",
        str(output_dir),
        "--formats",
        "gif",
    ]
    try:
        main()
    finally:
        sys.argv = old

    capsys.readouterr()
    output_names = sorted(path.name for path in output_dir.iterdir())
    assert output_names == ["same_optimized.gif", "same_optimized_2.gif"]


def test_cli_batch_output_pattern_collisions_are_deduplicated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image

    image_one = tmp_path / "img1.png"
    image_two = tmp_path / "img2.png"
    output_dir = tmp_path / "out"
    Image.new("RGB", (8, 8), "blue").save(image_one)
    Image.new("RGB", (8, 8), "green").save(image_two)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_one),
        str(image_two),
        "--output-dir",
        str(output_dir),
        "--formats",
        "gif",
        "--output-pattern",
        "optimized.{ext}",
    ]
    try:
        main()
    finally:
        sys.argv = old

    capsys.readouterr()
    output_names = sorted(path.name for path in output_dir.iterdir())
    assert output_names == ["optimized.gif", "optimized_2.gif"]


def test_cli_batch_same_basename_uses_input_order_for_output_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import perceptimg.cli as cli_module
    from PIL import Image
    from perceptimg.core.batch.config import BatchResult

    class _DummyOptimizationResult:
        def __init__(self, marker: bytes) -> None:
            self.image_bytes = marker
            self.report = SimpleNamespace(
                chosen_format="gif",
                size_before_kb=1.0,
                size_after_kb=0.5,
                ssim=0.99,
                psnr=40.0,
            )

    first_dir = tmp_path / "dir1"
    second_dir = tmp_path / "dir2"
    first_dir.mkdir()
    second_dir.mkdir()
    image_one = first_dir / "same.png"
    image_two = second_dir / "same.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.json"
    Image.new("RGB", (8, 8), "blue").save(image_one)
    Image.new("RGB", (8, 8), "green").save(image_two)

    def _out_of_order_batch(*args: object, **kwargs: object) -> BatchResult:
        return BatchResult(
            successful=[
                (image_two, _DummyOptimizationResult(b"B")),
                (image_one, _DummyOptimizationResult(b"A")),
            ],
            failed=[],
        )

    monkeypatch.setattr(cli_module, "optimize_batch", _out_of_order_batch)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_one),
        str(image_two),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
    ]
    try:
        main()
    finally:
        sys.argv = old

    capsys.readouterr()
    assert (output_dir / "same_optimized.gif").read_bytes() == b"A"
    assert (output_dir / "same_optimized_2.gif").read_bytes() == b"B"

    report = json.loads(report_path.read_text())
    assert report["results"] == [
        {
            "input": str(image_one),
            "output_path": str(output_dir / "same_optimized.gif"),
            "output_format": "gif",
            "size_before_kb": 1.0,
            "size_after_kb": 0.5,
            "ssim": 0.99,
            "psnr": 40.0,
        },
        {
            "input": str(image_two),
            "output_path": str(output_dir / "same_optimized_2.gif"),
            "output_format": "gif",
            "size_before_kb": 1.0,
            "size_after_kb": 0.5,
            "ssim": 0.99,
            "psnr": 40.0,
        },
    ]


def test_cli_batch_exact_duplicate_paths_use_input_occurrence_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import perceptimg.cli as cli_module
    from PIL import Image
    from perceptimg.core.batch.config import BatchResult

    class _DummyOptimizationResult:
        def __init__(self, marker: bytes) -> None:
            self.image_bytes = marker
            self.report = SimpleNamespace(
                chosen_format="webp",
                size_before_kb=1.0,
                size_after_kb=0.5,
                ssim=0.99,
                psnr=40.0,
            )

    image_path = tmp_path / "same.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.json"
    Image.new("RGB", (8, 8), "blue").save(image_path)

    def _out_of_order_batch(*args: object, **kwargs: object) -> BatchResult:
        return BatchResult(
            successful=[
                (image_path, _DummyOptimizationResult(b"SECOND")),
                (image_path, _DummyOptimizationResult(b"FIRST")),
            ],
            failed=[],
            successful_input_indices=[1, 0],
        )

    monkeypatch.setattr(cli_module, "optimize_batch", _out_of_order_batch)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_path),
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
    ]
    try:
        main()
    finally:
        sys.argv = old

    capsys.readouterr()
    assert (output_dir / "same_optimized.webp").read_bytes() == b"FIRST"
    assert (output_dir / "same_optimized_2.webp").read_bytes() == b"SECOND"

    report = json.loads(report_path.read_text())
    assert report["results"] == [
        {
            "input": str(image_path),
            "output_path": str(output_dir / "same_optimized.webp"),
            "output_format": "webp",
            "size_before_kb": 1.0,
            "size_after_kb": 0.5,
            "ssim": 0.99,
            "psnr": 40.0,
        },
        {
            "input": str(image_path),
            "output_path": str(output_dir / "same_optimized_2.webp"),
            "output_format": "webp",
            "size_before_kb": 1.0,
            "size_after_kb": 0.5,
            "ssim": 0.99,
            "psnr": 40.0,
        },
    ]


def test_cli_batch_report_format_csv_writes_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image

    image_path = tmp_path / "img1.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.csv"
    Image.new("RGB", (8, 8), "blue").save(image_path)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
        "--report-format",
        "csv",
    ]
    try:
        main()
    finally:
        sys.argv = old

    capsys.readouterr()
    report_text = report_path.read_text()
    assert report_text.startswith("input,status,output_path,output_format,")
    assert str(image_path) in report_text
    assert "successful" in report_text
    written_output = next(output_dir.iterdir())
    assert str(written_output) in report_text
    with pytest.raises(json.JSONDecodeError):
        json.loads(report_text)


def test_cli_batch_report_creates_missing_parent_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image

    image_path = tmp_path / "img1.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "nested" / "reports" / "report.json"
    Image.new("RGB", (8, 8), "blue").save(image_path)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
    ]
    try:
        main()
    finally:
        sys.argv = old

    capsys.readouterr()
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["successful"] == 1
    assert len(list(output_dir.iterdir())) == 1


def test_cli_batch_report_format_summary_writes_summary_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image

    image_path = tmp_path / "img1.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.txt"
    Image.new("RGB", (8, 8), "blue").save(image_path)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
        "--report-format",
        "summary",
    ]
    try:
        main()
    finally:
        sys.argv = old

    capsys.readouterr()
    report_text = report_path.read_text()
    assert "Processed 1 images:" in report_text
    assert "Successful: 1" in report_text
    with pytest.raises(json.JSONDecodeError):
        json.loads(report_text)


def test_cli_batch_report_data_includes_skipped_items() -> None:
    report = _batch_report_data(
        BatchResult(successful=[], failed=[], skipped=[Path("a.png")])
    )

    assert report["total"] == 1
    assert report["successful"] == 0
    assert report["failed"] == 0
    assert report["skipped"] == 1
    assert report["skipped_items"] == [{"input": "a.png"}]


def test_cli_batch_report_format_csv_includes_skipped_rows(tmp_path: Path) -> None:
    report_path = tmp_path / "report.csv"

    _write_batch_report(
        report_path,
        BatchResult(successful=[], failed=[], skipped=[Path("a.png")]),
        "csv",
    )

    report_text = report_path.read_text()
    assert "input,status,output_path,output_format," in report_text
    assert "a.png,skipped," in report_text


def test_cli_batch_summary_text_includes_skipped() -> None:
    summary = _batch_summary_text(
        BatchResult(successful=[], failed=[], skipped=[Path("a.png")])
    )

    assert "Processed 1 images:" in summary
    assert "Successful: 0" in summary
    assert "Skipped: 1" in summary


def test_cli_batch_summary_console_matches_report_when_skipped_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    import perceptimg.cli as cli_module

    image_path = tmp_path / "img1.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.txt"
    Image.new("RGB", (8, 8), "blue").save(image_path)

    skipped_result = BatchResult(successful=[], failed=[], skipped=[image_path])
    monkeypatch.setattr(cli_module, "optimize_batch", lambda *args, **kwargs: skipped_result)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
        "--report-format",
        "summary",
    ]
    try:
        main()
    finally:
        sys.argv = old

    captured = capsys.readouterr()
    report_text = report_path.read_text()
    assert report_text == _batch_summary_text(skipped_result)
    assert report_text in captured.err
    assert "Skipped: 1" in captured.err


def test_cli_batch_report_failure_preserves_written_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    import perceptimg.cli as cli_module

    image_path = tmp_path / "img1.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.json"
    Image.new("RGB", (8, 8), "blue").save(image_path)

    def _fail_report(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cli_module, "_write_batch_report", _fail_report)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "Failed to write batch report: disk full" in captured.err
    assert len(list(output_dir.iterdir())) == 1
    assert not report_path.exists()


def test_cli_batch_output_write_failure_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    import perceptimg.cli as cli_module

    image_one = tmp_path / "img1.png"
    image_two = tmp_path / "img2.png"
    output_dir = tmp_path / "out"
    report_path = tmp_path / "report.json"
    Image.new("RGB", (8, 8), "blue").save(image_one)
    Image.new("RGB", (8, 8), "green").save(image_two)

    real_write_output = cli_module._write_output
    calls = {"count": 0}

    def _fail_second_output(result: object, output_path: Path) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("disk full")
        real_write_output(result, output_path)

    monkeypatch.setattr(cli_module, "_write_output", _fail_second_output)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(image_one),
        str(image_two),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "Failed to write batch output: disk full" in captured.err
    assert len(list(output_dir.iterdir())) == 1
    assert not report_path.exists()


def test_cli_rejects_out_in_batch_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from PIL import Image

    image_one = tmp_path / "img1.png"
    image_two = tmp_path / "img2.png"
    report_path = tmp_path / "report.json"
    out_path = tmp_path / "single.webp"
    Image.new("RGB", (8, 8), "blue").save(image_one)
    Image.new("RGB", (8, 8), "green").save(image_two)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        str(image_one),
        str(image_two),
        "--out",
        str(out_path),
        "--report",
        str(report_path),
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "--out is only supported for single image mode" in captured.err
    assert not out_path.exists()
    assert not report_path.exists()


def test_cli_rejects_input_dir_with_positional_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image

    input_dir = tmp_path / "images"
    input_dir.mkdir()
    positional = tmp_path / "extra.png"
    report_path = tmp_path / "report.json"
    Image.new("RGB", (8, 8), "red").save(input_dir / "from_dir.png")
    Image.new("RGB", (8, 8), "blue").save(positional)

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        str(positional),
        "--input-dir",
        str(input_dir),
        "--report",
        str(report_path),
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "Use either --input-dir or positional input paths, not both" in captured.err
    assert not report_path.exists()


def test_cli_main_rejects_empty_input_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()

    old = sys.argv
    sys.argv = [
        "perceptimg",
        "--batch",
        "--input-dir",
        str(input_dir),
        "--report-format",
        "summary",
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        sys.argv = old

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "No image files found in directory:" in captured.err
    assert "Processing 0 images" not in captured.err
