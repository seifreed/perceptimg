import sys
from pathlib import Path


def test_load_policy_json_extension(tmp_path: Path) -> None:
    """Test loading policy from JSON file."""
    from perceptimg.cli import _load_policy

    path = tmp_path / "policy.json"
    path.write_text('{"max_size_kb": 5}', encoding="utf-8")
    policy = _load_policy(path)
    assert policy.max_size_kb == 5


def test_cli_main_module_requires_input(tmp_path: Path) -> None:
    """Test CLI main function with valid input."""
    from PIL import Image

    from perceptimg.cli import main

    input_path = tmp_path / "dummy.png"
    policy_path = tmp_path / "policy.json"

    Image.new("RGB", (8, 8), "white").save(input_path)
    policy_path.write_text('{"max_size_kb": 5}', encoding="utf-8")

    old_argv = sys.argv
    sys.argv = [
        "perceptimg",
        str(input_path),
        "--policy",
        str(policy_path),
        "--out",
        str(tmp_path / "out.png"),
    ]

    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
