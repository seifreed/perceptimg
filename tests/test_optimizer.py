from pathlib import Path

from PIL import Image

from perceptimg import Policy, optimize
from perceptimg.core.optimizer import Optimizer
from perceptimg.engines.avif_engine import AvifEngine
from perceptimg.engines.pillow_engine import PillowEngine
from perceptimg.engines.webp_engine import WebPEngine


def test_optimize_produces_report(tmp_path: Path):
    image_path = tmp_path / "input.png"
    image = Image.new("RGB", (48, 48), (120, 180, 200))
    image.save(image_path)

    policy = Policy(max_size_kb=50, min_ssim=0.8, preferred_formats=("jpeg", "png"))
    result = optimize(image_path, policy)

    assert result.image_bytes
    assert result.report.ssim >= 0.8
    assert result.report.size_after_kb <= 50
    assert "policy_satisfied" in result.report.reasons
    assert result.report.chosen_format in {"jpeg", "png", "webp", "avif"}


class UnavailableEngine(WebPEngine):
    @property
    def is_available(self) -> bool:
        return False


class UnavailableAvifEngine(AvifEngine):
    @property
    def is_available(self) -> bool:
        return False


def test_optimizer_handles_lossless_when_modern_formats_unavailable(tmp_path: Path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "red").save(image_path)
    policy = Policy(max_size_kb=80, allow_lossy=False)
    optimizer = Optimizer(engines=(UnavailableEngine(), UnavailableAvifEngine(), PillowEngine()))

    result = optimizer.optimize(image_path, policy)
    assert result.report.chosen_format in {"png", "gif", "tiff"}
    assert result.report.reasons


def test_report_to_dict_contains_expected_keys(tmp_path: Path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (24, 24), "blue").save(image_path)
    policy = Policy(max_size_kb=200)
    result = optimize(image_path, policy)
    report = result.report.to_dict()
    for key in (
        "chosen_format",
        "quality",
        "size_before_kb",
        "size_after_kb",
        "ssim",
        "psnr",
        "reasons",
    ):
        assert key in report
