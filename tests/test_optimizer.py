from pathlib import Path

from PIL import Image

import perceptimg.core.optimizer as optimizer_module
from perceptimg import Policy, optimize
from perceptimg.core.optimizer import Optimizer
from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.avif_engine import AvifEngine
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.engines.pillow_engine import PillowEngine
from perceptimg.engines.webp_engine import WebPEngine
from perceptimg.utils.image_io import image_to_bytes


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


class LateApngEngine(OptimizationEngine):
    format = "apng"

    def can_handle(self, fmt: str) -> bool:
        return fmt == "apng"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        data = image_to_bytes(image, format="PNG")
        return EngineResult(data=data, format="apng", quality=None, metadata={})


def test_thread_local_isolation_between_instances():
    """Two Optimizer instances should not share _last_engine_errors."""
    opt1 = Optimizer()
    opt2 = Optimizer()

    opt1._last_engine_errors.extend(["error from opt1"])
    assert opt2._last_engine_errors == []
    assert opt1._last_engine_errors == ["error from opt1"]


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


def test_optimizer_with_pillow_engine_only_uses_late_supported_formats() -> None:
    image = Image.new("RGB", (32, 32), "white")
    policy = Policy(max_size_kb=1000)
    optimizer = Optimizer(engines=(PillowEngine(),))

    result = optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)

    assert result.report.chosen_format in {"jpeg", "png", "tiff", "gif"}


def test_optimizer_with_pillow_engine_only_respects_low_max_candidates() -> None:
    image = Image.new("RGB", (32, 32), "white")
    policy = Policy(max_size_kb=1000)
    optimizer = Optimizer(engines=(PillowEngine(),))
    optimizer.strategy_generator.max_candidates = 2

    result = optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)

    assert result.report.chosen_format in {"jpeg", "png", "tiff", "gif"}


def test_optimizer_with_late_apng_engine_is_reachable() -> None:
    image = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
    policy = Policy(max_size_kb=1000)
    optimizer = Optimizer(engines=(LateApngEngine(),))

    result = optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)

    assert result.report.chosen_format == "apng"


def test_optimizer_with_late_apng_engine_is_reachable_with_single_candidate() -> None:
    image = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
    policy = Policy(max_size_kb=1000)
    optimizer = Optimizer(engines=(LateApngEngine(),))
    optimizer.strategy_generator.max_candidates = 1

    result = optimizer.optimize_from_analysis(image, optimizer.analyzer.analyze(image), policy)

    assert result.report.chosen_format == "apng"


def test_optimize_handles_small_images(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (4, 4), "purple").save(image_path)

    result = optimize(image_path, Policy(max_size_kb=50))

    assert result.image_bytes
    assert result.report.chosen_format


def test_optimizer_default_engine_provider_is_used_only_for_none_engines() -> None:
    def provider() -> tuple[OptimizationEngine]:
        return ()

    previous_provider = optimizer_module._default_engine_provider
    called = {"count": 0}

    def tracking_provider() -> tuple[OptimizationEngine]:
        called["count"] += 1
        return provider()

    optimizer_module.set_default_engine_provider(tracking_provider)
    try:
        Optimizer()
        assert called["count"] == 1

        Optimizer([])
        assert called["count"] == 1
    finally:
        optimizer_module.set_default_engine_provider(previous_provider)
