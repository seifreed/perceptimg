from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.metrics import MetricResult
from perceptimg.core.optimizer import Optimizer
from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.avif_engine import AvifEngine
from perceptimg.engines.base import EngineResult, OptimizationEngine
from perceptimg.engines.pillow_engine import PillowEngine
from perceptimg.engines.webp_engine import WebPEngine
from perceptimg.exceptions import OptimizationError
from perceptimg.utils import heuristics, validation


def test_satisfies_policy_thresholds() -> None:
    optimizer = Optimizer(engines=[])
    metrics_low_ssim = MetricResult(
        ssim=0.4,
        psnr=10.0,
        size_before_kb=1.0,
        size_after_kb=0.5,
        perceptual_score=0.1,
    )
    strategy = StrategyCandidate(
        format="jpeg",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    policy_ssim = Policy(min_ssim=0.9, allow_lossy=True)
    assert optimizer._satisfies_policy(metrics_low_ssim, policy_ssim, strategy) is False

    policy_lossless = Policy(allow_lossy=False)
    metrics_ok = MetricResult(
        ssim=0.99,
        psnr=10.0,
        size_before_kb=1.0,
        size_after_kb=0.5,
        perceptual_score=0.1,
    )
    assert optimizer._satisfies_policy(metrics_ok, policy_lossless, strategy) is False


class UnavailableEngine(OptimizationEngine):
    """Engine that's never available for testing error paths."""

    format = "jpeg"
    priority = 0

    @property
    def is_available(self) -> bool:
        return False

    def can_handle(self, fmt: str) -> bool:
        return fmt == "jpeg"

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        raise AssertionError("Should not be called")


def test_optimizer_optimize_no_candidates(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    optimizer = Optimizer(engines=[UnavailableEngine()])
    with pytest.raises(OptimizationError):
        optimizer.optimize(image_path, Policy())


class FailingAvifEngine(AvifEngine):
    """AVIF engine that's available but fails during encoding."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        raise OptimizationError("AVIF encoding failed: simulated failure")


class FailingWebPEngine(WebPEngine):
    """WebP engine that's available but fails during encoding."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        raise OptimizationError("WebP encoding failed: simulated failure")


class FailingPillowEngine(PillowEngine):
    """Pillow engine that fails during encoding."""

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        if strategy.format == "jpeg":
            raise OptimizationError("JPEG encoding failed: simulated failure")
        return super().optimize(image, strategy)


def test_avif_engine_error() -> None:
    engine = FailingAvifEngine()
    strategy = StrategyCandidate(
        format="avif",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_webp_engine_error() -> None:
    engine = FailingWebPEngine()
    strategy = StrategyCandidate(
        format="webp",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_pillow_engine_error() -> None:
    engine = FailingPillowEngine()
    strategy = StrategyCandidate(
        format="jpeg",
        quality=80,
        subsampling=2,
        progressive=True,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_compute_color_variance_empty_array() -> None:
    assert heuristics.compute_color_variance(np.array([])) == 0.0


def test_validation_none_passes() -> None:
    validation.ensure_between_0_1(None, "val")
    validation.ensure_non_empty(None, "items")
