import pytest
from PIL import Image

from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.avif_engine import AvifEngine
from perceptimg.engines.pillow_engine import PillowEngine
from perceptimg.engines.webp_engine import WebPEngine
from perceptimg.exceptions import OptimizationError


def test_pillow_engine_invalid_format_raises() -> None:
    engine = PillowEngine()
    strategy = StrategyCandidate(
        format="bmp",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_avif_engine_unavailable() -> None:
    class UnavailableAvif(AvifEngine):
        @property
        def is_available(self) -> bool:
            return False

    engine = UnavailableAvif()
    strategy = StrategyCandidate(
        format="avif",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_webp_engine_unavailable() -> None:
    class UnavailableWebP(WebPEngine):
        @property
        def is_available(self) -> bool:
            return False

    engine = UnavailableWebP()
    strategy = StrategyCandidate(
        format="webp",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_pillow_engine_gif_branch() -> None:
    engine = PillowEngine()
    image = Image.new("RGB", (8, 8), "red")
    strategy = StrategyCandidate(
        format="gif",
        quality=None,
        subsampling=None,
        progressive=False,
        lossless=True,
    )
    result = engine.optimize(image, strategy)
    assert result.data is not None
    assert result.format == "gif"
    assert result.quality is None
