import pytest
from PIL import Image

from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.apng_engine import ApngEngine
from perceptimg.engines.base import EngineResult
from perceptimg.engines.heif_engine import HeifEngine
from perceptimg.engines.jxl_engine import JxlEngine
from perceptimg.exceptions import OptimizationError


def test_jxl_engine_unavailable() -> None:
    class UnavailableJxl(JxlEngine):
        @property
        def is_available(self) -> bool:
            return False

    engine = UnavailableJxl()
    strategy = StrategyCandidate(
        format="jxl",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_heif_engine_unavailable() -> None:
    class UnavailableHeif(HeifEngine):
        @property
        def is_available(self) -> bool:
            return False

    engine = UnavailableHeif()
    strategy = StrategyCandidate(
        format="heif",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_apng_engine_unavailable() -> None:
    class UnavailableApng(ApngEngine):
        @property
        def is_available(self) -> bool:
            return False

    engine = UnavailableApng()
    strategy = StrategyCandidate(
        format="apng",
        quality=None,
        subsampling=None,
        progressive=False,
        lossless=True,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


class FakeJxlEngine(JxlEngine):
    """JXL engine that's available and returns fake data."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        return EngineResult(
            data=b"fake_jxl_data",
            format="jxl",
            quality=strategy.quality,
            metadata={"lossless": strategy.lossless},
        )


class FailingJxlEngine(JxlEngine):
    """JXL engine that fails during encoding."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        raise OptimizationError("JXL encoding failed: simulated failure")


class FakeHeifEngine(HeifEngine):
    """HEIF engine that's available and returns fake data."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        return EngineResult(
            data=b"fake_heif_data",
            format=strategy.format.lower(),
            quality=strategy.quality,
            metadata={"lossless": strategy.lossless},
        )


class FailingHeifEngine(HeifEngine):
    """HEIF engine that fails during encoding."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        raise OptimizationError("HEIF encoding failed: simulated failure")


class FakeApngEngine(ApngEngine):
    """APNG engine that's available and returns fake data."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        return EngineResult(
            data=b"fake_apng_data",
            format="apng",
            quality=None,
            metadata={"lossless": True},
        )


class FailingApngEngine(ApngEngine):
    """APNG engine that fails during encoding."""

    @property
    def is_available(self) -> bool:
        return True

    def optimize(self, image: Image.Image, strategy: StrategyCandidate) -> EngineResult:
        raise OptimizationError("APNG encoding failed: simulated failure")


def test_jxl_engine_success() -> None:
    engine = FakeJxlEngine()
    strategy = StrategyCandidate(
        format="jxl",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=True,
    )
    result = engine.optimize(Image.new("RGB", (8, 8)), strategy)
    assert result.data == b"fake_jxl_data"
    assert result.format == "jxl"


def test_jxl_engine_error() -> None:
    engine = FailingJxlEngine()
    strategy = StrategyCandidate(
        format="jxl",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_heif_engine_success() -> None:
    engine = FakeHeifEngine()
    strategy = StrategyCandidate(
        format="heic",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=True,
    )
    result = engine.optimize(Image.new("RGB", (8, 8)), strategy)
    assert result.data == b"fake_heif_data"
    assert result.format == "heic"


def test_heif_engine_error() -> None:
    engine = FailingHeifEngine()
    strategy = StrategyCandidate(
        format="heif",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)


def test_apng_engine_success() -> None:
    engine = FakeApngEngine()
    strategy = StrategyCandidate(
        format="apng",
        quality=None,
        subsampling=None,
        progressive=False,
        lossless=True,
    )
    result = engine.optimize(Image.new("RGB", (8, 8)), strategy)
    assert result.data == b"fake_apng_data"
    assert result.format == "apng"


def test_apng_engine_error() -> None:
    engine = FailingApngEngine()
    strategy = StrategyCandidate(
        format="apng",
        quality=None,
        subsampling=None,
        progressive=False,
        lossless=True,
    )
    with pytest.raises(OptimizationError):
        engine.optimize(Image.new("RGB", (8, 8)), strategy)
