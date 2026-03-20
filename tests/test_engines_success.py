from PIL import Image

from perceptimg.core.strategy import StrategyCandidate
from perceptimg.engines.avif_engine import AvifEngine
from perceptimg.engines.pillow_engine import PillowEngine
from perceptimg.engines.webp_engine import WebPEngine
from perceptimg.utils.image_io import bytes_to_image


def test_pillow_engine_success_jpeg() -> None:
    engine = PillowEngine()
    strategy = StrategyCandidate(
        format="jpeg",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=False,
    )
    result = engine.optimize(Image.new("RGB", (8, 8), "white"), strategy)
    img = bytes_to_image(result.data)
    assert img.size == (8, 8)


def test_webp_engine_handles_lossless_flag() -> None:
    engine = WebPEngine()
    if not engine.is_available:
        return
    strategy = StrategyCandidate(
        format="webp",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=True,
    )
    result = engine.optimize(Image.new("RGB", (8, 8), "white"), strategy)
    img = bytes_to_image(result.data)
    assert img.size == (8, 8)


def test_avif_engine_handles_lossless_flag() -> None:
    engine = AvifEngine()
    if not engine.is_available:
        return
    strategy = StrategyCandidate(
        format="avif",
        quality=80,
        subsampling=2,
        progressive=False,
        lossless=True,
    )
    result = engine.optimize(Image.new("RGB", (8, 8), "white"), strategy)
    img = bytes_to_image(result.data)
    assert img.size == (8, 8)
