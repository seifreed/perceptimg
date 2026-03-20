import logging

import pytest
from PIL import Image

from perceptimg.core.analyzer import AnalysisResult
from perceptimg.core.policy import Policy
from perceptimg.core.strategy import build_candidate, plan_qualities
from perceptimg.engines.apng_engine import ApngEngine
from perceptimg.engines.avif_engine import AvifEngine
from perceptimg.engines.heif_engine import HeifEngine
from perceptimg.engines.jxl_engine import JxlEngine
from perceptimg.engines.pillow_engine import PillowEngine
from perceptimg.engines.webp_engine import WebPEngine
from perceptimg.formats import apng as apng_fmt
from perceptimg.formats import avif as avif_fmt
from perceptimg.formats import gif as gif_fmt
from perceptimg.formats import heif as heif_fmt
from perceptimg.formats import jpeg as jpeg_fmt
from perceptimg.formats import jxl as jxl_fmt
from perceptimg.formats import png as png_fmt
from perceptimg.formats import tiff as tiff_fmt
from perceptimg.formats import webp as webp_fmt
from perceptimg.utils import heuristics, image_io, logging_config, validation


def test_validation_helpers() -> None:
    validation.ensure_positive(None, "x")
    with pytest.raises(validation.ValidationError):
        validation.ensure_positive(0, "x")
    with pytest.raises(validation.ValidationError):
        validation.ensure_between_0_1(2.0, "y")
    with pytest.raises(validation.ValidationError):
        validation.ensure_non_empty([], "z")


def test_logging_config_merge() -> None:
    logging_config.configure_logging(json_output=False, merge=True)
    logger = logging.getLogger("test")
    logger.info("hello")  # no crash


def test_formats_helpers() -> None:
    analysis = AnalysisResult(
        edge_density=0.1,
        color_variance=0.02,
        probable_text=True,
        probable_faces=False,
        resolution=(10, 10),
        aspect_ratio=1.0,
    )
    policy = Policy(preserve_text=True)
    assert jpeg_fmt.recommend_settings(policy, analysis)["quality"] >= 90
    assert png_fmt.recommend_settings(policy, analysis)["optimize"] is True
    assert webp_fmt.recommend_settings(policy, analysis)["lossless"] is True
    assert avif_fmt.recommend_settings(policy, analysis)["quality"] >= 80
    assert jxl_fmt.recommend_settings(policy, analysis)["quality"] >= 80
    assert heif_fmt.recommend_settings(policy, analysis)["quality"] >= 80
    assert tiff_fmt.recommend_settings(policy, analysis)["lossless"] is True
    assert gif_fmt.recommend_settings(policy, analysis)["optimize"] is True
    assert apng_fmt.recommend_settings(policy, analysis)["optimize"] is True


def test_heuristics_extreme_aspect_triggers_text() -> None:
    config = heuristics.HeuristicConfig()
    assert heuristics.detect_probable_text(0.2, 0.01, 10.0, config=config)
    assert heuristics.detect_probable_text(0.2, 0.01, 0.01, config=config)


def test_image_io_size_kb() -> None:
    img = Image.new("RGB", (2, 2), "black")
    data = image_io.image_to_bytes(img, format="PNG")
    assert image_io.size_kb(data) > 0


def test_strategy_plan_and_build() -> None:
    analysis = AnalysisResult(
        edge_density=0.1,
        color_variance=0.02,
        probable_text=True,
        probable_faces=True,
        resolution=(10, 10),
        aspect_ratio=1.0,
    )
    policy = Policy(preserve_text=True, preserve_faces=True)
    qualities = plan_qualities(policy, analysis)
    assert qualities[0] >= 90
    candidate = build_candidate("png", None, policy, analysis)
    assert candidate.lossless is True


def test_engine_is_available_flags() -> None:
    assert isinstance(WebPEngine().is_available, bool)
    assert isinstance(AvifEngine().is_available, bool)
    assert isinstance(PillowEngine().is_available, bool)
    assert isinstance(JxlEngine().is_available, bool)
    assert isinstance(HeifEngine().is_available, bool)
    assert isinstance(ApngEngine().is_available, bool)
