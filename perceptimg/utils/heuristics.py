"""Lightweight heuristics for image content signals.

These heuristics avoid heavyweight models while still providing usable signals
for policy- and analysis-driven decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(slots=True)
class HeuristicConfig:
    edge_density_text_threshold: float = 0.0707
    edge_density_strong_threshold: float = 0.1414
    color_variance_text_threshold: float = 0.03
    aspect_ratio_text_threshold: float = 2.5
    skin_ratio_threshold: float = 0.12
    skin_tone_r_min: float = 0.15
    skin_tone_r_max: float = 0.95
    skin_tone_g_min: float = 0.08
    skin_tone_g_max: float = 0.70
    skin_tone_b_min: float = 0.05
    skin_tone_b_max: float = 0.55
    skin_tone_rg_min_diff: float = 0.01
    skin_tone_rb_min_diff: float = 0.01
    elongated_aspect_threshold: float = 2.0


DEFAULT_CONFIG = HeuristicConfig()

# Maximum gradient magnitude from central differences on [0, 255] grayscale.
# np.gradient uses central differences: max per-axis value is 127.5.
# Diagonal magnitude: sqrt(127.5^2 + 127.5^2) = 127.5 * sqrt(2).
_MAX_GRADIENT_MAGNITUDE = 127.5 * (2**0.5)  # ~180.312


def to_rgb_array(image: Image.Image) -> np.ndarray:
    """Return an RGB numpy array in range [0, 255].

    For images with alpha channel (RGBA, LA), the alpha is discarded
    and only RGB channels are returned.
    """

    if image.mode == "RGBA":
        return np.asarray(image)[..., :3].astype(np.float32)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_edge_density(image: Image.Image, config: HeuristicConfig = DEFAULT_CONFIG) -> float:
    """Estimate edge density using simple gradients.

    Uses central differences on the grayscale image and normalizes by the
    maximum possible gradient magnitude.

    Note: Images smaller than 2x2 pixels return 0.0 as edge detection
    requires at least 2 pixels per dimension for gradient computation.
    """

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0  # No edges possible in tiny images
    dx = np.gradient(gray, axis=1)
    dy = np.gradient(gray, axis=0)
    magnitude = np.hypot(dx, dy)
    normalized = magnitude / _MAX_GRADIENT_MAGNITUDE
    return float(np.mean(normalized))


def compute_color_variance(rgb_array: np.ndarray) -> float:
    """Compute normalized color variance across channels."""

    if rgb_array.size == 0:
        return 0.0
    variance = float(np.var(rgb_array / 255.0))
    return variance


def detect_probable_text(
    edge_density: float,
    color_variance: float,
    aspect_ratio: float,
    config: HeuristicConfig = DEFAULT_CONFIG,
) -> bool:
    """Heuristic text detector based on edges and uniform regions.

    Note: Aspect ratio values like 10000.0 (height=0) or 0.0001 (width=0)
    are treated as extreme aspect ratios and will trigger text detection.
    """

    dense_edges = edge_density > config.edge_density_text_threshold
    strong_edges = edge_density > config.edge_density_strong_threshold
    low_variance = color_variance < config.color_variance_text_threshold
    # Note: extreme_aspect (threshold 2.5) is a subset of elongated (threshold 2.0)
    # We only need to check elongated since it's the broader condition
    threshold_elongated = config.elongated_aspect_threshold
    elongated = aspect_ratio >= threshold_elongated or aspect_ratio <= (1 / threshold_elongated)
    return bool(strong_edges or (dense_edges and (low_variance or elongated)))


def detect_probable_faces(rgb_array: np.ndarray, config: HeuristicConfig = DEFAULT_CONFIG) -> bool:
    """Estimate face likelihood via crude skin-tone masking.

    This is intentionally lightweight and should not be treated as accurate
    detection. It is used to bias strategy choices when policies ask to
    preserve faces.

    The heuristic detects a wide range of skin tones by checking:
    - R channel dominates (higher than G and B)
    - Values are within reasonable ranges for human skin tones

    Known Limitations (False Positives):
        - Desert/sand landscapes with warm tones
        - Wooden surfaces and furniture
        - Certain foods (bread, pastries, cooked meats)
        - Sunset/sunrise scenes with warm lighting
        - Artificial lighting with warm color temperatures
        - Brick walls and terracotta surfaces

    Known Limitations (False Negatives):
        - Very dark skin tones may not be detected (adjust thresholds via config)
        - Images with unusual white balance or color grading
        - Partial faces or profile views

    Use with Caution:
        This heuristic is suitable for content-aware optimization hints but
        should not replace proper face detection for critical applications.

    Note: This heuristic has limitations and may produce false positives
    for non-skin regions with similar color distributions.
    """

    if rgb_array.size == 0:
        return False
    if rgb_array.ndim < 3 or rgb_array.shape[2] < 3:
        return False
    normalized = rgb_array / 255.0
    r, g, b = normalized[..., 0], normalized[..., 1], normalized[..., 2]
    skin_mask = (
        (r >= config.skin_tone_r_min)
        & (r <= config.skin_tone_r_max)
        & (g >= config.skin_tone_g_min)
        & (g <= config.skin_tone_g_max)
        & (b >= config.skin_tone_b_min)
        & (b <= config.skin_tone_b_max)
        & ((r - g) >= config.skin_tone_rg_min_diff)
        & ((r - b) >= config.skin_tone_rb_min_diff)
    )
    skin_ratio = float(np.mean(skin_mask))
    return skin_ratio > config.skin_ratio_threshold


def compute_aspect_ratio(width: int, height: int) -> float:
    """Return width/height aspect ratio, handling degenerate cases.

    Returns:
        float: aspect ratio as width/height
            - For normal images: width/height
            - For height=0, width>0: 10_000.0 (capped maximum)
            - For width=0, height>0: 0.0001 (capped minimum, reciprocal of max)
            - For width=0, height=0: 1.0 (undefined aspect, neutral default)
    """

    _MAX_ASPECT = 10_000.0
    if width <= 0 and height <= 0:
        return 1.0  # undefined aspect, neutral default
    if height <= 0:
        return _MAX_ASPECT
    if width <= 0:
        return 1.0 / _MAX_ASPECT
    return float(width / height)
