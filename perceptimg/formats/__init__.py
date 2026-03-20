"""Format-specific recommendation helpers.

These helpers provide format-specific settings recommendations based on
policy and analysis. They are optional utilities that can be used for
advanced configuration or custom strategy generation.

Note: These are not used by the default optimization pipeline. The
StrategyGenerator in perceptimg.core.strategy handles strategy generation
for the default pipeline.
"""

from __future__ import annotations

from . import apng, avif, gif, heif, jpeg, jxl, png, tiff, webp

__all__ = ["apng", "avif", "gif", "heif", "jpeg", "jxl", "png", "tiff", "webp"]
