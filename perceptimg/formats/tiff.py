"""TIFF format characteristics and helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analyzer import AnalysisResult
from ..core.policy import Policy


@dataclass(slots=True)
class TIFFProfile:
    compression: str = "tiff_lzw"


def recommend_settings(policy: Policy, analysis: AnalysisResult) -> dict[str, str | bool]:
    _ = analysis
    return {
        "compression": TIFFProfile().compression,
        "lossless": True,
    }
