"""HEIF/HEIC format characteristics and helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analyzer import AnalysisResult
from ..core.policy import Policy


@dataclass(slots=True)
class HEIFProfile:
    default_quality: int = 80
    text_quality: int = 92


def recommend_settings(policy: Policy, analysis: AnalysisResult) -> dict[str, int | bool]:
    profile = HEIFProfile()
    quality = (
        profile.text_quality
        if (policy.preserve_text or analysis.probable_text)
        else profile.default_quality
    )
    return {
        "quality": quality,
        "lossless": not policy.allow_lossy,
    }
