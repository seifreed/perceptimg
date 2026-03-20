"""JPEG format characteristics and helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analyzer import AnalysisResult
from ..core.policy import Policy


@dataclass(slots=True)
class JPEGProfile:
    default_quality: int = 82
    text_quality: int = 90
    subsampling_for_text: int = 0
    subsampling_default: int = 2


def recommend_settings(policy: Policy, analysis: AnalysisResult) -> dict[str, int | bool]:
    profile = JPEGProfile()
    quality = (
        profile.text_quality
        if (policy.preserve_text or analysis.probable_text)
        else profile.default_quality
    )
    subsampling = (
        profile.subsampling_for_text
        if (analysis.probable_text or policy.preserve_faces)
        else profile.subsampling_default
    )
    return {
        "quality": quality,
        "subsampling": subsampling,
        "progressive": True,
    }
