"""WebP format helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analyzer import AnalysisResult
from ..core.policy import Policy


@dataclass(slots=True)
class WebPProfile:
    default_quality: int = 80
    lossless_quality: int = 100


def recommend_settings(policy: Policy, analysis: AnalysisResult) -> dict[str, int | bool]:
    del analysis
    profile = WebPProfile()
    lossless = not policy.allow_lossy or policy.preserve_text
    quality = profile.lossless_quality if lossless else profile.default_quality
    return {"quality": quality, "lossless": lossless, "method": 6}
