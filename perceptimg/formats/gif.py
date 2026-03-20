"""GIF format characteristics and helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analyzer import AnalysisResult
from ..core.policy import Policy


@dataclass(slots=True)
class GIFProfile:
    optimize: bool = True


def recommend_settings(policy: Policy, analysis: AnalysisResult) -> dict[str, bool]:
    _ = policy, analysis
    return {"optimize": GIFProfile().optimize}
