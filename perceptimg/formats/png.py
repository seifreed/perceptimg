"""PNG format helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analyzer import AnalysisResult
from ..core.policy import Policy


@dataclass(slots=True)
class PNGProfile:
    compress_level: int = 6
    optimize: bool = True


def recommend_settings(policy: Policy, analysis: AnalysisResult) -> dict[str, int | bool]:
    _ = (policy, analysis)
    profile = PNGProfile()
    return {"compress_level": profile.compress_level, "optimize": profile.optimize}
