"""Optimization reporting structures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .analyzer import AnalysisResult
from .policy import Policy
from .strategy import StrategyCandidate


@dataclass(slots=True)
class OptimizationReport:
    """Explainable summary of optimization decisions."""

    chosen_format: str
    quality: int | None
    size_before_kb: float
    size_after_kb: float
    ssim: float
    psnr: float
    perceptual_score: float
    reasons: list[str] = field(default_factory=list)
    policy: Policy | None = field(default=None, repr=False)
    analysis: AnalysisResult | None = field(default=None, repr=False)
    candidate: StrategyCandidate | None = field(default=None, repr=False)

    def to_dict(self, include_details: bool = False) -> Mapping[str, object]:
        """Return a serializable dictionary representation.

        Args:
            include_details: If True, include policy, analysis, and candidate details.
                Defaults to False for backward compatibility and cleaner output.
        """
        result: dict[str, object] = {
            "chosen_format": self.chosen_format,
            "quality": self.quality,
            "size_before_kb": self.size_before_kb,
            "size_after_kb": self.size_after_kb,
            "ssim": self.ssim,
            "psnr": self.psnr,
            "perceptual_score": self.perceptual_score,
            "reasons": list(self.reasons),
        }
        if include_details:
            if self.policy is not None:
                result["policy"] = dict(self.policy.to_dict())
            if self.analysis is not None:
                result["analysis"] = {
                    "edge_density": self.analysis.edge_density,
                    "color_variance": self.analysis.color_variance,
                    "probable_text": self.analysis.probable_text,
                    "probable_faces": self.analysis.probable_faces,
                    "resolution": list(self.analysis.resolution),
                    "aspect_ratio": self.analysis.aspect_ratio,
                }
            if self.candidate is not None:
                result["candidate"] = {
                    "format": self.candidate.format,
                    "quality": self.candidate.quality,
                    "subsampling": self.candidate.subsampling,
                    "progressive": self.candidate.progressive,
                    "lossless": self.candidate.lossless,
                    "reasons": list(self.candidate.reasons),
                }
        return result

    def __str__(self) -> str:
        details = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"OptimizationReport({details})"
