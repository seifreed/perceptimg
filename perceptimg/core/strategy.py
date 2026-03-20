"""Strategy generation for image optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..exceptions import StrategyError
from .analyzer import AnalysisResult
from .policy import _ALLOWED_FORMATS, Policy


@dataclass(slots=True)
class StrategyCandidate:
    """Represents a single optimization attempt."""

    format: str
    quality: int | None
    subsampling: int | None
    progressive: bool
    lossless: bool
    reasons: list[str] = field(default_factory=list)


DEFAULT_QUALITIES = (82, 74, 68)
DEFAULT_ORDER = ("jxl", "avif", "webp", "heif", "heic", "jpeg", "png", "tiff", "gif", "apng")

LOSSLESS_FORMATS = {"png", "tiff", "gif", "apng"}
FORMATS_WITH_LOSSLESS_MODE = {"webp", "avif", "heif", "heic", "jxl"}
LOSSY_ONLY_FORMATS = {"jpeg"}


def plan_qualities(policy: Policy, analysis: AnalysisResult) -> tuple[int | None, ...]:
    """Return an ordered quality plan driven by policy and analysis."""
    if not policy.allow_lossy:
        return (100,)
    qualities: list[int] = [82, 74, 68]
    if analysis.probable_text or policy.preserve_text:
        qualities.insert(0, 90)
    if policy.target_use_case == "mobile":
        qualities.append(60)
    return tuple(qualities)


def build_candidate(
    fmt: str,
    quality: int | None,
    policy: Policy,
    analysis: AnalysisResult,
) -> StrategyCandidate | None:
    """Build a strategy candidate for a format.

    Returns:
        StrategyCandidate if the format is compatible with the policy,
        None if the format cannot satisfy the policy (e.g., JPEG with allow_lossy=False).

    Note: Unknown formats will be processed with default parameters (lossless=False,
    progressive based on common defaults). The engine registry will handle unknown
    formats by returning no available engines.
    """
    reasons: list[str] = []

    if fmt in LOSSLESS_FORMATS:
        lossless = True
        quality = None
        progressive = False
    elif fmt == "jpeg":
        if not policy.allow_lossy:
            return None
        lossless = False
        progressive = True
    elif not policy.allow_lossy and fmt in FORMATS_WITH_LOSSLESS_MODE:
        lossless = True
        quality = None
        progressive = fmt in {"webp", "jxl"}
    else:
        lossless = False
        progressive = fmt in {"jpeg", "webp", "jxl"}

    if fmt == "webp" and lossless:
        reasons.append("webp_lossless")

    subsampling = (
        0 if (analysis.probable_text or policy.preserve_faces or policy.preserve_text) else 2
    )

    if policy.preserve_text:
        reasons.append("policy_preserve_text")
    if policy.preserve_faces:
        reasons.append("policy_preserve_faces")
    if analysis.probable_text:
        reasons.append("analysis_text_detected")
    if analysis.probable_faces:
        reasons.append("analysis_faces_likely")

    return StrategyCandidate(
        format=fmt,
        quality=quality,
        subsampling=subsampling,
        progressive=progressive,
        lossless=lossless,
        reasons=reasons,
    )


class StrategyGenerator:
    """Generate candidate strategies from policy and analysis signals."""

    def __init__(self, *, max_candidates: int = 8) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.max_candidates = max_candidates

    def generate(self, policy: Policy, analysis: AnalysisResult) -> list[StrategyCandidate]:
        formats = policy.preferred_format_order(DEFAULT_ORDER)
        if not formats:
            raise StrategyError("No candidate formats available")
        unknown = set(formats) - _ALLOWED_FORMATS
        if unknown:
            raise StrategyError(f"Unsupported formats in preferred order: {sorted(unknown)}")
        qualities = plan_qualities(policy, analysis)
        candidates: list[StrategyCandidate] = []
        seen_keys: set[tuple[str, int | None, bool, int | None]] = set()
        for fmt in formats:
            if fmt in LOSSLESS_FORMATS:
                candidate = build_candidate(fmt, None, policy, analysis)
                if candidate is None:
                    continue
                key = (
                    candidate.format,
                    candidate.quality,
                    candidate.lossless,
                    candidate.subsampling,
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(candidate)
                if len(candidates) >= self.max_candidates:
                    return candidates
            else:
                for quality in qualities:
                    if len(candidates) >= self.max_candidates:
                        return candidates
                    candidate = build_candidate(fmt, quality, policy, analysis)
                    if candidate is None:
                        continue
                    key = (
                        candidate.format,
                        candidate.quality,
                        candidate.lossless,
                        candidate.subsampling,
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    candidates.append(candidate)
        return candidates
