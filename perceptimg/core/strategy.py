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
        return (None,)
    qualities: list[int | None] = [82, 74, 68]
    if analysis.probable_text or policy.preserve_text:
        qualities.insert(0, 90)
    if policy.target_use_case == "mobile":
        qualities.append(60)
    qualities.append(None)  # lossless mode for formats that support it
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
        if quality is None:
            quality = DEFAULT_QUALITIES[0]
    elif not policy.allow_lossy and fmt in FORMATS_WITH_LOSSLESS_MODE:
        lossless = True
        quality = None
        progressive = fmt in {"webp", "jxl"}
    else:
        if not policy.allow_lossy:
            lossless = True
            quality = None
            progressive = False
        else:
            lossless = False
            progressive = fmt in {"jpeg", "webp", "jxl"}
            if quality is None:
                quality = DEFAULT_QUALITIES[0]

    if fmt == "webp" and lossless:
        reasons.append("webp_lossless")

    subsampling = (
        0 if (analysis.probable_text or analysis.probable_faces or policy.preserve_faces or policy.preserve_text) else 2
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

    @staticmethod
    def _distributed_indices(count: int, target: int) -> list[int]:
        """Pick indices across an ordered sequence while respecting a hard cap."""
        if target >= count:
            return list(range(count))
        if target == 1:
            return [0]

        indices: list[int] = []
        last_idx = -1
        for slot in range(target):
            raw_idx = round(slot * (count - 1) / (target - 1))
            min_allowed = last_idx + 1
            max_allowed = count - (target - slot)
            idx = max(min_allowed, min(raw_idx, max_allowed))
            indices.append(idx)
            last_idx = idx
        return indices

    def generate(
        self,
        policy: Policy,
        analysis: AnalysisResult,
        *,
        available_formats: set[str] | None = None,
    ) -> list[StrategyCandidate]:
        formats = policy.preferred_format_order(DEFAULT_ORDER)
        if not formats:
            raise StrategyError("No candidate formats available")
        unknown = set(formats) - _ALLOWED_FORMATS
        if unknown:
            raise StrategyError(f"Unsupported formats in preferred order: {sorted(unknown)}")
        if available_formats is not None:
            normalized_formats = {fmt.lower() for fmt in available_formats if fmt}
            supported_formats = [fmt for fmt in formats if fmt in normalized_formats]
            if supported_formats:
                formats = supported_formats
        qualities = plan_qualities(policy, analysis)
        candidates: list[StrategyCandidate] = []
        seen_keys: set[tuple[str, int | None, bool, int | None]] = set()
        quality_plan_by_format: dict[str, tuple[int | None, ...]] = {
            fmt: ((None,) if fmt in LOSSLESS_FORMATS else qualities) for fmt in formats
        }

        def append_candidate(fmt: str, quality: int | None) -> None:
            candidate = build_candidate(fmt, quality, policy, analysis)
            if candidate is None:
                return
            key = (
                candidate.format,
                candidate.quality,
                candidate.lossless,
                candidate.subsampling,
            )
            if key in seen_keys:
                return
            seen_keys.add(key)
            candidates.append(candidate)

        first_round_candidates: list[StrategyCandidate] = []
        for fmt in formats:
            candidate = build_candidate(fmt, quality_plan_by_format[fmt][0], policy, analysis)
            if candidate is None:
                continue
            first_round_candidates.append(candidate)

        if len(first_round_candidates) > self.max_candidates:
            selected_indices = self._distributed_indices(
                len(first_round_candidates), self.max_candidates
            )
            first_round_candidates = [first_round_candidates[idx] for idx in selected_indices]

        for candidate in first_round_candidates:
            append_candidate(candidate.format, candidate.quality)

        if len(candidates) >= self.max_candidates:
            return candidates

        max_rounds = max(len(plan) for plan in quality_plan_by_format.values())
        for round_index in range(1, max_rounds):
            for fmt in formats:
                plan = quality_plan_by_format[fmt]
                if round_index >= len(plan):
                    continue
                append_candidate(fmt, plan[round_index])
                if len(candidates) >= self.max_candidates:
                    return candidates

        return candidates
