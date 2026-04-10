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

    # Determine lossless mode based on format and policy
    if fmt in LOSSLESS_FORMATS:
        # PNG, TIFF, GIF, APNG are always lossless
        lossless = True
        final_quality = None
        progressive = False
    elif fmt == "jpeg":
        # JPEG is always lossy
        if not policy.allow_lossy:
            return None
        lossless = False
        progressive = True
        final_quality = quality if quality is not None else DEFAULT_QUALITIES[0]
    elif fmt in FORMATS_WITH_LOSSLESS_MODE:
        # WebP, AVIF, HEIF, JXL support both lossy and lossless modes
        # Note: Progressive encoding is only enabled for WebP and JXL because
        # Pillow's AVIF and HEIF encoders do not consistently support progressive
        # encoding across different libavif/libheif versions. This may change in
        # future Pillow versions as codec support improves.
        if policy.allow_lossy:
            lossless = False
            progressive = fmt in {"webp", "jxl"}
            final_quality = quality if quality is not None else DEFAULT_QUALITIES[0]
        else:
            lossless = True
            final_quality = None
            progressive = fmt in {"webp", "jxl"}
    else:
        # Unknown format - use defaults based on policy
        lossless = not policy.allow_lossy
        final_quality = (
            None if lossless else (quality if quality is not None else DEFAULT_QUALITIES[0])
        )
        progressive = False

    if fmt == "webp" and lossless:
        reasons.append("webp_lossless")

    subsampling = (
        0
        if (
            analysis.probable_text
            or analysis.probable_faces
            or policy.preserve_faces
            or policy.preserve_text
        )
        else 2
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
        quality=final_quality,
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
        """Return indices for selecting `target` items from `count` items.

        Priority order: select across the ordered list to preserve preferred formats
        while spreading selected indices for broader format coverage.

        Args:
            count: Total number of items in the sequence (must be > 0).
            target: Desired number of indices to select (must be > 0).

        Returns:
            List of selected indices, at most min(count, target) items.
        """
        if target >= count:
            return list(range(count))
        if target <= 1:
            return [0]
        if target == 2:
            return [0, count - 1]

        # Preserve the first few priority formats, then spread the remainder.
        keep_front = min(3, target)
        selected: set[int] = set(range(keep_front))
        remaining_slots = target - keep_front
        if remaining_slots <= 0:
            return [idx for idx in sorted(selected) if idx < count]

        candidate_span = count - keep_front
        if remaining_slots >= candidate_span:
            return list(range(count))

        if remaining_slots == 1:
            selected.add(count - 1)
            return sorted(selected)

        step = (candidate_span - 1) / (remaining_slots - 1)
        base = keep_front
        for i in range(remaining_slots):
            offset = round(i * step)
            selected.add(base + min(candidate_span - 1, max(0, offset)))

        if len(selected) < target:
            for candidate in range(base + 1, count - 1):
                if candidate in selected:
                    continue
                selected.add(candidate)
                if len(selected) >= target:
                    break

        return sorted(selected)[:target]

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
            supported_formats = tuple(fmt for fmt in formats if fmt in normalized_formats)
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
