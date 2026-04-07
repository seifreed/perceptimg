"""Policy definitions for perceptual optimization.

A Policy expresses desired outcomes and constraints without prescribing how to
achieve them. Policies are immutable and validated at creation time to avoid
runtime surprises.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, cast

class _UnsetType:
    """Sentinel to explicitly unset a field in merge().

    Use ``UNSET`` as a value in a dict passed to ``Policy.merge()`` to
    clear the corresponding field back to its default (``None``).
    """

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _UnsetType()

_ALLOWED_FORMATS = {
    "jpeg",
    "png",
    "webp",
    "avif",
    "jxl",
    "heif",
    "heic",
    "tiff",
    "gif",
    "apng",
}
_TargetUseCase = Literal["web", "mobile", "print", "general"]


def _validate_formats(formats: Sequence[str] | None) -> tuple[str, ...] | None:
    if formats is None:
        return None
    normalized = tuple(fmt.lower() for fmt in formats)
    unknown = set(normalized) - _ALLOWED_FORMATS
    if unknown:
        raise ValueError(f"Unsupported preferred_formats: {sorted(unknown)}")
    return normalized


_POLICY_PUBLIC_FIELDS = frozenset({
    "max_size_kb", "min_ssim", "preserve_text", "preserve_faces",
    "allow_lossy", "preferred_formats", "target_use_case",
})


@dataclass(frozen=True, slots=True)
class Policy:
    """Declarative optimization policy.

    Attributes:
        max_size_kb: Maximum allowed output size in kilobytes. None means unset.
        min_ssim: Minimum structural similarity index (0-1). None means unset.
        preserve_text: Whether to prioritize text crispness.
        preserve_faces: Whether to avoid artifacts around faces.
        allow_lossy: Whether lossy encoders are allowed.
        preferred_formats: Optional priority list of output formats.
        target_use_case: Target channel (web, mobile, print, general).
    """

    max_size_kb: int | None = None
    min_ssim: float | None = None
    preserve_text: bool = False
    preserve_faces: bool = False
    allow_lossy: bool = True
    preferred_formats: tuple[str, ...] | None = field(default=None, repr=False)
    target_use_case: _TargetUseCase = "web"
    _explicit_fields: frozenset[str] = field(
        default=frozenset(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.max_size_kb is not None:
            if math.isnan(self.max_size_kb) or math.isinf(self.max_size_kb):
                raise ValueError("max_size_kb must be a finite number")
            if self.max_size_kb <= 0:
                raise ValueError("max_size_kb must be positive when provided")
        if self.min_ssim is not None:
            if math.isnan(self.min_ssim) or math.isinf(self.min_ssim):
                raise ValueError("min_ssim must be a finite number")
            if not (0 <= self.min_ssim <= 1):
                raise ValueError("min_ssim must be within [0, 1]")
        object.__setattr__(self, "preferred_formats", _validate_formats(self.preferred_formats))

    def validate_for_size(self, input_size_kb: float) -> list[str]:
        """Validate policy for a given input size, returning list of warnings.

        Does not raise exceptions; returns warnings for potentially impossible constraints.
        """
        warnings: list[str] = []
        if (
            self.max_size_kb is not None
            and self.min_ssim is not None
            and self.min_ssim >= 0.95
            and self.max_size_kb < input_size_kb * 0.5
        ):
            warnings.append(
                f"Very aggressive compression (max_size_kb={self.max_size_kb} for "
                f"{input_size_kb:.1f}KB input) with min_ssim={self.min_ssim} may be impossible"
            )
        if (
            not self.allow_lossy
            and self.max_size_kb is not None
            and self.max_size_kb < input_size_kb
        ):
            warnings.append(
                f"Lossless policy with max_size_kb={self.max_size_kb} may exceed limit for "
                f"{input_size_kb:.1f}KB input"
            )
        return warnings

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dictionary representation."""

        data: dict[str, object] = asdict(self)
        data.pop("_explicit_fields", None)
        if self.preferred_formats is not None:
            data["preferred_formats"] = list(self.preferred_formats)
        return data

    def to_json(self) -> str:
        """Return the policy encoded as JSON."""

        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | MutableMapping[str, object]) -> Policy:
        """Create a policy from a mapping, validating inputs."""

        data = dict(payload)
        data.pop("_explicit_fields", None)
        explicit = frozenset(k for k in data if k in _POLICY_PUBLIC_FIELDS)
        data["_explicit_fields"] = explicit
        return cls(**cast(dict[str, Any], data))

    @classmethod
    def from_json(cls, payload: str) -> Policy:
        """Create a policy from a JSON string."""

        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Policy JSON must decode to a mapping")
        return cls.from_dict(data)

    def preferred_format_order(self, fallback: Iterable[str]) -> tuple[str, ...]:
        """Return ordered formats honoring policy preferences."""

        if self.preferred_formats:
            return self.preferred_formats
        return tuple(fmt for fmt in fallback if fmt in _ALLOWED_FORMATS)

    def with_updates(self, **updates: object) -> Policy:
        """Return a new Policy with selected fields updated."""

        data = self.to_dict()
        data.update(updates)
        return Policy.from_dict(data)

    def override(self, **updates: object) -> Policy:
        """Alias for with_updates for readability."""

        return self.with_updates(**updates)

    def merge(self, other: Policy | Mapping[str, object]) -> Policy:
        """Combine two policies, preferring values from ``other`` when set.

        ``other`` can be a ``Policy`` or a plain mapping.

        Override semantics:

        - None values: never override (treated as "unset")
        - Empty containers (empty tuple, list, dict): never override

        When ``other`` is a ``Policy``, only fields that were explicitly
        passed at construction time are considered as overrides.  This means
        ``base.merge(Policy())`` is a no-op, but
        ``base.merge(Policy(preserve_text=False))`` will set preserve_text
        to False even if that is the default value.
        Use a plain mapping for partial updates where unmentioned keys should
        be preserved.
        """

        base = self.to_dict()

        if isinstance(other, Policy):
            other_dict = {
                k: v
                for k, v in other.to_dict().items()
                if k in other._explicit_fields
            }
        else:
            other_dict = dict(other)
            other_dict.pop("_explicit_fields", None)

        is_policy = isinstance(other, Policy)
        override_data = {}
        for k, v in other_dict.items():
            if isinstance(v, _UnsetType):
                override_data[k] = None
                continue
            if v is None:
                continue
            override_data[k] = v

        base.update(override_data)
        return Policy.from_dict(base)


# Wrap __init__ to automatically track which fields were explicitly passed.
_Policy_orig_init = Policy.__init__
_POLICY_FIELD_NAMES = tuple(
    f.name for f in fields(Policy) if f.name != "_explicit_fields"
)


def _policy_init_wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    if not kwargs.get("_explicit_fields"):
        explicit: set[str] = set()
        for i in range(min(len(args), len(_POLICY_FIELD_NAMES))):
            explicit.add(_POLICY_FIELD_NAMES[i])
        explicit.update(k for k in kwargs if k in _POLICY_PUBLIC_FIELDS)
        kwargs["_explicit_fields"] = frozenset(explicit)
    _Policy_orig_init(self, *args, **kwargs)


Policy.__init__ = _policy_init_wrapper  # type: ignore[method-assign]
