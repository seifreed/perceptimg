"""Presentation-level factories and helpers for CLI-facing behavior."""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from collections.abc import Sequence
from importlib import import_module
from io import StringIO
from pathlib import Path
from typing import Any, Protocol, cast

from ..core.batch.config import BatchResult
from ..core.optimizer import OptimizationResult, Optimizer
from ..core.policy import Policy

_KNOWN_FORMATS: tuple[str, ...] = (
    "apng",
    "avif",
    "gif",
    "heif",
    "heic",
    "jpeg",
    "jxl",
    "png",
    "tiff",
    "webp",
)

FORMAT_EXTENSIONS = {"jpeg": "jpg", "tiff": "tif"}
DEFAULT_FORMAT_EXTENSION = "jpg"


class _OptimizerFactory(Protocol):
    """Factory signature used by CLI-facing optimizer assembly."""

    def __call__(
        self,
        *,
        ssim_weight: float,
        size_weight: float,
        prioritize_quality: bool,
        max_candidates: int,
    ) -> Optimizer: ...


class _PolicyFactory(Protocol):
    def from_json(self, payload: str) -> Policy: ...

    def from_dict(self, payload: dict[str, object]) -> Policy: ...


def _policy_factory(policy_factory: _PolicyFactory | None) -> _PolicyFactory:
    """Return a policy factory with safe late binding to the concrete policy class."""
    if policy_factory is not None:
        return policy_factory

    module = import_module("perceptimg.core.policy")
    return cast(_PolicyFactory, module.Policy)


def resolve_output_extension(format_name: str) -> str:
    """Return a stable file extension for a CLI-reported output format."""
    return FORMAT_EXTENSIONS.get(format_name, format_name)


def get_allowed_formats() -> tuple[str, ...]:
    """Return the supported output formats."""
    return _KNOWN_FORMATS


def parse_preferred_formats(formats: str | None) -> tuple[str, ...] | None:
    """Parse and normalize preferred formats from a CLI-style comma-separated string."""
    if not formats:
        return None
    return tuple(fmt.strip() for fmt in formats.split(",") if fmt.strip())


def validate_preferred_formats(formats: Sequence[str] | None) -> tuple[str, ...] | None:
    """Validate preferred formats and return them normalized."""
    if not formats:
        return None
    normalized = tuple(format_name.lower() for format_name in formats)
    return normalized


def build_optimizer(
    *,
    ssim_weight: float,
    size_weight: float,
    prioritize_quality: bool,
    max_candidates: int,
    optimizer_factory: _OptimizerFactory,
) -> Optimizer:
    """Build a CLI-compatible optimizer with configured strategy priorities."""
    return optimizer_factory(
        ssim_weight=ssim_weight,
        size_weight=size_weight,
        prioritize_quality=prioritize_quality,
        max_candidates=max_candidates,
    )


def get_extension(format_name: str) -> str:
    """Resolve output extension for user-facing file paths."""
    return resolve_output_extension(format_name)


def output_extension_from_reported_format(chosen_format: str) -> str:
    """Compute a default file extension for a chosen output format."""
    return f".{get_extension(chosen_format)}"


def _parse_preferred_formats(formats: str | None) -> tuple[str, ...] | None:
    """Backward-compatible private alias for CLI/presentation parsing."""
    return parse_preferred_formats(formats)


def _resolve_output_extension(format_name: str) -> str:
    """Backward-compatible private alias for extension parsing."""
    return resolve_output_extension(format_name)


def load_policy(
    policy_path: Path,
    *,
    policy_factory: _PolicyFactory | None = None,
) -> Policy:
    """Load and parse a policy file."""
    factory = _policy_factory(policy_factory)
    payload = policy_path.read_text(encoding="utf-8")
    if policy_path.suffix.lower() in {".json", ".policy"}:
        return factory.from_json(payload)
    data = json.loads(payload)
    return factory.from_dict(data)


def policy_from_flags(
    args: object,
    *,
    policy_factory: _PolicyFactory | None = None,
) -> Policy:
    """Build a policy from CLI-like flag arguments."""
    factory = _policy_factory(policy_factory)
    try:
        args_dict: dict[str, Any] = vars(args)
    except TypeError:
        args_dict = {}

    def _value(name: str, default: Any = None) -> Any:
        if name in args_dict:
            return args_dict[name]
        return getattr(args, name, default)

    if policy_path := _value("policy"):
        return load_policy(Path(policy_path), policy_factory=factory)

    base = {
        "max_size_kb": _value("max_size_kb"),
        "min_ssim": _value("min_ssim"),
        "preserve_text": _value("preserve_text", False),
        "preserve_faces": _value("preserve_faces", False),
        "allow_lossy": (
            _value("allow_lossy") if _value("allow_lossy") is not None else not _value("lossless")
        ),
        "target_use_case": _value("target_use_case") or "web",
    }
    if _value("formats"):
        base["preferred_formats"] = parse_preferred_formats(_value("formats"))
    return factory.from_dict(base)


def with_collision_suffix(path: Path, suffix_index: int) -> Path:
    filename = path.name
    if path.suffix:
        filename = f"{path.stem}_{suffix_index}{path.suffix}"
    else:
        filename = f"{filename}_{suffix_index}"
    return path.with_name(filename)


def reserve_batch_output_path(
    output_dir: Path,
    output_name: str,
    reserved: set[Path],
) -> Path:
    candidate = output_dir / output_name
    if candidate not in reserved:
        reserved.add(candidate)
        return candidate

    suffix_index = 2
    max_suffix = 10_000
    while suffix_index <= max_suffix:
        deduped = with_collision_suffix(candidate, suffix_index)
        if deduped not in reserved:
            reserved.add(deduped)
            return deduped
        suffix_index += 1
    raise RuntimeError(f"Too many filename collisions for {candidate}")


def plan_batch_successful_outputs(
    input_paths: Sequence[Path],
    successful: Sequence[tuple[Path, OptimizationResult]],
    output_dir: Path,
    output_pattern: str,
    successful_input_indices: Sequence[int] | None = None,
) -> list[tuple[Path, OptimizationResult, Path]]:
    """Map successful results to deterministic output paths."""
    reserved_outputs: set[Path] = set()

    def reserve_output(path: Path, result: OptimizationResult) -> Path:
        ext = resolve_output_extension(result.report.chosen_format)
        output_name = output_pattern.format(
            name=path.stem,
            ext=ext,
            format=result.report.chosen_format,
        )
        return reserve_batch_output_path(output_dir, output_name, reserved_outputs)

    if successful_input_indices is not None and len(successful_input_indices) != len(successful):
        import logging

        logging.getLogger(__name__).warning(
            "Input indices count (%d) != successful count (%d); " "falling back to path matching",
            len(successful_input_indices),
            len(successful),
        )
        successful_input_indices = None

    if successful_input_indices is not None and len(successful_input_indices) == len(successful):
        successful_by_index: dict[int, tuple[Path, OptimizationResult]] = {}
        for input_index, (path, result) in zip(
            successful_input_indices,
            successful,
            strict=True,
        ):
            if 0 <= input_index < len(input_paths):
                successful_by_index[input_index] = (path, result)

        planned_by_index: list[tuple[Path, OptimizationResult, Path]] = []
        for input_index, input_path in enumerate(input_paths):
            pair = successful_by_index.get(input_index)
            if pair is None:
                continue
            path, result = pair
            planned_by_index.append((path, result, reserve_output(input_path, result)))
        return planned_by_index

    successful_by_path: dict[str, deque[tuple[Path, OptimizationResult]]] = defaultdict(deque)
    for path, result in successful:
        successful_by_path[str(path)].append((path, result))

    planned_by_path: list[tuple[Path, OptimizationResult, Path]] = []
    for input_path in input_paths:
        bucket = successful_by_path[str(input_path)]
        if not bucket:
            continue
        path, result = bucket.popleft()
        planned_by_path.append((path, result, reserve_output(input_path, result)))

    for bucket in successful_by_path.values():
        while bucket:
            path, result = bucket.popleft()
            planned_by_path.append((path, result, reserve_output(path, result)))

    return planned_by_path


def batch_successful_report_rows(
    result: BatchResult,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> list[dict[str, object]]:
    if successful_outputs is None:
        return [
            {
                "input": str(path),
                "output_path": None,
                "output_format": res.report.chosen_format,
                "size_before_kb": res.report.size_before_kb,
                "size_after_kb": res.report.size_after_kb,
                "ssim": res.report.ssim,
                "psnr": res.report.psnr,
            }
            for path, res in result.successful
        ]

    return [
        {
            "input": str(path),
            "output_path": str(output_path),
            "output_format": res.report.chosen_format,
            "size_before_kb": res.report.size_before_kb,
            "size_after_kb": res.report.size_after_kb,
            "ssim": res.report.ssim,
            "psnr": res.report.psnr,
        }
        for path, res, output_path in successful_outputs
    ]


def batch_report_data(
    result: BatchResult,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> dict[str, object]:
    return {
        "total": result.total,
        "successful": len(result.successful),
        "failed": len(result.failed),
        "skipped": len(result.skipped),
        "success_rate": result.success_rate,
        "results": batch_successful_report_rows(result, successful_outputs),
        "errors": [{"input": str(path), "error": str(exc)} for path, exc in result.failed],
        "skipped_items": [{"input": str(path)} for path in result.skipped],
    }


def batch_summary_text(result: BatchResult) -> str:
    lines = [
        f"Processed {result.total} images:",
        f"Successful: {len(result.successful)}",
    ]
    if result.failed:
        lines.append(f"Failed: {len(result.failed)}")
    if result.skipped:
        lines.append(f"Skipped: {len(result.skipped)}")
    if result.successful:
        total_before = sum((r.report.size_before_kb or 0.0) for _, r in result.successful)
        total_after = sum((r.report.size_after_kb or 0.0) for _, r in result.successful)
        if total_before > 0:
            reduction = (1 - total_after / total_before) * 100
            lines.append(f"Total reduction: {reduction:.1f}%")
    return "\n".join(lines)


def write_batch_report(
    report_path: Path,
    result: BatchResult,
    report_format: str,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_format == "json":
        report_path.write_text(
            json.dumps(batch_report_data(result, successful_outputs), indent=2),
            encoding="utf-8",
        )
        return

    if report_format == "csv":
        fieldnames = [
            "input",
            "status",
            "output_path",
            "output_format",
            "size_before_kb",
            "size_after_kb",
            "ssim",
            "psnr",
            "error",
        ]
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in batch_successful_report_rows(result, successful_outputs):
            writer.writerow(
                {
                    "input": row["input"],
                    "status": "successful",
                    "output_path": row["output_path"] or "",
                    "output_format": row["output_format"],
                    "size_before_kb": row["size_before_kb"],
                    "size_after_kb": row["size_after_kb"],
                    "ssim": row["ssim"],
                    "psnr": row["psnr"],
                    "error": "",
                }
            )
        for path, exc in result.failed:
            writer.writerow(
                {
                    "input": str(path),
                    "status": "failed",
                    "output_path": "",
                    "output_format": "",
                    "size_before_kb": "",
                    "size_after_kb": "",
                    "ssim": "",
                    "psnr": "",
                    "error": str(exc),
                }
            )
        for path in result.skipped:
            writer.writerow(
                {
                    "input": str(path),
                    "status": "skipped",
                    "output_path": "",
                    "output_format": "",
                    "size_before_kb": "",
                    "size_after_kb": "",
                    "ssim": "",
                    "psnr": "",
                    "error": "",
                }
            )
        report_path.write_text(buffer.getvalue(), encoding="utf-8")
        return

    report_path.write_text(batch_summary_text(result), encoding="utf-8")


__all__ = [
    "DEFAULT_FORMAT_EXTENSION",
    "FORMAT_EXTENSIONS",
    "batch_report_data",
    "batch_successful_report_rows",
    "batch_summary_text",
    "build_optimizer",
    "get_allowed_formats",
    "get_extension",
    "load_policy",
    "output_extension_from_reported_format",
    "parse_preferred_formats",
    "plan_batch_successful_outputs",
    "policy_from_flags",
    "reserve_batch_output_path",
    "resolve_output_extension",
    "validate_preferred_formats",
    "with_collision_suffix",
    "write_batch_report",
    "_parse_preferred_formats",
    "_resolve_output_extension",
]
