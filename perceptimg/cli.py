"""CLI for perceptimg with single and batch processing support."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from . import Policy
from .core.batch import BatchProgress, BatchResult, estimate_batch_size, optimize_batch
from .core.metrics import MetricCalculator
from .core.optimizer import OptimizationResult, Optimizer
from .core.policy import _ALLOWED_FORMATS
from .core.strategy import StrategyGenerator
from .utils import logging_config

FORMAT_EXTENSIONS = {"jpeg": "jpg", "tiff": "tif"}


def _get_extension(format_name: str) -> str:
    return FORMAT_EXTENSIONS.get(format_name, format_name)


def _load_policy(policy_path: Path) -> Policy:
    payload = policy_path.read_text(encoding="utf-8")
    if policy_path.suffix.lower() in {".json", ".policy"}:
        return Policy.from_json(payload)
    data = json.loads(payload)
    return Policy.from_dict(data)


def _policy_from_flags(args: argparse.Namespace) -> Policy:
    if args.policy:
        return _load_policy(Path(args.policy))
    base = {
        "max_size_kb": args.max_size_kb,
        "min_ssim": args.min_ssim,
        "preserve_text": args.preserve_text,
        "preserve_faces": args.preserve_faces,
        "allow_lossy": (
            args.allow_lossy if args.allow_lossy is not None else not args.lossless
        ),
        "target_use_case": args.target_use_case or "web",
    }
    if args.formats:
        formats = tuple(fmt.strip() for fmt in args.formats.split(",") if fmt.strip())
        unknown = set(formats) - _ALLOWED_FORMATS
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            valid_list = ", ".join(sorted(_ALLOWED_FORMATS))
            raise ValueError(f"Unknown format(s): {unknown_list}. Valid formats: {valid_list}")
        base["preferred_formats"] = formats
    return Policy.from_dict(base)


def _write_output(result: OptimizationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.image_bytes)


def _with_collision_suffix(path: Path, suffix_index: int) -> Path:
    filename = path.name
    if path.suffix:
        filename = f"{path.stem}_{suffix_index}{path.suffix}"
    else:
        filename = f"{filename}_{suffix_index}"
    return path.with_name(filename)


def _reserve_batch_output_path(output_dir: Path, output_name: str, reserved: set[Path]) -> Path:
    candidate = output_dir / output_name
    if candidate not in reserved:
        reserved.add(candidate)
        return candidate

    suffix_index = 2
    max_suffix = 10_000
    while True:
        if suffix_index > max_suffix:
            raise RuntimeError(f"Too many filename collisions for {candidate}")
        deduped = _with_collision_suffix(candidate, suffix_index)
        if deduped not in reserved:
            reserved.add(deduped)
            return deduped
        suffix_index += 1


def _plan_batch_successful_outputs(
    input_paths: Sequence[Path],
    successful: Sequence[tuple[Path, OptimizationResult]],
    output_dir: Path,
    output_pattern: str,
    successful_input_indices: Sequence[int] | None = None,
) -> list[tuple[Path, OptimizationResult, Path]]:
    reserved_outputs: set[Path] = set()

    def reserve_output(path: Path, result: OptimizationResult) -> Path:
        ext = _get_extension(result.report.chosen_format)
        output_name = output_pattern.format(name=path.stem, ext=ext, format=result.report.chosen_format)
        return _reserve_batch_output_path(output_dir, output_name, reserved_outputs)

    if successful_input_indices is not None and len(successful_input_indices) != len(successful):
        import logging

        logging.getLogger(__name__).warning(
            "Input indices count (%d) != successful count (%d); falling back to path matching",
            len(successful_input_indices),
            len(successful),
        )
        successful_input_indices = None

    if successful_input_indices is not None and len(successful_input_indices) == len(successful):
        successful_by_index: dict[int, tuple[Path, OptimizationResult]] = {}
        for input_index, (path, result) in zip(successful_input_indices, successful, strict=True):
            if 0 <= input_index < len(input_paths):
                successful_by_index[input_index] = (path, result)

        planned: list[tuple[Path, OptimizationResult, Path]] = []
        for input_index, input_path in enumerate(input_paths):
            pair = successful_by_index.get(input_index)
            if pair is None:
                continue
            path, result = pair
            planned.append((path, result, reserve_output(input_path, result)))
        return planned

    successful_by_path: defaultdict[str, deque[tuple[Path, OptimizationResult]]] = defaultdict(deque)
    for path, result in successful:
        successful_by_path[str(path)].append((path, result))

    planned: list[tuple[Path, OptimizationResult, Path]] = []
    for input_path in input_paths:
        bucket = successful_by_path[str(input_path)]
        if not bucket:
            continue
        path, result = bucket.popleft()
        planned.append((path, result, reserve_output(input_path, result)))

    for bucket in successful_by_path.values():
        while bucket:
            path, result = bucket.popleft()
            planned.append((path, result, reserve_output(path, result)))

    return planned


def _progress_bar(progress: BatchProgress, *, show_errors: bool = True) -> None:
    bar_width = 40
    completed = progress.completed + progress.failed + progress.skipped
    ratio = completed / progress.total if progress.total > 0 else 0
    filled = int(bar_width * ratio)
    bar = "█" * filled + "░" * (bar_width - filled)

    status_parts = [f"[{bar}] {completed}/{progress.total} ({ratio:.0%})"]
    if progress.failed > 0 and show_errors:
        status_parts.append(f"❌ {progress.failed} failed")

    current = f"Processing: {progress.current_file}" if progress.current_file else ""
    print(f"\r{' '.join(status_parts)} {current}".ljust(100), end="", file=sys.stderr)
    if completed == progress.total:
        print(file=sys.stderr)


def _successful_report_rows(
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


def _batch_report_data(
    result: BatchResult,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> dict[str, object]:
    return {
        "total": result.total,
        "successful": len(result.successful),
        "failed": len(result.failed),
        "skipped": len(result.skipped),
        "success_rate": result.success_rate,
        "results": _successful_report_rows(result, successful_outputs),
        "errors": [{"input": str(path), "error": str(exc)} for path, exc in result.failed],
        "skipped_items": [{"input": str(path)} for path in result.skipped],
    }


def _batch_summary_text(result: BatchResult) -> str:
    lines = [
        f"Processed {result.total} images:",
        f"Successful: {len(result.successful)}",
    ]
    if result.failed:
        lines.append(f"Failed: {len(result.failed)}")
    if result.skipped:
        lines.append(f"Skipped: {len(result.skipped)}")
    if result.successful:
        total_before = sum(r.report.size_before_kb for _, r in result.successful)
        total_after = sum(r.report.size_after_kb for _, r in result.successful)
        if total_before > 0:
            reduction = (1 - total_after / total_before) * 100
            lines.append(f"Total reduction: {reduction:.1f}%")
    return "\n".join(lines)


def _write_batch_report(
    report_path: Path,
    result: BatchResult,
    report_format: str,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_format == "json":
        report_path.write_text(
            json.dumps(_batch_report_data(result, successful_outputs), indent=2),
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
        for row in _successful_report_rows(result, successful_outputs):
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

    report_path.write_text(_batch_summary_text(result), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="perceptimg: perceptual image optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    perceptimg input.png --max-size-kb 100 --min-ssim 0.9

  Batch processing:
    perceptimg --batch *.png --output-dir ./optimized --max-workers 4

  Estimate batch size:
    perceptimg --batch *.png --estimate

  With custom output pattern:
    perceptimg --batch images/*.jpg --output-pattern {name}_opt.{ext}
""",
    )
    parser.add_argument(
        "input",
        nargs="*",
        help="Input image path(s) or a directory",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode - process multiple images",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing images to process (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch mode",
    )
    parser.add_argument(
        "--output-pattern",
        type=str,
        default="{name}_optimized.{ext}",
        help="Output filename pattern: {name}, {ext}, {format} (batch mode)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum parallel workers for batch (default: CPU count)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        dest="continue_on_error",
        default=True,
        help="Continue processing on errors (batch mode)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_false",
        dest="continue_on_error",
        help="Stop batch processing on the first error",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate total size reduction without processing (batch mode)",
    )
    parser.add_argument(
        "--estimate-sample",
        type=int,
        default=3,
        help="Sample size for estimation (batch mode)",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output report path (format controlled by --report-format)",
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["json", "csv", "summary"],
        default="json",
        help="Report format for batch mode",
    )
    parser.add_argument("--policy", help="Path to policy JSON")
    parser.add_argument(
        "--out",
        required=False,
        help="Output path (single image only)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        default=False,
        help="Output logs in JSON format",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Log level (DEBUG, INFO, WARNING...)",
    )
    parser.add_argument("--max-size-kb", type=int, help="Maximum size in KB")
    parser.add_argument("--min-ssim", type=float, help="Minimum SSIM")
    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=0.7,
        help="SSIM weight for perceptual score (0-1)",
    )
    parser.add_argument(
        "--size-weight",
        type=float,
        default=0.3,
        help="Size reduction weight for perceptual score (0-1)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum candidate strategies (must be >= 1)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        help="Preferred format order, comma-separated (e.g., webp,avif,jpeg)",
    )
    parser.add_argument("--preserve-text", action="store_true", help="Prioritize text clarity")
    parser.add_argument("--preserve-faces", action="store_true", help="Prioritize faces")
    parser.add_argument("--allow-lossy", action="store_true", default=None, help="Allow lossy")
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Force lossless encoding",
    )
    parser.add_argument(
        "--target-use-case",
        type=str,
        choices=["web", "mobile", "print", "general"],
        help="Target use case",
    )
    parser.add_argument(
        "--prioritize-quality",
        action="store_true",
        help="Prioritize quality over size reduction",
    )
    parser.add_argument(
        "--no-cache-analysis",
        action="store_true",
        help="Disable analysis caching (batch mode)",
    )
    return parser.parse_args()


def _collect_batch_inputs(args: argparse.Namespace) -> list[Path]:
    input_paths = [Path(p) for p in args.input] if args.input else []
    input_dir = Path(args.input_dir) if args.input_dir else None
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".tiff", ".bmp"}

    if input_dir:
        if not input_dir.exists():
            print(f"Error: Batch input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        if not input_dir.is_dir():
            print(f"Error: --input-dir must be a directory: {input_dir}", file=sys.stderr)
            sys.exit(1)
        collected = sorted(
            (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions),
            key=lambda p: p.name,
        )
        if collected:
            return collected
        print(f"Error: No image files found in directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if input_paths:
        collected: list[Path] = []
        for path in input_paths:
            if path.is_file():
                collected.append(path)
            elif path.is_dir():
                collected.extend(
                    sorted(
                        (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in image_extensions),
                        key=lambda p: p.name,
                    )
                )
            else:
                print(f"Error: Batch input not found: {path}", file=sys.stderr)
                sys.exit(1)
        if collected:
            return collected

    print("Error: No valid input specified", file=sys.stderr)
    sys.exit(1)


def _process_batch(args: argparse.Namespace, policy: Policy) -> None:
    paths = _collect_batch_inputs(args)
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()

    if args.estimate:
        print(f"Estimating batch size for {len(paths)} images...", file=sys.stderr)
        estimate = estimate_batch_size(paths, policy, sample_size=args.estimate_sample)
        print(json.dumps(estimate, indent=2))
        return

    if args.report_format == "summary":
        print(f"Processing {len(paths)} images...", file=sys.stderr)

    def on_progress(progress: BatchProgress) -> None:
        if args.report_format != "csv":
            _progress_bar(progress)

    optimizer = Optimizer(
        metric_calculator=MetricCalculator(
            ssim_weight=args.ssim_weight,
            size_weight=args.size_weight,
        ),
        prioritize_quality=args.prioritize_quality,
    )
    optimizer.strategy_generator = StrategyGenerator(max_candidates=args.max_candidates)

    result = optimize_batch(
        paths,
        policy,
        max_workers=args.max_workers,
        on_progress=on_progress if args.report_format != "csv" else None,
        continue_on_error=args.continue_on_error,
        cache_analysis=not args.no_cache_analysis,
        optimizer=optimizer,
    )

    successful_outputs = _plan_batch_successful_outputs(
        paths,
        result.successful,
        output_dir,
        args.output_pattern,
        result.successful_input_indices,
    )
    try:
        for _, res, output_path in successful_outputs:
            _write_output(res, output_path)
    except OSError as exc:
        print(f"Error: Failed to write batch output: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.report:
        report_path = Path(args.report)
        try:
            _write_batch_report(report_path, result, args.report_format, successful_outputs)
        except OSError as exc:
            print(f"Error: Failed to write batch report: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.report_format == "summary":
        print(file=sys.stderr)
        print(_batch_summary_text(result), file=sys.stderr)


def _process_single(args: argparse.Namespace, policy: Policy) -> None:
    if not args.input:
        print("Error: Input file required for single image mode", file=sys.stderr)
        sys.exit(1)
    input_path = Path(args.input[0])
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    optimizer = Optimizer(
        metric_calculator=MetricCalculator(
            ssim_weight=args.ssim_weight,
            size_weight=args.size_weight,
        ),
        prioritize_quality=args.prioritize_quality,
    )
    optimizer.strategy_generator = StrategyGenerator(max_candidates=args.max_candidates)
    result = optimizer.optimize(input_path, policy)

    if args.out:
        output_path = Path(args.out)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}_optimized")
        output_path = output_path.with_suffix(f".{_get_extension(result.report.chosen_format)}")

    _write_output(result, output_path)
    print(json.dumps(result.report.to_dict(), indent=2))


def main() -> None:
    args = _parse_args()
    logging_config.configure_logging(
        json_output=args.log_json,
        level=args.log_level.upper(),
        merge=True,
    )

    if args.allow_lossy and args.lossless:
        print(
            "Warning: Both --allow-lossy and --lossless specified. "
            "Using --lossless (lossless mode).",
            file=sys.stderr,
        )
        args.allow_lossy = False

    if args.max_candidates < 1:
        print(
            f"Error: --max-candidates must be >= 1, got {args.max_candidates}",
            file=sys.stderr,
        )
        sys.exit(1)

    policy = _policy_from_flags(args)

    if args.input_dir is not None and args.input:
        print(
            "Error: Use either --input-dir or positional input paths, not both",
            file=sys.stderr,
        )
        sys.exit(1)

    inputs = [Path(p) for p in args.input] if args.input else []
    is_batch = (
        args.batch
        or args.input_dir is not None
        or len(inputs) > 1
        or any(path.is_dir() for path in inputs)
    )

    if is_batch:
        if args.out:
            print("Error: --out is only supported for single image mode", file=sys.stderr)
            sys.exit(1)
        _process_batch(args, policy)
    else:
        if len(inputs) != 1:
            print("Error: Input file required for single image mode", file=sys.stderr)
            sys.exit(1)
        _process_single(args, policy)


if __name__ == "__main__":
    main()
