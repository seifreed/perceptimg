"""CLI for perceptimg with single and batch processing support."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .api import (
    BatchProgress,
    BatchResult,
    OptimizationResult,
    Policy,
    estimate_batch_size,
    optimize_batch,
)
from .api import (
    _batch_report_data as _api_batch_report_data,
)
from .api import (
    _batch_summary_text as _api_batch_summary_text,
)
from .api import (
    _build_optimizer as _api_build_optimizer,
)
from .api import (
    _load_policy as _api_load_policy,
)
from .api import (
    _plan_batch_successful_outputs as _api_plan_batch_successful_outputs,
)
from .api import (
    _policy_from_flags as _api_policy_from_flags,
)
from .api import (
    _resolve_output_extension as _api_resolve_output_extension,
)
from .api import (
    _write_batch_report as _api_write_batch_report,
)
from .core.optimizer import Optimizer
from .utils import logging_config


def _load_policy(policy_path: Path) -> Policy:
    return _api_load_policy(policy_path)


def _policy_from_flags(args: argparse.Namespace) -> Policy:
    return _api_policy_from_flags(args)


def _build_optimizer(
    *,
    ssim_weight: float,
    size_weight: float,
    prioritize_quality: bool,
    max_candidates: int,
) -> Optimizer:
    return _api_build_optimizer(
        ssim_weight=ssim_weight,
        size_weight=size_weight,
        prioritize_quality=prioritize_quality,
        max_candidates=max_candidates,
    )


def _resolve_output_extension(format_name: str) -> str:
    return _api_resolve_output_extension(format_name)


def _plan_batch_successful_outputs(
    input_paths: Sequence[Path],
    successful: Sequence[tuple[Path, OptimizationResult]],
    output_dir: Path,
    output_pattern: str,
    successful_input_indices: Sequence[int] | None = None,
) -> list[tuple[Path, OptimizationResult, Path]]:
    return _api_plan_batch_successful_outputs(
        input_paths,
        successful,
        output_dir,
        output_pattern,
        successful_input_indices=successful_input_indices,
    )


def _batch_report_data(
    result: BatchResult,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> dict[str, object]:
    return _api_batch_report_data(result, successful_outputs=successful_outputs)


def _batch_summary_text(result: BatchResult) -> str:
    return _api_batch_summary_text(result)


def _write_batch_report(
    report_path: Path,
    result: BatchResult,
    report_format: str,
    successful_outputs: Sequence[tuple[Path, OptimizationResult, Path]] | None = None,
) -> None:
    _api_write_batch_report(
        report_path,
        result,
        report_format,
        successful_outputs=successful_outputs,
    )


def _write_output(result: OptimizationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.image_bytes)


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
        collected_paths = sorted(
            (
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ),
            key=lambda p: p.name,
        )
        if collected_paths:
            return collected_paths
        print(f"Error: No image files found in directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if input_paths:
        collected_items: list[Path] = []
        for path in input_paths:
            if path.is_file():
                collected_items.append(path)
            elif path.is_dir():
                collected_items.extend(
                    sorted(
                        (
                            p
                            for p in path.iterdir()
                            if p.is_file() and p.suffix.lower() in image_extensions
                        ),
                        key=lambda p: p.name,
                    )
                )
            else:
                print(f"Error: Batch input not found: {path}", file=sys.stderr)
                sys.exit(1)
        if collected_items:
            return collected_items

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

    optimizer = _build_optimizer(
        ssim_weight=args.ssim_weight,
        size_weight=args.size_weight,
        prioritize_quality=args.prioritize_quality,
        max_candidates=args.max_candidates,
    )

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

    optimizer = _build_optimizer(
        ssim_weight=args.ssim_weight,
        size_weight=args.size_weight,
        prioritize_quality=args.prioritize_quality,
        max_candidates=args.max_candidates,
    )
    result = optimizer.optimize(input_path, policy)

    if args.out:
        output_path = Path(args.out)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}_optimized")
        output_path = output_path.with_suffix(
            f".{_resolve_output_extension(result.report.chosen_format)}"
        )

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
