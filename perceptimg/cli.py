"""CLI for perceptimg with single and batch processing support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import Policy
from .core.batch import BatchProgress, estimate_batch_size, optimize_batch
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


def _policy_from_flags(args: argparse.Namespace) -> Policy | None:
    if args.policy:
        return _load_policy(Path(args.policy))
    base = {
        "max_size_kb": args.max_size_kb,
        "min_ssim": args.min_ssim,
        "preserve_text": args.preserve_text,
        "preserve_faces": args.preserve_faces,
        "allow_lossy": (
            True if args.allow_lossy is None and not args.lossless else not args.lossless
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


def _progress_bar(progress: BatchProgress, *, show_errors: bool = True) -> None:
    bar_width = 40
    completed = progress.completed + progress.failed
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
    parser.add_argument("input", nargs="?", help="Input image or directory (for batch)")
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
        default=True,
        help="Continue processing on errors (batch mode)",
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
        help="Output JSON report path (batch mode)",
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
        help="Output path (single image)",
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
    input_path = Path(args.input) if args.input else None
    input_dir = Path(args.input_dir) if args.input_dir else None

    if input_path and input_path.is_file():
        return [input_path]

    if input_dir:
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".tiff", ".bmp"}
        return [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]

    if input_path and input_path.is_dir():
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".tiff", ".bmp"}
        return [p for p in input_path.iterdir() if p.suffix.lower() in image_extensions]

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

    result = optimize_batch(
        paths,
        policy,
        max_workers=args.max_workers,
        on_progress=on_progress if args.report_format != "csv" else None,
        continue_on_error=args.continue_on_error,
        cache_analysis=not args.no_cache_analysis,
    )

    if args.report:
        report_data = {
            "total": result.total,
            "successful": len(result.successful),
            "failed": len(result.failed),
            "success_rate": result.success_rate,
            "results": [
                {
                    "input": str(path),
                    "output_format": res.report.chosen_format,
                    "size_before_kb": res.report.size_before_kb,
                    "size_after_kb": res.report.size_after_kb,
                    "ssim": res.report.ssim,
                    "psnr": res.report.psnr,
                }
                for path, res in result.successful
            ],
            "errors": [{"input": str(path), "error": str(exc)} for path, exc in result.failed],
        }
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report_data, indent=2))

    for path, res in result.successful:
        ext = _get_extension(res.report.chosen_format)
        name = path.stem
        output_name = args.output_pattern.format(
            name=name, ext=ext, format=res.report.chosen_format
        )
        output_path = output_dir / output_name
        _write_output(res, output_path)

    if args.report_format == "summary":
        print(f"\nProcessed {result.total} images:", file=sys.stderr)
        print(f"  ✅ Successful: {len(result.successful)}", file=sys.stderr)
        if result.failed:
            print(f"  ❌ Failed: {len(result.failed)}", file=sys.stderr)
        if result.successful:
            total_before = sum(r.report.size_before_kb for _, r in result.successful)
            total_after = sum(r.report.size_after_kb for _, r in result.successful)
            if total_before > 0:
                reduction = (1 - total_after / total_before) * 100
                print(f"  📉 Total reduction: {reduction:.1f}%", file=sys.stderr)


def _process_single(args: argparse.Namespace, policy: Policy) -> None:
    input_path = Path(args.input)
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

    if args.max_candidates < 1:
        print(
            f"Error: --max-candidates must be >= 1, got {args.max_candidates}",
            file=sys.stderr,
        )
        sys.exit(1)

    policy = _policy_from_flags(args)
    if policy is None:
        policy = Policy()

    if args.batch or args.input_dir or (args.input and Path(args.input).is_dir()):
        _process_batch(args, policy)
    else:
        if not args.input:
            print("Error: Input file required for single image mode", file=sys.stderr)
            sys.exit(1)
        _process_single(args, policy)


if __name__ == "__main__":
    main()
