"""Core batch processor with template method pattern."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

from ...exceptions import ImageLoadError, OptimizationError
from ...utils.image_io import load_image
from ..optimizer import OptimizationResult, Optimizer
from ..policy import Policy
from .cache import AnalysisCache
from .config import BatchConfig, BatchHooks, BatchProgress, BatchResult

ProcessResult = tuple[Path, OptimizationResult | Exception]


class BatchProcessor:
    """Core batch processing logic extracted to follow DRY principle.

    This class implements the Template Method pattern, allowing
    customization through hooks while keeping the core algorithm fixed.
    """

    def __init__(
        self,
        optimizer: Optimizer | None = None,
        cache: AnalysisCache | None = None,
    ):
        self._optimizer = optimizer or Optimizer()
        self._cache = cache

    def process_single(
        self,
        image_path: Path,
        policy: Policy,
        cache: AnalysisCache | None = None,
    ) -> tuple[Path, OptimizationResult | Exception]:
        """Process a single image.

        Args:
            image_path: Path to image file.
            policy: Optimization policy.
            cache: Optional analysis cache.

        Returns:
            Tuple of (path, result or exception).
        """
        try:
            if cache:
                image = load_image(image_path)
                cached_analysis = cache.get(image, image_path)
                if cached_analysis:
                    result = self._optimizer.optimize_from_analysis(
                        image, cached_analysis, policy, original_bytes=image_path.read_bytes()
                    )
                else:
                    analysis = self._optimizer.analyzer.analyze(image)
                    cache.set(image, analysis, image_path)
                    result = self._optimizer.optimize_from_analysis(
                        image, analysis, policy, original_bytes=image_path.read_bytes()
                    )
            else:
                result = self._optimizer.optimize(image_path, policy)
            return (image_path, result)
        except (OptimizationError, ImageLoadError, OSError, ValueError) as exc:
            return (image_path, exc)

    def execute(
        self,
        image_paths: Sequence[str | Path],
        config: BatchConfig,
        hooks: BatchHooks | None = None,
    ) -> BatchResult:
        """Execute batch processing with common pattern.

        This is the Template Method - the core algorithm that remains fixed
        while hooks allow customization.

        Args:
            image_paths: List of image paths to process.
            config: Batch configuration.
            hooks: Optional hooks for customization.

        Returns:
            BatchResult with successful and failed results.
        """
        hooks = hooks or BatchHooks()
        paths = [Path(p) for p in image_paths]
        progress = BatchProgress(total=len(paths), completed=0, failed=0)
        cache = AnalysisCache(maxsize=config.cache_maxsize) if config.cache_analysis else None

        successful: list[tuple[Path, OptimizationResult]] = []
        failed: list[tuple[Path, Exception]] = []
        successful_input_indices: list[int] = []
        failed_input_indices: list[int] = []
        progress_lock = threading.Lock()

        def process_with_hooks(image_path: Path) -> tuple[Path, OptimizationResult | Exception]:
            if hooks.rate_limiter:
                if not hooks.rate_limiter.acquire():
                    result: tuple[Path, OptimizationResult | Exception] = (
                        image_path,
                        TimeoutError("Rate limiter acquire timed out"),
                    )
                    if hooks.on_image_error:
                        hooks.on_image_error(image_path, result[1])
                    return result

            if hooks.on_image_start:
                hooks.on_image_start(image_path)
            result = self.process_single(image_path, config.policy, cache)

            if isinstance(result[1], Exception):
                if hooks.on_image_error:
                    hooks.on_image_error(image_path, result[1])
            else:
                if hooks.on_image_complete:
                    hooks.on_image_complete(image_path, result[1])

            return result

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(process_with_hooks, p): (index, p) for index, p in enumerate(paths)}
            abort_requested = False

            for future in as_completed(futures):
                input_index, path = futures[future]
                snapshot = None
                try:
                    _, result_or_exc = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    with progress_lock:
                        progress.current_file = str(path)
                        failed.append((path, exc))
                        failed_input_indices.append(input_index)
                        progress.failed += 1
                        progress.errors.append(f"{path}: {exc}")
                        if hooks.on_progress:
                            progress.sequence += 1
                            snapshot = progress.snapshot()
                else:
                    with progress_lock:
                        progress.current_file = str(path)
                        if isinstance(result_or_exc, Exception):
                            failed.append((path, result_or_exc))
                            failed_input_indices.append(input_index)
                            progress.failed += 1
                            progress.errors.append(f"{path}: {result_or_exc}")
                        else:
                            successful.append((path, result_or_exc))
                            successful_input_indices.append(input_index)
                            progress.completed += 1
                        if hooks.on_progress:
                            progress.sequence += 1
                            snapshot = progress.snapshot()

                if snapshot and hooks.on_progress:
                    hooks.on_progress(snapshot)

                if hooks.should_checkpoint and hooks.should_checkpoint():
                    if hooks.on_checkpoint:
                        hooks.on_checkpoint()

                if not config.continue_on_error and failed and not abort_requested:
                    abort_requested = True
                    executor.shutdown(wait=False, cancel_futures=True)

        return BatchResult(
            successful=successful,
            failed=failed,
            successful_input_indices=successful_input_indices,
            failed_input_indices=failed_input_indices,
        )
