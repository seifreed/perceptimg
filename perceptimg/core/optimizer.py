"""Optimization orchestrator."""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..engines.apng_engine import ApngEngine
from ..engines.avif_engine import AvifEngine
from ..engines.base import EngineResult, OptimizationEngine
from ..engines.heif_engine import HeifEngine
from ..engines.jxl_engine import JxlEngine
from ..engines.pillow_engine import PillowEngine
from ..engines.webp_engine import WebPEngine
from ..exceptions import OptimizationError
from ..utils import heuristics, image_io
from ..utils.image_io import bytes_to_image, load_image
from .analyzer import AnalysisResult, Analyzer
from .metrics import MetricCalculator, MetricResult
from .policy import Policy
from .report import OptimizationReport
from .strategy import StrategyCandidate, StrategyGenerator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OptimizationResult:
    """Final optimized artifact plus report."""

    image_bytes: bytes
    image: Image.Image
    report: OptimizationReport


class Optimizer:
    """Coordinates analysis, strategy generation, encoding, and selection."""

    _registry_lock = threading.Lock()

    def __init__(
        self,
        engines: Sequence[OptimizationEngine] | None = None,
        metric_calculator: MetricCalculator | None = None,
        analyzer: Analyzer | None = None,
        heuristic_config: heuristics.HeuristicConfig | None = None,
        prioritize_quality: bool = False,
    ) -> None:
        self._thread_local = threading.local()
        base_engines = engines or [
            JxlEngine(),
            AvifEngine(),
            WebPEngine(),
            HeifEngine(),
            ApngEngine(),
            PillowEngine(),
        ]
        self.engines: list[OptimizationEngine] = list(base_engines)
        self.engine_registry: dict[str, list[OptimizationEngine]] = {}
        for engine in self.engines:
            for fmt in self._formats_for(engine):
                self.engine_registry.setdefault(fmt, []).append(engine)
        self._sort_registry()
        config = heuristic_config if heuristic_config is not None else heuristics.HeuristicConfig()
        self.analyzer = analyzer or Analyzer(config=config)
        self.strategy_generator = StrategyGenerator()
        self.metric_calculator = metric_calculator or MetricCalculator()
        self.prioritize_quality = prioritize_quality

    @property
    def _last_engine_errors(self) -> list[str]:
        """Thread-local storage for engine errors."""
        errors = getattr(self._thread_local, "errors", None)
        if errors is None:
            self._thread_local.errors = []
        return self._thread_local.errors

    @_last_engine_errors.setter
    def _last_engine_errors(self, value: list[str]) -> None:
        self._thread_local.errors = value

    def _build_passthrough_result(
        self,
        original_image: Image.Image,
        original_bytes: bytes,
        policy: Policy,
        analysis: AnalysisResult,
        *,
        fallback_format: str | None = None,
    ) -> OptimizationResult | None:
        """Return the original image unchanged when no optimization improves size.

        Returns None if the original itself violates policy constraints like
        max_size_kb, so the caller can raise an appropriate error.
        """
        size_kb = len(original_bytes) / 1024.0
        if policy.max_size_kb is not None and size_kb > policy.max_size_kb:
            return None
        original_format = (original_image.format or fallback_format or "png").lower()
        perceptual_score = self.metric_calculator._perceptual_score(1.0, size_kb, size_kb)
        report = OptimizationReport(
            chosen_format=original_format,
            quality=None,
            size_before_kb=size_kb,
            size_after_kb=size_kb,
            ssim=1.0,
            psnr=100.0,
            perceptual_score=perceptual_score,
            reasons=["already_optimal"],
            policy=policy,
            analysis=analysis,
        )
        return OptimizationResult(
            image_bytes=original_bytes,
            image=original_image,
            report=report,
        )

    def optimize(self, image_path: str | Path, policy: Policy) -> OptimizationResult:
        self._last_engine_errors = []
        original_bytes = Path(image_path).read_bytes()
        original_image = load_image(image_path)
        analysis = self.analyzer.analyze(original_image)
        strategies = self._generate_strategies(policy, analysis)
        candidates = self._evaluate_candidates(original_image, original_bytes, strategies, policy)
        if not candidates:
            if self._all_rejected_larger:
                passthrough = self._build_passthrough_result(
                    original_image, original_bytes, policy, analysis,
                    fallback_format=Path(image_path).suffix.lstrip("."),
                )
                if passthrough is not None:
                    return passthrough
            error_msg = "No candidate met policy requirements"
            if self._last_engine_errors:
                error_msg += f". Engine errors: {'; '.join(self._last_engine_errors)}"
            raise OptimizationError(error_msg)
        chosen_metrics, chosen_candidate, engine_result = self._select_best(candidates)
        optimized_image = bytes_to_image(engine_result.data)
        report = OptimizationReport(
            chosen_format=engine_result.format,
            quality=engine_result.quality,
            size_before_kb=chosen_metrics.size_before_kb,
            size_after_kb=chosen_metrics.size_after_kb,
            ssim=chosen_metrics.ssim,
            psnr=chosen_metrics.psnr,
            perceptual_score=chosen_metrics.perceptual_score,
            reasons=chosen_candidate.reasons + ["policy_satisfied"],
            policy=policy,
            analysis=analysis,
            candidate=chosen_candidate,
        )
        return OptimizationResult(
            image_bytes=engine_result.data,
            image=optimized_image,
            report=report,
        )

    def optimize_from_analysis(
        self,
        image: Image.Image,
        analysis_result: AnalysisResult,
        policy: Policy,
        *,
        original_bytes: bytes | None = None,
    ) -> OptimizationResult:
        """Optimize using a precomputed analysis (public API)."""
        self._last_engine_errors = []
        bytes_in = original_bytes or image_io.image_to_bytes(image, format="PNG")
        strategies = self._generate_strategies(policy, analysis_result)
        candidates = self._evaluate_candidates(image, bytes_in, strategies, policy)
        if not candidates:
            if self._all_rejected_larger:
                passthrough = self._build_passthrough_result(
                    image, bytes_in, policy, analysis_result
                )
                if passthrough is not None:
                    return passthrough
            error_msg = "No candidate met policy requirements"
            if self._last_engine_errors:
                error_msg += f". Engine errors: {'; '.join(self._last_engine_errors)}"
            raise OptimizationError(error_msg)
        chosen_metrics, chosen_candidate, engine_result = self._select_best(candidates)
        optimized_image = bytes_to_image(engine_result.data)
        report = OptimizationReport(
            chosen_format=engine_result.format,
            quality=engine_result.quality,
            size_before_kb=chosen_metrics.size_before_kb,
            size_after_kb=chosen_metrics.size_after_kb,
            ssim=chosen_metrics.ssim,
            psnr=chosen_metrics.psnr,
            perceptual_score=chosen_metrics.perceptual_score,
            reasons=chosen_candidate.reasons + ["policy_satisfied"],
            policy=policy,
            analysis=analysis_result,
            candidate=chosen_candidate,
        )
        return OptimizationResult(
            image_bytes=engine_result.data,
            image=optimized_image,
            report=report,
        )

    def _generate_strategies(
        self, policy: Policy, analysis: AnalysisResult
    ) -> list[StrategyCandidate]:
        """Generate strategies while preserving compatibility with custom generators."""
        generate = self.strategy_generator.generate
        try:
            params = inspect.signature(generate).parameters
        except (TypeError, ValueError):
            params = {}

        if "available_formats" in params:
            available_formats = {
                fmt
                for engine in self.engines
                if engine.is_available
                for fmt in self._formats_for(engine)
                if fmt
            }
            return generate(policy, analysis, available_formats=available_formats)
        return generate(policy, analysis)

    def _evaluate_candidates(
        self,
        image: Image.Image,
        original_bytes: bytes,
        strategies: Iterable[StrategyCandidate],
        policy: Policy,
    ) -> list[tuple[MetricResult, StrategyCandidate, EngineResult]]:
        """Evaluate all candidates and return valid ones."""
        candidates: list[tuple[MetricResult, StrategyCandidate, EngineResult]] = []
        self._all_rejected_larger = False
        had_results = False
        all_larger = True
        for strategy in strategies:
            result, errors = self._try_engines(image, strategy)
            self._last_engine_errors.extend(errors)
            if result is None:
                continue
            had_results = True
            metrics = self.metric_calculator.compute(
                original=image,
                optimized=bytes_to_image(result.data),
                original_bytes=original_bytes,
                optimized_bytes=result.data,
            )
            if self._satisfies_policy(metrics, policy, strategy):
                candidates.append((metrics, strategy, result))
            elif metrics.size_after_kb <= metrics.size_before_kb:
                # Rejected for reasons other than size increase
                all_larger = False
        self._all_rejected_larger = had_results and not candidates and all_larger
        return candidates

    def _try_engines(
        self, image: Image.Image, strategy: StrategyCandidate
    ) -> tuple[EngineResult | None, list[str]]:
        """Try all engines for this format, return first success."""
        with Optimizer._registry_lock:
            engines = list(self.engine_registry.get(strategy.format.lower(), []))
        errors: list[str] = []
        for engine in engines:
            if not engine.is_available:
                continue
            try:
                return engine.optimize(image, strategy), errors
            except OptimizationError as exc:
                errors.append(f"{engine.__class__.__name__}: {exc}")
                logger.warning(
                    "Engine failed",
                    extra={
                        "format": strategy.format,
                        "engine": engine.__class__.__name__,
                        "error": str(exc),
                    },
                )
            except Exception as exc:
                errors.append(f"{engine.__class__.__name__}: {exc}")
                logger.error(
                    "Engine crashed unexpectedly",
                    extra={
                        "format": strategy.format,
                        "engine": engine.__class__.__name__,
                        "error": str(exc),
                    },
                )
        return None, errors

    def _select_best(
        self, candidates: list[tuple[MetricResult, StrategyCandidate, EngineResult]]
    ) -> tuple[MetricResult, StrategyCandidate, EngineResult]:
        """Select the best candidate from evaluated options."""
        if self.prioritize_quality:
            return max(candidates, key=lambda c: (c[0].ssim, -c[0].size_after_kb))
        return max(candidates, key=lambda c: (c[0].perceptual_score, -c[0].size_after_kb))

    def _satisfies_policy(
        self, metrics: MetricResult, policy: Policy, strategy: StrategyCandidate
    ) -> bool:
        """Check if metrics satisfy policy constraints."""
        if metrics.size_after_kb <= 0:
            return False
        if metrics.perceptual_score < 0.0:
            return False
        if policy.max_size_kb is not None and metrics.size_after_kb > policy.max_size_kb:
            return False
        if policy.min_ssim is not None and metrics.ssim < policy.min_ssim:
            return False
        if not policy.allow_lossy and not strategy.lossless:
            return False
        return True

    def evaluate_candidates_for_test(
        self,
        image: Image.Image,
        strategies: Iterable[StrategyCandidate],
        policy: Policy,
    ) -> list[tuple[MetricResult, StrategyCandidate, EngineResult]]:
        """Test helper to evaluate strategy candidates against a policy."""
        original_bytes = image_io.image_to_bytes(image, format="PNG")
        return self._evaluate_candidates(image, original_bytes, strategies, policy)

    @staticmethod
    def _formats_for(engine: OptimizationEngine) -> tuple[str, ...]:
        """Get supported formats for an engine."""
        known = getattr(engine, "SUPPORTED", None)
        if known:
            return tuple(str(fmt).lower() for fmt in known)
        return (getattr(engine, "format", ""),)

    def _sort_registry(self) -> None:
        """Sort engines by priority in descending order."""
        for fmt, engines in self.engine_registry.items():
            self.engine_registry[fmt] = sorted(engines, key=lambda e: e.priority, reverse=True)


def optimize(image_path: str | Path, policy: Policy) -> OptimizationResult:
    """Convenience function using the default optimizer stack."""
    return Optimizer().optimize(image_path, policy)


def optimize_image(
    image: Image.Image,
    policy: Policy,
    *,
    optimizer: Optimizer | None = None,
    original_bytes: bytes | None = None,
) -> OptimizationResult:
    """Optimize a PIL image in-memory without touching disk."""
    opt = optimizer or Optimizer()
    analysis = opt.analyzer.analyze(image)
    return opt.optimize_from_analysis(image, analysis, policy, original_bytes=original_bytes)


def optimize_bytes(
    data: bytes,
    policy: Policy,
    *,
    optimizer: Optimizer | None = None,
    original_format: str | None = None,
) -> OptimizationResult:
    """Optimize raw image bytes."""
    image = bytes_to_image(data)
    opt = optimizer or Optimizer()
    analysis = opt.analyzer.analyze(image)
    return opt.optimize_from_analysis(image, analysis, policy, original_bytes=data)


def register_engine(optimizer: Optimizer, engine: OptimizationEngine) -> None:
    """Register a custom engine for extensibility.

    Thread-safe: Uses a class-level lock to prevent concurrent modifications
    to the engine registry.
    """
    with Optimizer._registry_lock:
        optimizer.engines.append(engine)
        for fmt in Optimizer._formats_for(engine):
            engines = optimizer.engine_registry.setdefault(fmt, [])
            engines.append(engine)
            if len(engines) > 1:
                logger.info(
                    "Multiple engines registered for format",
                    extra={"format": fmt, "count": len(engines)},
                )
        optimizer._sort_registry()
