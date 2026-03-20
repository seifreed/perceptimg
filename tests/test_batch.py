"""Tests for batch processing functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from perceptimg import Policy
from perceptimg.core.batch import (
    AnalysisCache,
    BatchProgress,
    BatchResult,
    estimate_batch_size,
    optimize_batch,
    optimize_lazy,
)
from perceptimg.core.optimizer import Optimizer


def create_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    img = Image.new("RGB", size, "red")
    img.save(path)


class TestAnalysisCache:
    def test_cache_hit_same_image(self, tmp_path: Path) -> None:
        path = tmp_path / "test.png"
        create_test_image(path)
        image = Image.open(path)

        cache = AnalysisCache(maxsize=10)
        analysis = Optimizer().analyzer.analyze(image)

        cache.set(image, analysis, path)
        cached = cache.get(image, path)

        assert cached is not None
        assert cached.edge_density == analysis.edge_density

    def test_cache_miss_different_image(self, tmp_path: Path) -> None:
        path = tmp_path / "test.png"
        create_test_image(path)
        image = Image.open(path)

        cache = AnalysisCache(maxsize=10)
        cached = cache.get(image, path)

        assert cached is None

    def test_cache_eviction(self, tmp_path: Path) -> None:
        cache = AnalysisCache(maxsize=2)

        for i in range(3):
            path = tmp_path / f"test{i}.png"
            img = Image.new("RGB", (100 + i * 10, 100 + i * 10), f"#{i:02x}{i:02x}{i:02x}")
            img.save(path)
            image = Image.open(path)
            analysis = Optimizer().analyzer.analyze(image)
            cache.set(image, analysis, path)

        assert len(cache._cache) == 2

    def test_cache_clear(self, tmp_path: Path) -> None:
        path = tmp_path / "test.png"
        create_test_image(path)
        image = Image.open(path)

        cache = AnalysisCache(maxsize=10)
        analysis = Optimizer().analyzer.analyze(image)
        cache.set(image, analysis, path)

        cache.clear()
        assert len(cache._cache) == 0


class TestBatchProgress:
    def test_success_rate_zero_total(self) -> None:
        progress = BatchProgress(total=0, completed=0, failed=0)
        assert progress.success_rate == 0.0

    def test_success_rate_half(self) -> None:
        progress = BatchProgress(total=10, completed=5, failed=5)
        assert progress.success_rate == 0.5

    def test_success_rate_full(self) -> None:
        progress = BatchProgress(total=10, completed=10, failed=0)
        assert progress.success_rate == 1.0


class TestBatchResult:
    def test_total_count(self, tmp_path: Path) -> None:
        from PIL import Image

        from perceptimg.core.optimizer import OptimizationResult
        from perceptimg.core.report import OptimizationReport

        report = OptimizationReport(
            chosen_format="webp",
            quality=80,
            size_before_kb=10.0,
            size_after_kb=5.0,
            ssim=0.95,
            psnr=40.0,
            perceptual_score=0.9,
        )
        img = Image.new("RGB", (10, 10))
        result = BatchResult(
            successful=[
                (Path("a.png"), OptimizationResult(image_bytes=b"", image=img, report=report))
            ],
            failed=[(Path("b.png"), Exception("err"))],
        )
        assert result.total == 2

    def test_success_rate(self, tmp_path: Path) -> None:
        from PIL import Image

        from perceptimg.core.optimizer import OptimizationResult
        from perceptimg.core.report import OptimizationReport

        report = OptimizationReport(
            chosen_format="webp",
            quality=80,
            size_before_kb=10.0,
            size_after_kb=5.0,
            ssim=0.95,
            psnr=40.0,
            perceptual_score=0.9,
        )
        img = Image.new("RGB", (10, 10))
        result = BatchResult(
            successful=[
                (Path("a.png"), OptimizationResult(image_bytes=b"", image=img, report=report))
            ],
            failed=[(Path("b.png"), Exception("err"))],
        )
        assert result.success_rate == 0.5


class TestOptimizeBatch:
    def test_batch_all_success(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500, min_ssim=0.5)

        result = optimize_batch(paths, policy, max_workers=2)

        assert len(result.successful) == 3
        assert len(result.failed) == 0
        assert result.success_rate == 1.0

    def test_batch_partial_failure(self, tmp_path: Path) -> None:
        for i in range(2):
            create_test_image(tmp_path / f"img{i}.png")

        nonexistent = tmp_path / "nonexistent.png"
        paths = [tmp_path / "img0.png", nonexistent, tmp_path / "img1.png"]
        policy = Policy(max_size_kb=500, min_ssim=0.5)

        result = optimize_batch(paths, policy, continue_on_error=True)

        assert len(result.successful) == 2
        assert len(result.failed) == 1
        assert result.success_rate == 2 / 3

    def test_batch_progress_callback(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)
        progress_calls: list[BatchProgress] = []

        optimize_batch(paths, policy, on_progress=lambda p: progress_calls.append(p))

        assert len(progress_calls) == 3
        assert progress_calls[-1].completed == 3

    def test_batch_stop_on_error(self, tmp_path: Path) -> None:
        create_test_image(tmp_path / "good.png")
        paths = [tmp_path / "good.png", tmp_path / "bad.png"]
        policy = Policy(max_size_kb=500)

        result = optimize_batch(paths, policy, continue_on_error=False)

        assert len(result.successful) <= 1
        assert len(result.failed) >= 1

    def test_batch_with_cache(self, tmp_path: Path) -> None:
        create_test_image(tmp_path / "img.png")
        paths = [tmp_path / "img.png"]
        policy = Policy(max_size_kb=500)

        with patch("perceptimg.core.batch.processor.AnalysisCache") as mock_cache:
            mock_cache.return_value.get.return_value = None
            result = optimize_batch(paths, policy, cache_analysis=True)
            mock_cache.return_value.set.assert_called()

        assert len(result.successful) == 1


class TestOptimizeLazy:
    def test_lazy_yields_results(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)

        results = list(optimize_lazy(paths, policy))

        assert len(results) == 3
        successful = [r for r in results if not isinstance(r[1], Exception)]
        assert len(successful) == 3

    def test_lazy_handles_errors(self, tmp_path: Path) -> None:
        create_test_image(tmp_path / "good.png")
        paths = [tmp_path / "good.png", tmp_path / "bad.png"]
        policy = Policy(max_size_kb=500)

        results = list(optimize_lazy(paths, policy))

        assert len(results) == 2
        errors = [r for r in results if isinstance(r[1], Exception)]
        assert len(errors) == 1


class TestEstimateBatchSize:
    def test_estimate_from_sample(self, tmp_path: Path) -> None:
        for i in range(5):
            create_test_image(tmp_path / f"img{i}.png", size=(200, 200))

        paths = [tmp_path / f"img{i}.png" for i in range(5)]
        policy = Policy(max_size_kb=500)

        estimate = estimate_batch_size(paths, policy, sample_size=3)

        assert "estimated_total_kb_before" in estimate
        assert "estimated_total_kb_after" in estimate
        assert "estimated_reduction_percent" in estimate
        assert estimate["sample_size"] == 3

    def test_estimate_all_if_few(self, tmp_path: Path) -> None:
        create_test_image(tmp_path / "img.png")

        paths = [tmp_path / "img.png"]
        policy = Policy(max_size_kb=500)

        estimate = estimate_batch_size(paths, policy, sample_size=5)

        assert estimate["sample_size"] == 1
