"""Tests for batch processing functionality."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.application.batch import (
    estimate_batch_size,
    optimize_batch,
    optimize_lazy,
)
from perceptimg.core.batch import (
    AnalysisCache,
    BatchProgress,
    BatchResult,
)
from perceptimg.core.batch.config import BatchConfig, BatchHooks
from perceptimg.core.batch.processor import BatchProcessor
from perceptimg.core.optimizer import OptimizationResult, Optimizer
from perceptimg.core.rate_limiter import RateLimiter
from perceptimg.core.report import OptimizationReport


def create_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    img = Image.new("RGB", size, "red")
    img.save(path)


def make_stub_result() -> OptimizationResult:
    image = Image.new("RGB", (1, 1), "red")
    report = OptimizationReport(
        chosen_format="png",
        quality=None,
        size_before_kb=1.0,
        size_after_kb=1.0,
        ssim=1.0,
        psnr=99.0,
        perceptual_score=1.0,
        reasons=[],
    )
    return OptimizationResult(image_bytes=b"x", image=image, report=report)


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

    def test_total_count_includes_skipped(self) -> None:
        result = BatchResult(
            successful=[],
            failed=[],
            skipped=[Path("a.png"), Path("b.png")],
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

    def test_success_rate_with_skipped(self, tmp_path: Path) -> None:
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
            skipped=[Path("c.png")],
        )
        # 1 success / (1 success + 1 failed + 1 skipped) = 1/3
        assert abs(result.success_rate - 1 / 3) < 1e-9


class TestBatchProgressWithSkipped:
    def test_success_rate_includes_skipped(self) -> None:
        progress = BatchProgress(total=10, completed=5, failed=3, skipped=2)
        # 5 / (5 + 3 + 2) = 0.5
        assert progress.success_rate == 0.5

    def test_snapshot_copies_skipped(self) -> None:
        progress = BatchProgress(total=10, completed=5, failed=3, skipped=2)
        snap = progress.snapshot()
        assert snap.skipped == 2


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

    def test_batch_progress_callback_receives_snapshots(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        progress_calls: list[BatchProgress] = []
        optimize_batch(
            [tmp_path / f"img{i}.png" for i in range(3)],
            Policy(max_size_kb=500),
            on_progress=lambda p: progress_calls.append(p),
        )

        assert [call.completed for call in progress_calls] == [1, 2, 3]
        assert len({id(call) for call in progress_calls}) == 3

    def test_batch_progress_reports_completed_file(self) -> None:
        processor = BatchProcessor()
        ok_result = make_stub_result()
        progress_events: list[tuple[str | None, int, int]] = []

        def fake_process_single(
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            if image_path.name == "slow.png":
                time.sleep(0.05)
            else:
                time.sleep(0.01)
            return (image_path, ok_result)

        processor.process_single = fake_process_single
        processor.execute(
            [Path("slow.png"), Path("fast.png")],
            BatchConfig(policy=Policy(max_size_kb=500), max_workers=2, cache_analysis=False),
            BatchHooks(
                on_progress=lambda progress: progress_events.append(
                    (progress.current_file, progress.completed, progress.failed)
                )
            ),
        )

        assert progress_events == [("fast.png", 1, 0), ("slow.png", 2, 0)]

    def test_batch_stop_on_error(self, tmp_path: Path) -> None:
        ok_result = make_stub_result()

        def fake_process_single(
            self: BatchProcessor,
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            if image_path.name == "bad.png":
                time.sleep(0.01)
                return (image_path, ValueError("boom"))
            time.sleep(0.05)
            return (image_path, ok_result)

        with patch.object(
            BatchProcessor,
            "process_single",
            autospec=True,
            side_effect=fake_process_single,
        ):
            result = optimize_batch(
                [Path("good1.png"), Path("bad.png"), Path("good2.png")],
                Policy(max_size_kb=500),
                max_workers=3,
                continue_on_error=False,
                cache_analysis=False,
            )

        assert {path.name for path, _ in result.successful} == {"good1.png", "good2.png"}
        assert [path.name for path, _ in result.failed] == ["bad.png"]

    def test_batch_processor_stop_on_error_keeps_hook_results_consistent(self) -> None:
        processor = BatchProcessor()
        ok_result = make_stub_result()
        hook_events: list[tuple[str, str]] = []

        def fake_process_single(
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            if image_path.name == "bad.png":
                time.sleep(0.01)
                return (image_path, ValueError("boom"))
            time.sleep(0.05)
            return (image_path, ok_result)

        processor.process_single = fake_process_single
        result = processor.execute(
            [Path("good1.png"), Path("bad.png"), Path("good2.png")],
            BatchConfig(
                policy=Policy(max_size_kb=500),
                max_workers=3,
                continue_on_error=False,
                cache_analysis=False,
            ),
            BatchHooks(
                on_image_complete=lambda path, _: hook_events.append(("ok", path.name)),
                on_image_error=lambda path, _: hook_events.append(("err", path.name)),
            ),
        )

        assert {path.name for path, _ in result.successful} == {"good1.png", "good2.png"}
        assert [path.name for path, _ in result.failed] == ["bad.png"]
        assert set(hook_events) == {("ok", "good1.png"), ("ok", "good2.png"), ("err", "bad.png")}

    def test_batch_processor_respects_rate_limiter_timeout(self) -> None:
        processor = BatchProcessor()
        processed: list[str] = []

        def fake_process_single(
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            processed.append(image_path.name)
            return (image_path, make_stub_result())

        processor.process_single = fake_process_single
        with patch.object(RateLimiter, "acquire", autospec=True, return_value=False):
            result = processor.execute(
                [Path("a.png")],
                BatchConfig(policy=Policy(max_size_kb=500), max_workers=1, cache_analysis=False),
                BatchHooks(rate_limiter=RateLimiter()),
            )

        assert processed == []
        assert result.successful == []
        assert len(result.failed) == 1
        assert result.failed[0][0].name == "a.png"
        assert isinstance(result.failed[0][1], TimeoutError)

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

    def test_estimate_rejects_non_positive_sample_size(self, tmp_path: Path) -> None:
        create_test_image(tmp_path / "img.png")

        with pytest.raises(ValueError, match="sample_size must be >= 1"):
            estimate_batch_size([tmp_path / "img.png"], Policy(max_size_kb=500), sample_size=0)

    def test_estimate_handles_missing_files(self, tmp_path: Path) -> None:
        """estimate_batch_size should not crash when some files don't exist."""
        create_test_image(tmp_path / "exists.png")
        paths = [tmp_path / "exists.png", tmp_path / "missing.png"]
        policy = Policy(max_size_kb=500)
        estimate = estimate_batch_size(paths, policy, sample_size=2)
        assert estimate["estimated_total_kb_before"] >= 0
        assert "estimated_total_kb_after" in estimate


class TestCheckpointThreadSafety:
    def test_concurrent_mark_completed_no_lost_updates(self, tmp_path: Path) -> None:
        """Concurrent mark_completed calls should not lose updates."""
        from perceptimg.core.checkpoint import CheckpointManager, JobResult, JobStatus

        manager = CheckpointManager(tmp_path / "checkpoint.json")
        n = 50
        paths = [f"image_{i}.png" for i in range(n)]
        manager.start(paths)

        def mark(path: str) -> None:
            manager.mark_completed(path, JobResult(path=path, status=JobStatus.COMPLETED))

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(mark, p) for p in paths]
            for f in as_completed(futures):
                f.result()

        stats = manager.get_stats()
        assert stats["completed"] == n
        assert stats["pending"] == 0
        assert len(manager.get_results()) == n

    def test_should_checkpoint_not_skipped_on_concurrent_completions(self, tmp_path: Path) -> None:
        """should_checkpoint must fire even if counter jumps past interval boundary."""
        from perceptimg.core.checkpoint import CheckpointManager, JobResult, JobStatus

        manager = CheckpointManager(tmp_path / "checkpoint.json")
        paths = [f"image_{i}.png" for i in range(15)]
        manager.start(paths)

        # Complete 11 images (crosses the interval=5 boundary at 5 and 10)
        for p in paths[:11]:
            manager.mark_completed(p, JobResult(path=p, status=JobStatus.COMPLETED))

        # should_checkpoint must detect that we crossed the boundary
        triggered = manager.should_checkpoint(5)
        assert triggered, "should_checkpoint must fire when count crosses interval boundary"

        # Calling again without new completions should not fire
        assert not manager.should_checkpoint(5)
