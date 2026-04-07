"""Tests for enterprise features: checkpoint, retry, rate limiting, metrics."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.batch import (
    optimize_batch_with_checkpoint,
    optimize_batch_with_metrics,
    optimize_batch_with_rate_limit,
    optimize_batch_with_retry,
)
from perceptimg.core.batch.processor import BatchProcessor
from perceptimg.core.checkpoint import CheckpointManager, JobResult, JobStatus
from perceptimg.core.optimizer import OptimizationResult, Optimizer
from perceptimg.core.metrics_exporter import MetricsCollector, PrometheusMetricsExporter
from perceptimg.core.rate_limiter import RateLimitConfig, RateLimiter
from perceptimg.core.report import OptimizationReport
from perceptimg.core.retry import RetryConfig, RetryPolicy


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


class TestCheckpointManager:
    def test_start_creates_checkpoint(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        images = [tmp_path / "img1.png", tmp_path / "img2.png"]
        for img in images:
            create_test_image(img)

        manager.start([str(p) for p in images])

        assert manager._data is not None
        assert len(manager._data.pending) == 2
        assert checkpoint_path.exists()

    def test_load_resume(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        images = [tmp_path / "img1.png", tmp_path / "img2.png"]
        for img in images:
            create_test_image(img)

        manager.start([str(p) for p in images])
        assert manager._data is not None
        manager._data.pending.remove(str(images[0]))

        manager.save()
        manager2 = CheckpointManager(checkpoint_path)
        loaded = manager2.load()

        assert loaded
        assert manager2._data is not None
        assert len(manager2.get_pending()) == 1
        assert str(images[1]) in manager2.get_pending()

    def test_is_complete(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        images = [tmp_path / "img1.png"]
        create_test_image(images[0])

        manager.start([str(p) for p in images])
        assert not manager.is_complete()

        assert manager._data is not None
        manager._data.pending = []
        assert manager.is_complete()

    def test_clear(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        images = [tmp_path / "img1.png"]
        create_test_image(images[0])

        manager.start([str(p) for p in images])
        manager.clear()

        assert not checkpoint_path.exists()
        assert manager._data is None

    def test_merge_paths_adds_only_new_inputs(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        images = [tmp_path / "img1.png", tmp_path / "img2.png", tmp_path / "img3.png"]

        manager.start(images[:2])
        new_paths = manager.merge_paths(images)

        assert new_paths == [str(images[2])]
        assert manager._data is not None
        assert manager._data.total == 3
        assert manager.get_pending() == [str(path) for path in images]


class TestRetryPolicy:
    def test_success_no_retry(self) -> None:
        policy = RetryPolicy(RetryConfig(max_retries=3))
        result = policy.execute(lambda: 42)

        assert result.success
        assert result.result == 42
        assert result.attempts == 1

    def test_retry_on_failure(self) -> None:
        config = RetryConfig(max_retries=3, base_delay_ms=10)
        policy = RetryPolicy(config)
        attempts = [0]

        def failing():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("fail")
            return "success"

        result = policy.execute(failing)

        assert result.success
        assert result.attempts == 3

    def test_max_retries_exceeded(self) -> None:
        config = RetryConfig(max_retries=2, base_delay_ms=10)
        policy = RetryPolicy(config)

        def always_fail():
            raise ValueError("always fails")

        result = policy.execute(always_fail)

        assert not result.success
        assert result.attempts == 3

    def test_exponential_backoff(self) -> None:
        config = RetryConfig(max_retries=4, base_delay_ms=100, max_delay_ms=1000)
        policy = RetryPolicy(config)

        delays = [policy.calculate_delay(i) for i in range(1, 5)]

        assert delays[0] >= 100
        assert delays[1] >= 200
        assert delays[2] >= 400
        assert delays[3] <= 1000 + config.jitter_ms


class TestRateLimiter:
    def test_acquire_token(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_second=10, burst_size=5))

        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert limiter.try_acquire()

    def test_rate_limit_blocks(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_second=1, burst_size=1))

        limiter.try_acquire()
        assert not limiter.try_acquire()

    def test_reset(self) -> None:
        limiter = RateLimiter(RateLimitConfig(burst_size=5))

        for _ in range(5):
            limiter.try_acquire()

        assert not limiter.try_acquire()
        limiter.reset()
        assert limiter.try_acquire()

    def test_acquire_zero_timeout_is_non_blocking(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_second=1, burst_size=1, wait_timeout_ms=5000))

        assert limiter.acquire()
        start = time.monotonic()
        acquired = limiter.acquire(timeout_ms=0)
        elapsed = time.monotonic() - start

        assert not acquired
        assert elapsed < 0.2

    def test_rejects_invalid_requests_per_second(self) -> None:
        with pytest.raises(ValueError, match="requests_per_second must be a finite number > 0"):
            RateLimitConfig(requests_per_second=0)

    def test_rejects_invalid_burst_size(self) -> None:
        with pytest.raises(ValueError, match="burst_size must be >= 1"):
            RateLimitConfig(burst_size=0)

    def test_rejects_invalid_wait_timeout(self) -> None:
        with pytest.raises(ValueError, match="wait_timeout_ms must be >= 0"):
            RateLimitConfig(wait_timeout_ms=-1)


class TestMetricsCollector:
    def test_record_success(self) -> None:
        collector = MetricsCollector()
        collector.start_job(1)
        collector.record_success("webp", 1000, 500, 0.95, 100.0)
        collector.end_job()

        metrics = collector.collect()
        assert metrics["successful_images"] == 1
        assert metrics["total_bytes_before"] == 1000
        assert metrics["total_bytes_after"] == 500

    def test_average_ssim(self) -> None:
        collector = MetricsCollector()
        collector.start_job(2)
        collector.record_success("webp", 1000, 500, 0.90, 50.0)
        collector.record_success("webp", 1000, 500, 0.80, 50.0)
        collector.end_job()

        metrics = collector.collect()
        assert metrics["average_ssim"] == pytest.approx(0.85)

    def test_export_prometheus(self) -> None:
        exporter = PrometheusMetricsExporter()
        exporter.start_job(1)
        exporter.record_success("webp", 1000, 500, 0.95, 100.0)
        exporter.end_job()

        output = exporter.export()
        assert "perceptimg_batch_jobs_total 1" in output
        assert "perceptimg_batch_images_total 1" in output
        assert "perceptimg_batch_ssim_average" in output

    def test_export_prometheus_emits_single_header_for_labeled_metrics(self) -> None:
        exporter = PrometheusMetricsExporter()
        exporter.start_job(2)
        exporter.record_success("png", 1000, 500, 0.95, 100.0)
        exporter.record_success("webp", 1000, 400, 0.90, 90.0)
        exporter.record_failure("ValueError")
        exporter.record_failure("OSError")

        output = exporter.export()

        assert output.count("# HELP perceptimg_batch_formats_count") == 1
        assert output.count("# TYPE perceptimg_batch_formats_count") == 1
        assert 'perceptimg_batch_formats_count{format="png"} 1' in output
        assert 'perceptimg_batch_formats_count{format="webp"} 1' in output
        assert output.count("# HELP perceptimg_batch_errors_count") == 1
        assert output.count("# TYPE perceptimg_batch_errors_count") == 1
        assert 'perceptimg_batch_errors_count{error="ValueError"} 1' in output
        assert 'perceptimg_batch_errors_count{error="OSError"} 1' in output

    def test_start_job_accumulates_total_jobs(self) -> None:
        exporter = PrometheusMetricsExporter()

        exporter.start_job(1)
        exporter.end_job()
        exporter.start_job(2)
        exporter.end_job()

        stats = exporter.get_stats()

        assert stats["total_jobs"] == 2
        assert stats["total_images"] == 3  # accumulated: 1 + 2

    def test_export_prometheus_includes_skipped_images(self) -> None:
        exporter = PrometheusMetricsExporter()
        exporter.start_job(3)
        exporter.record_skip()

        output = exporter.export()

        assert output.count("# HELP perceptimg_batch_images_skipped") == 1
        assert output.count("# TYPE perceptimg_batch_images_skipped") == 1
        assert "perceptimg_batch_images_skipped 1" in output

    def test_end_job_preserves_accumulated_processing_time(self) -> None:
        exporter = PrometheusMetricsExporter()
        exporter.start_job(2)
        exporter.record_success("png", 1000, 500, 0.95, 100.0)
        exporter.record_success("webp", 1000, 400, 0.90, 200.0)

        assert exporter.get_stats()["total_processing_time_ms"] == 300.0

        exporter.end_job()

        assert exporter.get_stats()["total_processing_time_ms"] == 300.0


class TestBatchWithRetry:
    def test_success_with_retry(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)

        result = optimize_batch_with_retry(paths, policy)

        assert len(result.successful) == 3
        assert len(result.failed) == 0

    def test_stop_on_error_keeps_completed_results(self) -> None:
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

        with patch.object(BatchProcessor, "process_single", autospec=True, side_effect=fake_process_single):
            result = optimize_batch_with_retry(
                [Path("good1.png"), Path("bad.png"), Path("good2.png")],
                Policy(max_size_kb=500),
                retry_config=RetryConfig(max_retries=0),
                max_workers=3,
                continue_on_error=False,
            )

        assert {path.name for path, _ in result.successful} == {"good1.png", "good2.png"}
        assert [path.name for path, _ in result.failed] == ["bad.png"]

    def test_progress_reports_completed_file(self) -> None:
        ok_result = make_stub_result()
        progress_events: list[tuple[str | None, int, int]] = []

        def fake_process_single(
            self: BatchProcessor,
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            if image_path.name == "slow.png":
                time.sleep(0.05)
            else:
                time.sleep(0.01)
            return (image_path, ok_result)

        with patch.object(BatchProcessor, "process_single", autospec=True, side_effect=fake_process_single):
            optimize_batch_with_retry(
                [Path("slow.png"), Path("fast.png")],
                Policy(max_size_kb=500),
                max_workers=2,
                on_progress=lambda progress: progress_events.append(
                    (progress.current_file, progress.completed, progress.failed)
                ),
            )

        assert progress_events == [("fast.png", 1, 0), ("slow.png", 2, 0)]

    def test_progress_callback_receives_snapshots(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        progress_calls: list[object] = []
        optimize_batch_with_retry(
            [tmp_path / f"img{i}.png" for i in range(3)],
            Policy(max_size_kb=500),
            on_progress=lambda p: progress_calls.append(p),
        )

        assert [call.completed for call in progress_calls] == [1, 2, 3]
        assert len({id(call) for call in progress_calls}) == 3


class TestBatchWithRateLimit:
    def test_rate_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(5)]
        policy = Policy(max_size_kb=500)

        result = optimize_batch_with_rate_limit(
            paths, policy, rate_limit=RateLimitConfig(requests_per_second=100)
        )

        assert len(result.successful) == 5

    def test_stop_on_error_keeps_completed_results(self) -> None:
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

        with patch.object(BatchProcessor, "process_single", autospec=True, side_effect=fake_process_single):
            result = optimize_batch_with_rate_limit(
                [Path("good1.png"), Path("bad.png"), Path("good2.png")],
                Policy(max_size_kb=500),
                rate_limit=RateLimitConfig(requests_per_second=100, burst_size=100),
                max_workers=3,
                continue_on_error=False,
            )

        assert {path.name for path, _ in result.successful} == {"good1.png", "good2.png"}
        assert [path.name for path, _ in result.failed] == ["bad.png"]

    def test_rate_limit_timeout_fails_item_without_processing(self) -> None:
        processed: list[str] = []

        def fake_process_single(
            self: BatchProcessor,
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            processed.append(image_path.name)
            return (image_path, make_stub_result())

        with (
            patch.object(BatchProcessor, "process_single", autospec=True, side_effect=fake_process_single),
            patch.object(RateLimiter, "acquire", autospec=True, return_value=False),
        ):
            result = optimize_batch_with_rate_limit(
                [Path("a.png")],
                Policy(max_size_kb=500),
                rate_limit=RateLimitConfig(requests_per_second=100, burst_size=100),
                max_workers=1,
            )

        assert processed == []
        assert result.successful == []
        assert len(result.failed) == 1
        assert result.failed[0][0].name == "a.png"
        assert isinstance(result.failed[0][1], TimeoutError)


class TestBatchWithMetrics:
    def test_metrics_collection(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)

        result, metrics = optimize_batch_with_metrics(paths, policy)

        assert len(result.successful) == 3
        assert metrics["successful_images"] == 3
        assert metrics["total_bytes_before"] > 0

    def test_metrics_collection_records_processing_time(self) -> None:
        ok_result = make_stub_result()

        def fake_process_single(
            self: BatchProcessor,
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            time.sleep(0.01)
            return (image_path, ok_result)

        with patch.object(BatchProcessor, "process_single", autospec=True, side_effect=fake_process_single):
            result, metrics = optimize_batch_with_metrics(
                [Path("a.png"), Path("b.png")],
                Policy(max_size_kb=500),
                max_workers=1,
            )

        assert len(result.successful) == 2
        assert metrics["successful_images"] == 2
        assert metrics["total_processing_time_ms"] > 0

    def test_metrics_collection_records_processing_time_for_duplicate_paths(self) -> None:
        ok_result = make_stub_result()

        def fake_process_single(
            self: BatchProcessor,
            image_path: Path,
            policy: Policy,
            cache: object = None,
        ) -> tuple[Path, OptimizationResult | Exception]:
            time.sleep(0.01)
            return (image_path, ok_result)

        duplicate = Path("dup.png")
        with patch.object(BatchProcessor, "process_single", autospec=True, side_effect=fake_process_single):
            result, metrics = optimize_batch_with_metrics(
                [duplicate, duplicate],
                Policy(max_size_kb=500),
                max_workers=2,
            )

        assert len(result.successful) == 2
        assert metrics["successful_images"] == 2
        assert metrics["total_processing_time_ms"] > 0


class TestBatchWithCheckpoint:
    def test_checkpoint_saves_progress(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"

        result = optimize_batch_with_checkpoint(
            paths, policy, checkpoint_path, checkpoint_interval=1
        )

        assert len(result.successful) == 3
        assert checkpoint_path.exists()

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"

        result = optimize_batch_with_checkpoint(paths, policy, checkpoint_path)

        with patch("perceptimg.core.batch.__init__.BatchProcessor.execute") as mocked_execute:
            result2 = optimize_batch_with_checkpoint(paths, policy, checkpoint_path)
            mocked_execute.assert_not_called()

        assert len(result.successful) == 3
        assert len(result2.successful) == 3

    def test_resume_from_completed_checkpoint_processes_new_inputs(self, tmp_path: Path) -> None:
        first = tmp_path / "img0.png"
        second = tmp_path / "img1.png"
        create_test_image(first)
        create_test_image(second)

        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"

        initial = optimize_batch_with_checkpoint([first], policy, checkpoint_path)
        resumed = optimize_batch_with_checkpoint([first, second], policy, checkpoint_path)

        assert len(initial.successful) == 1
        assert len(resumed.successful) == 2
        assert {path.name for path, _ in resumed.successful} == {"img0.png", "img1.png"}

        manager = CheckpointManager(checkpoint_path)
        assert manager.load()
        assert manager.is_complete()
        assert manager.get_stats()["total"] == 2
        assert manager.get_pending() == []

    def test_resume_from_checkpoint_returns_only_requested_inputs(self, tmp_path: Path) -> None:
        first = tmp_path / "img0.png"
        second = tmp_path / "img1.png"
        create_test_image(first)
        create_test_image(second)

        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"

        optimize_batch_with_checkpoint([first, second], policy, checkpoint_path)

        with patch("perceptimg.core.batch.__init__.BatchProcessor.execute") as mocked_execute:
            subset = optimize_batch_with_checkpoint([first], policy, checkpoint_path)
            mocked_execute.assert_not_called()

        assert len(subset.successful) == 1
        assert [path.name for path, _ in subset.successful] == ["img0.png"]
        assert subset.failed == []

    def test_resume_from_checkpoint_preserves_requested_duplicate_inputs(
        self, tmp_path: Path
    ) -> None:
        image = tmp_path / "img0.png"
        create_test_image(image)

        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"

        initial = optimize_batch_with_checkpoint([image], policy, checkpoint_path)
        resumed = optimize_batch_with_checkpoint([image, image], policy, checkpoint_path)

        assert len(initial.successful) == 1
        assert len(resumed.successful) == 2
        assert [path.name for path, _ in resumed.successful] == ["img0.png", "img0.png"]
        assert resumed.failed == []

    def test_resume_from_checkpoint_preserves_skipped_results(self, tmp_path: Path) -> None:
        image = tmp_path / "img0.png"
        create_test_image(image)

        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        manager.start([image])
        manager.mark_completed(
            str(image),
            JobResult(path=str(image), status=JobStatus.SKIPPED),
        )
        manager.save()

        with patch("perceptimg.core.batch.__init__.BatchProcessor.execute") as mocked_execute:
            resumed = optimize_batch_with_checkpoint([image], policy, checkpoint_path)
            mocked_execute.assert_not_called()

        assert resumed.total == 1
        assert resumed.successful == []
        assert resumed.failed == []
        assert resumed.skipped == [image]

    def test_resume_from_checkpoint_counts_mixed_skipped_results(self, tmp_path: Path) -> None:
        skipped = tmp_path / "skipped.png"
        completed = tmp_path / "completed.png"
        failed = tmp_path / "failed.png"
        for path in [skipped, completed, failed]:
            create_test_image(path)

        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        manager.start([skipped, completed, failed])

        completed_result = Optimizer().optimize(completed, policy)
        manager.mark_completed(
            str(skipped),
            JobResult(path=str(skipped), status=JobStatus.SKIPPED),
        )
        manager.mark_completed(
            str(completed),
            JobResult(
                path=str(completed),
                status=JobStatus.COMPLETED,
                size_before_kb=completed_result.report.size_before_kb,
                size_after_kb=completed_result.report.size_after_kb,
                ssim=completed_result.report.ssim,
                format=completed_result.report.chosen_format,
                quality=completed_result.report.quality,
                psnr=completed_result.report.psnr,
                perceptual_score=completed_result.report.perceptual_score,
                reasons=list(completed_result.report.reasons),
                artifact_base64=base64.b64encode(completed_result.image_bytes).decode("ascii"),
            ),
        )
        manager.mark_completed(
            str(failed),
            JobResult(path=str(failed), status=JobStatus.FAILED, error="boom"),
        )
        manager.save()

        with patch("perceptimg.core.batch.__init__.BatchProcessor.execute") as mocked_execute:
            resumed = optimize_batch_with_checkpoint(
                [skipped, completed, failed], policy, checkpoint_path
            )
            mocked_execute.assert_not_called()

        assert resumed.total == 3
        assert [path for path in resumed.skipped] == [skipped]
        assert [path.name for path, _ in resumed.successful] == ["completed.png"]
        assert [path.name for path, _ in resumed.failed] == ["failed.png"]

    def test_resume_from_partial_checkpoint_includes_previous_results(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        manager.start(paths)

        first_result = Optimizer().optimize(paths[0], policy)
        manager.mark_completed(
            str(paths[0]),
            JobResult(
                path=str(paths[0]),
                status=JobStatus.COMPLETED,
                size_before_kb=first_result.report.size_before_kb,
                size_after_kb=first_result.report.size_after_kb,
                ssim=first_result.report.ssim,
                format=first_result.report.chosen_format,
                quality=first_result.report.quality,
                psnr=first_result.report.psnr,
                perceptual_score=first_result.report.perceptual_score,
                reasons=list(first_result.report.reasons),
                artifact_base64=base64.b64encode(first_result.image_bytes).decode("ascii"),
            ),
        )
        manager.save()

        result = optimize_batch_with_checkpoint(paths, policy, checkpoint_path)

        assert len(result.successful) == 3
        assert {path.name for path, _ in result.successful} == {"img0.png", "img1.png", "img2.png"}

    def test_resume_from_partial_checkpoint_ignores_unrequested_pending_paths(
        self, tmp_path: Path
    ) -> None:
        first = tmp_path / "img0.png"
        second = tmp_path / "img1.png"
        create_test_image(first)
        create_test_image(second)

        policy = Policy(max_size_kb=500)
        checkpoint_path = tmp_path / "checkpoint.json"
        manager = CheckpointManager(checkpoint_path)
        manager.start([first, second])

        first_result = Optimizer().optimize(first, policy)
        manager.mark_completed(
            str(first),
            JobResult(
                path=str(first),
                status=JobStatus.COMPLETED,
                size_before_kb=first_result.report.size_before_kb,
                size_after_kb=first_result.report.size_after_kb,
                ssim=first_result.report.ssim,
                format=first_result.report.chosen_format,
                quality=first_result.report.quality,
                psnr=first_result.report.psnr,
                perceptual_score=first_result.report.perceptual_score,
                reasons=list(first_result.report.reasons),
                artifact_base64=base64.b64encode(first_result.image_bytes).decode("ascii"),
            ),
        )
        manager.save()

        with patch("perceptimg.core.batch.__init__.BatchProcessor.execute") as mocked_execute:
            subset = optimize_batch_with_checkpoint([first], policy, checkpoint_path)
            mocked_execute.assert_not_called()

        assert len(subset.successful) == 1
        assert [path.name for path, _ in subset.successful] == ["img0.png"]
        assert subset.failed == []

    def test_invalid_checkpoint_interval_raises_value_error(self, tmp_path: Path) -> None:
        image_path = tmp_path / "img.png"
        create_test_image(image_path)

        with pytest.raises(ValueError, match="checkpoint_interval must be >= 1"):
            optimize_batch_with_checkpoint(
                [image_path],
                Policy(max_size_kb=500),
                tmp_path / "checkpoint.json",
                checkpoint_interval=0,
            )
