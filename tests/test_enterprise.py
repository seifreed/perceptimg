"""Tests for enterprise features: checkpoint, retry, rate limiting, metrics."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from perceptimg import Policy
from perceptimg.core.batch import (
    optimize_batch_with_checkpoint,
    optimize_batch_with_metrics,
    optimize_batch_with_rate_limit,
    optimize_batch_with_retry,
)
from perceptimg.core.checkpoint import CheckpointManager
from perceptimg.core.metrics_exporter import MetricsCollector, PrometheusMetricsExporter
from perceptimg.core.rate_limiter import RateLimitConfig, RateLimiter
from perceptimg.core.retry import RetryConfig, RetryPolicy


def create_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    img = Image.new("RGB", size, "red")
    img.save(path)


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
        assert "perceptimg_batch_images_total 1" in output
        assert "perceptimg_batch_ssim_average" in output


class TestBatchWithRetry:
    def test_success_with_retry(self, tmp_path: Path) -> None:
        for i in range(3):
            create_test_image(tmp_path / f"img{i}.png")

        paths = [tmp_path / f"img{i}.png" for i in range(3)]
        policy = Policy(max_size_kb=500)

        result = optimize_batch_with_retry(paths, policy)

        assert len(result.successful) == 3
        assert len(result.failed) == 0


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

        result2 = optimize_batch_with_checkpoint(paths, policy, checkpoint_path)

        assert len(result.successful) == 3
        assert len(result2.successful) == 3
