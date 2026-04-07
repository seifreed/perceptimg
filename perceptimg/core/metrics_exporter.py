"""Prometheus metrics exporter for monitoring batch processing."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MetricsConfig:
    """Configuration for Prometheus metrics.

    Attributes:
        namespace: Metric namespace prefix (default: "perceptimg").
        enable_default_metrics: Whether to enable default process metrics.
        port: Port for Prometheus server (default: 8000).
    """

    namespace: str = "perceptimg"
    enable_default_metrics: bool = True
    port: int = 8000


@dataclass(slots=True)
class BatchMetrics:
    """Batch processing metrics.

    Attributes:
        total_jobs: Total number of batch jobs.
        total_images: Total images processed.
        successful_images: Successfully processed images.
        failed_images: Failed images.
        total_bytes_before: Total bytes before optimization.
        total_bytes_after: Total bytes after optimization.
        total_processing_time_ms: Total processing time.
    """

    total_jobs: int = 0
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    skipped_images: int = 0
    total_bytes_before: int = 0
    total_bytes_after: int = 0
    total_processing_time_ms: float = 0.0
    total_ssim: float = 0.0
    job_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "total_images": self.total_images,
            "successful_images": self.successful_images,
            "failed_images": self.failed_images,
            "skipped_images": self.skipped_images,
            "total_bytes_before": self.total_bytes_before,
            "total_bytes_after": self.total_bytes_after,
            "total_processing_time_ms": self.total_processing_time_ms,
            "job_duration_ms": self.job_duration_ms,
            "in_progress": self.in_progress,
            "average_ssim": self.average_ssim,
            "average_compression_ratio": self.average_compression_ratio,
        }

    @property
    def in_progress(self) -> int:
        return self.total_images - self.successful_images - self.failed_images - self.skipped_images

    @property
    def average_ssim(self) -> float:
        if self.successful_images == 0:
            return 0.0
        return self.total_ssim / self.successful_images

    @property
    def average_compression_ratio(self) -> float:
        if self.total_bytes_before == 0:
            return 0.0
        return 1.0 - (self.total_bytes_after / self.total_bytes_before)

    def record_success(
        self,
        bytes_before: int,
        bytes_after: int,
        ssim: float,
        processing_time_ms: float,
    ) -> None:
        self.successful_images += 1
        self.total_bytes_before += bytes_before
        self.total_bytes_after += bytes_after
        self.total_ssim += ssim
        self.total_processing_time_ms += processing_time_ms

    def record_failure(self) -> None:
        self.failed_images += 1

    def record_skip(self) -> None:
        self.skipped_images += 1


class PrometheusMetricsExporter:
    """Exports metrics in Prometheus format.

    Example:
        >>> exporter = PrometheusMetricsExporter()
        >>> exporter.record_batch_start()
        >>> for image in images:
        ...     result = process(image)
        ...     exporter.record_image_processed(result)
        >>> exporter.record_batch_end()
        >>> print(exporter.export())  # Prometheus text format
    """

    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self._metrics = BatchMetrics()
        self._job_start_time: float | None = None
        self._format_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def start_job(self, total_images: int) -> None:
        """Record start of a batch job.

        Args:
            total_images: Total number of images to process.
        """
        with self._lock:
            self._metrics.total_jobs += 1
            self._metrics.total_images += total_images
            self._job_start_time = time.monotonic()

    def record_success(
        self,
        format: str,
        bytes_before: int,
        bytes_after: int,
        ssim: float,
        processing_time_ms: float,
    ) -> None:
        """Record a successful image processing.

        Args:
            format: Output format used.
            bytes_before: Original size in bytes.
            bytes_after: Optimized size in bytes.
            ssim: SSIM score.
            processing_time_ms: Processing time in milliseconds.
        """
        with self._lock:
            self._metrics.record_success(bytes_before, bytes_after, ssim, processing_time_ms)
            self._format_counts[format] = self._format_counts.get(format, 0) + 1

    def record_failure(self, error_type: str) -> None:
        """Record a failed image processing.

        Args:
            error_type: Type of error (e.g., "FileNotFoundError").
        """
        with self._lock:
            self._metrics.record_failure()
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

    def record_skip(self) -> None:
        """Record a skipped image."""
        with self._lock:
            self._metrics.record_skip()

    def end_job(self) -> None:
        """Record end of a batch job."""
        with self._lock:
            if self._job_start_time:
                self._metrics.job_duration_ms += (time.monotonic() - self._job_start_time) * 1000
                self._job_start_time = None

    def export(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string.
        """
        with self._lock:
            lines: list[str] = []
            ns = self.config.namespace

            m = self._metrics

            lines.append(f"# HELP {ns}_batch_jobs_total Total batch jobs processed")
            lines.append(f"# TYPE {ns}_batch_jobs_total counter")
            lines.append(f"{ns}_batch_jobs_total {m.total_jobs}")

            lines.append(f"# HELP {ns}_batch_images_total Total images processed")
            lines.append(f"# TYPE {ns}_batch_images_total counter")
            lines.append(f"{ns}_batch_images_total {m.total_images}")

            lines.append(f"# HELP {ns}_batch_images_successful Successfully processed images")
            lines.append(f"# TYPE {ns}_batch_images_successful counter")
            lines.append(f"{ns}_batch_images_successful {m.successful_images}")

            lines.append(f"# HELP {ns}_batch_images_failed Failed images")
            lines.append(f"# TYPE {ns}_batch_images_failed counter")
            lines.append(f"{ns}_batch_images_failed {m.failed_images}")

            lines.append(f"# HELP {ns}_batch_images_skipped Skipped images")
            lines.append(f"# TYPE {ns}_batch_images_skipped counter")
            lines.append(f"{ns}_batch_images_skipped {m.skipped_images}")

            lines.append(f"# HELP {ns}_batch_images_in_progress Images currently in progress")
            lines.append(f"# TYPE {ns}_batch_images_in_progress gauge")
            lines.append(f"{ns}_batch_images_in_progress {m.in_progress}")

            lines.append(f"# HELP {ns}_batch_bytes_before_total Total bytes before optimization")
            lines.append(f"# TYPE {ns}_batch_bytes_before_total counter")
            lines.append(f"{ns}_batch_bytes_before_total {m.total_bytes_before}")

            lines.append(f"# HELP {ns}_batch_bytes_after_total Total bytes after optimization")
            lines.append(f"# TYPE {ns}_batch_bytes_after_total counter")
            lines.append(f"{ns}_batch_bytes_after_total {m.total_bytes_after}")

            lines.append(f"# HELP {ns}_batch_ssim_average Average SSIM score")
            lines.append(f"# TYPE {ns}_batch_ssim_average gauge")
            lines.append(f"{ns}_batch_ssim_average {m.average_ssim:.4f}")

            lines.append(f"# HELP {ns}_batch_compression_ratio Average compression ratio")
            lines.append(f"# TYPE {ns}_batch_compression_ratio gauge")
            lines.append(f"{ns}_batch_compression_ratio {m.average_compression_ratio:.4f}")

            lines.append(f"# HELP {ns}_batch_processing_time_ms Total processing time in milliseconds")
            lines.append(f"# TYPE {ns}_batch_processing_time_ms counter")
            lines.append(f"{ns}_batch_processing_time_ms {m.total_processing_time_ms:.0f}")

            lines.append(f"# HELP {ns}_batch_job_duration_ms Total job wall-clock duration in milliseconds")
            lines.append(f"# TYPE {ns}_batch_job_duration_ms counter")
            lines.append(f"{ns}_batch_job_duration_ms {m.job_duration_ms:.0f}")

            if self._format_counts:
                lines.append(f"# HELP {ns}_batch_formats_count Images by output format")
                lines.append(f"# TYPE {ns}_batch_formats_count counter")
            for fmt, count in self._format_counts.items():
                lines.append(f'{ns}_batch_formats_count{{format="{fmt}"}} {count}')

            if self._error_counts:
                lines.append(f"# HELP {ns}_batch_errors_count Errors by type")
                lines.append(f"# TYPE {ns}_batch_errors_count counter")
            for error_type, count in self._error_counts.items():
                lines.append(f'{ns}_batch_errors_count{{error="{error_type}"}} {count}')

            return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get current metrics as a dictionary.

        Returns:
            Dict with current metrics.
        """
        with self._lock:
            return {
                **self._metrics.to_dict(),
                "formats": dict(self._format_counts),
                "errors": dict(self._error_counts),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = BatchMetrics()
            self._format_counts = {}
            self._error_counts = {}
            self._job_start_time = None


class MetricsCollector:
    """Collects metrics from multiple sources and aggregates them.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.add_callback(lambda: {"custom_metric": 42})
        >>> metrics = collector.collect()
    """

    def __init__(self, namespace: str = "perceptimg"):
        self.namespace = namespace
        self._exporter = PrometheusMetricsExporter(MetricsConfig(namespace=namespace))
        self._callbacks: list[Callable[[], dict[str, Any]]] = []
        self._callbacks_lock = threading.Lock()

    def add_callback(self, callback: Callable[[], dict[str, Any]]) -> None:
        """Add a callback that returns custom metrics.

        Args:
            callback: Function that returns a dict of metrics.
        """
        with self._callbacks_lock:
            self._callbacks.append(callback)

    def start_job(self, total_images: int) -> None:
        self._exporter.start_job(total_images)

    def record_success(
        self,
        format: str,
        bytes_before: int,
        bytes_after: int,
        ssim: float,
        processing_time_ms: float,
    ) -> None:
        self._exporter.record_success(format, bytes_before, bytes_after, ssim, processing_time_ms)

    def record_failure(self, error_type: str) -> None:
        self._exporter.record_failure(error_type)

    def record_skip(self) -> None:
        self._exporter.record_skip()

    def end_job(self) -> None:
        self._exporter.end_job()

    def export_prometheus(self) -> str:
        return self._exporter.export()

    def collect(self) -> dict[str, Any]:
        """Collect all metrics including custom callbacks.

        Returns:
            Dict with all metrics.
        """
        metrics = self._exporter.get_stats()
        with self._callbacks_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                custom = callback()
                metrics.update(custom)
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                logger.warning("Metrics callback %r failed: %s", callback, exc)
        return metrics
