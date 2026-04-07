"""Checkpoint management for resumable batch processing."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobStatus(StrEnum):
    """Status of a batch processing job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class JobResult:
    """Result of processing a single image."""

    path: str
    status: JobStatus
    error: str | None = None
    output_path: str | None = None
    size_before_kb: float | None = None
    size_after_kb: float | None = None
    ssim: float | None = None
    format: str | None = None
    quality: int | None = None
    psnr: float | None = None
    perceptual_score: float | None = None
    reasons: list[str] | None = None
    artifact_base64: str | None = None
    processing_time_ms: float | None = None


@dataclass
class CheckpointData:
    """Serializable checkpoint data."""

    version: int = 2
    job_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "results": self.results,
            "pending": self.pending,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        return cls(
            version=data.get("version", 1),
            job_id=data.get("job_id", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            total=data.get("total", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
            skipped=data.get("skipped", 0),
            results=data.get("results", []),
            pending=data.get("pending", []),
        )


class CheckpointManager:
    """Manages checkpoint files for resumable batch processing.

    Supports:
    - Periodic saving during processing
    - Resume from interruption
    - Multiple checkpoint formats (JSON, pickle)
    - Atomic writes to prevent corruption

    Example:
        >>> checkpoint = CheckpointManager(Path("checkpoint.json"))
        >>> checkpoint.start(["img1.png", "img2.png"])
        >>> checkpoint.mark_completed("img1.png", JobResult(...))
        >>> # After interruption:
        >>> checkpoint.load()
        >>> remaining = checkpoint.get_pending()
    """

    def __init__(self, checkpoint_path: Path | str | None = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._data: CheckpointData | None = None
        self._dirty = False
        self._lock = threading.Lock()
        self._last_checkpoint_count = 0
        self._checkpoint_pending = False

    def start(
        self,
        image_paths: Sequence[str | Path],
        job_id: str | None = None,
    ) -> None:
        """Initialize a new checkpoint with pending images.

        Args:
            image_paths: List of image paths to process.
            job_id: Optional job identifier. Generated if not provided.
        """
        self._last_checkpoint_count = 0
        self._checkpoint_pending = False
        self._data = CheckpointData(
            job_id=job_id or uuid.uuid4().hex[:8],
            created_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
            total=len(image_paths),
            pending=[str(p) for p in image_paths],
        )
        self._dirty = True
        if self.checkpoint_path:
            self._atomic_write()

    def load(self) -> bool:
        """Load checkpoint from file if it exists.

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists.
        """
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return False

        with self._lock:
            try:
                content = self.checkpoint_path.read_text(encoding="utf-8")
                data = json.loads(content)
                self._data = CheckpointData.from_dict(data)
                return True
            except (json.JSONDecodeError, KeyError, ValueError):
                return False

    def save(self) -> None:
        """Force save checkpoint to file."""
        with self._lock:
            if self._data and self.checkpoint_path:
                self._atomic_write()
                self._checkpoint_pending = False

    def _atomic_write(self) -> None:
        """Write checkpoint atomically to prevent corruption."""
        if not self._data or not self.checkpoint_path:
            return

        self._data.updated_at = datetime.now(UTC).isoformat()
        content = json.dumps(self._data.to_dict(), indent=2)

        fd, temp_path = tempfile.mkstemp(
            dir=self.checkpoint_path.parent,
            prefix=".tmp_checkpoint_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, self.checkpoint_path)
        except Exception:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise
        self._dirty = False

    def mark_completed(
        self,
        path: str,
        result: JobResult,
        *,
        checkpoint_interval: int | None = None,
    ) -> None:
        """Mark an image as completed.

        Thread-safe: acquires internal lock to protect shared state.

        Args:
            path: Image path that was processed.
            result: Result of processing.
            checkpoint_interval: If set, atomically check whether a checkpoint
                should be saved after this completion.
        """
        with self._lock:
            if not self._data:
                return

            path_str = str(path)
            try:
                self._data.pending.remove(path_str)
            except ValueError:
                pass

            self._data.results.append(
                {
                    "path": path_str,
                    "status": result.status.value,
                    "error": result.error,
                    "output_path": result.output_path,
                    "size_before_kb": result.size_before_kb,
                    "size_after_kb": result.size_after_kb,
                    "ssim": result.ssim,
                    "format": result.format,
                    "quality": result.quality,
                    "psnr": result.psnr,
                    "perceptual_score": result.perceptual_score,
                    "reasons": result.reasons,
                    "artifact_base64": result.artifact_base64,
                    "processing_time_ms": result.processing_time_ms,
                }
            )

            if result.status == JobStatus.COMPLETED:
                self._data.completed += 1
            elif result.status == JobStatus.FAILED:
                self._data.failed += 1
            elif result.status == JobStatus.SKIPPED:
                self._data.skipped += 1

            self._dirty = True

            if (
                checkpoint_interval is not None
                and checkpoint_interval >= 1
                and self.checkpoint_path
            ):
                completed_count = (
                    self._data.completed + self._data.failed + self._data.skipped
                )
                if (
                    completed_count > 0
                    and completed_count // checkpoint_interval
                    > self._last_checkpoint_count // checkpoint_interval
                ):
                    self._last_checkpoint_count = completed_count
                    self._atomic_write()
                    self._checkpoint_pending = False

    def mark_failed(
        self,
        path: str,
        error: str,
        *,
        checkpoint_interval: int | None = None,
    ) -> None:
        """Mark an image as failed.

        Args:
            path: Image path that failed.
            error: Error message.
            checkpoint_interval: If set, atomically check whether a checkpoint
                should be saved after this failure.
        """
        self.mark_completed(
            path,
            JobResult(path=str(path), status=JobStatus.FAILED, error=error),
            checkpoint_interval=checkpoint_interval,
        )

    def merge_paths(self, image_paths: Sequence[str | Path]) -> list[str]:
        """Merge new image paths into an existing checkpoint.

        Args:
            image_paths: Paths requested for the current run.

        Returns:
            Newly added pending paths.
        """
        with self._lock:
            if not self._data:
                return []

            # Count how many times each path already appears (completed/pending)
            known_counts = Counter(result["path"] for result in self._data.results)
            known_counts.update(self._data.pending)
            requested_counts = Counter(str(path) for path in image_paths)

            new_paths: list[str] = []
            for path in image_paths:
                path_str = str(path)
                if known_counts[path_str] >= requested_counts[path_str]:
                    continue
                known_counts[path_str] += 1
                self._data.pending.append(path_str)
                new_paths.append(path_str)

            if new_paths:
                self._data.total = len(self._data.results) + len(self._data.pending)
                self._dirty = True

            return new_paths

    def _get_pending_unlocked(self) -> list[str]:
        """Get pending paths assuming lock is already held."""
        if not self._data:
            return []
        return list(self._data.pending)

    def get_pending(self) -> list[str]:
        """Get list of pending image paths.

        Returns:
            List of paths that haven't been processed yet.
        """
        with self._lock:
            return self._get_pending_unlocked()

    def get_pending_for(self, image_paths: Sequence[str | Path]) -> list[str]:
        """Get pending image paths limited to the requested input set."""
        with self._lock:
            requested_remaining = Counter(str(path) for path in image_paths)

            for result in self._data.results if self._data else []:
                path = result["path"]
                if requested_remaining[path] > 0:
                    requested_remaining[path] -= 1

            pending_for: list[str] = []
            for path in self._get_pending_unlocked():
                if requested_remaining[path] <= 0:
                    continue
                pending_for.append(path)
                requested_remaining[path] -= 1
            return pending_for

    def _get_results_unlocked(self) -> list[JobResult]:
        """Get results assuming lock is already held."""
        if not self._data:
            return []
        return [
            JobResult(
                path=r["path"],
                status=JobStatus(r["status"]),
                error=r.get("error"),
                output_path=r.get("output_path"),
                size_before_kb=r.get("size_before_kb"),
                size_after_kb=r.get("size_after_kb"),
                ssim=r.get("ssim"),
                format=r.get("format"),
                quality=r.get("quality"),
                psnr=r.get("psnr"),
                perceptual_score=r.get("perceptual_score"),
                reasons=r.get("reasons"),
                artifact_base64=r.get("artifact_base64"),
                processing_time_ms=r.get("processing_time_ms"),
            )
            for r in self._data.results
        ]

    def get_results(self) -> list[JobResult]:
        """Get all processed results.

        Returns:
            List of results for completed/failed/skipped images.
        """
        with self._lock:
            return self._get_results_unlocked()

    def get_results_for(self, image_paths: Sequence[str | Path]) -> list[JobResult]:
        """Get processed results limited to the requested input set."""
        with self._lock:
            requested_remaining = Counter(str(path) for path in image_paths)
            selected: list[JobResult] = []
            for job in self._get_results_unlocked():
                if requested_remaining[job.path] <= 0:
                    continue
                selected.append(job)
                requested_remaining[job.path] -= 1
            return selected

    def get_stats(self) -> dict[str, int]:
        """Get processing statistics.

        Returns:
            Dict with total, completed, failed, skipped, pending counts.
        """
        with self._lock:
            if not self._data:
                return {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "pending": 0,
                }
            return {
                "total": self._data.total,
                "completed": self._data.completed,
                "failed": self._data.failed,
                "skipped": self._data.skipped,
                "pending": len(self._data.pending),
            }

    def is_complete(self) -> bool:
        """Check if all images have been processed.

        Returns:
            True if no pending images remain.
        """
        with self._lock:
            if not self._data:
                return True
            return len(self._data.pending) == 0

    def clear(self) -> None:
        """Delete checkpoint file if it exists."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self._data = None
        self._dirty = False

    def should_checkpoint(self, interval: int = 10) -> bool:
        """Check if we should save a checkpoint based on interval.

        When ``checkpoint_interval`` is passed to :meth:`mark_completed`, the
        decision is made atomically inside the lock and stored as a pending
        flag.  This method then simply checks and clears that flag, avoiding
        the race where concurrent ``mark_completed`` calls could skip an
        interval boundary.

        If ``mark_completed`` was called *without* ``checkpoint_interval``,
        the legacy interval-based check is used as a fallback.

        Args:
            interval: Save checkpoint every N completed images.

        Returns:
            True if checkpoint should be saved.
        """
        if interval < 1:
            raise ValueError(f"checkpoint interval must be >= 1, got {interval}")
        with self._lock:
            if self._checkpoint_pending:
                self._checkpoint_pending = False
                return True
            if not self._data or not self.checkpoint_path:
                return False
            completed_count = self._data.completed + self._data.failed + self._data.skipped
            if completed_count == 0:
                return False
            # Fallback for callers that don't pass checkpoint_interval to
            # mark_completed.  When mark_completed *does* set _checkpoint_pending,
            # the early return above handles it; this path only fires otherwise.
            # NOTE: ``interval`` here must match the one used in mark_completed,
            # because _last_checkpoint_count is shared state.
            if completed_count // interval > self._last_checkpoint_count // interval:
                self._last_checkpoint_count = completed_count
                return True
            return False

    def save_if_needed(self, interval: int = 10) -> bool:
        """Atomically check whether a checkpoint is due and save it.

        Combines :meth:`should_checkpoint` and :meth:`save` under a single
        lock acquisition so that no crash window exists between the decision
        and the write.

        Args:
            interval: Save checkpoint every N completed images.

        Returns:
            True if a checkpoint was saved.
        """
        if interval < 1:
            raise ValueError(f"checkpoint interval must be >= 1, got {interval}")
        with self._lock:
            need_save = False
            if self._checkpoint_pending:
                self._checkpoint_pending = False
                need_save = True
            elif self._data and self.checkpoint_path:
                completed_count = self._data.completed + self._data.failed + self._data.skipped
                if completed_count > 0 and completed_count // interval > self._last_checkpoint_count // interval:
                    self._last_checkpoint_count = completed_count
                    need_save = True
            if need_save and self._data and self.checkpoint_path:
                self._atomic_write()
                self._checkpoint_pending = False
                return True
            return False


def create_incremental_processor(
    image_paths: Sequence[str | Path],
    checkpoint_path: Path | str,
    job_id: str | None = None,
) -> tuple[CheckpointManager, Iterator[str | Path]]:
    """Create an incremental processor that yields only pending images.

    Args:
        image_paths: All image paths to process.
        checkpoint_path: Path to save checkpoint.
        job_id: Optional job identifier.

    Returns:
        Tuple of (checkpoint_manager, iterator of pending paths).

    Example:
        >>> manager, pending = create_incremental_processor(images, "checkpoint.json")
        >>> for path in pending:
        ...     result = process(path)
        ...     manager.mark_completed(path, result)
        ...     if manager.should_checkpoint():
        ...         manager.save()
    """
    manager = CheckpointManager(checkpoint_path)

    if manager.load():
        new_paths = manager.merge_paths(image_paths)
        if new_paths:
            manager.save()
        pending: list[str | Path] = list(manager.get_pending_for(image_paths))
    else:
        manager.start(image_paths, job_id)
        pending = list(image_paths)

    return manager, iter(pending)
