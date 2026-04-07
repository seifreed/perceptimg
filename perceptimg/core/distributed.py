"""Distributed job queue using Redis for horizontal scaling."""

from __future__ import annotations

import importlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobStatus(StrEnum):
    """Status of a distributed job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Job:
    """A job in the distributed queue.

    Attributes:
        id: Unique job identifier.
        image_path: Path to image to process.
        policy: Policy configuration as dict.
        status: Current job status.
        created_at: Job creation timestamp.
        started_at: Processing start timestamp.
        completed_at: Processing completion timestamp.
        worker_id: ID of worker processing job.
        result: Job result if completed.
        error: Error message if failed.
        retries: Number of retry attempts.
    """

    id: str
    image_path: str
    policy: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    worker_id: str = ""
    attempt_id: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "policy": self.policy,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "worker_id": self.worker_id,
            "attempt_id": self.attempt_id,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(
            id=data["id"],
            image_path=data["image_path"],
            policy=data["policy"],
            status=JobStatus(data.get("status", "queued")),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            worker_id=data.get("worker_id", ""),
            attempt_id=data.get("attempt_id", ""),
            result=data.get("result"),
            error=data.get("error"),
            retries=data.get("retries", 0),
        )


@dataclass
class RedisConfig:
    """Configuration for Redis connection.

    Attributes:
        host: Redis host (default: localhost).
        port: Redis port (default: 6379).
        db: Redis database number (default: 0).
        password: Redis password if required.
        queue_name: Name of the job queue (default: "perceptimg:jobs").
        result_ttl: Time to keep terminal jobs in seconds (default: 3600).
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    queue_name: str = "perceptimg:jobs"
    result_ttl: int = 3600
    max_retries: int = 3


class RedisJobQueue:
    """Distributed job queue backed by Redis.

    Allows multiple workers to process images in parallel across
    multiple machines.

    Example:
        >>> queue = RedisJobQueue(RedisConfig(host="redis.example.com"))
        >>> # Producer: enqueue jobs
        >>> queue.enqueue(["img1.png", "img2.png"], policy.to_dict())
        >>> # Consumer: process jobs
        >>> for job in queue.consume(worker_id="worker-1"):
        ...     result = process(job.image_path, policy)
        ...     queue.complete(job.id, "worker-1", result, attempt_id=job.attempt_id)
    """

    def __init__(self, config: RedisConfig | None = None):
        self.config = config or RedisConfig()
        self._redis: Any = None

    def _jobs_key(self) -> str:
        return f"{self.config.queue_name}:jobs"

    def _terminal_job_key(self, job_id: str) -> str:
        return f"{self.config.queue_name}:terminal:{job_id}"

    def _terminal_keys_pattern(self) -> str:
        return f"{self.config.queue_name}:terminal:*"

    def _pending_key(self) -> str:
        return f"{self.config.queue_name}:pending"

    def _processing_key(self) -> str:
        return f"{self.config.queue_name}:processing"

    def _completed_key(self) -> str:
        return f"{self.config.queue_name}:completed"

    def _failed_key(self) -> str:
        return f"{self.config.queue_name}:failed"

    def _apply_result_ttl(self, redis: Any, *keys: str) -> None:
        if self.config.result_ttl:
            for key in keys:
                redis.expire(key, self.config.result_ttl)

    @staticmethod
    def _decode_job(job_data: str | None) -> Job | None:
        if not job_data:
            return None
        try:
            return Job.from_dict(json.loads(job_data))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _load_live_job(self, redis: Any, job_id: str) -> Job | None:
        return self._decode_job(redis.hget(self._jobs_key(), job_id))

    def _load_terminal_job(self, redis: Any, job_id: str) -> Job | None:
        return self._decode_job(redis.hget(self._terminal_job_key(job_id), "data"))

    def _load_job(self, redis: Any, job_id: str) -> Job | None:
        live_job = self._load_live_job(redis, job_id)
        if live_job is not None:
            return live_job
        return self._load_terminal_job(redis, job_id)

    @staticmethod
    def _normalize_timeout(timeout: float | int) -> float | int:
        if timeout < -1:
            raise ValueError("timeout must be >= -1")
        return timeout

    def _record_missing_job_metadata(self, redis: Any, job_id: str) -> None:
        job = Job(
            id=job_id,
            image_path="",
            policy={},
            status=JobStatus.FAILED,
            completed_at=datetime.now(UTC).isoformat(),
            error="Missing job metadata",
        )
        self._persist_terminal_job(redis, job)

    def _remove_job_from_lists(self, redis: Any, job_id: str) -> None:
        redis.lrem(self._pending_key(), 0, job_id)
        redis.lrem(self._completed_key(), 0, job_id)
        redis.lrem(self._failed_key(), 0, job_id)

    def _remove_live_membership(self, redis: Any, job_id: str) -> None:
        redis.lrem(self._pending_key(), 0, job_id)
        redis.hdel(self._processing_key(), job_id)

    def _normalize_terminal_membership(
        self,
        redis: Any,
        job_id: str,
        status: JobStatus,
    ) -> None:
        self._remove_job_from_lists(redis, job_id)
        redis.hdel(self._processing_key(), job_id)

        if status == JobStatus.COMPLETED:
            redis.rpush(self._completed_key(), job_id)
            self._apply_result_ttl(redis, self._completed_key())
        elif status == JobStatus.FAILED:
            redis.rpush(self._failed_key(), job_id)

    def _persist_terminal_job(self, redis: Any, job: Job) -> None:
        terminal_key = self._terminal_job_key(job.id)
        self._normalize_terminal_membership(redis, job.id, job.status)
        pipe = redis.pipeline()
        pipe.hdel(self._jobs_key(), job.id)
        pipe.hset(terminal_key, "data", json.dumps(job.to_dict()))
        pipe.execute()
        if job.status == JobStatus.COMPLETED:
            self._apply_result_ttl(redis, terminal_key, self._completed_key())
        elif job.status == JobStatus.FAILED:
            self._apply_result_ttl(redis, terminal_key, self._failed_key())

    def _migrate_stale_live_terminal(self, redis: Any, job: Job) -> None:
        terminal_key = self._terminal_job_key(job.id)
        self._remove_live_membership(redis, job.id)
        if self._load_terminal_job(redis, job.id) is None:
            redis.hset(terminal_key, "data", json.dumps(job.to_dict()))
            self._apply_result_ttl(redis, terminal_key)
        redis.hdel(self._jobs_key(), job.id)

    def _iter_terminal_storage_job_ids(self, redis: Any) -> list[str]:
        keys: list[str] = []
        scan_iter = getattr(redis, "scan_iter", None)
        if callable(scan_iter):
            try:
                keys.extend(scan_iter(match=self._terminal_keys_pattern()))
            except TypeError:
                keys.extend(scan_iter(self._terminal_keys_pattern()))
        else:
            keys_fn = getattr(redis, "keys", None)
            if callable(keys_fn):
                keys.extend(keys_fn(self._terminal_keys_pattern()))

        prefix = self._terminal_job_key("")
        job_ids: list[str] = []
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode()
            if key.startswith(prefix):
                job_ids.append(key[len(prefix) :])
        return job_ids

    def _mark_invalid_live_transition(
        self,
        redis: Any,
        job: Job,
        *,
        action: str,
        error: str | None = None,
    ) -> None:
        message = f"Invalid state transition: {action} from {job.status.value}"
        if error:
            message = f"{message}: {error}"

        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(UTC).isoformat()
        job.started_at = ""
        job.worker_id = ""
        job.attempt_id = ""
        job.result = None
        job.error = message

        self._persist_terminal_job(redis, job)

    def _count_live_terminal_jobs(self, redis: Any, key: str, expected_status: JobStatus) -> int:
        seen: set[str] = set()
        # Check list-based IDs first
        for job_id in redis.lrange(key, 0, -1):
            if job_id in seen:
                continue
            job = self._load_job(redis, job_id)
            if job is not None and job.status == expected_status:
                seen.add(job_id)
        # Only scan terminal storage for IDs not already found
        for job_id in self._iter_terminal_storage_job_ids(redis):
            if job_id in seen:
                continue
            job = self._load_terminal_job(redis, job_id)
            if job is not None and job.status == expected_status:
                seen.add(job_id)
        return len(seen)

    def _count_live_pending_jobs(self, redis: Any) -> int:
        job_ids = redis.lrange(self._pending_key(), 0, -1)
        seen: set[str] = set()
        for job_id in job_ids:
            if job_id in seen:
                continue
            job_data = redis.hget(self._jobs_key(), job_id)
            if not job_data:
                continue
            try:
                job = Job.from_dict(json.loads(job_data))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if job.status == JobStatus.QUEUED:
                seen.add(job_id)
        return len(seen)

    def _count_live_processing_jobs(self, redis: Any) -> int:
        seen: set[str] = set()
        for job_id in redis.hkeys(self._processing_key()):
            if job_id in seen:
                continue
            processing_worker_id = redis.hget(self._processing_key(), job_id)
            if not processing_worker_id:
                continue
            job_data = redis.hget(self._jobs_key(), job_id)
            if not job_data:
                continue
            try:
                job = Job.from_dict(json.loads(job_data))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if job.status == JobStatus.PROCESSING and job.worker_id == processing_worker_id:
                seen.add(job_id)
        return len(seen)

    def _get_redis(self) -> Any:
        """Get Redis connection (lazy import)."""
        if self._redis is None:
            try:
                redis_module = importlib.import_module("redis")
                self._redis = redis_module.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    decode_responses=True,
                )
            except ImportError as e:
                raise ImportError(
                    "Redis support requires the 'redis' package. Install it with: pip install redis"
                ) from e
        return self._redis

    def enqueue(
        self,
        image_paths: list[str],
        policy: dict[str, Any],
        job_id_prefix: str | None = None,
    ) -> list[str]:
        """Enqueue jobs for processing.

        Args:
            image_paths: List of image paths to process.
            policy: Policy configuration as dict.
            job_id_prefix: Optional prefix for job IDs.

        Returns:
            List of enqueued job IDs.
        """
        redis = self._get_redis()
        job_ids: list[str] = []
        now = datetime.now(UTC).isoformat()

        for path in image_paths:
            job_id = f"{job_id_prefix or ''}{uuid.uuid4().hex[:12]}"
            job = Job(
                id=job_id,
                image_path=path,
                policy=policy,
                status=JobStatus.QUEUED,
                created_at=now,
            )
            redis.hset(self._jobs_key(), job_id, json.dumps(job.to_dict()))
            redis.rpush(self._pending_key(), job_id)
            job_ids.append(job_id)

        return job_ids

    def dequeue(self, worker_id: str, timeout: float | int = 0) -> Job | None:
        """Dequeue a job for processing.

        Args:
            worker_id: Worker identifier.
            timeout: Time to wait for job in seconds (0 = no wait, -1 = forever).

        Returns:
            Job if available, None otherwise.
        """
        redis = self._get_redis()
        timeout = self._normalize_timeout(timeout)
        deadline = time.monotonic() + float(timeout) if timeout > 0 else None
        is_first_wait = True
        max_stale_pops = 10 if timeout == 0 else 100
        stale_count = 0
        while True:
            if timeout == 0:
                job_id = redis.lpop(self._pending_key())
                if job_id is None:
                    return None
                if stale_count >= max_stale_pops:
                    redis.lpush(self._pending_key(), job_id)
                    return None
                result = (self._pending_key(), job_id)
            elif timeout == -1:
                result = redis.blpop(self._pending_key(), timeout=0)
            else:
                if deadline is None:
                    return None
                if is_first_wait:
                    remaining = float(timeout)
                    is_first_wait = False
                else:
                    remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                result = redis.blpop(self._pending_key(), timeout=remaining)

            if result is None:
                return None

            _, job_id = result
            job = self._load_live_job(redis, job_id)
            if job is None:
                if self._load_terminal_job(redis, job_id) is not None:
                    stale_count += 1
                    continue
                self._record_missing_job_metadata(redis, job_id)
                stale_count += 1
                continue

            if job.status != JobStatus.QUEUED:
                if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                    self._migrate_stale_live_terminal(redis, job)
                    stale_count += 1
                    continue
                redis.lrem(self._pending_key(), 0, job_id)
                stale_count += 1
                continue

            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(UTC).isoformat()
            job.completed_at = ""
            job.worker_id = worker_id
            job.attempt_id = uuid.uuid4().hex[:12]
            job.result = None
            job.error = None

            try:
                pipe = redis.pipeline(True)
                pipe.watch(self._jobs_key())
                # Re-verify job is still QUEUED after WATCH
                watched_data = pipe.hget(self._jobs_key(), job_id)
                if watched_data:
                    watched_job = self._decode_job(watched_data)
                    if watched_job is None or watched_job.status != JobStatus.QUEUED:
                        pipe.reset()
                        stale_count += 1
                        continue
                else:
                    pipe.reset()
                    stale_count += 1
                    continue
                pipe.multi()
                pipe.hset(self._jobs_key(), job_id, json.dumps(job.to_dict()))
                pipe.hset(self._processing_key(), job_id, worker_id)
                pipe.execute()
            except Exception as exc:
                if "WatchError" in type(exc).__name__:
                    # Another worker modified the job concurrently — retry
                    stale_count += 1
                    continue
                raise

            return job

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result: dict[str, Any],
        *,
        attempt_id: str,
    ) -> None:
        """Mark a job as completed.

        Args:
            job_id: Job identifier.
            worker_id: Worker identifier that owns the active attempt.
            result: Job result.
            attempt_id: Active attempt identifier.
        """
        redis = self._get_redis()

        job = self._load_job(redis, job_id)
        if job is None:
            return

        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            return
        if job.worker_id != worker_id:
            return
        if job.attempt_id != attempt_id:
            return
        if job.status != JobStatus.PROCESSING:
            self._mark_invalid_live_transition(redis, job, action="complete")
            return

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(UTC).isoformat()
        job.worker_id = ""
        job.attempt_id = ""
        job.error = None
        job.result = result

        self._persist_terminal_job(redis, job)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        retry: bool = False,
        *,
        attempt_id: str,
    ) -> None:
        """Mark a job as failed.

        Args:
            job_id: Job identifier.
            worker_id: Worker identifier that owns the active attempt.
            error: Error message.
            retry: Whether to retry the job.
            attempt_id: Active attempt identifier.
        """
        redis = self._get_redis()

        job = self._load_job(redis, job_id)
        if job is None:
            return

        if job.status in {JobStatus.FAILED, JobStatus.COMPLETED}:
            return
        if job.worker_id != worker_id:
            return
        if job.attempt_id != attempt_id:
            return
        if job.status != JobStatus.PROCESSING:
            self._mark_invalid_live_transition(redis, job, action="fail", error=error)
            return

        job.retries += 1
        job.error = error
        self._remove_job_from_lists(redis, job_id)

        if retry and job.retries < self.config.max_retries:
            job.status = JobStatus.QUEUED
            job.started_at = ""
            job.completed_at = ""
            job.worker_id = ""
            job.attempt_id = ""
            job.result = None
            pipe = redis.pipeline()
            pipe.rpush(self._pending_key(), job_id)
            pipe.hset(self._jobs_key(), job_id, json.dumps(job.to_dict()))
            pipe.hdel(self._processing_key(), job_id)
            pipe.execute()
        else:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(UTC).isoformat()
            job.worker_id = ""
            job.attempt_id = ""
            job.result = None
            self._persist_terminal_job(redis, job)

    def get_status(self, job_id: str) -> Job | None:
        """Get job status.

        Args:
            job_id: Job identifier.

        Returns:
            Job if found, None otherwise.
        """
        redis = self._get_redis()
        return self._load_job(redis, job_id)

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics.

        Returns:
            Dict with pending, processing, completed, failed counts.
        """
        redis = self._get_redis()

        return {
            "pending": self._count_live_pending_jobs(redis),
            "processing": self._count_live_processing_jobs(redis),
            "completed": self._count_live_terminal_jobs(
                redis, self._completed_key(), JobStatus.COMPLETED
            ),
            "failed": self._count_live_terminal_jobs(redis, self._failed_key(), JobStatus.FAILED),
        }

    def clear(self) -> None:
        """Clear all jobs from the queue."""
        redis = self._get_redis()
        terminal_ids = set(redis.lrange(self._completed_key(), 0, -1))
        terminal_ids.update(redis.lrange(self._failed_key(), 0, -1))
        terminal_keys = {self._terminal_job_key(job_id) for job_id in terminal_ids}

        scan_iter = getattr(redis, "scan_iter", None)
        if callable(scan_iter):
            try:
                terminal_keys.update(scan_iter(match=self._terminal_keys_pattern()))
            except TypeError:
                terminal_keys.update(scan_iter(self._terminal_keys_pattern()))
        else:
            keys = getattr(redis, "keys", None)
            if callable(keys):
                terminal_keys.update(keys(self._terminal_keys_pattern()))

        for key in terminal_keys:
            redis.delete(key)

        redis.delete(self._jobs_key())
        redis.delete(self._pending_key())
        redis.delete(self._processing_key())
        redis.delete(self._completed_key())
        redis.delete(self._failed_key())


class Worker:
    """Worker that processes jobs from a Redis queue.

    Example:
        >>> queue = RedisJobQueue(RedisConfig())
        >>> worker = Worker(queue, process_func=process_image)
        >>> worker.start()  # Runs until interrupted
    """

    def __init__(
        self,
        queue: RedisJobQueue,
        process_func: Callable[[str, dict[str, Any]], dict[str, Any]],
        worker_id: str | None = None,
        poll_interval: float = 1.0,
        max_jobs: int | None = None,
    ):
        import uuid

        if poll_interval < 0:
            raise ValueError("poll_interval must be >= 0")
        if max_jobs is not None and max_jobs < 0:
            raise ValueError("max_jobs must be >= 0")

        self.queue = queue
        self.process_func = process_func
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.max_jobs = max_jobs
        self._running = False
        self._jobs_processed = 0

    def start(self) -> None:
        """Start processing jobs."""
        self._running = True
        self._jobs_processed = 0

        while self._running:
            if self.max_jobs is not None and self._jobs_processed >= self.max_jobs:
                break

            job = self.queue.dequeue(self.worker_id, timeout=self.poll_interval)

            if job is None:
                continue

            self._jobs_processed += 1

            try:
                result = self.process_func(job.image_path, job.policy)
                self.queue.complete(job.id, self.worker_id, result, attempt_id=job.attempt_id)
            except Exception as e:
                _permanent = (FileNotFoundError, PermissionError, ValueError, TypeError)
                retryable = not isinstance(e, _permanent)
                self.queue.fail(job.id, self.worker_id, str(e), retry=retryable, attempt_id=job.attempt_id)

    def stop(self) -> None:
        """Stop processing jobs."""
        self._running = False


def create_worker_process(
    redis_config: RedisConfig | None = None,
) -> Worker:
    """Create a worker process for the distributed queue.

    Args:
        redis_config: Redis configuration.

    Returns:
        Worker instance ready to process jobs.
    """
    from perceptimg import Policy

    def process_image(image_path: str, policy_dict: dict[str, Any]) -> dict[str, Any]:
        from perceptimg.core.optimizer import Optimizer

        p = Policy.from_dict(policy_dict)
        optimizer = Optimizer()
        result = optimizer.optimize(Path(image_path), p)

        return {
            "image_path": image_path,
            "output_format": result.report.chosen_format,
            "size_before_kb": result.report.size_before_kb,
            "size_after_kb": result.report.size_after_kb,
            "ssim": result.report.ssim,
            "psnr": result.report.psnr,
        }

    queue = RedisJobQueue(redis_config)
    return Worker(queue, process_image)
