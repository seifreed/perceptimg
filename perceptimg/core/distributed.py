"""Distributed job queue using Redis for horizontal scaling."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
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
        result_ttl: Time to keep completed jobs in seconds (default: 3600).
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    queue_name: str = "perceptimg:jobs"
    result_ttl: int = 3600


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
        ...     queue.complete(job.id, result)
    """

    def __init__(self, config: RedisConfig | None = None):
        self.config = config or RedisConfig()
        self._redis: Any = None

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
        import uuid

        redis = self._get_redis()
        job_ids: list[str] = []
        now = datetime.utcnow().isoformat()

        for path in image_paths:
            job_id = f"{job_id_prefix or ''}{uuid.uuid4().hex[:12]}"
            job = Job(
                id=job_id,
                image_path=path,
                policy=policy,
                status=JobStatus.QUEUED,
                created_at=now,
            )
            redis.hset(f"{self.config.queue_name}:jobs", job_id, json.dumps(job.to_dict()))
            redis.rpush(f"{self.config.queue_name}:pending", job_id)
            job_ids.append(job_id)

        return job_ids

    def dequeue(self, worker_id: str, timeout: int = 0) -> Job | None:
        """Dequeue a job for processing.

        Args:
            worker_id: Worker identifier.
            timeout: Time to wait for job in seconds (0 = no wait, -1 = forever).

        Returns:
            Job if available, None otherwise.
        """
        redis = self._get_redis()

        result = redis.blpop(f"{self.config.queue_name}:pending", timeout=timeout)
        if result is None:
            return None

        _, job_id = result
        job_data = redis.hget(f"{self.config.queue_name}:jobs", job_id)
        if not job_data:
            return None

        job = Job.from_dict(json.loads(job_data))
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow().isoformat()
        job.worker_id = worker_id

        redis.hset(f"{self.config.queue_name}:jobs", job_id, json.dumps(job.to_dict()))
        redis.hset(f"{self.config.queue_name}:processing", job_id, worker_id)

        return job

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        """Mark a job as completed.

        Args:
            job_id: Job identifier.
            result: Job result.
        """
        redis = self._get_redis()

        job_data = redis.hget(f"{self.config.queue_name}:jobs", job_id)
        if not job_data:
            return

        job = Job.from_dict(json.loads(job_data))
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow().isoformat()
        job.result = result

        redis.hset(f"{self.config.queue_name}:jobs", job_id, json.dumps(job.to_dict()))
        redis.hdel(f"{self.config.queue_name}:processing", job_id)
        redis.rpush(f"{self.config.queue_name}:completed", job_id)

        if self.config.result_ttl:
            redis.expire(f"{self.config.queue_name}:jobs", self.config.result_ttl)

    def fail(self, job_id: str, error: str, retry: bool = False) -> None:
        """Mark a job as failed.

        Args:
            job_id: Job identifier.
            error: Error message.
            retry: Whether to retry the job.
        """
        redis = self._get_redis()

        job_data = redis.hget(f"{self.config.queue_name}:jobs", job_id)
        if not job_data:
            return

        job = Job.from_dict(json.loads(job_data))
        job.retries += 1
        job.error = error

        if retry and job.retries < 3:
            job.status = JobStatus.QUEUED
            redis.rpush(f"{self.config.queue_name}:pending", job_id)
        else:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow().isoformat()
            redis.rpush(f"{self.config.queue_name}:failed", job_id)

        redis.hset(f"{self.config.queue_name}:jobs", job_id, json.dumps(job.to_dict()))
        redis.hdel(f"{self.config.queue_name}:processing", job_id)

    def get_status(self, job_id: str) -> Job | None:
        """Get job status.

        Args:
            job_id: Job identifier.

        Returns:
            Job if found, None otherwise.
        """
        redis = self._get_redis()

        job_data = redis.hget(f"{self.config.queue_name}:jobs", job_id)
        if not job_data:
            return None

        return Job.from_dict(json.loads(job_data))

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics.

        Returns:
            Dict with pending, processing, completed, failed counts.
        """
        redis = self._get_redis()

        return {
            "pending": redis.llen(f"{self.config.queue_name}:pending"),
            "processing": redis.hlen(f"{self.config.queue_name}:processing"),
            "completed": redis.llen(f"{self.config.queue_name}:completed"),
            "failed": redis.llen(f"{self.config.queue_name}:failed"),
        }

    def clear(self) -> None:
        """Clear all jobs from the queue."""
        redis = self._get_redis()

        redis.delete(f"{self.config.queue_name}:jobs")
        redis.delete(f"{self.config.queue_name}:pending")
        redis.delete(f"{self.config.queue_name}:processing")
        redis.delete(f"{self.config.queue_name}:completed")
        redis.delete(f"{self.config.queue_name}:failed")


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
            if self.max_jobs and self._jobs_processed >= self.max_jobs:
                break

            job = self.queue.dequeue(self.worker_id, timeout=int(self.poll_interval))

            if job is None:
                continue

            try:
                result = self.process_func(job.image_path, job.policy)
                self.queue.complete(job.id, result)
                self._jobs_processed += 1
            except Exception as e:
                self.queue.fail(job.id, str(e), retry=True)

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
