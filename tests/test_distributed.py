from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

import pytest

from perceptimg.core.distributed import JobStatus, RedisConfig, RedisJobQueue, Worker


@dataclass
class _DummyJob:
    id: str = "job-1"
    image_path: str = "image.png"
    policy: dict[str, str] = None  # type: ignore[assignment]
    attempt_id: str = ""

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = {}


class _QueueThatStopsAfterOnePoll:
    def __init__(self) -> None:
        self.timeouts: list[float] = []
        self.worker: Worker | None = None

    def dequeue(self, worker_id: str, timeout: float = 0) -> None:
        self.timeouts.append(timeout)
        if self.worker is not None:
            self.worker._running = False
        return None


class _QueueWithSingleJob:
    def __init__(self) -> None:
        self.dequeue_calls = 0
        self.completed: list[tuple[str, dict[str, object]]] = []
        self.failed: list[tuple[str, str, bool]] = []
        self.worker: Worker | None = None

    def dequeue(self, worker_id: str, timeout: float = 0) -> _DummyJob | None:
        self.dequeue_calls += 1
        if self.dequeue_calls == 1:
            return _DummyJob()
        if self.worker is not None:
            self.worker._running = False
        return None

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result: dict[str, object],
        *,
        attempt_id: str | None = None,
    ) -> None:
        self.completed.append((job_id, {"worker_id": worker_id, **result}))

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        retry: bool = False,
        *,
        attempt_id: str | None = None,
    ) -> None:
        self.failed.append((job_id, f"{worker_id}:{error}", retry))


def test_worker_preserves_fractional_poll_interval() -> None:
    queue = _QueueThatStopsAfterOnePoll()
    worker = Worker(queue, process_func=lambda path, policy: {}, poll_interval=0.5)
    queue.worker = worker

    worker.start()

    assert queue.timeouts == [0.5]


def test_worker_zero_max_jobs_processes_nothing() -> None:
    queue = _QueueWithSingleJob()
    worker = Worker(queue, process_func=lambda path, policy: {}, max_jobs=0)
    queue.worker = worker

    worker.start()

    assert queue.dequeue_calls == 0
    assert queue.completed == []
    assert queue.failed == []


def test_worker_max_jobs_counts_failed_jobs() -> None:
    queue = _QueueWithSingleJob()
    worker = Worker(
        queue,
        process_func=lambda path, policy: (_ for _ in ()).throw(ValueError("boom")),
        max_jobs=1,
    )
    queue.worker = worker

    worker.start()

    assert queue.dequeue_calls == 1
    assert queue.completed == []
    assert len(queue.failed) == 1
    assert queue.failed[0][0] == "job-1"


def test_worker_does_not_retry_permanent_errors() -> None:
    """Permanent errors like FileNotFoundError should not be retried."""
    queue = _QueueWithSingleJob()
    worker = Worker(
        queue,
        process_func=lambda path, policy: (_ for _ in ()).throw(FileNotFoundError("gone")),
        max_jobs=1,
    )
    queue.worker = worker

    worker.start()

    assert len(queue.failed) == 1
    assert queue.failed[0][2] is False  # retry=False


def test_worker_retries_transient_errors() -> None:
    """Transient errors like ConnectionError should be retried."""
    queue = _QueueWithSingleJob()
    worker = Worker(
        queue,
        process_func=lambda path, policy: (_ for _ in ()).throw(ConnectionError("timeout")),
        max_jobs=1,
    )
    queue.worker = worker

    worker.start()

    assert len(queue.failed) == 1
    assert queue.failed[0][2] is True  # retry=True


def test_worker_rejects_negative_poll_interval() -> None:
    with pytest.raises(ValueError, match="poll_interval must be >= 0"):
        Worker(
            _QueueThatStopsAfterOnePoll(),
            process_func=lambda path, policy: {},
            poll_interval=-0.1,
        )


def test_worker_rejects_negative_max_jobs() -> None:
    with pytest.raises(ValueError, match="max_jobs must be >= 0"):
        Worker(
            _QueueThatStopsAfterOnePoll(),
            process_func=lambda path, policy: {},
            max_jobs=-1,
        )


class _DummyPipeline:
    """Buffers Redis commands and executes them atomically.

    Supports WATCH/MULTI/EXEC for optimistic locking.
    In immediate mode (before multi()), commands execute directly.
    After multi(), commands are buffered until execute().
    """

    def __init__(self, redis: _DummyRedis, transactional: bool = False) -> None:
        self._redis = redis
        self._commands: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self._transactional = transactional
        self._in_multi = False
        self._watched_keys: list[str] = []

    def watch(self, *keys: str) -> None:
        self._watched_keys.extend(keys)

    def multi(self) -> None:
        self._in_multi = True

    def reset(self) -> None:
        self._commands.clear()
        self._in_multi = False
        self._watched_keys.clear()

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._redis, name)

        def buffered(*args: Any, **kwargs: Any) -> Any:
            if self._transactional and not self._in_multi:
                # Before multi(), execute immediately (like real Redis pipeline)
                return method(*args, **kwargs)
            self._commands.append((method, args, kwargs))
            return self

        return buffered

    def execute(self) -> list[Any]:
        results: list[Any] = []
        for method, args, kwargs in self._commands:
            results.append(method(*args, **kwargs))
        self._commands.clear()
        self._in_multi = False
        self._watched_keys.clear()
        return results


class _DummyRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.lists: dict[str, list[str]] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self.blpop_calls: list[tuple[str, float | int]] = []
        self.lpop_calls: list[str] = []

    def hset(self, key: str, field: str, value: str) -> None:
        self.hashes.setdefault(key, {})[field] = value

    def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get(key, {}).get(field)

    def hdel(self, key: str, field: str) -> None:
        self.hashes.get(key, {}).pop(field, None)

    def hlen(self, key: str) -> int:
        return len(self.hashes.get(key, {}))

    def hkeys(self, key: str) -> list[str]:
        return list(self.hashes.get(key, {}).keys())

    def rpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).append(value)

    def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        if end == -1:
            return list(values[start:])
        return list(values[start : end + 1])

    def lrem(self, key: str, count: int, value: str) -> None:
        if key not in self.lists:
            return  # Match real Redis: LREM on non-existent key does nothing
        values = self.lists[key]
        if count == 0:
            self.lists[key] = [item for item in values if item != value]
            return
        removed = 0
        updated: list[str] = []
        for item in values:
            if item == value and removed < abs(count):
                removed += 1
                continue
            updated.append(item)
        self.lists[key] = updated

    def lpop(self, key: str) -> str | None:
        self.lpop_calls.append(key)
        values = self.lists.get(key, [])
        if not values:
            return None
        return values.pop(0)

    def blpop(self, key: str, timeout: float | int = 0) -> tuple[str, str] | None:
        self.blpop_calls.append((key, timeout))
        values = self.lists.get(key, [])
        if not values:
            return None
        return (key, values.pop(0))

    def expire(self, key: str, ttl: int) -> None:
        self.expire_calls.append((key, ttl))

    def delete(self, key: str) -> None:
        self.hashes.pop(key, None)
        self.lists.pop(key, None)

    def keys(self, pattern: str) -> list[str]:
        all_keys = set(self.hashes) | set(self.lists)
        return sorted(key for key in all_keys if fnmatch(key, pattern))

    def scan_iter(self, match: str | None = None) -> list[str]:
        if match is None:
            return self.keys("*")
        return self.keys(match)

    def pipeline(self, transaction: bool = False) -> _DummyPipeline:
        return _DummyPipeline(self, transactional=transaction)


class _DummyRedisJobQueue(RedisJobQueue):
    def __init__(self, redis: _DummyRedis, *, result_ttl: int = 0) -> None:
        self.config = RedisConfig(queue_name="perceptimg:test", result_ttl=result_ttl)
        self._redis = redis

    def _get_redis(self) -> _DummyRedis:
        return self._redis


def _terminal_job_key(job_id: str) -> str:
    return f"perceptimg:test:terminal:{job_id}"


def _stored_job_data(redis: _DummyRedis, job_id: str) -> dict[str, Any]:
    job_data = redis.hget("perceptimg:test:jobs", job_id)
    if job_data is None:
        job_data = redis.hget(_terminal_job_key(job_id), "data")
    assert job_data is not None
    return json.loads(job_data)


def test_dequeue_rejects_timeout_less_than_minus_one() -> None:
    queue = _DummyRedisJobQueue(_DummyRedis())

    with pytest.raises(ValueError, match="timeout must be >= -1"):
        queue.dequeue("worker-1", timeout=-2)


def test_dequeue_translates_minus_one_timeout_to_forever() -> None:
    redis = _DummyRedis()
    queue = _DummyRedisJobQueue(redis)

    queue.dequeue("worker-1", timeout=-1)

    assert redis.blpop_calls == [("perceptimg:test:pending", 0)]
    assert redis.lpop_calls == []


def test_dequeue_zero_timeout_is_nonblocking() -> None:
    redis = _DummyRedis()
    queue = _DummyRedisJobQueue(redis)

    queue.dequeue("worker-1", timeout=0)

    assert redis.lpop_calls == ["perceptimg:test:pending"]
    assert redis.blpop_calls == []


def test_dequeue_preserves_fractional_timeout() -> None:
    redis = _DummyRedis()
    queue = _DummyRedisJobQueue(redis)

    queue.dequeue("worker-1", timeout=0.25)

    assert redis.blpop_calls == [("perceptimg:test:pending", 0.25)]
    assert redis.lpop_calls == []


def test_dequeue_preserves_remaining_timeout_after_stale_id() -> None:
    class _StaleThenEmptyRedis(_DummyRedis):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def blpop(self, key: str, timeout: float | int = 0) -> tuple[str, str] | None:
            self.blpop_calls.append((key, timeout))
            self.calls += 1
            if self.calls == 1:
                return (key, "stale")
            return None

    redis = _StaleThenEmptyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "stale",
        json.dumps(_DummyJob(id="stale").__dict__ | {"status": JobStatus.COMPLETED.value}),
    )
    queue = _DummyRedisJobQueue(redis)

    result = queue.dequeue("worker-1", timeout=5)

    assert result is None
    assert len(redis.blpop_calls) == 2
    assert redis.blpop_calls[0] == ("perceptimg:test:pending", 5)
    assert 0 < float(redis.blpop_calls[1][1]) <= 5


def test_dequeue_marks_missing_job_metadata_as_failed() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "job-1")
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    result = queue.dequeue("worker-1", timeout=5)

    assert result is None
    assert redis.lists["perceptimg:test:failed"] == ["job-1"]
    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.FAILED.value
    assert job["error"] == "Missing job metadata"
    assert ("perceptimg:test:jobs", 10) not in redis.expire_calls
    assert (_terminal_job_key("job-1"), 10) in redis.expire_calls
    assert ("perceptimg:test:failed", 10) in redis.expire_calls


def test_dequeue_ignores_non_queued_jobs_left_in_pending() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "job-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    result = queue.dequeue("worker-1", timeout=0)

    assert result is None
    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["result"] == {"ok": True}
    assert redis.hget("perceptimg:test:jobs", "job-1") is None
    assert redis.hget(_terminal_job_key("job-1"), "data") is not None
    assert redis.lists.get("perceptimg:test:pending") == []
    assert redis.hashes.get("perceptimg:test:processing") is None


def test_dequeue_stale_live_terminal_migrates_to_terminal_ttl_without_rehydrating_indexes() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "job-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    result = queue.dequeue("worker-1", timeout=0)

    assert result is None
    assert redis.hget("perceptimg:test:jobs", "job-1") is None
    assert redis.hget(_terminal_job_key("job-1"), "data") is not None
    assert redis.lists.get("perceptimg:test:completed") in (None, [])
    assert redis.expire_calls == [(_terminal_job_key("job-1"), 10)]
    assert queue.get_stats() == {"pending": 0, "processing": 0, "completed": 1, "failed": 0}


def test_dequeue_terminal_pending_residue_does_not_refresh_terminal_ttl() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "job-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    result = queue.dequeue("worker-1", timeout=0)

    assert result is None
    assert redis.hget("perceptimg:test:jobs", "job-1") is None
    assert redis.hget(_terminal_job_key("job-1"), "data") is not None
    assert redis.lists.get("perceptimg:test:pending") == []
    assert redis.lists.get("perceptimg:test:completed") in (None, [])
    assert redis.expire_calls == [(_terminal_job_key("job-1"), 10)]


def test_dequeue_discards_terminal_pending_residue_without_recording_missing_metadata() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "job-1")
    redis.hset(
        _terminal_job_key("job-1"),
        "data",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    result = queue.dequeue("worker-1", timeout=0)

    assert result is None
    assert _stored_job_data(redis, "job-1")["status"] == JobStatus.COMPLETED.value
    assert redis.lists.get("perceptimg:test:pending") == []
    assert redis.lists.get("perceptimg:test:failed") in (None, [])
    assert ("perceptimg:test:failed", 10) not in redis.expire_calls


def test_dequeue_stale_live_terminal_does_not_refresh_existing_terminal_ttl() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "job-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    redis.hset(
        _terminal_job_key("job-1"),
        "data",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    result = queue.dequeue("worker-1", timeout=0)

    assert result is None
    assert redis.hget("perceptimg:test:jobs", "job-1") is None
    assert redis.hget(_terminal_job_key("job-1"), "data") is not None
    assert redis.lists.get("perceptimg:test:completed") in (None, [])
    assert redis.expire_calls == []
    assert queue.get_stats() == {"pending": 0, "processing": 0, "completed": 1, "failed": 0}


def test_get_stats_counts_orphan_terminal_key_without_completed_index() -> None:
    redis = _DummyRedis()
    redis.hset(
        _terminal_job_key("job-orphan"),
        "data",
        json.dumps(
            _DummyJob(id="job-orphan").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats == {"pending": 0, "processing": 0, "completed": 1, "failed": 0}


def test_dequeue_skips_stale_pending_and_returns_next_valid_job() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "stale")
    redis.rpush("perceptimg:test:pending", "live")
    redis.hset(
        "perceptimg:test:jobs",
        "stale",
        json.dumps(
            _DummyJob(id="stale").__dict__
            | {"status": JobStatus.COMPLETED.value, "result": {"ok": True}}
        ),
    )
    redis.hset(
        "perceptimg:test:jobs",
        "live",
        json.dumps(_DummyJob(id="live").__dict__ | {"status": JobStatus.QUEUED.value}),
    )
    queue = _DummyRedisJobQueue(redis)

    job = queue.dequeue("worker-1", timeout=0)

    assert job is not None
    assert job.id == "live"
    assert job.worker_id == "worker-1"
    assert redis.lists.get("perceptimg:test:pending") == []
    assert redis.hashes["perceptimg:test:processing"] == {"live": "worker-1"}


def test_complete_applies_ttl_to_completed_stats_key() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="attempt-1")

    assert ("perceptimg:test:jobs", 10) not in redis.expire_calls
    assert (_terminal_job_key("job-1"), 10) in redis.expire_calls
    assert ("perceptimg:test:completed", 10) in redis.expire_calls


def test_get_status_reads_terminal_snapshot_until_expiration() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="attempt-1")

    current = queue.get_status("job-1")
    assert current is not None
    assert current.status == JobStatus.COMPLETED
    assert current.result == {"ok": True}
    assert redis.hget("perceptimg:test:jobs", "job-1") is None

    redis.delete(_terminal_job_key("job-1"))

    assert queue.get_status("job-1") is None


def test_clear_deletes_orphan_terminal_job_keys() -> None:
    redis = _DummyRedis()
    redis.hset(
        _terminal_job_key("job-orphan"),
        "data",
        json.dumps(
            _DummyJob(id="job-orphan").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    queue.clear()

    assert queue.get_status("job-orphan") is None
    assert _terminal_job_key("job-orphan") not in redis.hashes


def test_complete_ignores_spurious_event_for_queued_job() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(_DummyJob(id="job-1").__dict__ | {"status": JobStatus.QUEUED.value}),
    )
    redis.rpush("perceptimg:test:pending", "job-1")
    queue = _DummyRedisJobQueue(redis)

    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="late-attempt")

    job = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert job["status"] == JobStatus.QUEUED.value
    assert job.get("error") is None
    assert job.get("result") is None
    assert redis.lists.get("perceptimg:test:pending") == ["job-1"]
    assert redis.lists.get("perceptimg:test:completed") in (None, [])
    assert redis.lists.get("perceptimg:test:failed") in (None, [])
    assert queue.get_stats() == {"pending": 1, "processing": 0, "completed": 0, "failed": 0}
    assert queue.dequeue("worker-1", timeout=0) is not None


def test_complete_preserves_failed_jobs_on_late_event() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.FAILED.value,
                "error": "orig",
            }
        ),
    )
    redis.rpush("perceptimg:test:failed", "job-1")
    queue = _DummyRedisJobQueue(redis)

    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="late-attempt")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.FAILED.value
    assert job["error"] == "orig"
    assert job.get("result") is None
    assert redis.lists["perceptimg:test:failed"] == ["job-1"]
    assert redis.lists.get("perceptimg:test:completed") in (None, [])


def test_complete_is_idempotent_and_cleans_stale_pending_entries() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    redis.rpush("perceptimg:test:pending", "job-1")
    queue = _DummyRedisJobQueue(redis)

    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="attempt-1")
    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="attempt-1")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.COMPLETED.value
    assert redis.lists.get("perceptimg:test:pending") == []
    assert redis.lists["perceptimg:test:completed"] == ["job-1"]
    assert queue.get_stats() == {"pending": 0, "processing": 0, "completed": 1, "failed": 0}


def test_complete_on_terminal_snapshot_does_not_rehydrate_completed_index_or_refresh_ttl() -> None:
    redis = _DummyRedis()
    redis.hset(
        _terminal_job_key("job-1"),
        "data",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="late-attempt")

    assert redis.lists.get("perceptimg:test:completed") in (None, [])
    assert redis.expire_calls == []
    current = queue.get_status("job-1")
    assert current is not None
    assert current.status == JobStatus.COMPLETED
    assert current.result == {"ok": True}


def test_fail_applies_ttl_only_for_terminal_failures() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.fail("job-1", "worker-1", "boom", retry=False, attempt_id="attempt-1")

    assert ("perceptimg:test:jobs", 10) not in redis.expire_calls
    assert (_terminal_job_key("job-1"), 10) in redis.expire_calls
    assert ("perceptimg:test:failed", 10) in redis.expire_calls

    redis_retry = _DummyRedis()
    redis_retry.hset(
        "perceptimg:test:jobs",
        "job-2",
        json.dumps(
            _DummyJob(id="job-2").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-2",
            }
        ),
    )
    queue_retry = _DummyRedisJobQueue(redis_retry, result_ttl=10)
    queue_retry.fail("job-2", "worker-1", "retry", retry=True, attempt_id="attempt-2")

    assert ("perceptimg:test:failed", 10) not in redis_retry.expire_calls


def test_terminal_ttl_does_not_expire_global_jobs_hash_with_live_jobs() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "queued-1",
        json.dumps(_DummyJob(id="queued-1").__dict__ | {"status": JobStatus.QUEUED.value}),
    )
    redis.rpush("perceptimg:test:pending", "queued-1")
    redis.hset(
        "perceptimg:test:jobs",
        "proc-1",
        json.dumps(
            _DummyJob(id="proc-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.fail("proc-1", "worker-1", "boom", retry=False, attempt_id="attempt-1")

    assert ("perceptimg:test:jobs", 10) not in redis.expire_calls
    assert (_terminal_job_key("proc-1"), 10) in redis.expire_calls
    assert ("perceptimg:test:failed", 10) in redis.expire_calls
    assert redis.hget("perceptimg:test:jobs", "queued-1") is not None
    assert redis.hget("perceptimg:test:jobs", "proc-1") is None


def test_fail_retry_preserves_existing_terminal_snapshot() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    redis.hset(
        _terminal_job_key("job-1"),
        "data",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.fail("job-1", "worker-1", "retry", retry=True, attempt_id="attempt-1")

    current = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert current["status"] == JobStatus.QUEUED.value
    assert redis.hget(_terminal_job_key("job-1"), "data") is not None
    assert redis.hashes.get("perceptimg:test:processing") == {}
    assert redis.lists.get("perceptimg:test:pending") == ["job-1"]


def test_fail_ignores_spurious_event_for_queued_job() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(_DummyJob(id="job-1").__dict__ | {"status": JobStatus.QUEUED.value}),
    )
    redis.rpush("perceptimg:test:pending", "job-1")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-1", "boom", retry=False, attempt_id="late-attempt")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.QUEUED.value
    assert job.get("retries", 0) == 0
    assert job.get("error") is None
    assert redis.lists.get("perceptimg:test:pending") == ["job-1"]
    assert redis.lists.get("perceptimg:test:failed") in (None, [])
    assert queue.get_stats() == {"pending": 1, "processing": 0, "completed": 0, "failed": 0}


def test_fail_preserves_completed_jobs_on_late_event() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    redis.rpush("perceptimg:test:completed", "job-1")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-1", "boom", retry=False, attempt_id="late-attempt")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["result"] == {"ok": True}
    assert job.get("error") is None
    assert redis.lists["perceptimg:test:completed"] == ["job-1"]
    assert redis.lists.get("perceptimg:test:failed") in (None, [])


def test_fail_is_idempotent_and_cleans_stale_pending_entries() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    redis.rpush("perceptimg:test:pending", "job-1")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-1", "boom", retry=False, attempt_id="attempt-1")
    queue.fail("job-1", "worker-1", "boom", retry=False, attempt_id="attempt-1")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.FAILED.value
    assert job["retries"] == 1
    assert redis.lists.get("perceptimg:test:pending") == []
    assert redis.lists["perceptimg:test:failed"] == ["job-1"]
    assert queue.get_stats() == {"pending": 0, "processing": 0, "completed": 0, "failed": 1}


def test_fail_on_terminal_snapshot_does_not_rehydrate_failed_index_or_refresh_ttl() -> None:
    redis = _DummyRedis()
    redis.hset(
        _terminal_job_key("job-1"),
        "data",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.FAILED.value,
                "error": "boom",
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.fail("job-1", "worker-1", "boom", retry=False, attempt_id="late-attempt")

    assert redis.lists.get("perceptimg:test:failed") in (None, [])
    assert redis.expire_calls == []
    current = queue.get_status("job-1")
    assert current is not None
    assert current.status == JobStatus.FAILED
    assert current.error == "boom"


def test_fail_on_completed_terminal_snapshot_does_not_rehydrate_indexes_or_refresh_ttl() -> None:
    redis = _DummyRedis()
    redis.hset(
        _terminal_job_key("job-1"),
        "data",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.COMPLETED.value,
                "result": {"ok": True},
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis, result_ttl=10)

    queue.fail("job-1", "worker-1", "late boom", retry=False, attempt_id="late-attempt")

    assert redis.lists.get("perceptimg:test:completed") in (None, [])
    assert redis.lists.get("perceptimg:test:failed") in (None, [])
    assert redis.expire_calls == []
    current = queue.get_status("job-1")
    assert current is not None
    assert current.status == JobStatus.COMPLETED
    assert current.result == {"ok": True}


def test_fail_retry_resets_processing_metadata() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "started_at": "2026-04-03T10:00:00",
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
                "error": None,
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-1", "boom", retry=True, attempt_id="attempt-1")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.QUEUED.value
    assert job["started_at"] == ""
    assert job["completed_at"] == ""
    assert job["worker_id"] == ""
    assert job["error"] == "boom"
    assert job["retries"] == 1
    assert redis.lists["perceptimg:test:pending"] == ["job-1"]


def test_complete_late_event_preserves_requeued_retry() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-old",
                "attempt_id": "attempt-old",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-old")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-old", "boom", retry=True, attempt_id="attempt-old")
    queue.complete("job-1", "worker-old", {"ok": True}, attempt_id="attempt-old")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.QUEUED.value
    assert job["retries"] == 1
    assert job["error"] == "boom"
    assert job["result"] is None
    assert redis.lists["perceptimg:test:pending"] == ["job-1"]
    assert "perceptimg:test:failed" not in redis.lists


def test_fail_late_event_preserves_requeued_retry() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-old",
                "attempt_id": "attempt-old",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-old")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-old", "boom", retry=True, attempt_id="attempt-old")
    queue.fail("job-1", "worker-old", "late boom", retry=False, attempt_id="attempt-old")

    job = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert job["status"] == JobStatus.QUEUED.value
    assert job["retries"] == 1
    assert job["error"] == "boom"
    assert redis.lists["perceptimg:test:pending"] == ["job-1"]
    assert "perceptimg:test:failed" not in redis.lists


def test_complete_old_worker_cannot_close_new_processing_attempt() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-old",
                "attempt_id": "attempt-old",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-old")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-old", "boom", retry=True, attempt_id="attempt-old")
    job = queue.dequeue("worker-new", timeout=0)
    assert job is not None

    queue.complete("job-1", "worker-old", {"ok": True, "from": "old"}, attempt_id="attempt-old")

    current = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert current["status"] == JobStatus.PROCESSING.value
    assert current["worker_id"] == "worker-new"
    assert current.get("result") is None
    assert redis.hashes["perceptimg:test:processing"] == {"job-1": "worker-new"}
    assert "perceptimg:test:completed" not in redis.lists


def test_complete_old_attempt_same_worker_cannot_close_new_processing_attempt() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-old",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-1", "boom", retry=True, attempt_id="attempt-old")
    job = queue.dequeue("worker-1", timeout=0)
    assert job is not None
    assert job.attempt_id != "attempt-old"

    queue.complete("job-1", "worker-1", {"ok": True, "from": "old"}, attempt_id="attempt-old")

    current = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert current["status"] == JobStatus.PROCESSING.value
    assert current["worker_id"] == "worker-1"
    assert current["attempt_id"] == job.attempt_id
    assert current.get("result") is None
    assert redis.hashes["perceptimg:test:processing"] == {"job-1": "worker-1"}
    assert "perceptimg:test:completed" not in redis.lists


def test_complete_requires_attempt_id_for_active_processing_job() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    queue = _DummyRedisJobQueue(redis)

    with pytest.raises(TypeError):
        queue.complete("job-1", "worker-1", {"ok": True})

    current = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert current["status"] == JobStatus.PROCESSING.value
    assert current["attempt_id"] == "attempt-1"
    assert redis.hashes["perceptimg:test:processing"] == {"job-1": "worker-1"}


def test_fail_old_worker_cannot_fail_new_processing_attempt() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-old",
                "attempt_id": "attempt-old",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-old")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-old", "boom", retry=True, attempt_id="attempt-old")
    job = queue.dequeue("worker-new", timeout=0)
    assert job is not None

    queue.fail("job-1", "worker-old", "late boom", retry=False, attempt_id="attempt-old")

    current = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert current["status"] == JobStatus.PROCESSING.value
    assert current["worker_id"] == "worker-new"
    assert current.get("error") is None
    assert redis.hashes["perceptimg:test:processing"] == {"job-1": "worker-new"}
    assert "perceptimg:test:failed" not in redis.lists


def test_fail_requires_attempt_id_for_active_processing_job() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    queue = _DummyRedisJobQueue(redis)

    with pytest.raises(TypeError):
        queue.fail("job-1", "worker-1", "boom", retry=False)

    current = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert current["status"] == JobStatus.PROCESSING.value
    assert current["attempt_id"] == "attempt-1"
    assert redis.hashes["perceptimg:test:processing"] == {"job-1": "worker-1"}


def test_fail_old_attempt_same_worker_cannot_fail_new_processing_attempt() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-old",
            }
        ),
    )
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    queue = _DummyRedisJobQueue(redis)

    queue.fail("job-1", "worker-1", "boom", retry=True, attempt_id="attempt-old")
    job = queue.dequeue("worker-1", timeout=0)
    assert job is not None
    assert job.attempt_id != "attempt-old"

    queue.fail("job-1", "worker-1", "late boom", retry=False, attempt_id="attempt-old")

    current = json.loads(redis.hget("perceptimg:test:jobs", "job-1"))
    assert current["status"] == JobStatus.PROCESSING.value
    assert current["worker_id"] == "worker-1"
    assert current["attempt_id"] == job.attempt_id
    assert current.get("error") is None
    assert redis.hashes["perceptimg:test:processing"] == {"job-1": "worker-1"}
    assert "perceptimg:test:failed" not in redis.lists


def test_complete_clears_stale_error_after_retry() -> None:
    redis = _DummyRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "started_at": "2026-04-03T10:00:00",
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
                "error": "boom",
                "retries": 1,
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    queue.complete("job-1", "worker-1", {"ok": True}, attempt_id="attempt-1")

    job = _stored_job_data(redis, "job-1")
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["error"] is None
    assert job["worker_id"] == ""
    assert job["result"] == {"ok": True}


def test_get_stats_ignores_stale_terminal_job_ids() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:completed", "stale-completed")
    redis.rpush("perceptimg:test:failed", "stale-failed")
    redis.rpush("perceptimg:test:completed", "live-completed")
    redis.rpush("perceptimg:test:failed", "live-failed")
    redis.hset(
        "perceptimg:test:jobs",
        "live-completed",
        json.dumps(_DummyJob(id="live-completed").__dict__ | {"status": JobStatus.COMPLETED.value}),
    )
    redis.hset(
        "perceptimg:test:jobs",
        "live-failed",
        json.dumps(_DummyJob(id="live-failed").__dict__ | {"status": JobStatus.FAILED.value}),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["completed"] == 1
    assert stats["failed"] == 1
    assert redis.lists["perceptimg:test:completed"] == ["stale-completed", "live-completed"]
    assert redis.lists["perceptimg:test:failed"] == ["stale-failed", "live-failed"]


def test_get_stats_removes_non_terminal_jobs_from_terminal_lists() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:completed", "queued-job")
    redis.rpush("perceptimg:test:failed", "processing-job")
    redis.hset(
        "perceptimg:test:jobs",
        "queued-job",
        json.dumps(_DummyJob(id="queued-job").__dict__ | {"status": JobStatus.QUEUED.value}),
    )
    redis.hset(
        "perceptimg:test:jobs",
        "processing-job",
        json.dumps(
            _DummyJob(id="processing-job").__dict__ | {"status": JobStatus.PROCESSING.value}
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["completed"] == 0
    assert stats["failed"] == 0
    assert redis.lists["perceptimg:test:completed"] == ["queued-job"]
    assert redis.lists["perceptimg:test:failed"] == ["processing-job"]


def test_get_stats_deduplicates_terminal_job_ids() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:completed", "job-1")
    redis.rpush("perceptimg:test:completed", "job-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(_DummyJob(id="job-1").__dict__ | {"status": JobStatus.COMPLETED.value}),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["completed"] == 1
    assert redis.lists["perceptimg:test:completed"] == ["job-1", "job-1"]


def test_get_stats_removes_non_queued_jobs_from_pending() -> None:
    redis = _DummyRedis()
    redis.rpush("perceptimg:test:pending", "job-1")
    redis.rpush("perceptimg:test:pending", "job-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(_DummyJob(id="job-1").__dict__ | {"status": JobStatus.COMPLETED.value}),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["pending"] == 0
    assert redis.lists["perceptimg:test:pending"] == ["job-1", "job-1"]


def test_get_stats_removes_non_processing_jobs_from_processing() -> None:
    redis = _DummyRedis()
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(_DummyJob(id="job-1").__dict__ | {"status": JobStatus.FAILED.value}),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["processing"] == 0
    assert redis.hashes.get("perceptimg:test:processing") == {"job-1": "worker-1"}


def test_get_stats_ignores_processing_entries_with_mismatched_worker_owner() -> None:
    redis = _DummyRedis()
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-2",
                "attempt_id": "attempt-1",
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["processing"] == 0
    assert redis.hashes.get("perceptimg:test:processing") == {"job-1": "worker-1"}


def test_get_stats_counts_processing_entries_with_matching_worker_owner() -> None:
    redis = _DummyRedis()
    redis.hset("perceptimg:test:processing", "job-1", "worker-1")
    redis.hset(
        "perceptimg:test:jobs",
        "job-1",
        json.dumps(
            _DummyJob(id="job-1").__dict__
            | {
                "status": JobStatus.PROCESSING.value,
                "worker_id": "worker-1",
                "attempt_id": "attempt-1",
            }
        ),
    )
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["processing"] == 1


def test_get_stats_does_not_drop_concurrent_pending_enqueue() -> None:
    class _RaceOnPendingReadRedis(_DummyRedis):
        def __init__(self) -> None:
            super().__init__()
            self._injected = False

        def lrange(self, key: str, start: int, end: int) -> list[str]:
            snapshot = super().lrange(key, start, end)
            if key == "perceptimg:test:pending" and not self._injected:
                self._injected = True
                self.hset(
                    "perceptimg:test:jobs",
                    "job-new",
                    json.dumps(
                        _DummyJob(id="job-new").__dict__ | {"status": JobStatus.QUEUED.value}
                    ),
                )
                self.rpush("perceptimg:test:pending", "job-new")
            return snapshot

    redis = _RaceOnPendingReadRedis()
    redis.hset(
        "perceptimg:test:jobs",
        "job-stale",
        json.dumps(_DummyJob(id="job-stale").__dict__ | {"status": JobStatus.COMPLETED.value}),
    )
    redis.rpush("perceptimg:test:pending", "job-stale")
    queue = _DummyRedisJobQueue(redis)

    stats = queue.get_stats()

    assert stats["pending"] == 0
    assert redis.lists["perceptimg:test:pending"] == ["job-stale", "job-new"]
