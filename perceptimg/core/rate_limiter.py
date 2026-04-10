"""Rate limiting with token bucket algorithm for controlled throughput."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite
from threading import Condition, Lock

_TOKEN_EPSILON = 1e-6


@dataclass(slots=True)
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_second: Maximum requests allowed per second (default: 10).
        burst_size: Maximum burst allowed (default: 20).
        wait_timeout_ms: Maximum time to wait for token (default: 5000).
    """

    requests_per_second: float = 10.0
    burst_size: int = 20
    wait_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if not isfinite(self.requests_per_second) or self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be a finite number > 0")
        if self.requests_per_second > 10000:
            raise ValueError(
                "requests_per_second must be <= 10000 "
                "(use a lower value or remove rate limiting)"
            )
        if self.burst_size < 1:
            raise ValueError("burst_size must be >= 1")
        if self.wait_timeout_ms < 0:
            raise ValueError("wait_timeout_ms must be >= 0")


class RateLimiter:
    """Token bucket rate limiter for controlling request throughput.

    Thread-safe implementation that allows bursts up to burst_size while
    maintaining an average rate of requests_per_second.

    Example:
        >>> limiter = RateLimiter(RateLimitConfig(requests_per_second=5))
        >>> for i in range(100):
        ...     limiter.acquire()  # Will block if rate exceeded
        ...     process_image()
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._tokens = float(self.config.burst_size)
        self._last_update = time.monotonic()
        self._cond = Condition(Lock())

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        self._tokens += elapsed * self.config.requests_per_second
        self._tokens = min(self._tokens, float(self.config.burst_size))

    def acquire(self, timeout_ms: int | None = None) -> bool:
        """Acquire a token, blocking if necessary.

        Args:
            timeout_ms: Maximum time to wait in milliseconds. Uses config default if None.

        Returns:
            True if token acquired, False if timeout expired.
        """
        if timeout_ms is None:
            timeout_ms = self.config.wait_timeout_ms
        deadline = time.monotonic() + timeout_ms / 1000.0

        with self._cond:
            while True:
                self._refill()

                if self._tokens >= 1.0 - _TOKEN_EPSILON:
                    self._tokens = max(0.0, self._tokens - 1.0)
                    return True

                if time.monotonic() >= deadline:
                    return False

                # Calculate time to next token
                tokens_needed = 1.0 - self._tokens
                wait_time = tokens_needed / self.config.requests_per_second
                wait_time = min(wait_time, deadline - time.monotonic())

                if wait_time <= 0:
                    return False
                self._cond.wait(timeout=wait_time)

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if rate limited.
        """
        with self._cond:
            self._refill()
            if self._tokens >= 1.0 - _TOKEN_EPSILON:
                self._tokens = max(0.0, self._tokens - 1.0)
                return True
            return False

    def get_tokens(self) -> float:
        """Get current number of available tokens.

        Returns:
            Number of tokens available (may be fractional).
        """
        with self._cond:
            self._refill()
            return self._tokens

    def reset(self) -> None:
        """Reset rate limiter to full burst capacity."""
        with self._cond:
            self._tokens = float(self.config.burst_size)
            self._last_update = time.monotonic()


class MultiRateLimiter:
    """Rate limiter for multiple resources with per-resource limits.

    Example:
        >>> limiter = MultiRateLimiter()
        >>> limiter.add_limit("api", RateLimitConfig(requests_per_second=10))
        >>> limiter.add_limit("disk", RateLimitConfig(requests_per_second=100))
        >>> limiter.acquire("api")
        >>> limiter.acquire("disk")
    """

    def __init__(self) -> None:
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = Lock()

    def add_limit(self, name: str, config: RateLimitConfig | None = None) -> None:
        """Add a rate limit for a named resource.

        Args:
            name: Resource name.
            config: Rate limit configuration.
        """
        with self._lock:
            self._limiters[name] = RateLimiter(config)

    def acquire(self, name: str, timeout_ms: int | None = None) -> bool:
        """Acquire a token for a named resource.

        Args:
            name: Resource name.
            timeout_ms: Maximum time to wait.

        Returns:
            True if token acquired, False if timeout.
        """
        limiter = self._limiters.get(name)
        if limiter:
            return limiter.acquire(timeout_ms)
        return True

    def try_acquire(self, name: str) -> bool:
        """Try to acquire a token for a named resource without blocking.

        Args:
            name: Resource name.

        Returns:
            True if token acquired, False if rate limited.
        """
        limiter = self._limiters.get(name)
        if limiter:
            return limiter.try_acquire()
        return True

    def reset(self, name: str | None = None) -> None:
        """Reset rate limiter(s).

        Args:
            name: Resource name, or None to reset all.
        """
        with self._lock:
            if name:
                limiter = self._limiters.get(name)
                if limiter:
                    limiter.reset()
            else:
                for limiter in self._limiters.values():
                    limiter.reset()
