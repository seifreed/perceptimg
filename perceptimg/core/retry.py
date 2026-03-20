"""Retry policies with exponential backoff for resilient batch processing."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

T = TypeVar("T")


class RetryDecision(StrEnum):
    """Decision after an error."""

    RETRY = "retry"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(slots=True)
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay_ms: Base delay in milliseconds before first retry (default: 100).
        max_delay_ms: Maximum delay in milliseconds between retries (default: 10000).
        exponential_base: Exponential multiplier for delay (default: 2).
        jitter_ms: Random jitter to add to delay (default: 50).
        retry_on: Exception types to retry on (default: all).
    """

    max_retries: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 10000
    exponential_base: float = 2.0
    jitter_ms: int = 50
    retry_on: tuple[type[Exception], ...] | None = None


@dataclass(slots=True)
class RetryResult:
    """Result of retry operation.

    Attributes:
        success: Whether the operation succeeded.
        result: The result if successful.
        error: The final error if failed.
        attempts: Number of attempts made.
        total_delay_ms: Total delay from retries in milliseconds.
    """

    success: bool
    result: Any = None
    error: Exception | None = None
    attempts: int = 1
    total_delay_ms: float = 0.0


class RetryPolicy:
    """Manages retry logic with exponential backoff.

    Example:
        >>> policy = RetryPolicy(RetryConfig(max_retries=3))
        >>> result = policy.execute(lambda: some_operation())
        >>> if result.success:
        ...     print(result.result)
        ... else:
        ...     print(f"Failed after {result.attempts} attempts: {result.error}")
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay in milliseconds for a given attempt.

        Uses exponential backoff with jitter to avoid thundering herd.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Delay in milliseconds.
        """
        delay = self.config.base_delay_ms * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay_ms)
        jitter = secrets.randbelow(self.config.jitter_ms + 1)
        return delay + jitter

    def should_retry(self, error: Exception) -> bool:
        """Determine if we should retry based on the error.

        Args:
            error: Exception that occurred.

        Returns:
            True if we should retry, False otherwise.
        """
        if self.config.retry_on is None:
            return True
        return isinstance(error, self.config.retry_on)

    def execute(
        self,
        operation: Callable[[], Any],
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> RetryResult:
        """Execute an operation with retry logic.

        Args:
            operation: Callable to execute.
            on_retry: Optional callback for retry events: (attempt, error, delay_ms).

        Returns:
            RetryResult with success status and result/error.
        """
        attempts = 0
        total_delay_ms = 0.0
        last_error: Exception | None = None

        while attempts < self.config.max_retries + 1:
            attempts += 1
            try:
                result = operation()
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_delay_ms=total_delay_ms,
                )
            except Exception as e:
                last_error = e

                if attempts <= self.config.max_retries and self.should_retry(e):
                    delay_ms = self.calculate_delay(attempts)
                    total_delay_ms += delay_ms

                    if on_retry:
                        on_retry(attempts, e, delay_ms)

                    time.sleep(delay_ms / 1000.0)
                else:
                    break

        return RetryResult(
            success=False,
            error=last_error,
            attempts=attempts,
            total_delay_ms=total_delay_ms,
        )

    async def execute_async(
        self,
        operation: Callable[[], Any],
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> RetryResult:
        """Execute an async operation with retry logic.

        Args:
            operation: Async callable to execute.
            on_retry: Optional callback for retry events.

        Returns:
            RetryResult with success status and result/error.
        """
        import asyncio

        attempts = 0
        total_delay_ms = 0.0
        last_error: Exception | None = None

        while attempts < self.config.max_retries + 1:
            attempts += 1
            try:
                result = await operation()
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_delay_ms=total_delay_ms,
                )
            except Exception as e:
                last_error = e

                if attempts <= self.config.max_retries and self.should_retry(e):
                    delay = self.config.base_delay_ms * (
                        self.config.exponential_base ** (attempts - 1)
                    )
                    delay = min(delay, self.config.max_delay_ms)
                    jitter = secrets.randbelow(self.config.jitter_ms + 1)
                    delay_ms = delay + jitter
                    total_delay_ms += delay_ms

                    if on_retry:
                        on_retry(attempts, e, delay_ms)

                    await asyncio.sleep(delay_ms / 1000.0)
                else:
                    break

        return RetryResult(
            success=False,
            error=last_error,
            attempts=attempts,
            total_delay_ms=total_delay_ms,
        )


@dataclass
class RetryableErrors:
    """Common sets of retryable errors for convenience."""

    TRANSIENT: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,
        ),
        repr=False,
    )

    FILE_IO: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            FileNotFoundError,
            PermissionError,
            IOError,
            OSError,
        ),
        repr=False,
    )

    ALL: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,),
        repr=False,
    )

    NONE: tuple[type[Exception], ...] = field(
        default_factory=lambda: (),
        repr=False,
    )
