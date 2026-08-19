"""Unit tests for Tenacity retry configuration."""

from __future__ import annotations

import httpx
from tenacity import RetryError

from src.execution.retry import al_api_retry


class TestAlApiRetry:
    def test_returns_retry_decorator(self) -> None:
        decorator = al_api_retry()
        assert callable(decorator)

    def test_default_max_attempts_is_three(self) -> None:
        call_count = 0

        @al_api_retry(max_attempts=3, min_wait=0.01, max_wait=0.05)
        async def failing_call() -> int:
            nonlocal call_count
            call_count += 1
            msg = f"Attempt {call_count}"
            raise httpx.ConnectError(msg)

        import asyncio

        try:
            asyncio.run(failing_call())
        except (RetryError, httpx.ConnectError):
            pass
        assert call_count == 3

    def test_does_not_retry_on_client_error(self) -> None:
        call_count = 0

        response = httpx.Response(400, request=httpx.Request("GET", "http://test"))

        @al_api_retry(max_attempts=3, min_wait=0.01, max_wait=0.05)
        async def failing_call() -> int:
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Bad Request", request=httpx.Request("GET", "http://test"), response=response
            )

        import asyncio

        try:
            asyncio.run(failing_call())
        except httpx.HTTPStatusError:
            pass
        assert call_count == 1

    def test_retries_on_server_error(self) -> None:
        call_count = 0

        response = httpx.Response(503, request=httpx.Request("GET", "http://test"))

        @al_api_retry(max_attempts=3, min_wait=0.01, max_wait=0.05)
        async def failing_call() -> int:
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Server Error", request=httpx.Request("GET", "http://test"), response=response
            )

        import asyncio

        try:
            asyncio.run(failing_call())
        except (RetryError, httpx.HTTPStatusError):
            pass
        assert call_count == 3

    def test_retries_on_timeout(self) -> None:
        call_count = 0

        @al_api_retry(max_attempts=2, min_wait=0.01, max_wait=0.05)
        async def failing_call() -> int:
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("Timeout")

        import asyncio

        try:
            asyncio.run(failing_call())
        except (RetryError, httpx.TimeoutException):
            pass
        assert call_count == 2

    def test_supports_custom_parameters(self) -> None:
        decorator = al_api_retry(max_attempts=5, min_wait=2.0, max_wait=60.0, backoff=3.0)
        assert callable(decorator)
