"""Tenacity retry configuration for Alpaca API calls."""

from __future__ import annotations

import logging

import httpx
import requests as _requests
import structlog
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger()


def _is_retryable(exception: BaseException) -> bool:
    try:
        from alpaca.common.exceptions import APIError

        if isinstance(exception, APIError):
            code = getattr(exception, "status_code", None)
            return code is None or code >= 500
    except ImportError:
        pass

    if isinstance(exception, httpx.HTTPStatusError):
        return int(exception.response.status_code) >= 500

    if isinstance(exception, _requests.exceptions.HTTPError):
        resp = getattr(exception, "response", None)
        if resp is not None:
            return int(resp.status_code) >= 500
        return True

    for name in type(exception).__mro__:
        cls_name = name.__name__
        if cls_name in ("ConnectError", "TimeoutException", "RemoteProtocolError", "NetworkError"):
            return True

    return False


def al_api_retry(
    max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 30.0, backoff: float = 2.0
):
    """Return a Tenacity retry decorator configured for Alpaca API calls.

    Retries on 5xx server errors and transient network failures.
    Does NOT retry on 4xx client errors (bad auth, invalid symbol, etc.).

    Args:
        max_attempts: Maximum number of attempts including the first call.
        min_wait: Minimum wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.
        backoff: Exponential backoff multiplier.

    Returns:
        A Tenacity retry decorator.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff, min=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
