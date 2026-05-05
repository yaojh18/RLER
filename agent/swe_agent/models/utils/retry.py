"""Retry utility for model queries."""

import logging
import os

from tenacity import AsyncRetrying, Retrying, before_sleep_log, retry_if_not_exception_type, stop_after_attempt, wait_exponential


def _retry_attempts(model_name: str | None) -> int:
    attempts = int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", 2))
    if model_name:
        normalized = model_name.lower()
        if normalized.startswith(("openai/", "gpt-")):
            attempts = max(attempts, 4)
    return attempts


def retry(
    *,
    logger: logging.Logger,
    abort_exceptions: list[type[Exception]],
    model_name: str | None = None,
    async_retry: bool = False,
) -> Retrying | AsyncRetrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.

    Returns:
        A tenacity.Retrying object.
    """
    retry_class = AsyncRetrying if async_retry else Retrying
    return retry_class(
        reraise=True,
        stop=stop_after_attempt(_retry_attempts(model_name)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
