"""Provider-neutral classification of cloud control-plane failures."""

import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Any


class FailureKind(str, Enum):
    """Why a provider operation failed; drives account health, failover and retries."""

    AUTH = "auth"
    """Credentials are confirmed invalid or revoked (e.g. HTTP 401)."""
    PERMISSION = "permission"
    """Credentials are valid but not allowed to do this (e.g. HTTP 403, missing entitlement)."""
    QUOTA = "quota"
    """Account quota, spend limit or billing balance is exhausted (e.g. HTTP 402)."""
    RATE_LIMIT = "rate_limit"
    """The control plane throttled the account (e.g. HTTP 429)."""
    CAPACITY = "capacity"
    """The requested GPU is out of stock for this account/region; not a credential fault."""
    INVALID_REQUEST = "invalid_request"
    """The request itself is wrong (bad GPU/model combination, HTTP 400/422); retrying elsewhere
    would fail the same way."""
    TRANSIENT = "transient"
    """Network errors, timeouts, HTTP 5xx: the request may or may not have taken effect."""

    @property
    def allows_failover(self) -> bool:
        """Whether another account may succeed where this one failed."""
        return self is not FailureKind.INVALID_REQUEST

    @property
    def is_definitive_rejection(self) -> bool:
        """The provider refused the request outright, so no resource can have been created."""
        return self in {
            FailureKind.AUTH,
            FailureKind.PERMISSION,
            FailureKind.QUOTA,
            FailureKind.RATE_LIMIT,
            FailureKind.INVALID_REQUEST,
        }


def kind_from_status(status: int | None) -> FailureKind:
    """Maps an HTTP status code to a failure kind (unknown statuses are transient)."""
    if status == 401:
        return FailureKind.AUTH
    if status == 403:
        return FailureKind.PERMISSION
    if status == 402:
        return FailureKind.QUOTA
    if status == 429:
        return FailureKind.RATE_LIMIT
    if status in (400, 404, 409, 422):
        return FailureKind.INVALID_REQUEST
    return FailureKind.TRANSIENT


def retry_after_from_error(error: Any) -> float | None:
    """Read HTTP numeric/date retry hints without rendering SDK exceptions."""
    headers = getattr(error, "headers", None) or getattr(getattr(error, "response", None), "headers", None)
    value = getattr(error, "retry_after", None)
    if value is None and headers is not None:
        value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        try:
            seconds = float(value)
        except ValueError:
            seconds = (parsedate_to_datetime(str(value)) - datetime.now(UTC)).total_seconds()
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (TypeError, ValueError, OverflowError):
        return None
