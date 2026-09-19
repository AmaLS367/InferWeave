"""Domain models, value objects, and evaluation logic for endpoint health and readiness probing."""

import logging
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProbeOutcome(str, Enum):
    """Specific outcome classification of a single health probe."""

    SUCCESS = "success"
    HTTP_ERROR = "http_error"
    CONNECTION_REFUSED = "connection_refused"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


class ReadinessState(str, Enum):
    """Overall readiness state across multiple healthcheck attempts."""

    PENDING = "pending"
    POLLING = "polling"
    READY = "ready"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class ProbeResult(BaseModel):
    """Immutable snapshot of a single health probe attempt."""

    is_healthy: bool = Field(..., description="Whether the probe passed healthy criteria")
    outcome: ProbeOutcome = Field(..., description="Categorical outcome of the probe attempt")
    status_code: int | None = Field(default=None, description="HTTP status code if reachable")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Round-trip latency in milliseconds")
    error_message: str | None = Field(default=None, description="Diagnostic error detail if probe failed")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the probe completed",
    )


class ReadinessReport(BaseModel):
    """Aggregated status of an ongoing or completed readiness polling session."""

    state: ReadinessState = Field(
        default=ReadinessState.PENDING, description="Current readiness evaluation state"
    )
    total_probes: int = Field(default=0, ge=0, description="Total number of probes executed")
    consecutive_successes: int = Field(
        default=0, ge=0, description="Current number of consecutive successful probes"
    )
    elapsed_seconds: float = Field(
        default=0.0, ge=0.0, description="Total elapsed polling duration in seconds"
    )
    last_probe: ProbeResult | None = Field(
        default=None, description="Most recent probe result"
    )
    history: list[ProbeResult] = Field(
        default_factory=list, description="Historical sequence of probe results"
    )


class HealthEvaluator:
    """Domain service evaluating probe outcomes and determining endpoint readiness.

    Pure business rules with no network or framework dependencies.
    """

    def evaluate_probe(
        self,
        status_code: int | None = None,
        latency_ms: float = 0.0,
        error: Exception | str | None = None,
        expected_status_codes: list[int] | None = None,
    ) -> ProbeResult:
        """Evaluates an HTTP response or network error into a standardized ProbeResult."""
        acceptable_codes = expected_status_codes or [200]

        if error is not None:
            err_type = (
                type(error).__name__.lower() if isinstance(error, Exception) else ""
            )
            err_str = f"{err_type} {error!s}".lower()
            error_msg = str(error)

            if "connect" in err_str or "refused" in err_str:
                outcome = ProbeOutcome.CONNECTION_REFUSED
            elif "timeout" in err_str or "timed out" in err_str:
                outcome = ProbeOutcome.TIMEOUT
            else:
                outcome = ProbeOutcome.UNKNOWN_ERROR

            return ProbeResult(
                is_healthy=False,
                outcome=outcome,
                status_code=status_code,
                latency_ms=max(0.0, latency_ms),
                error_message=error_msg,
            )

        if status_code is not None:
            if status_code in acceptable_codes:
                return ProbeResult(
                    is_healthy=True,
                    outcome=ProbeOutcome.SUCCESS,
                    status_code=status_code,
                    latency_ms=max(0.0, latency_ms),
                    error_message=None,
                )
            return ProbeResult(
                is_healthy=False,
                outcome=ProbeOutcome.HTTP_ERROR,
                status_code=status_code,
                latency_ms=max(0.0, latency_ms),
                error_message=f"Received HTTP status {status_code}, expected one of {acceptable_codes}",
            )

        return ProbeResult(
            is_healthy=False,
            outcome=ProbeOutcome.UNKNOWN_ERROR,
            status_code=None,
            latency_ms=max(0.0, latency_ms),
            error_message="Probe completed with no status code and no explicit error",
        )

    def evaluate_readiness(
        self,
        history: list[ProbeResult],
        required_consecutive_successes: int = 1,
    ) -> tuple[bool, int]:
        """Calculates consecutive healthy probes and determines readiness satisfaction.

        Returns:
            tuple[bool, int]: (is_ready, consecutive_successes_count)
        """
        if not history:
            return False, 0

        consecutive = 0
        for probe in reversed(history):
            if probe.is_healthy:
                consecutive += 1
            else:
                break

        is_ready = consecutive >= max(1, required_consecutive_successes)
        return is_ready, consecutive
