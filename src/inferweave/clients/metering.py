"""Per-logical-call measurements, isolated across concurrent tasks and shared transports."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class TransferMeasurement:
    # Submitted request bodies and fully buffered decoded response bodies, not wire bytes.
    request_body_bytes: int = 0
    response_body_bytes: int = 0
    incomplete: bool = False


current_transfer: ContextVar[TransferMeasurement | None] = ContextVar("inference_transfer", default=None)


@dataclass(frozen=True)
class UsageDelta:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    cancellations: int = 0
    total_latency_seconds: float = 0
    request_body_bytes: int = 0
    response_body_bytes: int = 0
    incomplete_transfers: int = 0
