"""Read-only local usage and indicative compute cost snapshots (never invoices)."""

import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DeploymentUsage(BaseModel):
    """Logical SDK calls observed since tracking started, including interrupted calls."""

    model_config = ConfigDict(frozen=True)

    tracking_started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requests: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    cancellations: int = Field(default=0, ge=0)
    total_latency_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    request_body_bytes: int = Field(default=0, ge=0)
    response_body_bytes: int = Field(default=0, ge=0)
    incomplete_transfers: int = Field(default=0, ge=0)

    @property
    def unfinished_requests(self) -> int:
        """Calls in flight or interrupted without a recorded outcome; never inferred failures."""
        return self.requests - self.successes - self.failures - self.cancellations

    @property
    def average_latency_seconds(self) -> float | None:
        finished = self.successes + self.failures + self.cancellations
        return self.total_latency_seconds / finished if finished else None


class ComputePricing(BaseModel):
    """Catalog quote for the requested GPU configuration, not a confirmed provider rate.

    The hourly rate is for the entire offered GPU configuration, not per GPU. Arbitrary
    catalog metadata is deliberately excluded from this persistent snapshot.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    gpu_type: str | None = None
    gpu_count: int = Field(ge=1)
    instance_type: str | None = None
    region: str | None = None
    price_per_hour_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    price_kind: Literal["on_demand", "spot", "unknown"] = "unknown"
    billing_granularity_seconds: int | None = Field(default=None, ge=1)
    source: Literal["catalog", "unavailable"] = "unavailable"
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CostEstimate(BaseModel):
    """Compute-only scenario; measured provider consumption and billing stay unavailable."""

    model_config = ConfigDict(frozen=True)

    currency: Literal["USD"] = "USD"
    estimated_cost_usd: float | None = None
    compute_seconds: float | None = None
    estimated_billable_seconds: float | None = None
    duration_source: Literal["unavailable", "caller_supplied", "dry_run"] = "unavailable"
    measured_gpu_seconds: float | None = None
    actual_billed_cost_usd: float | None = None
    unavailable_reasons: tuple[str, ...] = ()


class UsageReport(BaseModel):
    """Detached snapshot returned by SDK get_usage() and Deployment.usage()."""

    model_config = ConfigDict(frozen=True)

    deployment_id: str
    usage: DeploymentUsage | None
    pricing: ComputePricing | None
    cost: CostEstimate
    resource_status: Literal[
        "unknown", "not_provisioned", "retained_or_unknown", "down_acknowledged", "confirmed_absent"
    ] = "unknown"


def estimate_cost(
    pricing: ComputePricing | None, compute_seconds: float | None, *,
    dry_run: bool = False, uncertain: bool = False,
) -> CostEstimate:
    """Estimates only an explicitly supplied duration; never derives it from lifecycle/latency."""
    if compute_seconds is not None and (not math.isfinite(compute_seconds) or compute_seconds < 0):
        raise ValueError("compute_seconds must be finite and nonnegative.")
    if dry_run:
        return CostEstimate(
            estimated_cost_usd=0, compute_seconds=0, estimated_billable_seconds=0,
            duration_source="dry_run",
        )
    reasons = []
    if pricing is None or pricing.price_per_hour_usd is None or pricing.billing_granularity_seconds is None:
        reasons.append("pricing_unavailable")
    if compute_seconds is None:
        reasons.append("provider_compute_duration_unavailable")
    if uncertain:
        reasons.append("deployment_outcome_uncertain")
    if reasons:
        return CostEstimate(
            compute_seconds=compute_seconds,
            duration_source="caller_supplied" if compute_seconds is not None else "unavailable",
            unavailable_reasons=tuple(reasons),
        )
    assert pricing is not None and pricing.price_per_hour_usd is not None
    assert pricing.billing_granularity_seconds is not None and compute_seconds is not None
    increment = pricing.billing_granularity_seconds
    seconds = math.ceil(compute_seconds / increment) * increment
    amount = (seconds / 3600) * pricing.price_per_hour_usd
    if not math.isfinite(amount):
        raise ValueError("Computed estimate is too large.")
    return CostEstimate(
        estimated_cost_usd=amount, compute_seconds=compute_seconds,
        estimated_billable_seconds=seconds, duration_source="caller_supplied",
    )
