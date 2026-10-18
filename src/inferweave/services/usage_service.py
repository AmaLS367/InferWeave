"""Atomic local request metering and conservative catalog cost scenarios."""

import math
from collections.abc import Iterable
from dataclasses import asdict
from typing import Literal

from inferweave.accounts.manager import AccountManager
from inferweave.clients.metering import UsageDelta
from inferweave.core.credentials import known_secret_values, sanitize_text
from inferweave.core.exceptions import DeploymentNotFoundError
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.models.usage import (
    ComputePricing,
    DeploymentUsage,
    UsageReport,
    estimate_cost,
)
from inferweave.ports.deployment_repository import DeploymentRepositoryPort


def capture_pricing(
    request: DeploymentRequest, profile: ModelProfile, provider: str, *, uses_spot: bool,
    secrets: Iterable[str] = (),
) -> ComputePricing:
    """Snapshot this candidate's quote only if it matches the actual requested configuration."""
    count = request.num_gpus or profile.hardware.gpu_count
    gpu = request.gpu_type
    secret_values = known_secret_values(secrets)
    quote = ComputePricing(
        provider=sanitize_text(provider, secret_values),
        gpu_type=sanitize_text(gpu, secret_values) if gpu else None, gpu_count=count,
    )
    offer = request._selected_offer
    if (
        offer is None or gpu is None or offer.provider.lower() != provider.lower()
        or offer.gpu_spec.name.lower() != gpu.lower() or offer.gpu_count != count
    ):
        return quote
    regions = request.options.provider.preferred_regions if request.options else []
    if offer.region and (not regions or regions[0] != offer.region):
        return quote
    rate = offer.spot_price_per_hour if uses_spot else offer.price_per_hour
    if rate is not None and not math.isfinite(rate):
        rate = None
    return quote.model_copy(update={
        "instance_type": sanitize_text(offer.instance_type, secret_values),
        "region": sanitize_text(offer.region, secret_values) if offer.region else None,
        "price_per_hour_usd": rate,
        "price_kind": "spot" if uses_spot else "on_demand",
        "billing_granularity_seconds": offer.billing_granularity_sec,
        "source": "catalog",
    })


class UsageService:
    def __init__(self, repository: DeploymentRepositoryPort, accounts: AccountManager) -> None:
        self._repository = repository
        self._accounts = accounts

    async def record(self, deployment_id: str, delta: UsageDelta) -> None:
        def add(record: DeploymentRecord) -> None:
            usage = record.usage or DeploymentUsage()
            record.usage = usage.model_copy(update={
                key: getattr(usage, key) + value for key, value in asdict(delta).items()
            })
        if await self._repository.update(deployment_id, add) is None:
            raise DeploymentNotFoundError(deployment_id)

    async def report(self, deployment_id: str, compute_seconds: float | None = None) -> UsageReport:
        record = await self._repository.get(deployment_id)
        if record is None:
            raise DeploymentNotFoundError(deployment_id)
        if not record.is_dry_run:
            self._accounts.resolve_owner(record)
        record = record.to_sanitized_record(self._accounts.secret_values())
        uncertain = record.needs_reconciliation or record.creation_may_continue
        resource_status: Literal[
            "unknown", "not_provisioned", "retained_or_unknown", "down_acknowledged", "confirmed_absent"
        ] = "unknown"
        if record.is_dry_run:
            resource_status = "not_provisioned"
        elif not uncertain:
            if record.resources_confirmed_absent_at is not None:
                resource_status = "confirmed_absent"
            elif record.last_stop_action == AutostopAction.DOWN:
                resource_status = "down_acknowledged"
            elif record.last_stop_action == AutostopAction.STOP:
                resource_status = "retained_or_unknown"
        return UsageReport(
            deployment_id=record.id,
            usage=record.usage if self._repository.supports_atomic_updates else None,
            pricing=record.pricing,
            cost=estimate_cost(
                record.pricing, compute_seconds, dry_run=record.is_dry_run, uncertain=uncertain,
            ),
            resource_status=resource_status,
        )
